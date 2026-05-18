from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from conditioning import ConditionEmbeddingLayer
from optical.models.multiscale import (
    _DepthwisePointwiseBlock,
    _build_hidden_channels,
    _build_positional_timestep_embedding,
    _build_spatial_stage_sizes,
    _init_linear_or_conv,
    _interpolate_like,
    OpticalPrefixReadoutDecoder,
)


class IterativeMultiscaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        output_height: int,
        output_width: int,
        num_steps: int,
        step_embedding_dim: int,
        condition_layer: ConditionEmbeddingLayer | None = None,
        condition_embed_dim: int | None = None,
        latent_stage_channels: Sequence[int] | None = None,
        prev_image_channels: Sequence[int] | None = None,
        use_prev_image: bool = True,
        fusion_hidden_dim: int = 128,
        weight_init: str = "kaiming_uniform",
        output_weight_init: str = "xavier_uniform",
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.num_steps = int(num_steps)
        self.step_embedding_dim = int(step_embedding_dim)
        self.condition_layer = condition_layer
        self.condition_embed_dim = None if condition_embed_dim is None else int(condition_embed_dim)
        self.use_prev_image = bool(use_prev_image)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.weight_init = str(weight_init)
        self.output_weight_init = str(output_weight_init)
        self.upsample_mode = str(upsample_mode)

        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if self.latent_height <= 0 or self.latent_width <= 0:
            raise ValueError("latent_height and latent_width must be positive")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_height and output_width must be positive")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.step_embedding_dim <= 0:
            raise ValueError("step_embedding_dim must be positive")
        if self.condition_layer is None and self.condition_embed_dim is not None:
            raise ValueError("condition_embed_dim must be omitted when condition_layer is disabled")
        if self.condition_layer is not None and (self.condition_embed_dim is None or self.condition_embed_dim <= 0):
            raise ValueError("condition_embed_dim must be positive when condition_layer is enabled")

        self.stage_sizes = _build_spatial_stage_sizes(
            input_height=self.latent_height,
            input_width=self.latent_width,
            output_height=self.output_height,
            output_width=self.output_width,
        )
        self.latent_stage_channels = _build_hidden_channels(
            stage_count=len(self.stage_sizes),
            hidden_dim=max(self.fusion_hidden_dim, 64),
            hidden_channels=latent_stage_channels,
        )
        if prev_image_channels is None:
            prev_channels = (32, 24, 16)
        else:
            prev_channels = tuple(int(value) for value in prev_image_channels)
        if not prev_channels or any(value <= 0 for value in prev_channels):
            raise ValueError("prev_image_channels must contain positive integers")
        self.prev_image_channels = prev_channels
        prev_feature_channels = list(reversed(self.prev_image_channels))
        while len(prev_feature_channels) < len(self.stage_sizes):
            prev_feature_channels.insert(0, prev_feature_channels[0])
        self.prev_feature_channels = tuple(prev_feature_channels[: len(self.stage_sizes)])

        self.latent_stem = nn.Conv2d(self.latent_channels, self.latent_stage_channels[0], kernel_size=1, bias=True)
        if self.use_prev_image:
            self.prev_stem = nn.Conv2d(1, self.prev_image_channels[0], kernel_size=3, stride=1, padding=1, bias=True)
            prev_blocks: list[nn.Module] = []
            for in_channels, out_channels in zip(self.prev_image_channels[:-1], self.prev_image_channels[1:]):
                prev_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.GroupNorm(num_groups=1, num_channels=out_channels),
                        nn.SiLU(),
                    )
                )
            self.prev_blocks = nn.ModuleList(prev_blocks)
            self.prev_projectors = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.prev_feature_channels[index],
                        self.latent_stage_channels[index],
                        kernel_size=1,
                        bias=True,
                    )
                    for index in range(len(self.stage_sizes))
                ]
            )
        else:
            self.prev_stem = None
            self.prev_blocks = None
            self.prev_projectors = None
        self.stage_projectors = nn.ModuleList(
            [
                nn.Identity()
                if index == 0 or self.latent_stage_channels[index - 1] == self.latent_stage_channels[index]
                else nn.Conv2d(
                    self.latent_stage_channels[index - 1],
                    self.latent_stage_channels[index],
                    kernel_size=1,
                    bias=True,
                )
                for index in range(len(self.stage_sizes))
            ]
        )

        conditioning_input_dim = self.step_embedding_dim + (
            0 if self.condition_embed_dim is None else self.condition_embed_dim
        )
        self.conditioning_projector = nn.Sequential(
            nn.Linear(conditioning_input_dim, self.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _DepthwisePointwiseBlock(
                    in_channels=self.latent_stage_channels[index],
                    out_channels=self.latent_stage_channels[index],
                    condition_dim=self.fusion_hidden_dim,
                )
                for index in range(len(self.stage_sizes))
            ]
        )
        self.head = nn.Conv2d(self.latent_stage_channels[-1], 1, kernel_size=1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _init_linear_or_conv(self.latent_stem, self.weight_init)
        if self.prev_stem is not None:
            _init_linear_or_conv(self.prev_stem, self.weight_init)
        if self.prev_blocks is not None:
            for block in self.prev_blocks:
                conv = block[0]
                _init_linear_or_conv(conv, self.weight_init)
        if self.prev_projectors is not None:
            for projector in self.prev_projectors:
                _init_linear_or_conv(projector, self.weight_init)
        for projector in self.stage_projectors:
            if isinstance(projector, nn.Conv2d):
                _init_linear_or_conv(projector, self.weight_init)
        for module in self.conditioning_projector:
            if isinstance(module, nn.Linear):
                _init_linear_or_conv(module, self.weight_init)
        for block in self.blocks:
            _init_linear_or_conv(block.depthwise, self.weight_init)
            _init_linear_or_conv(block.pointwise, self.weight_init)
            if block.modulation is not None:
                _init_linear_or_conv(block.modulation.projector, self.weight_init)
        _init_linear_or_conv(self.head, self.output_weight_init)
        if self.condition_layer is not None:
            embedding = getattr(self.condition_layer, "embedding", None)
            if isinstance(embedding, nn.Embedding):
                nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            projector = getattr(self.condition_layer, "projector", None)
            if isinstance(projector, nn.Linear):
                _init_linear_or_conv(projector, self.weight_init)
            elif isinstance(projector, nn.Sequential):
                for module in projector:
                    if isinstance(module, nn.Linear):
                        _init_linear_or_conv(module, self.weight_init)

    def _encode_prev_image(self, prev_image: torch.Tensor) -> list[torch.Tensor]:
        if not self.use_prev_image or self.prev_stem is None or self.prev_blocks is None:
            raise RuntimeError("_encode_prev_image should not be called when use_prev_image is disabled")
        if prev_image.dim() != 4 or int(prev_image.shape[1]) != 1:
            raise ValueError(f"prev_image must be [B,1,H,W], got {tuple(prev_image.shape)}")
        hidden = self.prev_stem(prev_image.to(dtype=torch.float32))
        features: list[torch.Tensor] = [hidden]
        for block in self.prev_blocks:
            hidden = F.avg_pool2d(hidden, kernel_size=2, stride=2)
            hidden = block(hidden)
            features.append(hidden)
        low_to_high_features = list(reversed(features))
        while len(low_to_high_features) < len(self.stage_sizes):
            low_to_high_features.insert(0, low_to_high_features[0])
        return low_to_high_features[: len(self.stage_sizes)]

    def encode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 4 or int(latent.shape[1]) != self.latent_channels:
            raise ValueError(
                f"latent must be [B,{self.latent_channels},H,W], got {tuple(latent.shape)}"
            )
        if tuple(latent.shape[-2:]) != (self.latent_height, self.latent_width):
            raise ValueError(
                f"latent spatial shape must be {(self.latent_height, self.latent_width)}, "
                f"got {tuple(latent.shape[-2:])}"
            )
        return self.latent_stem(latent.to(dtype=torch.float32))

    def encode_condition_base(
        self,
        *,
        class_labels: torch.Tensor | None,
        condition: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.condition_layer is None:
            return None
        resolved_condition = class_labels if condition is None else condition
        if resolved_condition is None:
            raise ValueError("condition must be provided when conditioning is enabled")
        return self.condition_layer(resolved_condition.to(device=device))

    def _build_conditioning_repr_from_base(
        self,
        *,
        batch_size: int,
        device: torch.device,
        timesteps: torch.Tensor | int,
        condition_base: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(timesteps, int):
            timestep_tensor = torch.full((batch_size,), int(timesteps), device=device, dtype=torch.long)
        else:
            timestep_tensor = timesteps.to(device=device, dtype=torch.long).reshape(-1)
        if int(timestep_tensor.shape[0]) != batch_size:
            raise ValueError("timestep batch size must match image batch size")
        timestep_repr = _build_positional_timestep_embedding(timestep_tensor, self.step_embedding_dim)

        if self.condition_layer is None:
            fused_input = timestep_repr
        else:
            if condition_base is None:
                raise ValueError("condition_base must be provided when conditioning is enabled")
            fused_input = torch.cat([condition_base, timestep_repr], dim=1)
        return self.conditioning_projector(fused_input.to(dtype=torch.float32))

    def forward_from_encoded(
        self,
        *,
        prev_image: torch.Tensor,
        latent_base: torch.Tensor,
        timesteps: torch.Tensor | int,
        condition_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent_base.dim() != 4 or int(latent_base.shape[1]) != self.latent_stage_channels[0]:
            raise ValueError(
                "latent_base must be the encoded latent stem output, "
                f"got {tuple(latent_base.shape)}"
            )
        batch_size = int(latent_base.shape[0])
        prev_features = self._encode_prev_image(prev_image) if self.use_prev_image else None
        conditioning_repr = self._build_conditioning_repr_from_base(
            batch_size=batch_size,
            device=latent_base.device,
            timesteps=timesteps,
            condition_base=condition_base,
        )

        hidden = latent_base
        for index, (target_size, block) in enumerate(zip(self.stage_sizes, self.blocks)):
            hidden = _interpolate_like(hidden, size=target_size, mode=self.upsample_mode)
            hidden = self.stage_projectors[index](hidden)
            if prev_features is not None and self.prev_projectors is not None:
                prev_feature = _interpolate_like(prev_features[index], size=target_size, mode=self.upsample_mode)
                hidden = hidden + self.prev_projectors[index](prev_feature)
            hidden = block(hidden, conditioning_repr)
        return self.head(hidden)

    def forward(
        self,
        *,
        prev_image: torch.Tensor,
        latent: torch.Tensor,
        timesteps: torch.Tensor | int,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent_base = self.encode_latent(latent)
        condition_base = self.encode_condition_base(
            class_labels=class_labels,
            condition=condition,
            device=latent.device,
        )
        return self.forward_from_encoded(
            prev_image=prev_image,
            latent_base=latent_base,
            timesteps=timesteps,
            condition_base=condition_base,
        )


class IterativeOpticalDecoder(nn.Module):
    def __init__(self, *, optical_decoder: OpticalPrefixReadoutDecoder) -> None:
        super().__init__()
        self.optical_decoder = optical_decoder

    @property
    def slm_input_height(self) -> int:
        return int(self.optical_decoder.slm_input_height)

    @property
    def slm_input_width(self) -> int:
        return int(self.optical_decoder.slm_input_width)

    def forward(
        self,
        control_map: torch.Tensor,
        *,
        error_factor: float | None = None,
    ) -> dict[str, torch.Tensor]:
        optical_output = self.optical_decoder(control_map, error_factor=error_factor)
        final_detector = optical_output["final_detector"]
        if not isinstance(final_detector, torch.Tensor):
            raise TypeError("optical decoder final_detector must be a tensor")
        return {
            "final_detector": final_detector,
        }


class IterativeMultiscaleOpticalModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: IterativeMultiscaleEncoder,
        decoder: IterativeOpticalDecoder,
        state_normalization: str = "mean_power",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.state_normalization = str(state_normalization)
        if self.state_normalization != "mean_power":
            raise ValueError(
                "state_normalization must be 'mean_power' for the first implementation, "
                f"got {self.state_normalization!r}"
            )

    @staticmethod
    def _to_image(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 4 and int(image.shape[1]) == 1:
            return image
        if image.dim() == 3:
            return image.unsqueeze(1)
        raise ValueError(f"state image must be [B,1,H,W] or [B,H,W], got {tuple(image.shape)}")

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        image = self._to_image(state).to(dtype=torch.float32)
        mean_power = image.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
        return image / mean_power

    def forward_step(
        self,
        *,
        prev_image: torch.Tensor,
        latent: torch.Tensor,
        step_id: torch.Tensor | int,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        error_factor: float | None = None,
    ) -> dict[str, torch.Tensor]:
        prev_image = self._to_image(prev_image)
        latent_base = self.encoder.encode_latent(latent)
        condition_base = self.encoder.encode_condition_base(
            class_labels=class_labels,
            condition=condition,
            device=latent.device,
        )
        control_map = self.encoder.forward_from_encoded(
            prev_image=prev_image,
            latent_base=latent_base,
            timesteps=step_id,
            condition_base=condition_base,
        )
        prediction = self.decoder(control_map, error_factor=error_factor)["final_detector"]
        return {
            "control_map": control_map,
            "final_detector": prediction,
        }

    def forward(
        self,
        *,
        latent: torch.Tensor,
        num_steps: int,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        error_factor: float | None = None,
        detach_prev_state: bool = False,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        batch_size = int(latent.shape[0])
        if initial_state is None:
            prev_state = torch.zeros(
                (batch_size, 1, self.encoder.output_height, self.encoder.output_width),
                device=latent.device,
                dtype=torch.float32,
            )
        else:
            prev_state = self._to_image(initial_state).to(device=latent.device, dtype=torch.float32)
        latent_base = self.encoder.encode_latent(latent)
        condition_base = self.encoder.encode_condition_base(
            class_labels=class_labels,
            condition=condition,
            device=latent.device,
        )

        predictions: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        control_maps: list[torch.Tensor] = []
        for step_index in range(int(num_steps)):
            control_map = self.encoder.forward_from_encoded(
                prev_image=prev_state,
                latent_base=latent_base,
                timesteps=step_index,
                condition_base=condition_base,
            )
            prediction = self.decoder(control_map, error_factor=error_factor)["final_detector"]
            normalized_state = self.normalize_state(prediction)
            predictions.append(prediction)
            states.append(normalized_state)
            control_maps.append(control_map)
            prev_state = normalized_state.detach() if detach_prev_state else normalized_state
        return {
            "predictions": tuple(predictions),
            "states": tuple(states),
            "control_maps": tuple(control_maps),
        }
