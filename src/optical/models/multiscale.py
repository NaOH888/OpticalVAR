from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from conditioning import ConditionEmbeddingLayer, ConditionalLatentFusion, LatentEmbeddingLayer
from optical.core import PropagateContext, PropagationConfig, PropagationErrorConfig
from optical.core.base import OpticLayer, SourceLayer
from optical.layers.detector import DetectorLayer


def _build_positional_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim!r}")
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.float().unsqueeze(1)

    exponent = -math.log(float(max_period)) * torch.arange(
        half_dim,
        device=timesteps.device,
        dtype=torch.float32,
    ) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
    if embedding_dim % 2 != 0:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionalPhaseSLMEncoder(nn.Module):
    """Digital encoder that maps latent noise and optional conditions to an SLM phase map."""

    def __init__(
        self,
        *,
        input_channels: int,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        hidden_dim: int = 512,
        phase_alpha_pi: float = 2.0,
        time_conditional: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: int = 128,
        class_conditional: bool = False,
        condition_mode: str | None = None,
        num_classes: int = 0,
        condition_input_dim: int | None = None,
        class_embed_dim: int = 128,
        class_condition_channels: int = 4,
        condition_hidden_dim: int | None = None,
        weight_init: str = "kaiming_uniform",
        output_weight_init: str = "xavier_uniform",
        embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.hidden_dim = int(hidden_dim)
        self.phase_alpha_pi = float(phase_alpha_pi)
        self.time_conditional = bool(time_conditional)
        self.time_embedding_type = str(time_embedding_type)
        self.time_embedding_dim = int(time_embedding_dim)
        self.class_conditional = bool(class_conditional)
        self.condition_mode = (
            "class_index" if self.class_conditional and condition_mode is None else condition_mode
        )
        self.num_classes = int(num_classes)
        self.condition_input_dim = None if condition_input_dim is None else int(condition_input_dim)
        self.class_embed_dim = int(class_embed_dim)
        self.class_condition_channels = int(class_condition_channels)
        self.condition_hidden_dim = (
            None if condition_hidden_dim is None else int(condition_hidden_dim)
        )
        self.weight_init = str(weight_init)
        self.output_weight_init = str(output_weight_init)
        self.embedding_init_std = float(embedding_init_std)

        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels!r}")
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input_height and input_width must be positive")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_height and output_width must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if self.time_conditional and self.time_embedding_type not in {"positional", "fourier"}:
            raise ValueError(
                "time_embedding_type must be 'positional' or 'fourier' when time_conditional=True, "
                f"got {time_embedding_type!r}"
            )
        if self.class_condition_channels <= 0 and self.condition_mode is not None:
            raise ValueError(
                "class_condition_channels must be positive when conditioning is enabled, "
                f"got {class_condition_channels!r}"
            )
        if self.condition_mode is None and self.num_classes != 0:
            raise ValueError("num_classes must be 0 when conditioning is disabled")
        if self.condition_mode == "class_index" and self.num_classes <= 0:
            raise ValueError("num_classes must be positive when condition_mode='class_index'")
        if self.condition_mode == "attribute_vector" and (
            self.condition_input_dim is None or self.condition_input_dim <= 0
        ):
            raise ValueError("condition_input_dim must be positive when condition_mode='attribute_vector'")

        self.condition_embedding = (
            ConditionEmbeddingLayer(
                mode=str(self.condition_mode),
                output_dim=self.class_condition_channels,
                num_classes=self.num_classes if self.condition_mode == "class_index" else None,
                input_dim=self.condition_input_dim if self.condition_mode == "attribute_vector" else None,
                embed_dim=self.class_embed_dim,
                hidden_dim=self.condition_hidden_dim if self.condition_hidden_dim is not None else self.hidden_dim,
            )
            if self.condition_mode is not None
            else None
        )
        if self.time_conditional and self.time_embedding_type == "fourier":
            half_dim = max(self.time_embedding_dim // 2, 1)
            fourier_weight = torch.randn((half_dim,), dtype=torch.float32)
            self.register_buffer("_fourier_time_weight", fourier_weight, persistent=True)
        else:
            self.register_buffer("_fourier_time_weight", torch.empty(0), persistent=False)

        conditioned_channels = self.input_channels + (
            self.class_condition_channels if self.condition_mode is not None else 0
        )
        feature_dim = conditioned_channels * self.input_height * self.input_width
        if self.time_conditional:
            feature_dim += self.time_embedding_dim

        output_dim = self.output_height * self.output_width
        self.fc1 = nn.Linear(feature_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, output_dim)
        self.act = nn.SiLU()
        self._reset_parameters()

    @property
    def phase_period_rad(self) -> float:
        return float(self.phase_alpha_pi * math.pi)

    def _init_linear(self, layer: nn.Linear, mode: str) -> None:
        if mode == "kaiming_uniform":
            init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif mode == "kaiming_normal":
            init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif mode == "xavier_uniform":
            init.xavier_uniform_(layer.weight)
        elif mode == "xavier_normal":
            init.xavier_normal_(layer.weight)
        else:
            raise ValueError(f"Unsupported encoder init mode: {mode!r}")
        if layer.bias is not None:
            init.zeros_(layer.bias)

    def _reset_parameters(self) -> None:
        self._init_linear(self.fc1, self.weight_init)
        self._init_linear(self.fc2, self.weight_init)
        self._init_linear(self.fc3, self.output_weight_init)
        if self.condition_embedding is not None:
            embedding = getattr(self.condition_embedding, "embedding", None)
            if isinstance(embedding, nn.Embedding):
                init.normal_(embedding.weight, mean=0.0, std=self.embedding_init_std)
            projector = getattr(self.condition_embedding, "projector", None)
            if isinstance(projector, nn.Linear):
                self._init_linear(projector, self.weight_init)
            elif isinstance(projector, nn.Sequential):
                for module in projector:
                    if isinstance(module, nn.Linear):
                        self._init_linear(module, self.weight_init)

    def _encode_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.time_embedding_type == "positional":
            return _build_positional_timestep_embedding(timesteps, self.time_embedding_dim)

        frequencies = self._fourier_time_weight.to(device=timesteps.device, dtype=torch.float32)
        arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0) * (2.0 * math.pi)
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
        if int(embedding.shape[1]) < self.time_embedding_dim:
            embedding = F.pad(embedding, (0, self.time_embedding_dim - int(embedding.shape[1])))
        return embedding[:, : self.time_embedding_dim]

    def forward(
        self,
        sample: torch.Tensor,
        *,
        timesteps: torch.Tensor | int | None = None,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sample.dim() != 4 or int(sample.shape[1]) != self.input_channels:
            raise ValueError(
                f"sample must be [B,{self.input_channels},H,W], got {tuple(sample.shape)}"
            )
        batch_size, _, height, width = sample.shape
        if height != self.input_height or width != self.input_width:
            raise ValueError(
                f"sample spatial shape must be {(self.input_height, self.input_width)}, "
                f"got {(height, width)}"
            )

        conditioned_sample = sample.to(dtype=torch.float32)
        features: list[torch.Tensor] = []
        if self.time_conditional:
            if timesteps is None:
                raise ValueError("timesteps must be provided when time_conditional=True")
            if isinstance(timesteps, int):
                timestep_tensor = torch.full((batch_size,), int(timesteps), device=sample.device, dtype=torch.long)
            else:
                timestep_tensor = timesteps.to(device=sample.device, dtype=torch.long).reshape(-1)
            if int(timestep_tensor.shape[0]) != batch_size:
                raise ValueError("timesteps batch size must match sample batch size")
            features.append(self._encode_timesteps(timestep_tensor).to(dtype=sample.dtype))
        elif timesteps is not None:
            raise ValueError("timesteps must not be provided when time_conditional=False")

        resolved_condition = class_labels if condition is None else condition
        if self.condition_mode is not None:
            if resolved_condition is None:
                raise ValueError("condition must be provided when conditioning is enabled")
            if self.condition_embedding is None:
                raise RuntimeError("condition embedding is not initialized")
            condition_embedding = self.condition_embedding(resolved_condition.to(device=sample.device))
            if int(condition_embedding.shape[0]) != batch_size:
                raise ValueError("condition batch size must match sample batch size")
            condition_map = condition_embedding.view(batch_size, self.class_condition_channels, 1, 1)
            condition_map = condition_map.expand(-1, -1, self.input_height, self.input_width)
            conditioned_sample = torch.cat((conditioned_sample, condition_map.to(dtype=conditioned_sample.dtype)), dim=1)
        elif class_labels is not None or condition is not None:
            raise ValueError("condition must not be provided when conditioning is disabled")

        features.insert(0, conditioned_sample.reshape(batch_size, -1))
        hidden = torch.cat(features, dim=1)
        hidden = self.act(self.fc1(hidden))
        hidden = self.act(self.fc2(hidden))
        raw_phase = self.fc3(hidden).reshape(batch_size, 1, self.output_height, self.output_width)
        return torch.remainder(raw_phase, self.phase_period_rad)


class PhaseMapEncoder(nn.Module):
    """Map a fused latent representation [B, D] to an SLM phase map."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_height: int,
        output_width: int,
        hidden_dim: int = 512,
        phase_alpha_pi: float = 2.0,
        weight_init: str = "kaiming_uniform",
        output_weight_init: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.hidden_dim = int(hidden_dim)
        self.phase_alpha_pi = float(phase_alpha_pi)
        self.weight_init = str(weight_init)
        self.output_weight_init = str(output_weight_init)

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim!r}")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_height and output_width must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")

        output_dim = self.output_height * self.output_width
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, output_dim)
        self.act = nn.SiLU()
        self._reset_parameters()

    @property
    def phase_period_rad(self) -> float:
        return float(self.phase_alpha_pi * math.pi)

    def _init_linear(self, layer: nn.Linear, mode: str) -> None:
        if mode == "kaiming_uniform":
            init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif mode == "kaiming_normal":
            init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif mode == "xavier_uniform":
            init.xavier_uniform_(layer.weight)
        elif mode == "xavier_normal":
            init.xavier_normal_(layer.weight)
        else:
            raise ValueError(f"Unsupported phase map encoder init mode: {mode!r}")
        if layer.bias is not None:
            init.zeros_(layer.bias)

    def _reset_parameters(self) -> None:
        self._init_linear(self.fc1, self.weight_init)
        self._init_linear(self.fc2, self.weight_init)
        self._init_linear(self.fc3, self.output_weight_init)

    def forward(self, fused_repr: torch.Tensor) -> torch.Tensor:
        if fused_repr.dim() != 2 or int(fused_repr.shape[1]) != self.input_dim:
            raise ValueError(
                f"fused_repr must be [B,{self.input_dim}], got {tuple(fused_repr.shape)}"
            )
        hidden = fused_repr.to(dtype=torch.float32)
        hidden = self.act(self.fc1(hidden))
        hidden = self.act(self.fc2(hidden))
        raw_phase = self.fc3(hidden).reshape(hidden.shape[0], 1, self.output_height, self.output_width)
        return torch.remainder(raw_phase, self.phase_period_rad)


class LatentPhaseMapEncoder(nn.Module):
    """latent -> latent embedding -> optional condition fusion -> phase map."""

    def __init__(
        self,
        *,
        latent_layer: LatentEmbeddingLayer,
        phase_map_encoder: PhaseMapEncoder,
        condition_layer: ConditionEmbeddingLayer | None = None,
        fusion_layer: ConditionalLatentFusion | None = None,
    ) -> None:
        super().__init__()
        self.latent_layer = latent_layer
        self.phase_map_encoder = phase_map_encoder
        self.condition_layer = condition_layer
        self.fusion_layer = fusion_layer

    @property
    def output_height(self) -> int:
        return self.phase_map_encoder.output_height

    @property
    def output_width(self) -> int:
        return self.phase_map_encoder.output_width

    @property
    def phase_period_rad(self) -> float:
        return self.phase_map_encoder.phase_period_rad

    def forward(
        self,
        sample: torch.Tensor,
        *,
        timesteps: torch.Tensor | int | None = None,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timesteps is not None:
            raise ValueError("LatentPhaseMapEncoder does not support timestep conditioning")
        latent_repr = self.latent_layer(sample)
        if self.condition_layer is None:
            if class_labels is not None or condition is not None:
                raise ValueError("condition must not be provided when conditioning is disabled")
            fused_repr = latent_repr
        else:
            resolved_condition = class_labels if condition is None else condition
            if resolved_condition is None:
                raise ValueError("condition must be provided when conditioning is enabled")
            if self.fusion_layer is None:
                raise RuntimeError("fusion_layer must be provided when conditioning is enabled")
            condition_repr = self.condition_layer(resolved_condition.to(device=latent_repr.device))
            fused_repr = self.fusion_layer(latent_repr, condition_repr)
        return self.phase_map_encoder(fused_repr)


class HierarchicalRVQPhaseMapEncoder(nn.Module):
    """使用分层 RVQ code 和条件向量共同生成多尺度相位分量，并融合成最终相位图。"""

    def __init__(
        self,
        *,
        num_codebooks: int,
        codebook_size: int,
        code_embed_dim: int,
        output_height: int,
        output_width: int,
        condition_layer: ConditionEmbeddingLayer | None = None,
        stage_hidden_dim: int = 512,
        stage_fusion_hidden_dim: int | None = None,
        upsample_mode: str = "bilinear",
        phase_alpha_pi: float = 2.0,
        output_weight_init: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.code_embed_dim = int(code_embed_dim)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.condition_layer = condition_layer
        self.stage_hidden_dim = int(stage_hidden_dim)
        self.stage_fusion_hidden_dim = int(
            stage_fusion_hidden_dim if stage_fusion_hidden_dim is not None else self.stage_hidden_dim
        )
        self.upsample_mode = str(upsample_mode)
        self.phase_alpha_pi = float(phase_alpha_pi)
        self.output_weight_init = str(output_weight_init)

        if self.num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {num_codebooks!r}")
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size!r}")
        if self.code_embed_dim <= 0:
            raise ValueError(f"code_embed_dim must be positive, got {code_embed_dim!r}")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_height and output_width must be positive")
        if self.stage_hidden_dim <= 0 or self.stage_fusion_hidden_dim <= 0:
            raise ValueError("stage hidden dimensions must be positive")

        self.stage_sizes = self._build_stage_sizes(
            output_height=self.output_height,
            output_width=self.output_width,
            num_codebooks=self.num_codebooks,
        )
        condition_dim = 0 if self.condition_layer is None else int(self.condition_layer.output_dim)

        self.code_embedding = nn.ModuleList(
            nn.Embedding(self.codebook_size, self.code_embed_dim) for _ in range(self.num_codebooks)
        )
        self.stage_condition_projectors = (
            None
            if condition_dim == 0
            else nn.ModuleList(
                nn.Sequential(
                    nn.Linear(condition_dim, self.stage_hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(self.num_codebooks)
            )
        )
        self.stage_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(
                    self.code_embed_dim + (self.stage_hidden_dim if condition_dim > 0 else 0),
                    self.stage_fusion_hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(self.stage_fusion_hidden_dim, stage_height * stage_width),
            )
            for stage_height, stage_width in self.stage_sizes
        )
        self._reset_parameters()

    @property
    def phase_period_rad(self) -> float:
        return float(self.phase_alpha_pi * math.pi)

    @staticmethod
    def _build_stage_sizes(
        *,
        output_height: int,
        output_width: int,
        num_codebooks: int,
    ) -> tuple[tuple[int, int], ...]:
        divisor = 2 ** max(num_codebooks - 1, 0)
        if output_height % divisor != 0 or output_width % divisor != 0:
            raise ValueError(
                "output_height and output_width must be divisible by 2**(num_codebooks-1), "
                f"got {(output_height, output_width)} with num_codebooks={num_codebooks}"
            )
        base_height = output_height // divisor
        base_width = output_width // divisor
        return tuple(
            (base_height * (2 ** index), base_width * (2 ** index))
            for index in range(num_codebooks)
        )

    def _init_linear(self, layer: nn.Linear, mode: str) -> None:
        if mode == "kaiming_uniform":
            init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif mode == "kaiming_normal":
            init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif mode == "xavier_uniform":
            init.xavier_uniform_(layer.weight)
        elif mode == "xavier_normal":
            init.xavier_normal_(layer.weight)
        else:
            raise ValueError(f"Unsupported hierarchical encoder init mode: {mode!r}")
        if layer.bias is not None:
            init.zeros_(layer.bias)

    def _reset_parameters(self) -> None:
        for embedding in self.code_embedding:
            init.normal_(embedding.weight, mean=0.0, std=0.02)
        if self.stage_condition_projectors is not None:
            for projector in self.stage_condition_projectors:
                for module in projector:
                    if isinstance(module, nn.Linear):
                        self._init_linear(module, "kaiming_uniform")
        for head in self.stage_heads:
            linear_layers = [module for module in head if isinstance(module, nn.Linear)]
            for module in linear_layers[:-1]:
                self._init_linear(module, "kaiming_uniform")
            self._init_linear(linear_layers[-1], self.output_weight_init)

    def forward(
        self,
        sample: torch.Tensor,
        *,
        timesteps: torch.Tensor | int | None = None,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timesteps
        codes = sample.to(dtype=torch.long)
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        if codes.dim() != 2 or int(codes.shape[1]) != self.num_codebooks:
            raise ValueError(
                f"sample must be discrete RVQ codes [B,{self.num_codebooks}] or [{self.num_codebooks}], "
                f"got {tuple(codes.shape)}"
            )
        batch_size = int(codes.shape[0])

        condition_repr = None
        if self.condition_layer is not None:
            resolved_condition = class_labels if condition is None else condition
            if resolved_condition is None:
                raise ValueError("condition must be provided when conditioning is enabled")
            condition_repr = self.condition_layer(resolved_condition.to(device=codes.device))
            if int(condition_repr.shape[0]) != batch_size:
                raise ValueError("condition batch size must match latent batch size")
        elif class_labels is not None or condition is not None:
            raise ValueError("condition must not be provided when conditioning is disabled")

        stage_maps: list[torch.Tensor] = []
        align_corners = False if self.upsample_mode in {"bilinear", "bicubic"} else None
        for index, ((stage_height, stage_width), embedding, head) in enumerate(
            zip(self.stage_sizes, self.code_embedding, self.stage_heads)
        ):
            code_repr = embedding(codes[:, index]).to(dtype=torch.float32)
            if condition_repr is not None:
                if self.stage_condition_projectors is None:
                    raise RuntimeError("stage_condition_projectors must exist when conditioning is enabled")
                stage_condition = self.stage_condition_projectors[index](condition_repr.to(dtype=torch.float32))
                stage_input = torch.cat((code_repr, stage_condition), dim=1)
            else:
                stage_input = code_repr
            stage_map = head(stage_input).reshape(batch_size, 1, stage_height, stage_width)
            if (stage_height, stage_width) != (self.output_height, self.output_width):
                stage_map = F.interpolate(
                    stage_map,
                    size=(self.output_height, self.output_width),
                    mode=self.upsample_mode,
                    align_corners=align_corners,
                )
            stage_maps.append(stage_map)

        raw_phase = torch.stack(stage_maps, dim=0).sum(dim=0)
        return torch.remainder(raw_phase, self.phase_period_rad)


class OpticalPrefixReadoutDecoder(nn.Module):
    """Optical decoder that reads each prefix at the detector plane."""

    def __init__(
        self,
        *,
        slm_layer: SourceLayer,
        optical_layers: Sequence[OpticLayer],
        detector_layer: DetectorLayer,
        distance_slm_to_first_layer_m: float,
        distance_between_layers_m: Sequence[float],
        distance_last_layer_to_detector_m: float,
        propagation_config: PropagationConfig,
        error_config: PropagationErrorConfig,
        default_error_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.slm_layer = slm_layer
        self.optical_layers = nn.ModuleList(list(optical_layers))
        self.detector_layer = detector_layer
        self.distance_slm_to_first_layer_m = float(distance_slm_to_first_layer_m)
        self.distance_last_layer_to_detector_m = float(distance_last_layer_to_detector_m)
        self.distance_between_layers_m = tuple(float(value) for value in distance_between_layers_m)
        self.propagation_config = propagation_config
        self.error_config = error_config
        self.default_error_factor = float(default_error_factor)

        if len(self.optical_layers) == 0:
            raise ValueError("optical_layers must contain at least one layer")
        if len(self.distance_between_layers_m) != len(self.optical_layers) - 1:
            raise ValueError(
                "distance_between_layers_m length must be len(optical_layers) - 1, "
                f"got {len(self.distance_between_layers_m)} vs {len(self.optical_layers) - 1}"
            )

        self._propagation_context = self._build_propagation_context()

    @property
    def slm_input_height(self) -> int:
        if hasattr(self.slm_layer, "pixel_count_y"):
            return int(self.slm_layer.pixel_count_y)
        return int(self.slm_layer.sy)

    @property
    def slm_input_width(self) -> int:
        if hasattr(self.slm_layer, "pixel_count_x"):
            return int(self.slm_layer.pixel_count_x)
        return int(self.slm_layer.sx)

    @property
    def num_prefix_readouts(self) -> int:
        return len(self.optical_layers)

    def _build_propagation_context(self) -> PropagateContext:
        context = PropagateContext(
            propagation_config=self.propagation_config,
            error_config=self.error_config,
            error_factor=self.default_error_factor,
        )
        z = 0.0
        context.add_layer(self.slm_layer, z)

        z += self.distance_slm_to_first_layer_m
        for layer_index, layer in enumerate(self.optical_layers):
            context.add_layer(layer, z)
            if layer_index < len(self.optical_layers) - 1:
                z += self.distance_between_layers_m[layer_index]

        z += self.distance_last_layer_to_detector_m
        context.add_layer(self.detector_layer, z)
        return context

    def forward(
        self,
        slm_input: torch.Tensor,
        *,
        error_factor: float | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        if slm_input.dim() != 4 or int(slm_input.shape[1]) != 1:
            raise ValueError(f"slm_input must be [B,1,H,W], got {tuple(slm_input.shape)}")
        if tuple(slm_input.shape[-2:]) != (self.slm_input_height, self.slm_input_width):
            raise ValueError(
                "slm_input spatial shape must match SLM pixel grid: "
                f"expected={(self.slm_input_height, self.slm_input_width)}, "
                f"got={tuple(slm_input.shape[-2:])}"
            )

        effective_error_factor = self.default_error_factor if error_factor is None else float(error_factor)
        original_error_factor = self._propagation_context.error_factor
        self._propagation_context.error_factor = effective_error_factor
        try:
            readouts = self._propagation_context.propagate_with_prefix_readouts(slm_input)
        finally:
            self._propagation_context.error_factor = original_error_factor
        prefix_readouts = tuple(
            readouts[f"prefix_readout_{index + 1}"] for index in range(self.num_prefix_readouts)
        )
        readouts["prefix_readouts"] = prefix_readouts
        return readouts


class OpticalMultiscaleModel(nn.Module):
    """End-to-end model: digital encoder -> optical decoder with prefix readouts."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        optical_decoder: nn.Module,
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.optical_decoder = optical_decoder
        self.upsample_mode = str(upsample_mode)

    def _upsample_to_slm_grid(self, encoder_output: torch.Tensor) -> torch.Tensor:
        if encoder_output.dim() != 4 or int(encoder_output.shape[1]) != 1:
            raise ValueError(
                "encoder output must be [B,1,H,W] before loading onto the SLM, "
                f"got {tuple(encoder_output.shape)}"
            )
        target_size = (
            self.optical_decoder.slm_input_height,
            self.optical_decoder.slm_input_width,
        )
        if tuple(encoder_output.shape[-2:]) == target_size:
            return encoder_output
        return F.interpolate(encoder_output, size=target_size, mode=self.upsample_mode)

    def forward(
        self,
        sample: torch.Tensor,
        *,
        timesteps: torch.Tensor | int | None = None,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        error_factor: float | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        encoder_output = self.encoder(
            sample,
            timesteps=timesteps,
            class_labels=class_labels,
            condition=condition,
        )
        slm_input = self._upsample_to_slm_grid(encoder_output)
        optical_output = self.optical_decoder(slm_input, error_factor=error_factor)
        return {
            "encoder_output": encoder_output,
            "slm_input": slm_input,
            **optical_output,
        }
