from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from conditioning import ConditionEmbeddingLayer
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


def _init_linear_or_conv(module: nn.Module, mode: str) -> None:
    if isinstance(module, nn.Conv2d):
        weight = module.weight
    elif isinstance(module, nn.Linear):
        weight = module.weight
    else:
        raise TypeError(f"Unsupported module type for initialization: {type(module).__name__}")
    if mode == "kaiming_uniform":
        init.kaiming_uniform_(weight, nonlinearity="relu")
    elif mode == "kaiming_normal":
        init.kaiming_normal_(weight, nonlinearity="relu")
    elif mode == "xavier_uniform":
        init.xavier_uniform_(weight)
    elif mode == "xavier_normal":
        init.xavier_normal_(weight)
    else:
        raise ValueError(f"Unsupported init mode: {mode!r}")
    if getattr(module, "bias", None) is not None:
        init.zeros_(module.bias)


def _interpolate_like(
    tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(tensor, size=size, mode=mode, align_corners=False)
    return F.interpolate(tensor, size=size, mode=mode)


def _build_spatial_stage_sizes(
    *,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
) -> tuple[tuple[int, int], ...]:
    sizes = [(int(input_height), int(input_width))]
    current_h = int(input_height)
    current_w = int(input_width)
    while current_h < int(output_height) or current_w < int(output_width):
        next_h = min(current_h * 2, int(output_height))
        next_w = min(current_w * 2, int(output_width))
        if next_h == current_h and next_w == current_w:
            break
        sizes.append((next_h, next_w))
        current_h, current_w = next_h, next_w
    if sizes[-1] != (int(output_height), int(output_width)):
        sizes.append((int(output_height), int(output_width)))
    return tuple(sizes)


def _build_hidden_channels(
    *,
    stage_count: int,
    hidden_dim: int,
    hidden_channels: Sequence[int] | None,
) -> tuple[int, ...]:
    if hidden_channels is not None:
        resolved = tuple(int(value) for value in hidden_channels)
        if len(resolved) != stage_count:
            raise ValueError(
                f"hidden_channels must have length {stage_count}, got {len(resolved)}"
            )
        if any(value <= 0 for value in resolved):
            raise ValueError(f"hidden_channels must be positive, got {resolved!r}")
        return resolved
    base_channels = min(int(hidden_dim), 96)
    resolved = [base_channels]
    for index in range(1, stage_count):
        resolved.append(max(base_channels // (2**index), 16))
    return tuple(int(value) for value in resolved)


class _FiLMModulation(nn.Module):
    def __init__(self, *, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.feature_dim = int(feature_dim)
        self.projector = nn.Linear(self.condition_dim, 2 * self.feature_dim)

    def forward(self, features: torch.Tensor, condition_repr: torch.Tensor) -> torch.Tensor:
        modulation = self.projector(condition_repr).to(dtype=features.dtype)
        gamma, beta = modulation.chunk(2, dim=1)
        gamma = gamma.view(features.shape[0], self.feature_dim, 1, 1)
        beta = beta.view(features.shape[0], self.feature_dim, 1, 1)
        return features * (1.0 + gamma) + beta


class _DepthwisePointwiseBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        condition_dim: int | None,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            int(in_channels),
            int(in_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=int(in_channels),
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=int(out_channels))
        self.modulation = (
            None
            if condition_dim is None
            else _FiLMModulation(condition_dim=int(condition_dim), feature_dim=int(out_channels))
        )
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor, condition_repr: torch.Tensor | None) -> torch.Tensor:
        hidden = self.depthwise(features)
        hidden = self.pointwise(hidden)
        hidden = self.norm(hidden)
        if self.modulation is not None:
            if condition_repr is None:
                raise ValueError("condition_repr must be provided when modulation is enabled")
            hidden = self.modulation(hidden, condition_repr)
        return self.act(hidden)


class SpatialPhaseMapEncoder(nn.Module):
    """Map a spatial latent map directly to a single-channel phase map."""

    def __init__(
        self,
        *,
        input_channels: int,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        hidden_dim: int = 512,
        hidden_channels: Sequence[int] | None = None,
        phase_alpha_pi: float = 2.0,
        condition_layer: ConditionEmbeddingLayer | None = None,
        condition_dim: int | None = None,
        weight_init: str = "kaiming_uniform",
        output_weight_init: str = "xavier_uniform",
        apply_wrap: bool = True,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.hidden_dim = int(hidden_dim)
        self.phase_alpha_pi = float(phase_alpha_pi)
        self.condition_layer = condition_layer
        self.condition_dim = None if condition_dim is None else int(condition_dim)
        self.weight_init = str(weight_init)
        self.output_weight_init = str(output_weight_init)
        self.apply_wrap = bool(apply_wrap)
        self.upsample_mode = str(upsample_mode)

        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels!r}")
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input_height and input_width must be positive")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_height and output_width must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if self.condition_layer is None and self.condition_dim is not None:
            raise ValueError("condition_dim must be omitted when condition_layer is disabled")
        if self.condition_layer is not None and (self.condition_dim is None or self.condition_dim <= 0):
            raise ValueError("condition_dim must be positive when condition_layer is enabled")

        self.stage_sizes = _build_spatial_stage_sizes(
            input_height=self.input_height,
            input_width=self.input_width,
            output_height=self.output_height,
            output_width=self.output_width,
        )
        self.hidden_channels = _build_hidden_channels(
            stage_count=len(self.stage_sizes),
            hidden_dim=self.hidden_dim,
            hidden_channels=hidden_channels,
        )

        self.stem = nn.Conv2d(self.input_channels, self.hidden_channels[0], kernel_size=1, bias=True)
        blocks: list[nn.Module] = []
        for index, out_channels in enumerate(self.hidden_channels):
            in_channels = self.hidden_channels[index - 1] if index > 0 else self.hidden_channels[0]
            blocks.append(
                _DepthwisePointwiseBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    condition_dim=self.condition_dim,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(self.hidden_channels[-1], 1, kernel_size=1, bias=True)
        self._reset_parameters()

    @property
    def phase_period_rad(self) -> float:
        return float(self.phase_alpha_pi * math.pi)

    def _reset_parameters(self) -> None:
        _init_linear_or_conv(self.stem, self.weight_init)
        for block in self.blocks:
            if not isinstance(block, _DepthwisePointwiseBlock):
                continue
            _init_linear_or_conv(block.depthwise, self.weight_init)
            _init_linear_or_conv(block.pointwise, self.weight_init)
            if block.modulation is not None:
                _init_linear_or_conv(block.modulation.projector, self.weight_init)
        _init_linear_or_conv(self.head, self.output_weight_init)
        if self.condition_layer is not None:
            embedding = getattr(self.condition_layer, "embedding", None)
            if isinstance(embedding, nn.Embedding):
                init.normal_(embedding.weight, mean=0.0, std=0.02)
            projector = getattr(self.condition_layer, "projector", None)
            if isinstance(projector, nn.Linear):
                _init_linear_or_conv(projector, self.weight_init)
            elif isinstance(projector, nn.Sequential):
                for module in projector:
                    if isinstance(module, nn.Linear):
                        _init_linear_or_conv(module, self.weight_init)

    def forward(
        self,
        sample: torch.Tensor,
        *,
        timesteps: torch.Tensor | int | None = None,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timesteps is not None:
            raise ValueError("SpatialPhaseMapEncoder does not support timestep conditioning")
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

        condition_repr: torch.Tensor | None = None
        resolved_condition = class_labels if condition is None else condition
        if self.condition_layer is not None:
            if resolved_condition is None:
                raise ValueError("condition must be provided when conditioning is enabled")
            condition_repr = self.condition_layer(resolved_condition.to(device=sample.device))
            if int(condition_repr.shape[0]) != batch_size:
                raise ValueError("condition batch size must match sample batch size")
        elif class_labels is not None or condition is not None:
            raise ValueError("condition must not be provided when conditioning is disabled")

        hidden = self.stem(sample.to(dtype=torch.float32))
        for index, block in enumerate(self.blocks):
            target_size = self.stage_sizes[index]
            hidden = _interpolate_like(hidden, size=target_size, mode=self.upsample_mode)
            hidden = block(hidden, condition_repr)
        raw_phase = self.head(hidden)
        raw_phase = _interpolate_like(
            raw_phase,
            size=(self.output_height, self.output_width),
            mode=self.upsample_mode,
        )
        if self.apply_wrap:
            return torch.remainder(raw_phase, self.phase_period_rad)
        return raw_phase


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
