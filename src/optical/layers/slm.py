from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from optical.core.config import FDTDSourceConfig, SourceConfig
from optical.core.base import SourceLayer
from optical.layers.source import LightSourceLayer

if TYPE_CHECKING:
    from optical.backends.fdtd.api import FDTDBuilder, FDTDLayerContext


class SLMDeviceLayer(SourceLayer):
    """
    离散 SLM 器件适配层。

    当前职责边界只保留两部分：
    1. 在 SLM 像素网格上执行相位量化；
    2. 根据 `fill_factor` 把像素死区映射为传播网格上的有效开口。

    其中 `fill_factor` 对应器件自身的像素开口率，不属于误差模型；
    当前实现不再在这里叠加串扰、静态/时变相位噪声或全局振幅抖动。
    """

    def __init__(
        self,
        *,
        pixel_pitch_x_m: float,
        pixel_pitch_y_m: float,
        pixel_count_x: int,
        pixel_count_y: int,
        dx: float,
        fill_factor: float,
        phase_alpha: float,
        phase_bit_depth: int | None,
        source_config: SourceConfig,
        fdtd_config: FDTDSourceConfig | None = None,
        phase_range_rad: float | None = None,
        use_ste_during_training: bool = True,
    ):
        """
        :param pixel_pitch_x_m: SLM 像素在 x 方向的物理 pitch，单位 m。
        :param pixel_pitch_y_m: SLM 像素在 y 方向的物理 pitch，单位 m。
        :param pixel_count_x: SLM 在 x 方向的像素个数。
        :param pixel_count_y: SLM 在 y 方向的像素个数。
        :param dx: 标量传播画布的采样间距，单位 m。
        :param fill_factor: 像素有效开口率，用于表达器件自身的 dead space / padding。
        :param phase_alpha: 当 `phase_range_rad` 未显式给定时，默认相位范围为 `phase_alpha * pi`。
        :param phase_bit_depth: SLM 相位量化位宽；为 `None` 时表示不量化。
        :param source_config: SLM 持有的光源配置，包括波长、输入语义和默认振幅。
        :param phase_range_rad: 相位量化的有效范围，单位 rad；为空时回退到 `phase_alpha * pi`。
        :param use_ste_during_training: 训练态是否对量化使用直通估计器。
        """
        super().__init__(dx=dx)
        self.source_config = source_config
        self.pixel_pitch_x_m = float(pixel_pitch_x_m)
        self.pixel_pitch_y_m = float(pixel_pitch_y_m)
        self.pixel_count_x = int(pixel_count_x)
        self.pixel_count_y = int(pixel_count_y)
        self.fill_factor = float(fill_factor)
        self.phase_alpha = float(phase_alpha)
        self.bit_depth = phase_bit_depth
        self.phase_range_rad = phase_range_rad
        self.use_ste_during_training = bool(use_ste_during_training)
        self.fdtd_config = fdtd_config or FDTDSourceConfig()

        if self.pixel_pitch_x_m <= 0.0 or self.pixel_pitch_y_m <= 0.0:
            raise ValueError("pixel_pitch_x_m and pixel_pitch_y_m must be positive")
        if self.pixel_count_x <= 0 or self.pixel_count_y <= 0:
            raise ValueError("pixel_count_x and pixel_count_y must be positive")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive")
        if self.fill_factor <= 0.0 or self.fill_factor > 1.0:
            raise ValueError("fill_factor must be in (0, 1]")
        if self.source_config.light_mode not in {"phase", "amplitude", "intensity"}:
            raise ValueError(
                "source_config.light_mode must be one of {'phase', 'amplitude', 'intensity'}, "
                f"got {self.source_config.light_mode!r}"
            )

        self.samples_per_pixel_x = self._resolve_samples_per_pixel(self.pixel_pitch_x_m, self.dx, "x")
        self.samples_per_pixel_y = self._resolve_samples_per_pixel(self.pixel_pitch_y_m, self.dx, "y")

        self.width = self.pixel_pitch_x_m * float(self.pixel_count_x)
        self.height = self.pixel_pitch_y_m * float(self.pixel_count_y)
        self.sx = self.pixel_count_x * self.samples_per_pixel_x
        self.sy = self.pixel_count_y * self.samples_per_pixel_y

        self.light_source = LightSourceLayer(
            width_m=self.width,
            height_m=self.height,
            dx_m=self.dx,
            config=self.source_config,
            fdtd_config=self.fdtd_config,
        )

        aperture_mask = self._build_aperture_mask()
        self.register_buffer("_aperture_mask", aperture_mask, persistent=False)

    @staticmethod
    def _resolve_samples_per_pixel(pixel_pitch_m: float, dx: float, axis_name: str) -> int:
        """
        把像素 pitch 解析为传播网格上每个像素对应的采样点数。

        当前仍采用严格整数映射，避免在 SLM 像素到传播画布之间额外引入插值语义。
        """
        ratio = float(pixel_pitch_m) / float(dx)
        rounded = int(round(ratio))
        if rounded < 1:
            raise ValueError(
                f"pixel_pitch_{axis_name}_m / dx must be >= 1, got {ratio:.6f}. "
                "当前传播网格不能比 SLM 像素更粗。"
            )
        if not math.isclose(ratio, float(rounded), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"pixel_pitch_{axis_name}_m / dx must be close to an integer, got {ratio:.6f}. "
                "当前版本要求每个 SLM 像素对应固定整数个传播采样点。"
            )
        return rounded

    @staticmethod
    def _build_center_block_mask(full_size: int, active_size: int) -> torch.Tensor:
        """在一个像素单元内构造居中的有效开口窗口。"""
        if active_size <= 0 or active_size > full_size:
            raise ValueError(f"active_size must be in [1, full_size], got {active_size} vs {full_size}")
        mask = torch.zeros((full_size,), dtype=torch.float32)
        start = (full_size - active_size) // 2
        end = start + active_size
        mask[start:end] = 1.0
        return mask

    def _build_aperture_mask(self) -> torch.Tensor:
        """
        构造整个 SLM 的有效开口掩模。

        每个像素先对应传播画布上的一个 pitch 单元；当 `fill_factor<1` 时，
        仅保留该单元中心的一块有效区域，用于表达像素间 padding / dead space。
        """
        active_x = max(1, int(round(self.fill_factor * float(self.samples_per_pixel_x))))
        active_y = max(1, int(round(self.fill_factor * float(self.samples_per_pixel_y))))
        active_x = min(active_x, self.samples_per_pixel_x)
        active_y = min(active_y, self.samples_per_pixel_y)

        cell_x = self._build_center_block_mask(self.samples_per_pixel_x, active_x)
        cell_y = self._build_center_block_mask(self.samples_per_pixel_y, active_y)
        cell_mask = torch.outer(cell_y, cell_x)
        full_mask = cell_mask.repeat(self.pixel_count_y, self.pixel_count_x)
        return full_mask.unsqueeze(0).unsqueeze(0)

    def _expand_pixels_to_field_grid(self, img: torch.Tensor) -> torch.Tensor:
        """把器件像素网格复制展开为传播网格上的分段常值场。"""
        return img.repeat_interleave(self.samples_per_pixel_y, dim=-2).repeat_interleave(
            self.samples_per_pixel_x,
            dim=-1,
        )

    @staticmethod
    def _ste_round(value: torch.Tensor) -> torch.Tensor:
        """训练阶段使用的直通估计器 round。"""
        return value + (torch.round(value) - value).detach()

    def _quantize_phase(
        self,
        phase: torch.Tensor,
        bit_depth: int | None,
        phase_range_rad: float,
        use_ste: bool,
    ) -> torch.Tensor:
        """把连续相位量化到有限灰度级。"""
        if bit_depth is None:
            return phase
        bits = int(bit_depth)
        if bits <= 1:
            return phase

        max_level = float((1 << bits) - 1)
        phase_clamped = phase.clamp(0.0, float(phase_range_rad))
        phase_norm = phase_clamped / float(phase_range_rad)
        phase_levels = phase_norm * max_level
        if use_ste:
            phase_quant = self._ste_round(phase_levels) / max_level
        else:
            phase_quant = torch.round(phase_levels) / max_level
        return phase_quant * float(phase_range_rad)

    def _apply_fill_factor(self, field: torch.Tensor) -> torch.Tensor:
        """在传播网格上施加由 `fill_factor` 定义的像素有效开口。"""
        mask = self._aperture_mask.to(device=field.device, dtype=field.real.dtype)
        return field * mask

    def _expand_output_channels_if_needed(self, field: torch.Tensor) -> torch.Tensor:
        """
        当输入只提供单通道像素图时，按当前波长数把输出复场广播到多通道。
        """
        target_channels = len(self.source_config.wavelengths_m)
        current_channels = int(field.shape[1])
        if current_channels == target_channels:
            return field
        if current_channels == 1:
            return field.expand(-1, target_channels, -1, -1)
        raise ValueError(
            f"SLMDeviceLayer output channel mismatch: current={current_channels}, "
            f"target(wavelengths)={target_channels}"
        )

    def forward(self, img: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        """
        输入张量必须位于 SLM 像素网格上。

        期望空间尺寸：
        - `[..., pixel_count_y, pixel_count_x]`

        当前实现不再依赖外部 `ctx`：
        - 波长数来自 `source_config.wavelengths_m`；
        - 输入语义和默认振幅来自 `source_config`。
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.dim() != 4:
            raise ValueError(f"SLMDeviceLayer expects 4D input [B,C,H,W], got {tuple(img.shape)}")
        if int(img.shape[-2]) != self.pixel_count_y or int(img.shape[-1]) != self.pixel_count_x:
            raise ValueError(
                "SLMDeviceLayer input spatial size mismatch: "
                f"expected={(self.pixel_count_y, self.pixel_count_x)}, got={tuple(img.shape[-2:])}"
            )

        img = img.to(dtype=torch.float32)
        light_mode = self.source_config.light_mode
        bit_depth = self.bit_depth
        phase_range_rad = self.phase_range_rad
        if phase_range_rad is None:
            phase_range_rad = self.phase_alpha * math.pi
        phase_range_rad = float(phase_range_rad)
        use_ste = bool(self.use_ste_during_training)
        use_ste = bool(use_ste and self.training)

        if light_mode == "phase":
            phase = self._quantize_phase(
                img,
                bit_depth=bit_depth,
                phase_range_rad=phase_range_rad,
                use_ste=use_ste,
            )
            expanded_phase = self._expand_pixels_to_field_grid(phase)
            field = self.light_source(expanded_phase)
            field = self._apply_fill_factor(field)
            return self._expand_output_channels_if_needed(field)

        amp_input = LightSourceLayer._prepare_amplitude_input(img, light_mode=light_mode)
        expanded_amp = self._expand_pixels_to_field_grid(amp_input)
        field = self.light_source(expanded_amp)
        field = self._apply_fill_factor(field)
        return self._expand_output_channels_if_needed(field)

    def build_fdtd(self, builder: "FDTDBuilder", layer_ctx: "FDTDLayerContext") -> float:
        # 当前 FDTD 仍沿用内部连续光源实现；像素开口率只在 Python 前向中体现。
        return self.light_source.build_fdtd(builder, layer_ctx)
