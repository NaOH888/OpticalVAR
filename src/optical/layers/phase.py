from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from optical.core.base import OpticLayer


class DiffractivePhaseLayer(OpticLayer):
    """可训练相位层：支持参数网格与传播网格解耦和按波长缩放。"""

    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        dx_m: float,
        channels: int,
        wavelengths_m: tuple[float, ...],
        alpha_pi: float = 2.0,
        share_across_channels: bool = True,
        reference_wavelength_m: float | None = None,
        phase_grid_height: int | None = None,
        phase_grid_width: int | None = None,
        initial_phase_map_rad: torch.Tensor | None = None,
    ):
        """
        构造纯相位层，并允许“可训练相位网格”独立于传播网格尺寸。

        :param width: 当前相位层的物理宽度，单位 m。
        :param height: 当前相位层的物理高度，单位 m。
        :param dx: 传播网格采样间距，单位 m。
        :param channels: 输入复场的波长通道数。
        :param alpha_pi: 导出 canonical 相位时使用的周期系数，实际周期约为 `alpha_pi * pi`。
        :param share_across_channels: 是否在不同波长通道间共享同一张 canonical 相位图。
        :param reference_wavelength_m: 若非空，则把 canonical 相位按参考波长缩放到各通道。
        :param phase_grid_height: 可训练相位参数网格的高度；为空时退化为传播网格高度。
        :param phase_grid_width: 可训练相位参数网格的宽度；为空时退化为传播网格宽度。
        :param initial_phase_map_rad: 可选初始相位图，单位 rad，形状应与相位参数网格匹配。
        """
        super().__init__(dx=dx_m)
        self.width = width_m
        self.height = height_m
        self.sx = int(round(width_m / dx_m))
        self.sy = int(round(height_m / dx_m))

        self.channels = channels
        self.wavelengths_m = tuple(float(x) for x in wavelengths_m)
        self.alpha_pi = float(alpha_pi)
        self.share_across_channels = bool(share_across_channels)
        self.reference_wavelength_m = reference_wavelength_m

        phase_ch = 1 if self.share_across_channels else channels
        self.phase_grid_height = self.sy if phase_grid_height is None else int(phase_grid_height)
        self.phase_grid_width = self.sx if phase_grid_width is None else int(phase_grid_width)
        if self.phase_grid_height <= 0 or self.phase_grid_width <= 0:
            raise ValueError(
                "phase_grid_height and phase_grid_width must be positive, "
                f"got {(self.phase_grid_height, self.phase_grid_width)}"
            )

        if initial_phase_map_rad is None:
            initial_phase_map_rad = torch.full(
                (1, phase_ch, self.phase_grid_height, self.phase_grid_width),
                fill_value=0.0,
                dtype=torch.float32,
            )
        else:
            initial_phase_map_rad = self._normalize_initial_phase_map(initial_phase_map_rad, phase_ch)

        # 直接优化 raw phase，forward 依赖 exp(j*phase) 的周期性；导出时再统一取 canonical 相位。
        self.raw_phase = nn.Parameter(initial_phase_map_rad.clone())

    @property
    def max_phase_rad(self) -> float:
        """返回导出 canonical 相位时使用的周期。"""
        return float(self.alpha_pi * torch.pi)

    def _normalize_initial_phase_map(self, phase_map_rad: torch.Tensor, phase_ch: int) -> torch.Tensor:
        """把初始化相位图规范成 `[1,C,H,W]` 形式，并校验空间尺寸。"""
        if phase_map_rad.dim() == 2:
            phase_map_rad = phase_map_rad.unsqueeze(0).unsqueeze(0)
        elif phase_map_rad.dim() == 3:
            phase_map_rad = phase_map_rad.unsqueeze(0)
        if phase_map_rad.dim() != 4:
            raise ValueError(
                "initial_phase_map_rad must be [H,W] / [C,H,W] / [1,C,H,W], "
                f"got {tuple(phase_map_rad.shape)}"
            )
        if int(phase_map_rad.shape[0]) != 1:
            raise ValueError(f"initial_phase_map_rad batch must be 1, got {tuple(phase_map_rad.shape)}")
        if int(phase_map_rad.shape[1]) != int(phase_ch):
            raise ValueError(
                "initial_phase_map_rad channel count does not match phase layer channel setting: "
                f"expected={phase_ch}, got={tuple(phase_map_rad.shape)}"
            )
        if tuple(phase_map_rad.shape[-2:]) != (self.phase_grid_height, self.phase_grid_width):
            raise ValueError(
                "initial_phase_map_rad spatial size does not match phase grid: "
                f"expected={(self.phase_grid_height, self.phase_grid_width)}, "
                f"got={tuple(phase_map_rad.shape[-2:])}"
            )
        return phase_map_rad.to(dtype=torch.float32)

    def export_phase_map(self) -> torch.Tensor:
        """导出当前层的 canonical 相位图，形状为 `[1,C,H,W]`。"""
        return self._build_canonical_phase_map().detach().clone()

    def _build_canonical_phase_map(self) -> torch.Tensor:
        """构造不含波长扩展/缩放的 canonical 相位图。"""
        return torch.remainder(self.raw_phase, self.max_phase_rad)

    def _build_raw_phase_map(self) -> torch.Tensor:
        """返回训练中真正参与传播的无界 raw phase。"""
        return self.raw_phase

    def _build_phase_map(self, wavelengths: list[float], device) -> torch.Tensor:
        """按当前通道共享/参考波长设置构造真正参与传播的相位图。"""
        phase = self._build_raw_phase_map().to(device=device)

        if self.share_across_channels:
            phase = phase.expand(1, len(wavelengths), -1, -1)

        if self.reference_wavelength_m is not None:
            scale = torch.tensor(
                [self.reference_wavelength_m / float(wl) for wl in wavelengths],
                device=device,
                dtype=phase.dtype,
            ).view(1, -1, 1, 1)
            phase = phase * scale
        return phase

    @staticmethod
    def _resize_real_map(real_map: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """把实值二维网格映射到目标尺寸，粗到细用最近邻，细到粗用面积平均。"""
        if tuple(real_map.shape[-2:]) == tuple(target_hw):
            return real_map

        src_h, src_w = (int(real_map.shape[-2]), int(real_map.shape[-1]))
        dst_h, dst_w = (int(target_hw[0]), int(target_hw[1]))
        if src_h >= dst_h and src_w >= dst_w:
            return F.adaptive_avg_pool2d(real_map, output_size=target_hw)
        return F.interpolate(real_map, size=target_hw, mode="nearest")

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs):
        """对输入复场施加纯相位调制，并在需要时映射到传播网格。"""
        if input_field.dim() != 4 or not torch.is_complex(input_field):
            raise ValueError("DiffractivePhaseLayer 输入必须是复张量 [B,C,H,W]")
        _, c, h, w = input_field.shape

        if len(self.wavelengths_m) != c:
            raise ValueError("wavelengths 长度与输入通道数不一致")

        phase = self._build_phase_map(list(self.wavelengths_m), input_field.device)
        modulation = torch.exp(1j * phase)
        if tuple(modulation.shape[-2:]) != (h, w):
            real = self._resize_real_map(modulation.real, (h, w))
            imag = self._resize_real_map(modulation.imag, (h, w))
            modulation = torch.complex(real, imag)
        return input_field * modulation
