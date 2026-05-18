from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from optical.core.base import OpticLayer


class DiffractiveAmplitudeLayer(OpticLayer):
    """可训练纯振幅层：学习 [0, 1] 范围内的振幅透过率。"""

    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        dx_m: float,
        channels: int,
        share_across_channels: bool = True,
        amplitude_grid_height: int | None = None,
        amplitude_grid_width: int | None = None,
        initial_amplitude_map: torch.Tensor | None = None,
    ) -> None:
        super().__init__(dx=dx_m)
        self.width = float(width_m)
        self.height = float(height_m)
        self.sx = int(round(self.width / dx_m))
        self.sy = int(round(self.height / dx_m))

        self.channels = int(channels)
        self.share_across_channels = bool(share_across_channels)

        amplitude_ch = 1 if self.share_across_channels else self.channels
        self.amplitude_grid_height = self.sy if amplitude_grid_height is None else int(amplitude_grid_height)
        self.amplitude_grid_width = self.sx if amplitude_grid_width is None else int(amplitude_grid_width)
        if self.amplitude_grid_height <= 0 or self.amplitude_grid_width <= 0:
            raise ValueError(
                "amplitude_grid_height and amplitude_grid_width must be positive, "
                f"got {(self.amplitude_grid_height, self.amplitude_grid_width)}"
            )

        if initial_amplitude_map is None:
            initial_amplitude_map = torch.full(
                (1, amplitude_ch, self.amplitude_grid_height, self.amplitude_grid_width),
                fill_value=1.0,
                dtype=torch.float32,
            )
        else:
            initial_amplitude_map = self._normalize_initial_amplitude_map(
                initial_amplitude_map,
                amplitude_ch=amplitude_ch,
            )

        initial_amplitude_map = initial_amplitude_map.clamp(1.0e-4, 1.0 - 1.0e-4)
        self.raw_amplitude_logits = nn.Parameter(torch.logit(initial_amplitude_map))

    def _normalize_initial_amplitude_map(
        self,
        amplitude_map: torch.Tensor,
        *,
        amplitude_ch: int,
    ) -> torch.Tensor:
        if amplitude_map.dim() == 2:
            amplitude_map = amplitude_map.unsqueeze(0).unsqueeze(0)
        elif amplitude_map.dim() == 3:
            amplitude_map = amplitude_map.unsqueeze(0)
        if amplitude_map.dim() != 4:
            raise ValueError(
                "initial_amplitude_map must be [H,W] / [C,H,W] / [1,C,H,W], "
                f"got {tuple(amplitude_map.shape)}"
            )
        if int(amplitude_map.shape[0]) != 1:
            raise ValueError(f"initial_amplitude_map batch must be 1, got {tuple(amplitude_map.shape)}")
        if int(amplitude_map.shape[1]) != int(amplitude_ch):
            raise ValueError(
                "initial_amplitude_map channel count does not match amplitude layer channel setting: "
                f"expected={amplitude_ch}, got={tuple(amplitude_map.shape)}"
            )
        if tuple(amplitude_map.shape[-2:]) != (self.amplitude_grid_height, self.amplitude_grid_width):
            raise ValueError(
                "initial_amplitude_map spatial size does not match amplitude grid: "
                f"expected={(self.amplitude_grid_height, self.amplitude_grid_width)}, "
                f"got={tuple(amplitude_map.shape[-2:])}"
            )
        return amplitude_map.to(dtype=torch.float32)

    @property
    def raw_amplitude(self) -> torch.Tensor:
        """导出当前振幅图，范围稳定在 [0,1]。"""
        return torch.sigmoid(self.raw_amplitude_logits)

    def export_amplitude_map(self) -> torch.Tensor:
        return self.raw_amplitude.detach().clone()

    @staticmethod
    def _resize_real_map(real_map: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if tuple(real_map.shape[-2:]) == tuple(target_hw):
            return real_map

        src_h, src_w = (int(real_map.shape[-2]), int(real_map.shape[-1]))
        dst_h, dst_w = (int(target_hw[0]), int(target_hw[1]))
        if src_h >= dst_h and src_w >= dst_w:
            return F.adaptive_avg_pool2d(real_map, output_size=target_hw)
        return F.interpolate(real_map, size=target_hw, mode="nearest")

    def _build_amplitude_map(self, *, device: torch.device, channels: int) -> torch.Tensor:
        amplitude = self.raw_amplitude.to(device=device)
        if self.share_across_channels:
            amplitude = amplitude.expand(1, channels, -1, -1)
        return amplitude

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        if input_field.dim() != 4 or not torch.is_complex(input_field):
            raise ValueError("DiffractiveAmplitudeLayer 输入必须是复张量 [B,C,H,W]")
        _, c, h, w = input_field.shape
        if c != self.channels:
            raise ValueError(
                f"DiffractiveAmplitudeLayer channel mismatch: expected={self.channels}, got={c}"
            )

        amplitude = self._build_amplitude_map(device=input_field.device, channels=c)
        if tuple(amplitude.shape[-2:]) != (h, w):
            amplitude = self._resize_real_map(amplitude, (h, w))
        return input_field * amplitude
