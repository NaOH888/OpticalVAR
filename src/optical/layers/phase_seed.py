from __future__ import annotations

import torch
import torch.nn.functional as F

from optical.core.base import SourceLayer


class PhaseSeedLayer(SourceLayer):
    """相位种子层：把输入直接归一化为相位图并转为复场，可作为传播链首层。"""

    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        dx_m: float,
        wavelengths_m: tuple[float, ...],
        source_amp: float,
        alpha_pi: float = 2.0,
    ):
        super().__init__(dx=dx_m)
        self.width = width_m
        self.height = height_m
        self.sx = int(round(width_m / dx_m))
        self.sy = int(round(height_m / dx_m))
        self.wavelengths_m = tuple(float(x) for x in wavelengths_m)
        self.source_amp = float(source_amp)
        self.alpha_pi = float(alpha_pi)

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        """
        用法:
        - 直接把输入第一通道归一化到 `[0, alpha*pi]` 作为相位；
        - 不再额外引入旁路相位输入。
        """
        if input_field.dim() == 2:
            input_field = input_field.unsqueeze(0).unsqueeze(0)
        elif input_field.dim() == 3:
            input_field = input_field.unsqueeze(0)
        if input_field.dim() != 4:
            raise ValueError(f"PhaseSeedLayer expects [B,C,H,W], [C,H,W], or [H,W], got {tuple(input_field.shape)}")
        b = input_field.shape[0]

        # 输入直接作为相位种子来源，不再旁路注入另一张额外相位图。
        phase = input_field[:, :1, :, :]
        phase = (phase - phase.amin(dim=(2, 3), keepdim=True)) / (
            phase.amax(dim=(2, 3), keepdim=True) - phase.amin(dim=(2, 3), keepdim=True) + 1e-8
        )
        phase = phase * (self.alpha_pi * torch.pi)

        phase = F.interpolate(phase, size=(self.sy, self.sx), mode="bilinear", align_corners=True)
        field = self.source_amp * torch.exp(1j * phase)

        if len(self.wavelengths_m) > 1:
            field = field.expand(-1, len(self.wavelengths_m), -1, -1)
        return field
