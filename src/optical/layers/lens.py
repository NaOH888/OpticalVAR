from __future__ import annotations

import torch

from optical.core.base import OpticLayer


class ConvexLensLayer(OpticLayer):
    """薄透镜层：施加 `exp(-j*k*(x^2+y^2)/(2f))` 二次相位。"""

    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        dx_m: float,
        focal_length_m: float,
        wavelengths_m: tuple[float, ...],
    ):
        super().__init__(dx=dx_m)
        self.width = width_m
        self.height = height_m
        self.f = focal_length_m
        self.wavelengths_m = tuple(float(x) for x in wavelengths_m)
        self.sx = int(round(width_m / dx_m))
        self.sy = int(round(height_m / dx_m))

        x = (torch.arange(self.sx) - (self.sx - 1) / 2.0) * dx_m
        y = (torch.arange(self.sy) - (self.sy - 1) / 2.0) * dx_m
        self.register_buffer("X", x.repeat(self.sy, 1))
        self.register_buffer("Y", y.view(-1, 1).repeat(1, self.sx))

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs):
        original_dim = input_field.dim()
        if original_dim == 2:
            input_field = input_field.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            input_field = input_field.unsqueeze(0)

        b, c, h, w = input_field.shape
        if h != self.sy or w != self.sx:
            raise ValueError(f"输入尺寸 {(h, w)} 与透镜尺寸 {(self.sy, self.sx)} 不一致")

        if len(self.wavelengths_m) != c:
            raise ValueError("wavelengths 长度与输入通道数不一致")

        x2y2 = self.X.to(input_field.device) ** 2 + self.Y.to(input_field.device) ** 2
        phase = []
        for i in range(c):
            k = 2.0 * torch.pi / float(self.wavelengths_m[i])
            phase.append(torch.exp(-1j * k * x2y2 / (2.0 * self.f)))
        phase = torch.stack(phase, dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        output = input_field * phase

        if original_dim == 3:
            return output.squeeze(0)
        if original_dim == 2:
            return output.squeeze(0).squeeze(0)
        return output
