from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from optical.core.config import DetectorConfig, FDTDMonitorConfig
from optical.core.base import OpticLayer

if TYPE_CHECKING:
    from optical.backends.fdtd.api import FDTDBuilder, FDTDLayerContext


class DetectorLayer(OpticLayer):
    """Sample the complex field onto a detector grid and store intensity."""

    def __init__(
        self,
        *,
        config: DetectorConfig,
        dx_m: float,
        fdtd_config: FDTDMonitorConfig | None = None,
    ):
        super().__init__(dx=dx_m)
        self.config = config
        self.fdtd_config = fdtd_config or FDTDMonitorConfig(name="detector")
        self.width_num = int(config.width_num)
        self.height_num = int(config.height_num)
        self.detector_unit_len = float(config.detector_unit_len_m)

        self.width = float(self.width_num * self.detector_unit_len)
        self.height = float(self.height_num * self.detector_unit_len)
        self.sx = int(round(self.width / dx_m))
        self.sy = int(round(self.height / dx_m))
        self.E = None
        self.I = None

    def measure(self, input_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = input_field.shape
        out_h = self.height_num
        out_w = self.width_num

        xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, out_w, device=input_field.device)
        ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, out_h, device=input_field.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

        real = F.grid_sample(input_field.real, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        imag = F.grid_sample(input_field.imag, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        measured_field = torch.complex(real, imag)
        measured_intensity = measured_field.abs() ** 2
        return measured_field, measured_intensity

    def record_measurement(self, input_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.E, self.I = self.measure(input_field)
        return self.E, self.I

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs):
        self.record_measurement(input_field)
        return torch.zeros_like(input_field)

    def build_fdtd(self, builder: "FDTDBuilder", layer_ctx: "FDTDLayerContext") -> float:
        # Detector 本身是零厚度抽象观测面，默认放在递推得到的真实底面 z。
        monitor_z = float(layer_ctx.z_bottom)
        builder.add_profile_monitor(
            name=self.fdtd_config.name,
            monitor_type=self.fdtd_config.monitor_type,
            x_span=float(self.width),
            y_span=float(self.height),
            z=monitor_z,
        )
        return monitor_z
