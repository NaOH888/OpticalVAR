from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from optical.core.config import FDTDSourceConfig, SourceConfig
from optical.core.base import SourceLayer

if TYPE_CHECKING:
    from optical.backends.fdtd.api import FDTDBuilder, FDTDLayerContext


class LightSourceLayer(SourceLayer):
    """
    连续光源层。

    职责边界：
    1. 把输入张量重采样到当前传播网格；
    2. 按 `light_mode` 解释输入的物理语义；
    3. 直接组装复振幅光场。

    这里不再承担任何离散 SLM 器件行为：
    - 不做输入归一化；
    - 不做相位量化；
    - 不做像素开口率、dead space 等器件级几何处理。

    这些器件语义统一放到 `SLMDeviceLayer` 或更外层处理。
    """

    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        dx_m: float,
        config: SourceConfig,
        fdtd_config: FDTDSourceConfig | None = None,
    ):
        super().__init__(dx=dx_m)
        self.config = config
        self.fdtd_config = fdtd_config or FDTDSourceConfig()
        self.width = float(width_m)
        self.height = float(height_m)
        self.sx = int(round(self.width / dx_m))
        self.sy = int(round(self.height / dx_m))

    def _resolve_light_mode(self) -> str:
        """解析输入张量对应的物理语义。"""
        mode = str(self.config.light_mode).lower()
        if mode not in {"phase", "amplitude", "intensity"}:
            raise ValueError(f"Unsupported SourceConfig.light_mode={self.config.light_mode!r}")
        return mode

    @staticmethod
    def _prepare_amplitude_input(img: torch.Tensor, light_mode: str) -> torch.Tensor:
        """
        根据当前输入语义构造目标振幅。

        - `amplitude`：输入直接视作振幅；
        - `intensity`：输入视作强度，内部转换为 `sqrt(intensity)`。
        """
        if light_mode == "intensity":
            return torch.sqrt(torch.clamp_min(img, 0.0))
        return img

    @staticmethod
    def _compose_field(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """根据振幅与相位直接组装复场。"""
        return amplitude * torch.exp(1j * phase)

    def forward(self, img: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        """
        输入解释方式：
        - `light_mode='phase'`：输入直接视作相位，单位 rad；
        - `light_mode='amplitude'`：输入直接视作振幅；
        - `light_mode='intensity'`：输入直接视作强度，内部只做物理上必要的 `sqrt`。
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.dim() != 4:
            raise ValueError(f"LightSourceLayer expects 4D input [B,C,H,W], got {tuple(img.shape)}")

        light_mode = self._resolve_light_mode()
        img = F.interpolate(img, (self.sy, self.sx), mode="bilinear", align_corners=True).to(dtype=torch.float32)
        source_amp = float(self.config.amplitude)
        if light_mode == "phase":
            amplitude = torch.full_like(img, fill_value=source_amp)
            return self._compose_field(amplitude, img)

        amp_input = self._prepare_amplitude_input(img, light_mode=light_mode)
        amplitude = source_amp * amp_input
        phase = torch.zeros_like(amplitude)
        return self._compose_field(amplitude, phase)

    def build_fdtd(self, builder: "FDTDBuilder", layer_ctx: "FDTDLayerContext") -> float:
        wavelength_values = [float(x) for x in self.config.wavelengths_m]
        wavelength_start = min(wavelength_values)
        wavelength_stop = max(wavelength_values)

        # 光源当前仍视作零厚度平面，位置直接取递推得到的真实底面 z。
        source_z = float(layer_ctx.z_bottom)
        builder.add_plane_source(
            name=self.fdtd_config.name,
            injection_axis=self.fdtd_config.injection_axis,
            direction=self.fdtd_config.direction,
            x_span=float(self.width),
            y_span=float(self.height),
            z=source_z,
            wavelength_start=wavelength_start,
            wavelength_stop=wavelength_stop,
        )
        return source_z
