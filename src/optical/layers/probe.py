from __future__ import annotations

import torch

from optical.core.base import OpticLayer
from optical.core.config import FDTDProbeConfig


class FieldProbeLayer(OpticLayer):
    """
    中间场探针层。

    职责：
    - 在标量传播主链中记录当前平面的复场与强度；
    - 不改变传播结果，直接把输入场原样透传给后续层；
    - 便于多层超表面实验观察“每一片器件之后”的场分布。

    运行副作用：
    - `self.E` 保存当前平面的复场；
    - `self.I` 保存当前平面的强度 `|E|^2`；
    - 不额外引入传播参数或配置状态。
    """

    def __init__(
        self,
        dx: float,
        name: str | None = None,
        fdtd_config: FDTDProbeConfig | None = None,
    ):
        super().__init__(dx=dx)
        self.name = name or "field_probe"
        self.fdtd_config = fdtd_config or FDTDProbeConfig(name=self.name)
        self.E: torch.Tensor | None = None
        self.I: torch.Tensor | None = None
        self.width: float | None = None
        self.height: float | None = None

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs):
        """
        记录当前复场并原样透传。

        这里不做任何采样、裁剪或插值，目的是保持探针层对传播结果零侵入。
        如果后续需要记录重采样后的观测结果，应继续使用 `DetectorLayer`。
        """
        self.E = input_field
        self.I = input_field.abs() ** 2
        return input_field

    def build_fdtd(self, builder: "FDTDBuilder", layer_ctx: "FDTDLayerContext") -> float:
        """
        为多层 FDTD 导出提供兼容实现。

        当前默认行为是不导出中间 probe monitor，只返回该抽象平面的 `z_bottom`，
        目的是让带有 `FieldProbeLayer` 的多层场景能够顺利导出 FDTD。

        若需要在 FDTD 中显式观察中间平面，可通过 `fdtd_config.enabled`
        开启 monitor，并确保当前 probe 已具备 `width/height`。
        """
        probe_z = float(layer_ctx.z_bottom)
        if not bool(self.fdtd_config.enabled):
            return probe_z

        if self.width is None or self.height is None:
            raise ValueError(
                f"FieldProbeLayer '{self.name}' requires width/height before exporting FDTD monitor"
            )

        builder.add_profile_monitor(
            name=self.fdtd_config.name,
            monitor_type=self.fdtd_config.monitor_type,
            x_span=float(self.width),
            y_span=float(self.height),
            z=probe_z,
        )
        return probe_z
