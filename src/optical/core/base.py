from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from optical.backends.fdtd.api import FDTDBuilder, FDTDLayerContext


class OpticLayer(nn.Module):
    """Base class for optical layers used by both scalar and FDTD backends."""

    def __init__(self, dx: float):
        super().__init__()
        self.dx = dx

    def forward(self, input_field: torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    def build_fdtd(self, builder: "FDTDBuilder", layer_ctx: "FDTDLayerContext") -> float | None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support FDTD export")

    def fdtd_physical_thickness(self, layer_ctx: "FDTDLayerContext") -> float:
        """
        返回当前层在 FDTD 几何中的真实厚度。

        默认把层视作零厚度抽象平面；实体层再在子类里覆盖。
        """
        return 0.0


class SourceLayer(OpticLayer):
    """Marker base class for source layers."""
