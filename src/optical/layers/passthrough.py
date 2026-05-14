from __future__ import annotations

import torch

from optical.core.base import OpticLayer


class PassThroughLayer(OpticLayer):
    """透传层：不改变场，仅提供目标平面的网格尺寸与位置。"""

    def __init__(self, width: float, height: float, dx: float):
        super().__init__(dx=dx)
        self.width = width
        self.height = height
        self.sx = int(round(width / dx))
        self.sy = int(round(height / dx))

    def forward(self, input_field: torch.Tensor, *_args, **_kwargs):
        return input_field
