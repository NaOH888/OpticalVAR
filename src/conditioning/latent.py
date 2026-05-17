from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousMapLatentProjector(nn.Module):
    """将连续 latent map 或连续 latent 向量投影成统一的 latent 表示向量。"""

    def __init__(
        self,
        *,
        output_dim: int,
        input_dim: int | None = None,
        latent_height: int | None = None,
        latent_width: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.input_dim = None if input_dim is None else int(input_dim)
        self.latent_height = None if latent_height is None else int(latent_height)
        self.latent_width = None if latent_width is None else int(latent_width)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else self.output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim!r}")

        if self.input_dim is not None:
            flattened_dim = self.input_dim
        elif self.latent_height is not None and self.latent_width is not None:
            flattened_dim = self.latent_height * self.latent_width
        else:
            raise ValueError(
                "ContinuousMapLatentProjector requires input_dim or both latent_height and latent_width"
            )
        self.projector = nn.Sequential(
            nn.Linear(flattened_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # 连续 latent 统一展平为 [B, D]，再映射到固定维度表示。
        features = latent.to(dtype=torch.float32)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        elif features.dim() in {3, 4}:
            features = features.reshape(features.shape[0], -1)
        if features.dim() != 2:
            raise ValueError(
                "continuous latent must be [B,C,H,W], [B,H,W], [B,D], or [D], "
                f"got {tuple(features.shape)}"
            )
        if self.input_dim is not None and int(features.shape[1]) != self.input_dim:
            raise ValueError(f"continuous latent input_dim must be {self.input_dim}, got {int(features.shape[1])}")
        return self.projector(features).to(dtype=torch.float32)


class LatentEmbeddingLayer(nn.Module):
    """对外统一的 latent 嵌入层，具体嵌入方式由注入的 projector 决定。"""

    def __init__(self, *, projector: nn.Module) -> None:
        super().__init__()
        self.projector = projector

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projector(latent)


class ConditionalLatentFusion(nn.Module):
    """将 latent 表示与条件表示融合成后续网络可直接消费的统一表示。"""

    def __init__(
        self,
        *,
        latent_dim: int,
        condition_dim: int,
        output_dim: int,
        mode: str = "concat",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.output_dim = int(output_dim)
        self.mode = str(mode)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else max(self.output_dim, self.latent_dim + self.condition_dim))
        if self.mode not in {"concat", "add"}:
            raise ValueError(f"Unsupported fusion mode: {self.mode!r}")

        if self.mode == "add":
            if self.latent_dim != self.condition_dim:
                raise ValueError("add fusion requires latent_dim == condition_dim")
            self.projector = (
                nn.Identity()
                if self.latent_dim == self.output_dim
                else nn.Linear(self.latent_dim, self.output_dim)
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.latent_dim + self.condition_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    def forward(self, latent_repr: torch.Tensor, condition_repr: torch.Tensor) -> torch.Tensor:
        # 目前支持两种融合：
        # 1. add: 适合同维表示，直接逐元素相加
        # 2. concat: 拼接后再过 MLP，表达能力更强，也是当前主推荐方式
        if latent_repr.dim() != 2 or condition_repr.dim() != 2:
            raise ValueError("latent_repr and condition_repr must both be [B,D]")
        if int(latent_repr.shape[0]) != int(condition_repr.shape[0]):
            raise ValueError("latent_repr and condition_repr batch size must match")
        if self.mode == "add":
            return self.projector(latent_repr + condition_repr).to(dtype=torch.float32)
        return self.projector(torch.cat((latent_repr, condition_repr), dim=1)).to(dtype=torch.float32)


class ConditionalLatentInputAdapter(nn.Module):
    """将条件嵌入、latent 嵌入和融合三步打包成一个统一入口。"""

    def __init__(
        self,
        *,
        condition_layer: nn.Module,
        latent_layer: nn.Module,
        fusion_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.condition_layer = condition_layer
        self.latent_layer = latent_layer
        self.fusion_layer = fusion_layer

    def forward(self, *, latent: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        # 返回中间结果而不只返回 fused_repr，便于训练调试和后续可视化分析。
        latent_repr = self.latent_layer(latent)
        condition_repr = self.condition_layer(condition)
        fused_repr = self.fusion_layer(latent_repr, condition_repr)
        return {
            "latent_repr": latent_repr,
            "condition_repr": condition_repr,
            "fused_repr": fused_repr,
        }
