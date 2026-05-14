from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEmbeddingLayer(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        output_dim: int,
        num_classes: int | None = None,
        input_dim: int | None = None,
        embed_dim: int = 128,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.output_dim = int(output_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else max(self.output_dim, self.embed_dim))
        self.num_classes = None if num_classes is None else int(num_classes)
        self.input_dim = None if input_dim is None else int(input_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim!r}")

        if self.mode == "class_index":
            if self.num_classes is None or self.num_classes <= 0:
                raise ValueError("num_classes must be positive when mode='class_index'")
            self.embedding = nn.Embedding(self.num_classes, self.embed_dim)
            self.projector = (
                nn.Identity()
                if self.embed_dim == self.output_dim
                else nn.Linear(self.embed_dim, self.output_dim)
            )
        elif self.mode == "attribute_vector":
            if self.input_dim is None or self.input_dim <= 0:
                raise ValueError("input_dim must be positive when mode='attribute_vector'")
            self.embedding = None
            self.projector = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            raise ValueError(f"Unsupported condition mode: {self.mode!r}")

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if self.mode == "class_index":
            labels = condition.to(dtype=torch.long).reshape(-1)
            if self.embedding is None:
                raise RuntimeError("class embedding is not initialized")
            embedded = self.embedding(labels).to(dtype=torch.float32)
            projected = self.projector(embedded)
            return projected.to(dtype=torch.float32)

        features = condition.to(dtype=torch.float32)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() != 2:
            raise ValueError(
                "attribute_vector condition must be [B,D] or [D], "
                f"got {tuple(features.shape)}"
            )
        if self.input_dim is not None and int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"attribute_vector dimension must be {self.input_dim}, got {int(features.shape[1])}"
            )
        projected = self.projector(features)
        return projected.to(dtype=torch.float32)
