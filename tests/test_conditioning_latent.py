from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conditioning import (
    ConditionEmbeddingLayer,
    ConditionalLatentFusion,
    ConditionalLatentInputAdapter,
    ContinuousMapLatentProjector,
    LatentEmbeddingLayer,
)


class ConditioningLatentTests(unittest.TestCase):
    def test_continuous_map_embedding_projects_to_vector(self) -> None:
        layer = LatentEmbeddingLayer(
            projector=ContinuousMapLatentProjector(
                latent_height=4,
                latent_width=4,
                output_dim=12,
                hidden_dim=16,
            )
        )
        output = layer(torch.rand((2, 1, 4, 4), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (2, 12))
        self.assertTrue(torch.isfinite(output).all())

    def test_condition_latent_adapter_returns_all_representations(self) -> None:
        condition_layer = ConditionEmbeddingLayer(
            mode="attribute_vector",
            input_dim=5,
            output_dim=10,
            hidden_dim=12,
        )
        latent_layer = LatentEmbeddingLayer(
            projector=ContinuousMapLatentProjector(
                input_dim=12,
                output_dim=10,
                hidden_dim=12,
            )
        )
        fusion_layer = ConditionalLatentFusion(
            latent_dim=10,
            condition_dim=10,
            output_dim=14,
            mode="concat",
            hidden_dim=18,
        )
        adapter = ConditionalLatentInputAdapter(
            condition_layer=condition_layer,
            latent_layer=latent_layer,
            fusion_layer=fusion_layer,
        )
        result = adapter(
            latent=torch.rand((2, 3, 2, 2), dtype=torch.float32),
            condition=torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32),
        )
        self.assertEqual(tuple(result["latent_repr"].shape), (2, 10))
        self.assertEqual(tuple(result["condition_repr"].shape), (2, 10))
        self.assertEqual(tuple(result["fused_repr"].shape), (2, 14))


if __name__ == "__main__":
    unittest.main()
