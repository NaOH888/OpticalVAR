from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conditioning import ConditionEmbeddingLayer


class ConditioningLatentTests(unittest.TestCase):
    def test_attribute_vector_condition_embedding_projects_to_vector(self) -> None:
        layer = ConditionEmbeddingLayer(
            mode="attribute_vector",
            input_dim=5,
            output_dim=12,
            hidden_dim=16,
        )
        output = layer(torch.rand((2, 5), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (2, 12))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_class_index_condition_embedding_projects_to_vector(self) -> None:
        layer = ConditionEmbeddingLayer(
            mode="class_index",
            num_classes=10,
            output_dim=8,
            embed_dim=6,
        )
        output = layer(torch.tensor([1, 7], dtype=torch.long))
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
