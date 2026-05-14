from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vae import build_cvae


class ConditionalVAESmokeTests(unittest.TestCase):
    def test_forward_and_decode_shapes(self) -> None:
        model = build_cvae(
            {
                "input_channels": 1,
                "image_size": 32,
                "latent_height": 4,
                "latent_width": 4,
                "condition_mode": "class_index",
                "num_classes": 10,
                "condition_embed_dim": 8,
                "condition_channels": 2,
                "encoder_hidden_channels": [16, 32, 64],
                "decoder_hidden_channels": [64, 32, 16],
                "reconstruction_loss": "bce",
            }
        )
        batch = {
            "data": torch.rand((2, 1, 32, 32), dtype=torch.float32),
            "labels": torch.tensor([1, 4], dtype=torch.long),
        }
        output = model(batch)
        encoded = model.encode(batch["data"], batch["labels"], sample_posterior=False)
        decoded = model.decode(encoded.z, batch["labels"])

        self.assertEqual(tuple(output.recon_x.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(output.z.shape), (2, 16))
        self.assertEqual(tuple(encoded.z.shape), (2, 16))
        self.assertEqual(tuple(decoded.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(output.loss))

    def test_forward_and_decode_support_attribute_vector_condition(self) -> None:
        model = build_cvae(
            {
                "input_channels": 1,
                "image_size": 32,
                "latent_height": 4,
                "latent_width": 4,
                "condition_mode": "attribute_vector",
                "condition_input_dim": 5,
                "condition_embed_dim": 8,
                "condition_hidden_dim": 16,
                "condition_channels": 2,
                "encoder_hidden_channels": [16, 32, 64],
                "decoder_hidden_channels": [64, 32, 16],
                "reconstruction_loss": "bce",
            }
        )
        condition = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32)
        batch = {
            "data": torch.rand((2, 1, 32, 32), dtype=torch.float32),
            "labels": condition,
        }
        output = model(batch)
        encoded = model.encode(batch["data"], batch["labels"], sample_posterior=False)
        decoded = model.decode(encoded.z, batch["labels"])
        self.assertEqual(tuple(decoded.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(output.loss))


if __name__ == "__main__":
    unittest.main()
