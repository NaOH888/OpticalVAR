from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.reconstruct_pretrained_autoencoder import _parse_indices, _psnr, _tensor_gray_to_rgb, _tensor_rgb_to_gray


class ReconstructPretrainedAutoencoderTests(unittest.TestCase):
    def test_parse_indices_supports_csv(self) -> None:
        self.assertEqual(_parse_indices("0, 3,7"), [0, 3, 7])
        self.assertEqual(_parse_indices(""), [])

    def test_gray_to_rgb_and_back_preserves_shape(self) -> None:
        image = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32).reshape(1, 1, 4, 4)
        rgb = _tensor_gray_to_rgb(image)
        self.assertEqual(tuple(rgb.shape), (1, 3, 4, 4))
        gray = _tensor_rgb_to_gray(rgb)
        self.assertEqual(tuple(gray.shape), (1, 1, 4, 4))
        self.assertTrue(torch.allclose(gray, image, atol=1e-6))

    def test_psnr_is_infinite_for_identical_images(self) -> None:
        image = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        self.assertTrue(math.isinf(_psnr(image, image)))


if __name__ == "__main__":
    unittest.main()
