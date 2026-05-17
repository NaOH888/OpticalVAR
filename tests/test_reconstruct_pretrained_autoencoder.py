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

from scripts._pretrained_autoencoder_utils import l2_loss as _l2_loss
from scripts._pretrained_autoencoder_utils import psnr as _psnr
from scripts._pretrained_autoencoder_utils import tensor_gray_to_rgb as _tensor_gray_to_rgb
from scripts._pretrained_autoencoder_utils import tensor_rgb_to_gray as _tensor_rgb_to_gray
from scripts.reconstruct_pretrained_autoencoder import _parse_indices


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

    def test_l2_loss_is_zero_for_identical_images(self) -> None:
        image = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        self.assertEqual(_l2_loss(image, image), 0.0)


if __name__ == "__main__":
    unittest.main()
