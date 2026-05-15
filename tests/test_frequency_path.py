from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optical.data import FrequencyPathDataset, MultiScaleFrequencyTargetTransform


class _TupleImageDataset(Dataset):
    def __init__(self) -> None:
        self.images = [
            torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0,
            torch.flip(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0, dims=(-1,)),
        ]
        self.labels = [3, 7]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.labels[index]


class FrequencyPathTests(unittest.TestCase):
    def test_transform_returns_aligned_scale_and_band_targets(self) -> None:
        transform = MultiScaleFrequencyTargetTransform(
            num_levels=3,
            max_freq_fraction=1.0,
            transition_width=0.05,
        )
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0

        output = transform(image)

        self.assertIn("target_final", output)
        self.assertIn("target_scales", output)
        self.assertIn("target_bands", output)
        self.assertEqual(len(output["target_scales"]), 3)
        self.assertEqual(len(output["target_bands"]), 3)
        self.assertTrue(torch.allclose(output["target_scale_3"], image))
        self.assertTrue(torch.allclose(output["target_final"], image))
        self.assertTrue(
            torch.allclose(
                output["target_band_1"] + output["target_band_2"] + output["target_band_3"],
                image,
                atol=1e-5,
            )
        )

    def test_transform_accepts_explicit_cutoffs(self) -> None:
        transform = MultiScaleFrequencyTargetTransform(
            num_levels=3,
            transition_width=0.05,
            cutoffs=[0.12, 0.34],
        )

        output = transform(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0)

        self.assertEqual(transform.cutoffs, (0.12, 0.34))
        self.assertEqual(len(output["target_scales"]), 3)
        self.assertEqual(len(output["target_bands"]), 3)

    def test_dataset_wraps_tuple_dataset_and_attaches_targets(self) -> None:
        dataset = FrequencyPathDataset(
            _TupleImageDataset(),
            MultiScaleFrequencyTargetTransform(num_levels=2),
        )

        sample = dataset[0]

        self.assertIn("input", sample)
        self.assertIn("label", sample)
        self.assertIn("target_final", sample)
        self.assertIn("target_scale_1", sample)
        self.assertIn("target_scale_2", sample)
        self.assertIn("target_band_1", sample)
        self.assertIn("target_band_2", sample)
        self.assertEqual(int(sample["label"]), 3)
        self.assertEqual(tuple(sample["input"].shape), (1, 4, 4))
        self.assertEqual(tuple(sample["target_final"].shape), (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
