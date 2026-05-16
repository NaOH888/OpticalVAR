from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optical.data import MultiScaleFrequencyTargetTransform
from optical.losses import OpticalMultiscaleLoss
from vae import build_perceptual_loss


class OpticalMultiscaleLossTests(unittest.TestCase):
    def test_loss_is_zero_when_predictions_match_targets(self) -> None:
        transform = MultiScaleFrequencyTargetTransform(num_levels=3)
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0
        targets = transform(image)
        outputs = {
            "final_detector": targets["target_final"].unsqueeze(1),
            "prefix_readout_1": targets["target_scale_1"].unsqueeze(1),
            "prefix_readout_2": targets["target_scale_2"].unsqueeze(1),
            "prefix_readout_3": targets["target_scale_3"].unsqueeze(1),
        }
        criterion = OpticalMultiscaleLoss(
            num_levels=3,
            final_weight=1.0,
            scale_weight=1.0,
            band_weight=1.0,
        )

        losses = criterion(outputs, targets)

        self.assertTrue(torch.allclose(losses["total_loss"], torch.zeros(()), atol=1e-6))
        self.assertTrue(torch.allclose(losses["final_loss"], torch.zeros(()), atol=1e-6))
        self.assertTrue(torch.allclose(losses["scale_loss"], torch.zeros(()), atol=1e-6))
        self.assertTrue(torch.allclose(losses["band_loss"], torch.zeros(()), atol=1e-6))
        self.assertTrue(torch.allclose(losses["tv_loss"], torch.zeros(()), atol=1e-6))
        self.assertTrue(torch.allclose(losses["background_loss"], torch.zeros(()), atol=1e-6))
        self.assertEqual(len(losses["scale_losses"]), 3)
        self.assertEqual(len(losses["band_losses"]), 3)

    def test_loss_increases_when_prefix_prediction_is_perturbed(self) -> None:
        transform = MultiScaleFrequencyTargetTransform(num_levels=2)
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0
        targets = transform(image)
        outputs = {
            "final_detector": targets["target_final"].unsqueeze(1),
            "prefix_readout_1": (targets["target_scale_1"] + 0.1).unsqueeze(1),
            "prefix_readout_2": targets["target_scale_2"].unsqueeze(1),
        }
        criterion = OpticalMultiscaleLoss(
            num_levels=2,
            final_weight=1.0,
            scale_weight=1.0,
            band_weight=0.0,
        )

        losses = criterion(outputs, targets)

        self.assertGreater(float(losses["total_loss"]), 0.0)
        self.assertGreater(float(losses["scale_loss"]), 0.0)
        self.assertEqual(len(losses["scale_losses"]), 2)
        self.assertEqual(len(losses["band_losses"]), 0)

    def test_loss_accepts_batched_dataset_targets(self) -> None:
        transform = MultiScaleFrequencyTargetTransform(num_levels=2)
        image_a = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0
        image_b = torch.flip(image_a, dims=(-1,))
        targets_a = transform(image_a)
        targets_b = transform(image_b)
        batched_targets = {
            key: torch.stack((targets_a[key], targets_b[key]), dim=0)
            for key in ("target_final", "target_scale_1", "target_scale_2", "target_band_1", "target_band_2")
        }
        outputs = {
            "final_detector": batched_targets["target_final"],
            "prefix_readout_1": batched_targets["target_scale_1"],
            "prefix_readout_2": batched_targets["target_scale_2"],
        }
        criterion = OpticalMultiscaleLoss(
            num_levels=2,
            final_weight=1.0,
            scale_weight=1.0,
            band_weight=1.0,
        )

        losses = criterion(outputs, batched_targets)

        self.assertTrue(torch.allclose(losses["total_loss"], torch.zeros(()), atol=1e-6))
        self.assertEqual(len(losses["scale_losses"]), 2)
        self.assertEqual(len(losses["band_losses"]), 2)

    def test_tv_loss_is_reported_when_enabled(self) -> None:
        image = torch.zeros((1, 4, 4), dtype=torch.float32)
        outputs = {
            "final_detector": image.unsqueeze(1),
            "prefix_readout_1": image.unsqueeze(1),
        }
        targets = {
            "target_final": image,
            "target_scale_1": image,
            "target_band_1": image,
        }
        outputs["final_detector"] = outputs["final_detector"].clone()
        outputs["final_detector"][0, 0, 1, 1] = 1.0
        criterion = OpticalMultiscaleLoss(
            num_levels=1,
            final_weight=1.0,
            scale_weight=0.0,
            band_weight=0.0,
            tv_weight=0.5,
        )

        losses = criterion(outputs, targets)

        self.assertGreater(float(losses["tv_loss"]), 0.0)
        self.assertGreater(float(losses["total_loss"]), float(losses["final_loss"]))

    def test_scale_aligned_losses_ignore_global_intensity_scaling(self) -> None:
        transform = MultiScaleFrequencyTargetTransform(num_levels=2)
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0
        targets = transform(image)
        outputs = {
            "final_detector": (targets["target_final"] * 2.5).unsqueeze(1),
            "prefix_readout_1": (targets["target_scale_1"] * 2.5).unsqueeze(1),
            "prefix_readout_2": (targets["target_scale_2"] * 2.5).unsqueeze(1),
        }
        criterion = OpticalMultiscaleLoss(
            num_levels=2,
            final_weight=1.0,
            scale_weight=1.0,
            band_weight=0.0,
        )

        losses = criterion(outputs, targets)

        self.assertTrue(torch.allclose(losses["final_loss"], torch.zeros(()), atol=1e-6))
        self.assertTrue(torch.allclose(losses["scale_loss"], torch.zeros(()), atol=1e-6))

    def test_background_loss_is_reported_when_enabled(self) -> None:
        image = torch.zeros((1, 4, 4), dtype=torch.float32)
        outputs = {
            "final_detector": image.unsqueeze(1).clone(),
            "prefix_readout_1": image.unsqueeze(1),
        }
        targets = {
            "target_final": image,
            "target_scale_1": image,
            "target_band_1": image,
        }
        outputs["final_detector"][0, 0, 0, 0] = 1.0
        criterion = OpticalMultiscaleLoss(
            num_levels=1,
            final_weight=1.0,
            scale_weight=0.0,
            band_weight=0.0,
            background_weight=0.25,
            background_threshold=0.05,
        )

        losses = criterion(outputs, targets)

        self.assertGreater(float(losses["background_loss"]), 0.0)
        self.assertGreater(float(losses["total_loss"]), float(losses["final_loss"]))

    def test_perceptual_loss_is_reported_when_enabled(self) -> None:
        image = torch.rand((1, 32, 32), dtype=torch.float32)
        outputs = {
            "final_detector": image.unsqueeze(1).clone(),
            "prefix_readout_1": image.unsqueeze(1),
        }
        targets = {
            "target_final": torch.zeros_like(image),
            "target_scale_1": image,
            "target_band_1": image,
        }
        criterion = OpticalMultiscaleLoss(
            num_levels=1,
            final_weight=1.0,
            scale_weight=0.0,
            band_weight=0.0,
            perceptual_weight=0.1,
            perceptual_loss_fn=build_perceptual_loss(
                {
                    "perceptual_weight": 0.1,
                    "perceptual_weights": "none",
                    "perceptual_feature_layers": [3],
                }
            ),
        )

        losses = criterion(outputs, targets)

        self.assertGreaterEqual(float(losses["perceptual_loss"]), 0.0)
        self.assertGreater(float(losses["total_loss"]), float(losses["final_loss"]))

    def test_level_weights_can_be_updated_after_construction(self) -> None:
        criterion = OpticalMultiscaleLoss(
            num_levels=3,
            level_weights=(1.0, 1.0, 1.0),
        )

        criterion.set_level_weights((3.0, 2.0, 1.0))

        self.assertEqual(criterion.level_weights, (3.0, 2.0, 1.0))

    def test_latent_diversity_loss_penalizes_collapsed_outputs(self) -> None:
        image = torch.zeros((2, 8, 8), dtype=torch.float32)
        outputs = {
            "final_detector": image.unsqueeze(1),
            "prefix_readout_1": image.unsqueeze(1),
        }
        targets = {
            "target_final": image,
            "target_scale_1": image,
            "target_band_1": image,
        }
        latent_input = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
        criterion = OpticalMultiscaleLoss(
            num_levels=1,
            final_weight=1.0,
            scale_weight=0.0,
            band_weight=0.0,
            latent_diversity_weight=1.0,
            latent_diversity_margin=0.2,
        )

        losses = criterion(outputs, targets, latent_input=latent_input)

        self.assertGreater(float(losses["latent_diversity_loss"]), 0.0)
        self.assertGreater(float(losses["total_loss"]), float(losses["final_loss"]))


if __name__ == "__main__":
    unittest.main()
