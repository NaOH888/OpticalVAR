from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train_optical_iterative_multiscale import IterativeStepLoss, main, train


class TrainOpticalIterativeMultiscaleScriptTests(unittest.TestCase):
    def _write_tiny_training_case(self, tmp_path: Path) -> tuple[Path, Path]:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        npz_path = dataset_dir / "tiny_autoencoder.npz"
        manifest_path = dataset_dir / "tiny_autoencoder.json"
        outputs_dir = tmp_path / "outputs"
        config_path = tmp_path / "train_iterative_config.json"

        teacher_images = np.linspace(
            0.0,
            1.0,
            num=4 * 1 * 8 * 8,
            dtype=np.float32,
        ).reshape(4, 1, 8, 8)
        latents = np.linspace(
            -1.0,
            1.0,
            num=4 * 4 * 2 * 2,
            dtype=np.float32,
        ).reshape(4, 4, 2, 2)
        labels = np.linspace(
            0.0,
            1.0,
            num=4 * 6,
            dtype=np.float32,
        ).reshape(4, 6)
        sample_ids = np.arange(4, dtype=np.int64)
        np.savez(
            npz_path,
            teacher_images=teacher_images,
            latents=latents,
            labels=labels,
            sample_ids=sample_ids,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "image_key": "teacher_images",
                    "label_key": "labels",
                    "latent_key": "latents",
                    "latent_source": "autoencoderkl",
                    "latent_type": "continuous_map",
                    "sample_id_key": "sample_ids",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(
                {
                    "runtime": {
                        "seed": 7,
                        "device": "cpu",
                    },
                    "dataset": {
                        "manifest_path": str(manifest_path),
                        "channel_mode": "keep",
                        "batch_size": 2,
                        "shuffle": False,
                        "num_workers": 0,
                        "max_items": 4,
                    },
                    "multiscale": {
                        "num_levels": 3,
                        "cutoff_mode": "power_equalized",
                        "max_freq_fraction": 1.0,
                        "transition_width": 0.05,
                    },
                    "iterative": {
                        "num_steps": 3,
                        "detach_prev_state": False,
                        "initial_state": "zeros",
                        "step_embedding_dim": 8,
                        "state_normalization": "mean_power",
                    },
                    "encoder": {
                        "latent_channels": [24, 16, 12],
                        "prev_image_channels": [12, 8],
                        "condition_mode": "attribute_vector",
                        "condition_input_dim": 6,
                        "condition_embed_dim": 12,
                        "fusion_hidden_dim": 16,
                        "output_height": 8,
                        "output_width": 8,
                        "weight_init": "kaiming_uniform",
                        "output_weight_init": "xavier_uniform",
                        "upsample_mode": "bilinear",
                    },
                    "optical": {
                        "source": {
                            "wavelengths_m": [5.32e-7],
                            "light_mode": "phase",
                            "amplitude": 1.0,
                        },
                        "slm": {
                            "pixel_pitch_x_m": 1.0e-6,
                            "pixel_pitch_y_m": 1.0e-6,
                            "pixel_count_x": 8,
                            "pixel_count_y": 8,
                            "dx_m": 1.0e-6,
                            "fill_factor": 1.0,
                            "phase_alpha": 2.0,
                            "phase_bit_depth": None,
                        },
                        "phase_layer": {
                            "alpha_pi": 2.0,
                            "modulation_mode": "phase",
                            "share_across_channels": True,
                            "init_mode": "uniform",
                            "init_min_rad": 0.0,
                            "init_max_rad": 2.0,
                            "phase_grid_height": 8,
                            "phase_grid_width": 8,
                        },
                        "detector": {
                            "width_num": 8,
                            "height_num": 8,
                            "detector_unit_len_m": 1.0e-6,
                        },
                        "propagation": {
                            "canvas_h": None,
                            "canvas_w": None,
                            "canvas_factor": 1.0,
                            "refractive_index": 1.0,
                            "use_bandlimit_window": False,
                            "evanescent_mode": "keep",
                            "fft_norm": "ortho",
                        },
                        "error": {
                            "delta_z_m": 0.0,
                            "shift_x_m": 0.0,
                            "shift_y_m": 0.0,
                            "error_factor": 1.0,
                        },
                        "distances_m": {
                            "slm_to_first_layer_m": 2.0e-6,
                            "between_layers_m": [2.0e-6, 2.0e-6],
                            "last_layer_to_detector_m": 2.0e-6,
                        },
                    },
                    "loss": {
                        "loss_type": "l1",
                        "perceptual_weight": 0.0,
                    },
                    "training": {
                        "epochs": 1,
                        "lr": 1.0e-3,
                        "grad_clip_norm": 1.0,
                        "weight_decay": 0.0,
                        "log_interval": 1,
                        "max_steps_per_epoch": 1,
                        "output_dir": str(outputs_dir),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return config_path, outputs_dir

    def test_script_runs_one_epoch_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = self._write_tiny_training_case(tmp_path)

            main(["--config", str(config_path)])

            self.assertTrue((outputs_dir / "latest.pt").exists())
            self.assertTrue((outputs_dir / "history.jsonl").exists())
            history_lines = (outputs_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
            payload = json.loads(history_lines[-1])
            self.assertIn("total_loss", payload)
            self.assertIn("final_loss", payload)
            self.assertIn("scale_loss", payload)
            self.assertIn("scale_losses", payload)
            self.assertEqual(len(payload["scale_losses"]), 2)
            self.assertIn("band_loss", payload)
            self.assertIn("tv_loss", payload)
            self.assertIn("background_loss", payload)
            self.assertIn("perceptual_loss", payload)
            self.assertIn("latent_diversity_loss", payload)
            self.assertIn("condition_correlation_loss", payload)
            self.assertIn("optimizer_lrs", payload)

    def test_train_requires_num_steps_match_num_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, _ = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["iterative"]["num_steps"] = 2

            with self.assertRaises(ValueError):
                train(config, config_path=config_path)

    def test_train_allows_optical_num_layers_different_from_num_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["optical"]["num_layers"] = 2
            config["optical"]["distances_m"]["between_layers_m"] = [2.0e-6]

            result = train(config, config_path=config_path)

            self.assertTrue((outputs_dir / "latest.pt").exists())
            self.assertIn("metrics", result)

    def test_train_supports_disabling_prev_image_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["iterative"]["use_prev_image"] = False

            result = train(config, config_path=config_path)

            self.assertTrue((outputs_dir / "latest.pt").exists())
            self.assertIn("metrics", result)

    def test_train_supports_landmark_heatmap_condition_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["encoder"]["use_landmark_heatmap"] = True
            config["encoder"]["condition_attribute_dim"] = 4
            config["encoder"]["condition_landmark_dim"] = 2
            config["encoder"]["landmark_heatmap_sigma_px"] = 2.0

            result = train(config, config_path=config_path)

            self.assertTrue((outputs_dir / "latest.pt").exists())
            self.assertIn("metrics", result)

    def test_train_supports_step1_latent_diversity_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["loss"]["latent_diversity_weight"] = 0.5
            config["loss"]["latent_diversity_margin"] = 0.1

            result = train(config, config_path=config_path)

            self.assertTrue((outputs_dir / "latest.pt").exists())
            self.assertIn("metrics", result)
            self.assertIn("latent_diversity_loss", result["metrics"])

    def test_train_supports_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))

            first_result = train(config, config_path=config_path)
            latest_checkpoint = Path(first_result["latest_checkpoint"])
            self.assertTrue(latest_checkpoint.exists())

            config["training"]["epochs"] = 2
            second_result = train(config, config_path=config_path, resume_path_override=latest_checkpoint)

            self.assertEqual(second_result["metrics"]["epoch"], 2)
            self.assertEqual(second_result["resumed_from"], str(latest_checkpoint))
            history_lines = (outputs_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(history_lines), 2)

    def test_train_supports_split_encoder_decoder_learning_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["training"]["lr"] = 1.0e-3
            config["training"]["encoder_lr"] = 2.0e-4
            config["training"]["decoder_lr"] = 8.0e-4

            result = train(config, config_path=config_path)

            self.assertTrue((outputs_dir / "latest.pt").exists())
            self.assertEqual(result["metrics"]["optimizer_lrs"]["encoder"], 2.0e-4)
            self.assertEqual(result["metrics"]["optimizer_lrs"]["decoder"], 8.0e-4)

    def test_iterative_step_loss_supports_step_weights(self) -> None:
        criterion = IterativeStepLoss(
            num_steps=3,
            loss_type="l1",
            step_weights=[4.0, 2.0],
            perceptual_weight=0.0,
            state_normalization="mean_power",
        )
        predictions = (
            torch.tensor(
                [[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 1.0], [0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 1.0], [0.0, 0.0]]], [[[0.0, 1.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 1.0], [1.0, 0.0]]], [[[0.0, 1.0], [1.0, 1.0]]]],
                dtype=torch.float32,
            ),
        )
        batch = {
            "target_scale_1": torch.tensor(
                [[[[1.0, 1.0], [0.0, 0.0]]], [[[1.0, 0.0], [0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            "target_scale_2": torch.tensor(
                [[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 0.0], [1.0, 1.0]]]],
                dtype=torch.float32,
            ),
            "target_scale_3": torch.tensor(
                [[[[1.0, 0.0], [0.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
        }
        output = criterion(predictions=predictions, batch=batch)

        scale_losses = output["scale_losses"]
        self.assertEqual(len(scale_losses), 2)
        expected_weighted = (
            4.0 * float(scale_losses[0].detach().cpu())
            + 2.0 * float(scale_losses[1].detach().cpu())
        ) / 6.0
        self.assertAlmostEqual(float(output["scale_loss"].detach().cpu()), expected_weighted, places=6)

    def test_iterative_step_loss_keeps_perceptual_out_of_scale_and_final(self) -> None:
        class ConstantPerceptual(torch.nn.Module):
            def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return prediction.new_tensor(2.0)

        criterion = IterativeStepLoss(
            num_steps=2,
            loss_type="l1",
            step_weights=[1.0],
            perceptual_weight=0.5,
            perceptual_loss_fn=ConstantPerceptual(),
            state_normalization="mean_power",
        )
        predictions = (
            torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32),
        )
        batch = {
            "target_scale_1": torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32),
            "target_scale_2": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        }

        output = criterion(predictions=predictions, batch=batch)

        step_losses = output["scale_losses"]
        expected_scale = float(step_losses[0].detach().cpu())
        self.assertAlmostEqual(float(output["scale_loss"].detach().cpu()), expected_scale, places=6)
        self.assertAlmostEqual(
            float(output["final_loss"].detach().cpu()),
            float(criterion._scale_aligned_loss(predictions[-1], batch["target_scale_2"]).detach().cpu()),
            places=6,
        )
        self.assertAlmostEqual(float(output["perceptual_loss"].detach().cpu()), 2.0, places=6)
        self.assertAlmostEqual(
            float(output["total_loss"].detach().cpu()),
            expected_scale + float(output["final_loss"].detach().cpu()) + 1.0,
            places=6,
        )

    def test_iterative_step_loss_supports_all_step_latent_div_weights(self) -> None:
        criterion = IterativeStepLoss(
            num_steps=3,
            loss_type="l1",
            step_weights=[1.0, 1.0],
            latent_diversity_weight=1.0,
            latent_diversity_margin=0.1,
            latent_div_step_weights=[1.0, 2.0, 4.0],
            state_normalization="mean_power",
        )
        predictions = (
            torch.tensor(
                [[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 1.0], [0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 1.0], [0.0, 0.0]]], [[[0.0, 0.0], [1.0, 0.0]]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[1.0, 1.0], [1.0, 0.0]]], [[[0.0, 0.0], [1.0, 1.0]]]],
                dtype=torch.float32,
            ),
        )
        batch = {
            "target_scale_1": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
            "target_scale_2": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
            "target_scale_3": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        }
        latent_input = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

        output = criterion(predictions=predictions, batch=batch, latent_input=latent_input)

        per_step = [
            float(
                criterion._latent_diversity_loss(
                    prediction=prediction,
                    latent_input=latent_input,
                ).detach().cpu()
            )
            for prediction in predictions
        ]
        expected = (1.0 * per_step[0] + 2.0 * per_step[1] + 4.0 * per_step[2]) / 7.0
        self.assertAlmostEqual(float(output["latent_diversity_loss"].detach().cpu()), expected, places=6)

    def test_iterative_step_loss_supports_condition_correlation_loss(self) -> None:
        class TinyPerceptual(torch.nn.Module):
            def extract_features(self, image: torch.Tensor) -> list[torch.Tensor]:
                return [image]

            def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return torch.mean(torch.abs(prediction - target))

        criterion = IterativeStepLoss(
            num_steps=2,
            loss_type="l1",
            step_weights=[1.0],
            perceptual_weight=0.0,
            perceptual_loss_fn=TinyPerceptual(),
            condition_corr_weight=1.0,
            condition_corr_step_weights=[1.0, 2.0],
            state_normalization="mean_power",
        )
        predictions = (
            torch.tensor(
                [
                    [[[0.0, 0.0], [0.0, 0.0]]],
                    [[[1.0, 0.0], [0.0, 0.0]]],
                    [[[2.0, 0.0], [0.0, 0.0]]],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [[[0.0, 0.0], [0.0, 0.0]]],
                    [[[0.5, 0.0], [0.0, 0.0]]],
                    [[[1.0, 0.0], [0.0, 0.0]]],
                ],
                dtype=torch.float32,
            ),
        )
        batch = {
            "target_scale_1": torch.zeros((3, 1, 2, 2), dtype=torch.float32),
            "target_scale_2": torch.zeros((3, 1, 2, 2), dtype=torch.float32),
        }
        condition_input = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        output = criterion(predictions=predictions, batch=batch, condition_input=condition_input)

        self.assertGreaterEqual(float(output["condition_correlation_loss"].detach().cpu()), 0.0)
        self.assertIn("condition_correlation_loss", output)


if __name__ == "__main__":
    unittest.main()
