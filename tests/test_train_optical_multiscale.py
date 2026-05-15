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

from scripts.train_optical_multiscale import (
    _build_dataset_and_loader,
    _build_model,
    _build_model_inputs,
    _move_batch_to_device,
    main,
    train,
)


class TrainOpticalMultiscaleScriptTests(unittest.TestCase):
    def _write_tiny_training_case(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        npz_path = dataset_dir / "tiny_fashion.npz"
        manifest_path = dataset_dir / "tiny_fashion.json"
        outputs_dir = tmp_path / "outputs"
        config_path = tmp_path / "train_config.json"

        images = np.linspace(0.0, 1.0, num=4 * 3 * 8 * 8, dtype=np.float32).reshape(4, 3, 8, 8)
        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        np.savez(npz_path, images=images, labels=labels)
        manifest_path.write_text(
            json.dumps({"image_key": "images", "label_key": "labels"}),
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(
                {
                    "runtime": {
                        "seed": 7,
                        "device": "cpu"
                    },
                    "dataset": {
                        "manifest_path": str(manifest_path),
                        "channel_mode": "first",
                        "batch_size": 2,
                        "shuffle": False,
                        "num_workers": 0,
                        "max_items": 4
                    },
                    "multiscale": {
                        "num_levels": 2,
                        "max_freq_fraction": 1.0,
                        "transition_width": 0.05
                    },
                    "encoder": {
                        "noise_channels": 1,
                        "latent_seed": 12345,
                        "hidden_dim": 32,
                        "latent_embed_dim": 24,
                        "condition_embed_dim": 12,
                        "fused_dim": 20,
                        "input_height": 8,
                        "input_width": 8,
                        "output_height": 8,
                        "output_width": 8,
                        "phase_alpha_pi": 2.0,
                        "class_conditional": True,
                        "num_classes": 10,
                        "class_embed_dim": 8,
                        "class_condition_channels": 2,
                        "weight_init": "kaiming_uniform",
                        "output_weight_init": "xavier_uniform",
                        "embedding_init_std": 0.02
                    },
                    "optical": {
                        "source": {
                            "wavelengths_m": [5.32e-7],
                            "light_mode": "phase",
                            "amplitude": 1.0
                        },
                        "slm": {
                            "pixel_pitch_x_m": 1.0e-6,
                            "pixel_pitch_y_m": 1.0e-6,
                            "pixel_count_x": 8,
                            "pixel_count_y": 8,
                            "dx_m": 1.0e-6,
                            "fill_factor": 1.0,
                            "phase_alpha": 2.0,
                            "phase_bit_depth": None
                        },
                        "phase_layer": {
                            "alpha_pi": 2.0,
                            "share_across_channels": True,
                            "init_mode": "uniform",
                            "initial_phase_value_rad": 0.0,
                            "init_min_rad": 0.0,
                            "init_max_rad": 2.0,
                            "phase_grid_height": 4,
                            "phase_grid_width": 4
                        },
                        "detector": {
                            "width_num": 8,
                            "height_num": 8,
                            "detector_unit_len_m": 1.0e-6
                        },
                        "propagation": {
                            "canvas_h": None,
                            "canvas_w": None,
                            "canvas_factor": 1.0,
                            "refractive_index": 1.0,
                            "use_bandlimit_window": False,
                            "evanescent_mode": "keep",
                            "fft_norm": "ortho"
                        },
                        "error": {
                            "delta_z_m": 0.0,
                            "shift_x_m": 0.0,
                            "shift_y_m": 0.0,
                            "error_factor": 1.0
                        },
                        "distances_m": {
                            "slm_to_first_layer_m": 2.0e-6,
                            "between_layers_m": 2.0e-6,
                            "last_layer_to_detector_m": 2.0e-6
                        }
                    },
                    "loss": {
                        "final_weight": 1.0,
                        "scale_weight": 1.0,
                        "band_weight": 1.0,
                        "tv_weight": 0.01,
                        "background_weight": 0.02,
                        "background_threshold": 0.05,
                        "loss_type": "mse",
                        "band_mode": "prefix_difference"
                    },
                    "training": {
                        "epochs": 1,
                        "lr": 1.0e-3,
                        "weight_decay": 0.0,
                        "log_interval": 1,
                        "max_steps_per_epoch": 1,
                        "save_every_epoch": False,
                        "output_dir": str(outputs_dir)
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest_path, outputs_dir, config_path

    def test_script_runs_one_epoch_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, outputs_dir, config_path = self._write_tiny_training_case(tmp_path)

            main(["--config", str(config_path)])

            self.assertTrue((outputs_dir / "latest.pt").exists())
            self.assertTrue((outputs_dir / "history.jsonl").exists())
            self.assertTrue((outputs_dir / "resolved_config.json").exists())
            history_lines = (outputs_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(history_lines)
            payload = json.loads(history_lines[-1])
            self.assertIn("tv_loss", payload)
            self.assertIn("background_loss", payload)

    def test_train_can_resume_with_updated_loss_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, outputs_dir, config_path = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))

            first_result = train(config, config_path=config_path)
            self.assertEqual(int(first_result["start_epoch"]), 0)
            first_checkpoint_path = Path(first_result["latest_checkpoint"])
            first_checkpoint = torch.load(first_checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(int(first_checkpoint["epoch"]), 1)

            config["training"]["epochs"] = 2
            config["loss"]["scale_weight"] = 3.0
            config["training"]["resume_optimizer"] = True
            second_result = train(
                config,
                config_path=config_path,
                resume_path_override=first_checkpoint_path,
            )

            self.assertEqual(int(second_result["start_epoch"]), 1)
            self.assertEqual(second_result["resumed_from"], str(first_checkpoint_path))
            resumed_checkpoint = torch.load(first_checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(int(resumed_checkpoint["epoch"]), 2)
            self.assertAlmostEqual(float(resumed_checkpoint["config"]["loss"]["scale_weight"]), 3.0)

            history_lines = (outputs_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(history_lines), 2)

    def test_fixed_latent_depends_on_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, config_path = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            dataset, _, _ = _build_dataset_and_loader(
                config,
                config_dir=config_path.parent,
                repo_root=PROJECT_ROOT,
            )
            sample_a = dataset[0]
            sample_b = dataset[1]
            batch_a = next(iter(torch.utils.data.DataLoader([sample_a], batch_size=1)))
            batch_b = next(iter(torch.utils.data.DataLoader([sample_b], batch_size=1)))
            batch_a = _move_batch_to_device(batch_a, torch.device("cpu"))
            batch_b = _move_batch_to_device(batch_b, torch.device("cpu"))
            model = _build_model(config, sample_item=sample_a).to(torch.device("cpu"))

            latent_a_1, labels_a_1 = _build_model_inputs(
                batch_a,
                model=model,
                config=config,
                device=torch.device("cpu"),
            )
            latent_a_2, labels_a_2 = _build_model_inputs(
                batch_a,
                model=model,
                config=config,
                device=torch.device("cpu"),
            )
            latent_b, labels_b = _build_model_inputs(
                batch_b,
                model=model,
                config=config,
                device=torch.device("cpu"),
            )

            self.assertTrue(torch.allclose(latent_a_1, latent_a_2))
            self.assertFalse(torch.allclose(latent_a_1, latent_b))
            self.assertEqual(int(labels_a_1[0]), int(batch_a["label"][0]))
            self.assertEqual(int(labels_a_2[0]), int(batch_a["label"][0]))
            self.assertEqual(int(labels_b[0]), int(batch_b["label"][0]))
            dataset.base_dataset.close()

    def test_dataset_latent_overrides_fixed_latent_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, config_path = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = _build_model(
                config,
                sample_item={
                    "target_final": torch.zeros((1, 8, 8), dtype=torch.float32),
                    "latent": torch.zeros((1, 8, 8), dtype=torch.float32),
                },
            ).to(torch.device("cpu"))
            batch = {
                "target_final": torch.zeros((2, 1, 8, 8), dtype=torch.float32),
                "latent": torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8),
                "label": torch.tensor([3, 5], dtype=torch.long),
            }

            latent, labels = _build_model_inputs(
                batch,
                model=model,
                config=config,
                device=torch.device("cpu"),
            )

            self.assertTrue(torch.allclose(latent, batch["latent"]))
            self.assertEqual(int(labels[0]), 3)
            self.assertEqual(int(labels[1]), 5)

    def test_discrete_latent_codes_are_accepted_by_model_inputs_and_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, config_path = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["encoder"]["rvq_codebook_size"] = 16
            sample_item = {
                "target_final": torch.zeros((1, 8, 8), dtype=torch.float32),
                "latent": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "label": torch.tensor(3, dtype=torch.long),
            }
            model = _build_model(config, sample_item=sample_item).to(torch.device("cpu"))
            batch = {
                "target_final": torch.zeros((2, 1, 8, 8), dtype=torch.float32),
                "latent": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
                "label": torch.tensor([3, 5], dtype=torch.long),
            }

            latent, labels = _build_model_inputs(
                batch,
                model=model,
                config=config,
                device=torch.device("cpu"),
            )
            outputs = model(latent, class_labels=labels)

            self.assertEqual(latent.dtype, torch.long)
            self.assertEqual(int(labels[0]), 3)
            self.assertEqual(tuple(outputs["encoder_output"].shape), (2, 1, 8, 8))

    def test_dataset_builder_can_load_cutoffs_from_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, config_path = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            cutoffs_path = tmp_path / "cutoffs.json"
            cutoffs_path.write_text(
                json.dumps({"cutoffs": [0.12]}, indent=2),
                encoding="utf-8",
            )
            config["multiscale"]["cutoffs_path"] = str(cutoffs_path)
            config["multiscale"].pop("max_freq_fraction", None)
            dataset, _, transform = _build_dataset_and_loader(
                config,
                config_dir=config_path.parent,
                repo_root=PROJECT_ROOT,
            )

            self.assertEqual(transform.cutoffs, (0.12,))
            sample = dataset[0]
            self.assertIn("target_band_1", sample)
            self.assertIn("target_band_2", sample)
            dataset.base_dataset.close()

    def test_phase_layer_pitch_and_grid_can_be_independent_from_slm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, _, config_path = self._write_tiny_training_case(tmp_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["optical"]["phase_layer"]["phase_grid_height"] = 2
            config["optical"]["phase_layer"]["phase_grid_width"] = 3
            config["optical"]["phase_layer"]["phase_pitch_x_m"] = 2.0e-6
            config["optical"]["phase_layer"]["phase_pitch_y_m"] = 3.0e-6

            model = _build_model(
                config,
                sample_item={
                    "target_final": torch.zeros((1, 8, 8), dtype=torch.float32),
                },
            )
            phase_layer = model.optical_decoder.optical_layers[0]

            self.assertEqual(int(phase_layer.phase_grid_height), 2)
            self.assertEqual(int(phase_layer.phase_grid_width), 3)
            self.assertAlmostEqual(float(phase_layer.width), 6.0e-6)
            self.assertAlmostEqual(float(phase_layer.height), 6.0e-6)
            self.assertEqual(int(phase_layer.sx), 6)
            self.assertEqual(int(phase_layer.sy), 6)

if __name__ == "__main__":
    unittest.main()
