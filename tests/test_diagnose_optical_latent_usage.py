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

from scripts.diagnose_optical_latent_usage import main as diagnose_main
from scripts.train_optical_multiscale import _build_dataset_and_loader, _build_model


class DiagnoseOpticalLatentUsageScriptTests(unittest.TestCase):
    def _write_tiny_continuous_case(self, tmp_path: Path) -> tuple[Path, Path]:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        latent_npz = dataset_dir / "tiny_autoencoderkl.npz"
        latent_manifest = dataset_dir / "tiny_autoencoderkl.json"
        checkpoint_path = tmp_path / "latest.pt"

        teacher_images = np.linspace(0.0, 1.0, num=4 * 1 * 8 * 8, dtype=np.float32).reshape(4, 1, 8, 8)
        labels = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        latents = np.linspace(0.0, 1.0, num=4 * 4 * 2 * 2, dtype=np.float32).reshape(4, 4, 2, 2)
        sample_ids = np.arange(4, dtype=np.int64)

        np.savez(
            latent_npz,
            teacher_images=teacher_images,
            latents=latents,
            labels=labels,
            sample_ids=sample_ids,
        )
        latent_manifest.write_text(
            json.dumps(
                {
                    "dataset_name": "tiny_autoencoderkl",
                    "split": "train",
                    "image_key": "teacher_images",
                    "label_key": "labels",
                    "sample_id_key": "sample_ids",
                    "latent_source": "autoencoderkl",
                    "latent_type": "continuous_map",
                    "latent_key": "latents",
                    "latent_spec": {
                        "shape": [4, 2, 2],
                        "model_id": "fake/sd-vae",
                        "latent_mode": "mode",
                        "scaling_factor": 0.18215,
                    },
                    "npz_files": [latent_npz.name],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        config = {
            "runtime": {
                "seed": 7,
                "device": "cpu",
            },
            "dataset": {
                "manifest_path": str(latent_manifest),
                "channel_mode": "keep",
                "batch_size": 2,
                "shuffle": False,
                "num_workers": 0,
                "max_items": 4,
            },
            "multiscale": {
                "num_levels": 2,
                "max_freq_fraction": 1.0,
                "transition_width": 0.05,
            },
            "encoder": {
                "hidden_dim": 32,
                "latent_embed_dim": 16,
                "condition_embed_dim": 8,
                "fused_dim": 20,
                "fusion_mode": "concat",
                "fusion_hidden_dim": 24,
                "condition_mode": "attribute_vector",
                "condition_input_dim": 2,
                "class_embed_dim": 8,
                "condition_hidden_dim": 12,
                "output_height": 8,
                "output_width": 8,
                "phase_alpha_pi": 2.0,
                "weight_init": "kaiming_uniform",
                "output_weight_init": "xavier_uniform",
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
                    "share_across_channels": True,
                    "init_mode": "uniform",
                    "initial_phase_value_rad": 0.0,
                    "init_min_rad": 0.0,
                    "init_max_rad": 2.0,
                    "phase_grid_height": 4,
                    "phase_grid_width": 4,
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
                    "between_layers_m": 2.0e-6,
                    "last_layer_to_detector_m": 2.0e-6,
                },
            },
            "loss": {
                "final_weight": 1.0,
                "scale_weight": 1.0,
                "band_weight": 0.5,
                "tv_weight": 0.0,
                "background_weight": 0.0,
                "background_threshold": 0.05,
                "loss_type": "mse",
                "band_mode": "prefix_difference",
            },
            "training": {
                "epochs": 1,
                "lr": 1.0e-3,
                "weight_decay": 0.0,
                "log_interval": 1,
                "max_steps_per_epoch": 1,
                "save_every_epoch": False,
                "output_dir": str(tmp_path / "outputs"),
            },
        }

        dataset, _, _ = _build_dataset_and_loader(
            config,
            config_dir=tmp_path,
            repo_root=PROJECT_ROOT,
        )
        try:
            sample_item = dataset[0]
            model = _build_model(config, sample_item=sample_item).to(torch.device("cpu"))
            checkpoint = {
                "model": model.state_dict(),
                "config": config,
                "metrics": {
                    "epoch": 0,
                    "total_loss": 0.0,
                },
            }
            torch.save(checkpoint, checkpoint_path)
        finally:
            dataset.base_dataset.close()

        return checkpoint_path, latent_manifest

    def test_diagnose_script_runs_for_continuous_latent_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path, latent_manifest = self._write_tiny_continuous_case(tmp_path)
            output_dir = tmp_path / "diagnose_outputs"

            diagnose_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--data-manifest",
                    str(latent_manifest),
                    "--anchor-index",
                    "0",
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue((output_dir / "anchor_target.png").exists())
            self.assertTrue((output_dir / "anchor_phase.png").exists())
            self.assertTrue((output_dir / "latent_vary_final_grid.png").exists())
            self.assertTrue((output_dir / "condition_vary_final_grid.png").exists())
            self.assertTrue((output_dir / "summary.json").exists())

            payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["anchor_index"], 0)
            self.assertEqual(len(payload["latent_only"]), 2)
            self.assertEqual(len(payload["condition_only"]), 2)
            self.assertIsInstance(payload["anchor_label"], list)


if __name__ == "__main__":
    unittest.main()
