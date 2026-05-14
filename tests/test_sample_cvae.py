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

from scripts.sample_cvae import main as sample_cvae_main
from vae import build_cvae


class SampleCVAEScriptTests(unittest.TestCase):
    def test_sample_cvae_supports_reconstruction_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            np.savez(
                tmp_path / "tiny.npz",
                images=np.random.rand(2, 1, 32, 32).astype(np.float32),
                labels=np.array([1, 4], dtype=np.int64),
            )
            (tmp_path / "tiny.json").write_text(
                json.dumps({"image_key": "images", "label_key": "labels"}),
                encoding="utf-8",
            )
            config = {
                "runtime": {"seed": 42, "device": "cpu"},
                "dataset": {
                    "manifest_path": str((tmp_path / "tiny.json").resolve()),
                    "channel_mode": "keep",
                    "batch_size": 2,
                    "shuffle": False,
                    "num_workers": 0,
                    "max_items": None,
                },
                "model": {
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
                },
            }
            model = build_cvae(config["model"])
            checkpoint_path = tmp_path / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
            output_dir = tmp_path / "outputs"
            sample_cvae_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--sample-index",
                    "0",
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )
            self.assertTrue((output_dir / "sample_0000_target.png").exists())
            self.assertTrue((output_dir / "sample_0000_recon.png").exists())
            self.assertTrue((output_dir / "sample_0000_prior.png").exists())

    def test_sample_cvae_supports_class_prior_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            np.savez(
                tmp_path / "tiny.npz",
                images=np.random.rand(2, 1, 32, 32).astype(np.float32),
                labels=np.array([1, 4], dtype=np.int64),
            )
            (tmp_path / "tiny.json").write_text(
                json.dumps({"image_key": "images", "label_key": "labels"}),
                encoding="utf-8",
            )
            config = {
                "runtime": {"seed": 42, "device": "cpu"},
                "dataset": {
                    "manifest_path": str((tmp_path / "tiny.json").resolve()),
                    "channel_mode": "keep",
                    "batch_size": 2,
                    "shuffle": False,
                    "num_workers": 0,
                    "max_items": None,
                },
                "model": {
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
                },
            }
            model = build_cvae(config["model"])
            checkpoint_path = tmp_path / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
            output_dir = tmp_path / "outputs"
            sample_cvae_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--random-prior",
                    "--label",
                    "3",
                    "--num-samples",
                    "4",
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )
            self.assertTrue((output_dir / "label_03_grid.png").exists())
