from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.export_autoencoder_latents import export_latents


class _FakeLatentDist:
    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent

    def sample(self) -> torch.Tensor:
        return self._latent


class _FakeEncodeOutput:
    def __init__(self, latent: torch.Tensor) -> None:
        self.latent_dist = _FakeLatentDist(latent)


class _FakeDecodeOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _FakeAutoencoderKL:
    config = type("Config", (), {"scaling_factor": 0.18215})()

    @classmethod
    def from_pretrained(cls, model_id: str):
        instance = cls()
        instance.model_id = model_id
        return instance

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def encode(self, vae_input: torch.Tensor) -> _FakeEncodeOutput:
        latent_base = torch.nn.functional.avg_pool2d(vae_input[:, :1], kernel_size=2, stride=2)
        latent = latent_base.repeat(1, 4, 1, 1)
        return _FakeEncodeOutput(latent)

    def decode(self, latent: torch.Tensor) -> _FakeDecodeOutput:
        gray = torch.nn.functional.interpolate(latent[:, :1], scale_factor=2, mode="nearest")
        sample = torch.cat([gray, gray, gray], dim=1).clamp(-1.0, 1.0)
        return _FakeDecodeOutput(sample)


class ExportAutoencoderLatentsTests(unittest.TestCase):
    def test_export_latents_writes_manifest_and_npz_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            image_npz = dataset_dir / "tiny_gray.npz"
            image_manifest = dataset_dir / "tiny_gray.json"
            config_path = tmp_path / "export_autoencoder.json"

            images = np.linspace(0.0, 1.0, num=8 * 1 * 4 * 4, dtype=np.float32).reshape(8, 1, 4, 4)
            labels = np.stack([np.arange(8) % 2, (np.arange(8) + 1) % 2], axis=1).astype(np.float32)
            sample_ids = np.arange(8, dtype=np.int64)
            np.savez(image_npz, images=images, labels=labels, sample_ids=sample_ids)
            image_manifest.write_text(
                json.dumps(
                    {
                        "dataset_name": "tiny",
                        "split": "train",
                        "image_key": "images",
                        "label_key": "labels",
                        "sample_id_key": "sample_ids",
                        "label_names": ["attr_a", "attr_b"],
                        "label_semantics": "tiny_attrs",
                        "condition_components": {"attributes_dim": 2, "landmarks_dim": 0, "total_dim": 2},
                    }
                ),
                encoding="utf-8",
            )

            config = {
                "runtime": {"seed": 7, "device": "cpu"},
                "dataset": {
                    "dataset_name": "tiny",
                    "split": "train",
                    "manifest_path": str(image_manifest),
                    "channel_mode": "keep",
                    "max_items": None,
                },
                "autoencoder": {
                    "model_id": "fake/model",
                    "latent_mode": "mode",
                    "batch_size": 4,
                },
                "export": {
                    "shard_size": 4,
                    "output_manifest_path": str(output_dir / "tiny_autoencoder.json"),
                },
            }
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            with patch("scripts.export_autoencoder_latents.load_autoencoder_cls", return_value=_FakeAutoencoderKL):
                result = export_latents(config, config_path=config_path)

            manifest_path = Path(result["output_manifest"])
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["latent_source"], "autoencoderkl")
            self.assertEqual(payload["latent_type"], "continuous_map")
            self.assertEqual(payload["latent_spec"]["shape"], [4, 2, 2])
            self.assertEqual(payload["num_items"], 8)
            self.assertEqual(len(payload["npz_files"]), 2)
            shard = np.load(output_dir / payload["npz_files"][0], allow_pickle=False)
            try:
                self.assertEqual(shard["teacher_images"].shape, (4, 1, 4, 4))
                self.assertEqual(shard["latents"].shape, (4, 4, 2, 2))
                self.assertEqual(shard["labels"].shape, (4, 2))
            finally:
                shard.close()


if __name__ == "__main__":
    unittest.main()
