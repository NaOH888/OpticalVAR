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

from optical.data import NpzImageDataset


class NpzDatasetTests(unittest.TestCase):
    def test_from_manifest_can_reduce_rgb_to_single_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            npz_path = tmp_path / "tiny.json".replace(".json", ".npz")
            manifest_path = tmp_path / "tiny.json"
            images = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
            labels = np.array([1, 5], dtype=np.int64)
            np.savez(npz_path, images=images, labels=labels)
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_key": "images",
                        "label_key": "labels"
                    }
                ),
                encoding="utf-8",
            )

            dataset = NpzImageDataset.from_manifest(manifest_path, channel_mode="first")
            sample = dataset[0]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(sample["image"].shape), (1, 4, 4))
            self.assertEqual(int(sample["sample_id"]), 0)
            self.assertEqual(int(sample["label"]), 1)
            self.assertTrue(torch.allclose(sample["image"][0], torch.as_tensor(images[0, 0])))
            dataset.close()

    def test_from_manifest_can_return_latent_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            npz_path = tmp_path / "tiny_pairs.npz"
            manifest_path = tmp_path / "tiny_pairs.json"
            images = np.ones((2, 1, 4, 4), dtype=np.float32)
            labels = np.array([2, 7], dtype=np.int64)
            latents = np.arange(2 * 1 * 2 * 2, dtype=np.float32).reshape(2, 1, 2, 2)
            np.savez(npz_path, teacher_images=images, labels=labels, latents=latents)
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_key": "teacher_images",
                        "label_key": "labels",
                        "latent_key": "latents"
                    }
                ),
                encoding="utf-8",
            )

            dataset = NpzImageDataset.from_manifest(manifest_path)
            sample = dataset[1]

            self.assertEqual(tuple(sample["image"].shape), (1, 4, 4))
            self.assertEqual(tuple(sample["latent"].shape), (1, 2, 2))
            self.assertEqual(int(sample["label"]), 7)
            self.assertTrue(torch.allclose(sample["latent"], torch.as_tensor(latents[1])))
            dataset.close()

    def test_from_manifest_can_return_rvq_codes_as_integer_latent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            npz_path = tmp_path / "tiny_rvq.npz"
            manifest_path = tmp_path / "tiny_rvq.json"
            images = np.ones((2, 1, 4, 4), dtype=np.float32)
            labels = np.array([2, 7], dtype=np.int64)
            rvq_codes = np.array([[1, 5, 9], [3, 4, 8]], dtype=np.int64)
            np.savez(npz_path, teacher_images=images, labels=labels, rvq_codes=rvq_codes)
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_key": "teacher_images",
                        "label_key": "labels",
                        "latent_source": "rvq",
                        "latent_type": "discrete_code",
                        "latent_key": "rvq_codes",
                        "latent_spec": {
                            "num_stages": 3,
                            "codebook_size": 256,
                            "shape": [3],
                        },
                    }
                ),
                encoding="utf-8",
            )

            dataset = NpzImageDataset.from_manifest(manifest_path)
            sample = dataset[1]

            self.assertEqual(dataset.latent_source, "rvq")
            self.assertEqual(dataset.latent_type, "discrete_code")
            self.assertEqual(tuple(sample["latent"].shape), (3,))
            self.assertEqual(sample["latent"].dtype, torch.long)
            self.assertTrue(torch.equal(sample["latent"], torch.tensor([3, 4, 8], dtype=torch.long)))
            dataset.close()

    def test_from_manifest_can_read_sharded_npz_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shard_0 = tmp_path / "tiny_part0000.npz"
            shard_1 = tmp_path / "tiny_part0001.npz"
            manifest_path = tmp_path / "tiny.json"
            np.savez(
                shard_0,
                images=np.ones((2, 1, 2, 2), dtype=np.float32),
                labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
                sample_ids=np.array([10, 11], dtype=np.int64),
            )
            np.savez(
                shard_1,
                images=np.full((1, 1, 2, 2), 2.0, dtype=np.float32),
                labels=np.array([[1, 1]], dtype=np.float32),
                sample_ids=np.array([12], dtype=np.int64),
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_key": "images",
                        "label_key": "labels",
                        "sample_id_key": "sample_ids",
                        "npz_files": [shard_0.name, shard_1.name],
                    }
                ),
                encoding="utf-8",
            )

            dataset = NpzImageDataset.from_manifest(manifest_path)
            sample = dataset[2]

            self.assertEqual(len(dataset), 3)
            self.assertEqual(tuple(sample["image"].shape), (1, 2, 2))
            self.assertEqual(int(sample["sample_id"]), 12)
            self.assertTrue(torch.allclose(sample["label"], torch.tensor([1.0, 1.0])))
            dataset.close()


if __name__ == "__main__":
    unittest.main()
