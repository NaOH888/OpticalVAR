from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optical.data import ReferencedImageLatentDataset
from scripts.export_rvq_pairs import export_pairs


class ExportRVQPairsTests(unittest.TestCase):
    def test_export_pairs_references_existing_image_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            image_npz = dataset_dir / "tiny_gray.npz"
            image_manifest = dataset_dir / "tiny_gray.json"
            config_path = tmp_path / "export_rvq.json"

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
                        "label_names": ["attr_a", "attr_b", "left_eye_x", "left_eye_y"],
                        "label_semantics": "attrs_plus_landmarks",
                        "condition_components": {
                            "attributes_dim": 2,
                            "landmarks_dim": 2,
                            "total_dim": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = {
                "runtime": {
                    "seed": 7,
                },
                "dataset": {
                    "dataset_name": "tiny",
                    "split": "train",
                    "manifest_path": str(image_manifest),
                    "channel_mode": "keep",
                    "max_items": None,
                },
                "rvq": {
                    "pca_dim": 4,
                    "num_stages": 2,
                    "codebook_size": 3,
                    "batch_size": 4,
                    "pca_fit_max_items": 4,
                    "rvq_fit_max_items": 4,
                },
                "export": {
                    "shard_size": 4,
                    "output_manifest_path": str(output_dir / "tiny_rvq.json"),
                },
            }
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

            result = export_pairs(config, config_path=config_path)
            latent_manifest = Path(result["output_manifest"])
            payload = json.loads(latent_manifest.read_text(encoding="utf-8"))

            self.assertEqual(payload["latent_source"], "rvq")
            self.assertEqual(payload["latent_type"], "discrete_code")
            self.assertEqual(payload["num_items"], 8)
            self.assertEqual(len(payload["npz_files"]), 2)
            self.assertTrue((output_dir / payload["latent_spec"]["rvq_model_file"]).exists())
            self.assertEqual(payload["image_manifest_path"], "../dataset/tiny_gray.json")
            self.assertEqual(payload["label_names"], ["attr_a", "attr_b", "left_eye_x", "left_eye_y"])
            self.assertEqual(payload["condition_components"]["total_dim"], 4)

            dataset = ReferencedImageLatentDataset.from_latent_manifest(latent_manifest)
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (1, 4, 4))
            self.assertEqual(tuple(sample["latent"].shape), (2,))
            self.assertEqual(int(sample["sample_id"]), 0)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
