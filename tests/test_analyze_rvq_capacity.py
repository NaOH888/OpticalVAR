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

from scripts.analyze_rvq_capacity import analyze_rvq_capacity
from scripts.export_rvq_pairs import export_pairs


class AnalyzeRVQCapacityTests(unittest.TestCase):
    def test_analyze_rvq_capacity_runs_on_exported_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            analysis_dir = tmp_path / "analysis"
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
                    }
                ),
                encoding="utf-8",
            )

            config = {
                "runtime": {"seed": 7},
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

            summary = analyze_rvq_capacity(
                manifest_path=Path(result["output_manifest"]),
                output_dir=analysis_dir,
                sample_index=0,
                donor_index=1,
                cumulative_stage_counts=[1, 2],
                swap_stages=[0, 1],
            )

            self.assertEqual(summary["num_stages"], 2)
            self.assertEqual(len(summary["residual_norms"]), 2)
            self.assertEqual(len(summary["swap_summary"]), 2)
            self.assertTrue((analysis_dir / "summary.json").exists())
            self.assertTrue((analysis_dir / "cumulative_reconstructions.png").exists())
            self.assertTrue((analysis_dir / "increment_reconstructions.png").exists())
            self.assertTrue((analysis_dir / "residual_curve.png").exists())
            self.assertTrue((analysis_dir / "stage_swaps.png").exists())
            self.assertTrue((analysis_dir / "stage_swap_diffs.png").exists())


if __name__ == "__main__":
    unittest.main()
