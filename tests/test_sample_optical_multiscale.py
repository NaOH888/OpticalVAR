from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tests.test_train_optical_multiscale as train_test_module

from scripts.sample_optical_multiscale import main as sample_main
from scripts.train_optical_multiscale import main as train_main


class SampleOpticalMultiscaleScriptTests(unittest.TestCase):
    def test_sample_script_uses_fixed_latent_and_saves_prefix_images(self) -> None:
        helper = train_test_module.TrainOpticalMultiscaleScriptTests()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path, outputs_dir, config_path = helper._write_tiny_training_case(tmp_path)
            sample_outputs_dir = tmp_path / "sample_outputs"

            train_main(["--config", str(config_path)])
            sample_main(
                [
                    "--checkpoint",
                    str(outputs_dir / "latest.pt"),
                    "--data-manifest",
                    str(manifest_path),
                    "--sample-index",
                    "1",
                    "--output-dir",
                    str(sample_outputs_dir),
                    "--device",
                    "cpu",
                    "--latent-seed",
                    "11",
                ]
            )

            self.assertTrue((sample_outputs_dir / "sample_0001_noise.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_target.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_prefix_01.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_target_band_01.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_prefix_band_compare_01.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_final_detector.png").exists())

    def test_sample_script_supports_random_latent_with_label(self) -> None:
        helper = train_test_module.TrainOpticalMultiscaleScriptTests()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, outputs_dir, config_path = helper._write_tiny_training_case(tmp_path)
            sample_outputs_dir = tmp_path / "random_sample_outputs"

            train_main(["--config", str(config_path)])
            sample_main(
                [
                    "--checkpoint",
                    str(outputs_dir / "latest.pt"),
                    "--random-latent",
                    "--label",
                    "3",
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(sample_outputs_dir),
                    "--device",
                    "cpu",
                    "--latent-seed",
                    "99",
                ]
            )

            self.assertTrue((sample_outputs_dir / "random_label_03_grid.png").exists())
            self.assertFalse((sample_outputs_dir / "random_label_03_0000_noise.png").exists())
            self.assertFalse((sample_outputs_dir / "random_label_03_0001_overview.png").exists())

    def test_sample_script_can_save_random_latent_details_when_requested(self) -> None:
        helper = train_test_module.TrainOpticalMultiscaleScriptTests()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, outputs_dir, config_path = helper._write_tiny_training_case(tmp_path)
            sample_outputs_dir = tmp_path / "random_sample_outputs_detail"

            train_main(["--config", str(config_path)])
            sample_main(
                [
                    "--checkpoint",
                    str(outputs_dir / "latest.pt"),
                    "--random-latent",
                    "--label",
                    "3",
                    "--num-samples",
                    "2",
                    "--detail-sample",
                    "--output-dir",
                    str(sample_outputs_dir),
                    "--device",
                    "cpu",
                    "--latent-seed",
                    "99",
                ]
            )

            self.assertTrue((sample_outputs_dir / "random_label_03_grid.png").exists())
            self.assertTrue((sample_outputs_dir / "random_label_03_0000_noise.png").exists())
            self.assertTrue((sample_outputs_dir / "random_label_03_0000_prefix_01.png").exists())
            self.assertTrue((sample_outputs_dir / "random_label_03_0000_final_detector.png").exists())
            self.assertTrue((sample_outputs_dir / "random_label_03_0001_overview.png").exists())


if __name__ == "__main__":
    unittest.main()
