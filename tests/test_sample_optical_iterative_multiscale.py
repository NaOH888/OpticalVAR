from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tests.test_train_optical_iterative_multiscale as train_test_module

from scripts.sample_optical_iterative_multiscale import main as sample_main
from scripts.train_optical_iterative_multiscale import main as train_main


class SampleOpticalIterativeMultiscaleScriptTests(unittest.TestCase):
    def test_sample_script_saves_step_predictions_and_states(self) -> None:
        helper = train_test_module.TrainOpticalIterativeMultiscaleScriptTests()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = helper._write_tiny_training_case(tmp_path)
            sample_outputs_dir = tmp_path / "sample_outputs"

            train_main(["--config", str(config_path)])
            sample_main(
                [
                    "--checkpoint",
                    str(outputs_dir / "latest.pt"),
                    "--data-manifest",
                    str(tmp_path / "dataset" / "tiny_autoencoder.json"),
                    "--sample-index",
                    "1",
                    "--output-dir",
                    str(sample_outputs_dir),
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue((sample_outputs_dir / "sample_0001_target.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_target_scale_01.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_target_scale_03.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_step_01.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_step_03.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_state_01.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_state_03.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_overview.png").exists())
            self.assertTrue((sample_outputs_dir / "sample_0001_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
