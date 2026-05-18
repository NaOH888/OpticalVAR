from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tests.test_train_optical_iterative_multiscale as train_test_module

from scripts.diagnose_optical_iterative_multiscale import main as diagnose_main
from scripts.train_optical_iterative_multiscale import main as train_main


class DiagnoseOpticalIterativeMultiscaleScriptTests(unittest.TestCase):
    def test_diagnose_script_runs_for_iterative_checkpoint(self) -> None:
        helper = train_test_module.TrainOpticalIterativeMultiscaleScriptTests()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path, outputs_dir = helper._write_tiny_training_case(tmp_path)
            diagnose_output_dir = tmp_path / "diagnose_outputs"

            train_main(["--config", str(config_path)])
            diagnose_main(
                [
                    "--checkpoint",
                    str(outputs_dir / "latest.pt"),
                    "--data-manifest",
                    str(tmp_path / "dataset" / "tiny_autoencoder.json"),
                    "--anchor-index",
                    "0",
                    "--num-samples",
                    "2",
                    "--output-dir",
                    str(diagnose_output_dir),
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue((diagnose_output_dir / "anchor_target.png").exists())
            self.assertTrue((diagnose_output_dir / "anchor_prediction_step_01.png").exists())
            self.assertTrue((diagnose_output_dir / "latent_vary_prediction_step_01.png").exists())
            self.assertTrue((diagnose_output_dir / "condition_vary_prediction_step_03.png").exists())
            self.assertTrue((diagnose_output_dir / "summary.json").exists())

            payload = json.loads((diagnose_output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["anchor_index"], 0)
            self.assertEqual(len(payload["latent_only"]), 2)
            self.assertEqual(len(payload["condition_only"]), 2)
            self.assertEqual(len(payload["latent_only_mean"]["prediction_mad_per_step"]), 3)
            self.assertEqual(len(payload["condition_only_mean"]["control_map_mad_per_step"]), 3)


if __name__ == "__main__":
    unittest.main()
