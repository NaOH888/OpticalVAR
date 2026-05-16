from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.prepare_celeba_grayscale import _read_landmark_table


class PrepareCelebaGrayscaleTests(unittest.TestCase):
    def test_read_landmark_table_normalizes_aligned_coordinates_to_unit_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            landmark_path = tmp_path / "list_landmarks_align_celeba.txt"
            landmark_path.write_text(
                "\n".join(
                    [
                        "2",
                        "lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y",
                        "000001.jpg 11 31 167 31 89 89 41 167 137 167",
                        "000002.jpg 1 21 177 21 89 109 21 197 157 197",
                    ]
                ),
                encoding="utf-8",
            )

            landmark_names, landmark_map = _read_landmark_table(
                landmark_path,
                selected_filenames={"000001.jpg"},
                crop_size=176,
            )

            self.assertEqual(len(landmark_names), 10)
            normalized = landmark_map["000001.jpg"]
            self.assertEqual(tuple(normalized.shape), (10,))
            self.assertTrue(np.all(normalized >= 0.0))
            self.assertTrue(np.all(normalized <= 1.0))
            self.assertAlmostEqual(float(normalized[0]), (11.0 - 1.0) / 175.0)
            self.assertAlmostEqual(float(normalized[1]), (31.0 - 21.0) / 175.0)


if __name__ == "__main__":
    unittest.main()
