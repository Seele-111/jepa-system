import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_jepa_format_transfer import load_format_groups, split_source_calibration
from train_segment_locator import VideoRecord


class FormatTransferTests(unittest.TestCase):
    def test_load_format_groups_uses_joint_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.json"
            path.write_text(json.dumps({"videos": [{"video_name": "a.mp4", "fps": "8", "resolution": "720x480"}]}))
            self.assertEqual(load_format_groups(path)["a.mp4"], "fps_8_resolution_720x480")

    def test_source_split_is_disjoint_and_complete(self):
        records = [VideoRecord(str(i), np.zeros((3, 3), dtype=np.float32), np.asarray([0, i % 2, 1])) for i in range(8)]
        fit, calibration = split_source_calibration(records, [0, 1, 2, 3, 4, 5], 0.25, 42)
        self.assertFalse(set(fit) & set(calibration))
        self.assertEqual(set(fit) | set(calibration), set(range(6)))


if __name__ == "__main__":
    unittest.main()
