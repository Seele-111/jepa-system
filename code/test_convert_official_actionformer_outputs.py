import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from convert_official_actionformer_outputs import (
    build_prediction_fold,
    decode_video,
    select_calibration_params,
)


class OfficialActionFormerOutputTests(unittest.TestCase):
    def test_decode_uses_half_open_boundaries_and_score_order(self):
        candidates = [(1.2, 3.1, 0.9), (0.0, 5.0, 0.2)]
        self.assertEqual(decode_video(candidates, 5, 0.5, 1), [(1, 3)])

    def test_calibration_and_outer_export_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = {
                "database": {
                    "cal": {"subset": "calibration", "duration": 5, "annotations": [{"segment": [1, 3]}]},
                    "outer": {"subset": "validation", "duration": 5, "annotations": [{"segment": [2, 4]}]},
                }
            }
            annotation_path = root / "annotation.json"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            calibration = {"video-id": ["cal", "cal"], "t-start": np.array([1.0, 0.0]), "t-end": np.array([3.0, 5.0]), "label": np.array([0, 0]), "score": np.array([0.9, 0.1])}
            outer = {"video-id": ["outer"], "t-start": np.array([2.0]), "t-end": np.array([4.0]), "label": np.array([0]), "score": np.array([0.8])}
            cal_path, outer_path = root / "cal.pkl", root / "outer.pkl"
            with cal_path.open("wb") as handle:
                pickle.dump(calibration, handle)
            with outer_path.open("wb") as handle:
                pickle.dump(outer, handle)
            fold, audit = build_prediction_fold(0, annotation_path, cal_path, outer_path, {"outer": "outer.mp4"})
            self.assertEqual(fold["val_names"], ["outer.mp4"])
            self.assertEqual(fold["predictions"], [[[2, 3]]])
            self.assertEqual(audit["calibration"]["tp"], 1)

    def test_selector_prefers_high_precision_single_detection(self):
        detections = {"v": [(1, 3, 0.9), (0, 5, 0.1)]}
        labels = {"v": np.asarray([0, 1, 1, 0, 0])}
        params, metrics = select_calibration_params(detections, labels)
        self.assertEqual(params["max_predictions"], 1)
        self.assertEqual(metrics["fp"], 0)


if __name__ == "__main__":
    unittest.main()
