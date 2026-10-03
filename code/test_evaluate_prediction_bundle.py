import unittest

import numpy as np

from evaluate_prediction_bundle import evaluate_bundle, exclude_names


class PredictionBundleTests(unittest.TestCase):
    def test_bundle_reports_overall_and_generator_groups(self):
        bundle = {
            "task": "test",
            "folds": [
                {
                    "fold": 0,
                    "val_names": ["a.mp4", "b.mp4"],
                    "predictions": [[[1, 3]], []],
                    "labels": [[0, 1, 1, 1, 0], [0, 0, 0]],
                }
            ],
        }
        report = evaluate_bundle(bundle, 0.3, {"a.mp4": "Hunyuan", "b.mp4": "Luma"})
        self.assertEqual(report["videos"], 2)
        self.assertIn("overall", report)
        self.assertEqual(set(report["groups"]), {"Hunyuan", "Luma"})
        self.assertEqual(report["overall"]["segment"]["tp"], 1)

    def test_exclude_names_removes_overlapping_videos(self):
        bundle = {
            "folds": [{"fold": 0, "val_names": ["a.mp4", "b.mp4"], "predictions": [[], []], "labels": [[0], [1]]}]
        }
        filtered = exclude_names(bundle, {"a.mp4"})
        self.assertEqual(filtered["folds"][0]["val_names"], ["b.mp4"])
        self.assertEqual(filtered["folds"][0]["labels"], [[1]])


if __name__ == "__main__":
    unittest.main()
