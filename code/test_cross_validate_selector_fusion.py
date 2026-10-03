import unittest

import numpy as np

from cross_validate_selector_fusion import _build_prediction_export
from train_segment_locator import VideoRecord


class CrossValidateSelectorFusionPredictionExportTests(unittest.TestCase):
    def test_build_prediction_export_omits_labels_by_default(self):
        records = [
            VideoRecord("video_a.mp4", np.zeros((6, 3), dtype=np.float32), np.array([0, 1, 1, 0, 0, 0])),
            VideoRecord("video_b.mp4", np.zeros((5, 3), dtype=np.float32), np.array([0, 0, 0, 1, 1])),
        ]
        folds = [
            {
                "fold": 0,
                "train_idx": [1],
                "val_idx": [0],
                "val_names": ["video_a.mp4"],
                "validation": {"segment": {"f1": 0.5}},
            }
        ]

        export = _build_prediction_export(
            data_dir="/tmp/event_data",
            records=records,
            fold_summaries=folds,
            predictions_by_fold=[[[(1, 2), (4, 5)]]],
            include_labels=False,
        )

        self.assertEqual(export["data_dir"], "/tmp/event_data")
        self.assertEqual(export["folds"][0]["val_names"], ["video_a.mp4"])
        self.assertEqual(export["folds"][0]["predictions"], [[[1, 2], [4, 5]]])
        self.assertNotIn("labels", export["folds"][0])

    def test_build_prediction_export_can_include_labels_for_offline_cv(self):
        records = [
            VideoRecord("video_a.mp4", np.zeros((4, 3), dtype=np.float32), np.array([0, 1, 1, 0])),
        ]

        export = _build_prediction_export(
            data_dir="/tmp/event_data",
            records=records,
            fold_summaries=[{"fold": 0, "train_idx": [], "val_idx": [0], "val_names": ["video_a.mp4"]}],
            predictions_by_fold=[[[(1, 2)]]],
            include_labels=True,
        )

        self.assertEqual(export["folds"][0]["labels"], [[0, 1, 1, 0]])


if __name__ == "__main__":
    unittest.main()
