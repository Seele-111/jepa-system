import unittest

import numpy as np

from run_strict_dp_event_set_oof import build_dp_folds_from_base_export
from train_segment_locator import VideoRecord


class RunStrictDPEventSetOOFTests(unittest.TestCase):
    def test_build_dp_folds_uses_export_fold_indices_and_labels(self):
        records = [
            VideoRecord("v0", np.zeros((5, 3), dtype=np.float32), np.array([1, 1, 0, 0, 0])),
            VideoRecord("v1", np.zeros((5, 3), dtype=np.float32), np.array([0, 0, 1, 1, 0])),
        ]
        export = {
            "folds": [
                {
                    "fold": 7,
                    "train_idx": [1],
                    "val_idx": [0],
                    "predictions": [[[0, 1]]],
                    "labels": [[1, 1, 0, 0, 0]],
                }
            ]
        }
        raw = {0: ([(0, 1), (2, 3)], np.array([0.9, 0.2], dtype=np.float32))}

        folds = build_dp_folds_from_base_export(export, records, raw)

        self.assertEqual(folds[0]["fold"], 7)
        self.assertEqual(folds[0]["train_idx"], [1])
        self.assertEqual(folds[0]["val_idx"], [0])
        self.assertEqual(folds[0]["base"], [[(0, 1)]])
        self.assertEqual(folds[0]["candidates"], [[(0, 1), (2, 3)]])
        self.assertTrue(np.array_equal(folds[0]["scores"][0], np.array([0.9, 0.2], dtype=np.float32)))
        self.assertTrue(np.array_equal(folds[0]["labels"][0], np.array([1, 1, 0, 0, 0])))

    def test_build_dp_folds_reconstructs_missing_train_indices(self):
        records = [
            VideoRecord(f"v{idx}", np.zeros((5, 3), dtype=np.float32), np.zeros(5, dtype=np.int64))
            for idx in range(3)
        ]
        export = {
            "folds": [
                {
                    "fold": 0,
                    "val_idx": [1],
                    "predictions": [[]],
                }
            ]
        }
        raw = {1: ([], np.zeros(0, dtype=np.float32))}

        folds = build_dp_folds_from_base_export(export, records, raw)

        self.assertEqual(folds[0]["train_idx"], [0, 2])


if __name__ == "__main__":
    unittest.main()
