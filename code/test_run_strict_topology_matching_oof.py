import unittest

import numpy as np

from run_strict_topology_matching_oof import build_matching_folds, flatten, load_prediction_export
from train_segment_locator import VideoRecord


class RunStrictTopologyMatchingOOFTests(unittest.TestCase):
    def test_build_matching_folds_reconstructs_train_indices_and_uses_fold_labels(self):
        records = [
            VideoRecord("v0", np.zeros((6, 3), dtype=np.float32), np.array([1, 1, 0, 0, 0, 0])),
            VideoRecord("v1", np.zeros((6, 3), dtype=np.float32), np.array([0, 0, 1, 1, 0, 0])),
            VideoRecord("v2", np.zeros((6, 3), dtype=np.float32), np.array([0, 0, 0, 0, 1, 1])),
        ]
        export = {
            "folds": [
                {
                    "fold": 5,
                    "val_idx": [1],
                    "predictions": [[[0, 5]]],
                    "labels": [[0, 0, 1, 1, 0, 0]],
                }
            ]
        }
        raw_by_video = {
            1: ([(2, 3), (0, 5)], np.array([0.7, 0.95], dtype=np.float32)),
        }
        evidence_by_video = {
            1: np.ones((6, 2), dtype=np.float32),
        }

        folds = build_matching_folds(export, records, raw_by_video, evidence_by_video)

        self.assertEqual(folds[0]["fold"], 5)
        self.assertEqual(folds[0]["train_idx"], [0, 2])
        self.assertEqual(folds[0]["val_idx"], [1])
        self.assertEqual(folds[0]["base"], [[(0, 5)]])
        self.assertEqual(folds[0]["candidates"], [[(2, 3), (0, 5)]])
        self.assertTrue(np.array_equal(folds[0]["scores"][0], np.array([0.7, 0.95], dtype=np.float32)))
        self.assertTrue(np.array_equal(folds[0]["evidence"][0], np.ones((6, 2), dtype=np.float32)))
        self.assertTrue(np.array_equal(folds[0]["labels"][0], np.array([0, 0, 1, 1, 0, 0])))

    def test_flatten_preserves_fold_order(self):
        folds = [{"base": [[(0, 1)]], "labels": [np.array([1, 1])]}, {"base": [[]], "labels": [np.zeros(2)]}]

        self.assertEqual(flatten(folds, "base"), [[(0, 1)], []])
        self.assertEqual(len(flatten(folds, "labels")), 2)

    def test_load_prediction_export_converts_segments_and_labels(self):
        export = {
            "folds": [
                {
                    "fold": 0,
                    "val_idx": [0],
                    "predictions": [[[1, 3]]],
                    "labels": [[0, 1, 1, 1]],
                }
            ]
        }

        data = load_prediction_export(export)

        self.assertEqual(data["folds"][0]["predictions"], [[(1, 3)]])
        self.assertTrue(np.array_equal(data["folds"][0]["labels"][0], np.array([0, 1, 1, 1])))


if __name__ == "__main__":
    unittest.main()
