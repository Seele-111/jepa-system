import unittest

import numpy as np

from jepa_watershed_split import select_watershed_split_params, watershed_split_predictions


class JEPAWatershedSplitTests(unittest.TestCase):
    def test_watershed_split_replaces_long_parent_with_evidence_islands(self):
        predictions = [[(0, 12)]]
        evidence = [np.array([0.1, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.75, 0.85, 0.8, 0.1, 0.1, 0.1], dtype=np.float32)]

        split = watershed_split_predictions(
            predictions,
            evidence,
            evidence_threshold=0.7,
            smooth_window=1,
            min_gap=0,
            min_length=2,
            pad=0,
            parent_min_length=8,
        )

        self.assertEqual(split, [[(1, 3), (7, 9)]])

    def test_watershed_split_keeps_parent_when_only_one_island_exists(self):
        predictions = [[(0, 8)]]
        evidence = [np.array([0.1, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)]

        split = watershed_split_predictions(
            predictions,
            evidence,
            evidence_threshold=0.7,
            smooth_window=1,
            min_gap=0,
            min_length=2,
            pad=0,
            parent_min_length=8,
        )

        self.assertEqual(split, [[(0, 8)]])

    def test_select_watershed_split_params_enables_improving_split(self):
        predictions = [[(0, 12)]]
        evidence = [np.array([0.1, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.75, 0.85, 0.8, 0.1, 0.1, 0.1], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.int64)]

        config, metrics, split = select_watershed_split_params(
            predictions,
            evidence,
            labels,
            evidence_thresholds=[0.7],
            smooth_windows=[1],
            min_gaps=[0],
            min_lengths=[2],
            pads=[0],
            parent_min_lengths=[8],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(split, [[(1, 3), (7, 9)]])
        self.assertEqual(metrics["segment"]["tp"], 2)

    def test_select_watershed_split_params_disables_harmful_split(self):
        predictions = [[(0, 12)]]
        evidence = [np.array([0.1, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.75, 0.85, 0.8, 0.1, 0.1, 0.1], dtype=np.float32)]
        labels = [np.array([1] * 13, dtype=np.int64)]

        config, metrics, split = select_watershed_split_params(
            predictions,
            evidence,
            labels,
            evidence_thresholds=[0.7],
            smooth_windows=[1],
            min_gaps=[0],
            min_lengths=[2],
            pads=[0],
            parent_min_lengths=[8],
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(split, predictions)
        self.assertEqual(metrics["segment"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
