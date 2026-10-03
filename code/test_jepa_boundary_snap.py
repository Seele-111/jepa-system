import unittest

import numpy as np

from jepa_boundary_snap import select_boundary_snap_params, snap_segments_to_candidates


class JEPABoundarySnapTests(unittest.TestCase):
    def test_snap_segments_expands_mainline_to_high_score_overlapping_candidate(self):
        mainline = [[(10, 14)]]
        candidates = [[(8, 18), (30, 35)]]
        scores = [np.array([0.9, 0.95], dtype=np.float32)]

        snapped = snap_segments_to_candidates(
            mainline,
            candidates,
            scores,
            min_score=0.5,
            min_mainline_coverage=0.8,
            max_length_ratio=3.0,
        )

        self.assertEqual(snapped, [[(8, 18)]])

    def test_snap_segments_rejects_unrelated_high_score_candidate(self):
        mainline = [[(10, 14)]]
        candidates = [[(30, 35)]]
        scores = [np.array([0.99], dtype=np.float32)]

        snapped = snap_segments_to_candidates(
            mainline,
            candidates,
            scores,
            min_score=0.5,
            min_mainline_coverage=0.8,
            max_length_ratio=3.0,
        )

        self.assertEqual(snapped, [[(10, 14)]])

    def test_select_boundary_snap_params_can_choose_improving_snap(self):
        mainline = [[(10, 14)]]
        candidates = [[(8, 18)]]
        scores = [np.array([0.9], dtype=np.float32)]
        labels = [np.array([0] * 8 + [1] * 11 + [0] * 5, dtype=np.int64)]

        config, metrics, snapped = select_boundary_snap_params(
            mainline,
            candidates,
            scores,
            labels,
            min_scores=[0.5],
            min_mainline_coverages=[0.8],
            max_length_ratios=[3.0],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(snapped, [[(8, 18)]])
        self.assertGreater(metrics["segment"]["f1"], 0.99)

    def test_select_boundary_snap_params_can_disable_harmful_snap(self):
        mainline = [[(10, 14)]]
        candidates = [[(8, 22)]]
        scores = [np.array([0.9], dtype=np.float32)]
        labels = [np.array([0] * 10 + [1] * 5 + [0] * 10, dtype=np.int64)]

        config, metrics, snapped = select_boundary_snap_params(
            mainline,
            candidates,
            scores,
            labels,
            min_scores=[0.5],
            min_mainline_coverages=[0.8],
            max_length_ratios=[4.0],
            iou_threshold=0.5,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(snapped, mainline)
        self.assertGreater(metrics["segment"]["f1"], 0.99)


if __name__ == "__main__":
    unittest.main()
