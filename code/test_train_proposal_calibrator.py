import unittest

import numpy as np

from train_proposal_calibrator import (
    PROPOSAL_FEATURE_NAMES,
    evaluate_segment_predictions,
    extract_proposal_features,
    generate_candidates,
    interval_iou,
    label_candidates,
    nms_segments,
)


class ProposalCalibratorTests(unittest.TestCase):
    def test_generate_candidates_merges_multithreshold_segments(self):
        scores = np.array([0.0, 0.7, 0.8, 0.1, 0.75, 0.0, 0.0, 0.9])

        candidates = generate_candidates(scores, thresholds=[0.5, 0.7], min_gap=1, min_length=2)

        self.assertEqual(candidates, [(1, 4)])

    def test_extract_proposal_features_are_finite_and_named(self):
        signals = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.1, 0.9],
                [0.2, 0.8, 0.4],
                [0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )

        features = extract_proposal_features(signals, (1, 3))

        self.assertEqual(features.shape, (len(PROPOSAL_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())

    def test_interval_iou_uses_inclusive_boundaries(self):
        self.assertAlmostEqual(interval_iou((1, 4), (3, 5)), 2 / 5)

    def test_label_candidates_uses_best_iou(self):
        candidates = [(0, 1), (3, 5), (8, 9)]
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])

        y, best = label_candidates(candidates, labels, iou_threshold=0.3)

        np.testing.assert_array_equal(y, np.array([0, 1, 1]))
        np.testing.assert_allclose(best, np.array([0.0, 1.0, 1.0]))

    def test_nms_segments_suppresses_overlapping_lower_score_segments(self):
        segments = [(1, 5, 0.9), (2, 6, 0.8), (10, 12, 0.7)]

        kept = nms_segments(segments, iou_threshold=0.3)

        self.assertEqual(kept, [(1, 5), (10, 12)])

    def test_evaluate_segment_predictions_matches_ground_truth_once(self):
        pred = [[(0, 0), (2, 2), (4, 5)]]
        labels = [np.array([1, 1, 1, 0, 1, 1], dtype=np.int64)]

        metrics = evaluate_segment_predictions(pred, labels, iou_threshold=0.3)

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)


if __name__ == "__main__":
    unittest.main()
