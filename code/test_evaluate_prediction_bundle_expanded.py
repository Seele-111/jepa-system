import unittest

import numpy as np

from evaluate_prediction_bundle_expanded import _match, expanded_metrics


class ExpandedPredictionEvaluationTests(unittest.TestCase):
    def test_matching_and_boundary_metrics(self):
        rows = [
            ("a.mp4", [(1, 3), (7, 7)], np.asarray([0, 1, 1, 1, 0, 0, 1, 1])),
            ("normal.mp4", [(1, 1)], np.zeros(4, dtype=np.int64)),
        ]
        report = expanded_metrics(rows, 0.3, (0.25, 0.75))
        self.assertEqual(report["overall"]["segment"]["tp"], 2)
        self.assertEqual(report["overall"]["segment"]["fp"], 1)
        self.assertEqual(report["matched_event_iou"]["count"], 2)
        self.assertAlmostEqual(report["boundary_error_frames"]["start"]["mean"], 0.5)
        self.assertEqual(report["normal_video_false_positive_rate"]["rate"], 1.0)
        self.assertEqual(report["video_strata"]["normal"]["videos"], 1)

    def test_empty_normal_stratum_is_explicitly_not_estimable(self):
        rows = [("a.mp4", [], np.asarray([1, 1]))]
        report = expanded_metrics(rows, 0.3, (0.25, 0.75))
        self.assertIsNone(report["normal_video_false_positive_rate"]["rate"])
        self.assertIn("not estimable", report["normal_video_false_positive_rate"]["note"])

    def test_match_uses_prediction_order_like_main_evaluator(self):
        matches, unmatched_pred, unmatched_gt = _match([(0, 2), (4, 5)], [(0, 2), (4, 5)], 0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual(unmatched_pred, [])
        self.assertEqual(unmatched_gt, [])


if __name__ == "__main__":
    unittest.main()
