import unittest

import numpy as np

from analyze_oof_fn_attribution import classify_unmatched_gt_reason


class AnalyzeOOFFNAttributionTests(unittest.TestCase):
    def test_classifies_swallowed_event_when_prediction_overlaps_gt_but_fails_iou(self):
        reason = classify_unmatched_gt_reason(
            gt=(10, 14),
            predictions=[(0, 30)],
            candidates=[(10, 14)],
            candidate_scores=np.array([0.9], dtype=np.float32),
            iou_threshold=0.3,
            active_threshold=0.5,
        )

        self.assertEqual(reason["reason"], "wide_prediction_matching_conflict")

    def test_classifies_candidate_scored_too_low(self):
        reason = classify_unmatched_gt_reason(
            gt=(10, 14),
            predictions=[],
            candidates=[(10, 14)],
            candidate_scores=np.array([0.1], dtype=np.float32),
            iou_threshold=0.3,
            active_threshold=0.5,
        )

        self.assertEqual(reason["reason"], "candidate_scored_too_low")


if __name__ == "__main__":
    unittest.main()
