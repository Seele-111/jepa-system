import unittest

import numpy as np

from analyze_fn_candidate_attribution import (
    _classify_fn_reason,
    _match_prediction_segments,
    _segment_overlap_fraction,
)


class FnCandidateAttributionTest(unittest.TestCase):
    def test_match_prediction_segments_reports_unmatched_gt(self):
        labels = np.zeros(20, dtype=np.int64)
        labels[2:5] = 1
        labels[10:14] = 1

        result = _match_prediction_segments([(2, 4), (15, 16)], labels, iou_threshold=0.3)

        self.assertEqual(result["matched_gt_indices"], [0])
        self.assertEqual(result["unmatched_gt"], [(10, 13)])
        self.assertEqual(result["false_positive_predictions"], [(15, 16)])

    def test_segment_overlap_fraction_uses_candidate_length(self):
        self.assertAlmostEqual(_segment_overlap_fraction((5, 9), (7, 11)), 3 / 5)
        self.assertAlmostEqual(_segment_overlap_fraction((0, 2), (3, 5)), 0.0)

    def test_classify_fn_reason_distinguishes_low_rank_from_generation_miss(self):
        self.assertEqual(
            _classify_fn_reason(
                best_candidate_iou=0.0,
                best_candidate_score=0.0,
                best_candidate_rank=None,
                selected_iou=0.0,
                fused_iou=0.0,
                protected_suppressed=False,
            ),
            "candidate_generation_miss",
        )
        self.assertEqual(
            _classify_fn_reason(
                best_candidate_iou=0.45,
                best_candidate_score=0.21,
                best_candidate_rank=87,
                selected_iou=0.0,
                fused_iou=0.0,
                protected_suppressed=False,
            ),
            "candidate_scored_too_low",
        )

    def test_classify_fn_reason_distinguishes_fusion_suppression(self):
        self.assertEqual(
            _classify_fn_reason(
                best_candidate_iou=0.62,
                best_candidate_score=0.91,
                best_candidate_rank=1,
                selected_iou=0.51,
                fused_iou=0.0,
                protected_suppressed=True,
            ),
            "fusion_suppressed_selected_candidate",
        )

    def test_classify_fn_reason_distinguishes_matching_conflict(self):
        self.assertEqual(
            _classify_fn_reason(
                best_candidate_iou=1.0,
                best_candidate_score=0.87,
                best_candidate_rank=18,
                selected_iou=0.47,
                fused_iou=0.45,
                protected_suppressed=False,
            ),
            "segment_matching_conflict",
        )


if __name__ == "__main__":
    unittest.main()
