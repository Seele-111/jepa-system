import unittest

import numpy as np

from jepa_ot_event_set_matcher import (
    score_ot_candidates,
    select_ot_event_set_params,
)


class JEPAOTEventSetMatcherTests(unittest.TestCase):
    def test_ot_scores_reward_candidates_covering_jepa_evidence_islands(self):
        evidence = np.zeros((10, 3), dtype=np.float32)
        evidence[1:3, :] = 1.0
        candidates = [(1, 2), (5, 6)]
        selector_scores = np.array([0.1, 0.9], dtype=np.float32)

        scores = score_ot_candidates(
            base_segments=[],
            candidates=candidates,
            selector_scores=selector_scores,
            evidence=evidence,
            evidence_threshold=0.5,
            selector_weight=0.0,
            contrast_weight=0.5,
            active_fraction_weight=0.5,
            agreement_weight=0.5,
            island_coverage_weight=1.0,
            peak_alignment_weight=0.5,
            base_overlap_penalty=0.0,
            length_penalty=0.0,
            smooth_window=1,
            min_evidence_island_length=1,
        )

        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_ot_selector_replaces_wide_parent_with_two_evidence_matched_children(self):
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0], dtype=np.int64)]
        base_predictions = [[(0, 9)]]
        candidates = [[(1, 2), (7, 8), (4, 5)]]
        selector_scores = [np.array([0.05, 0.05, 0.95], dtype=np.float32)]
        evidence = np.zeros((10, 3), dtype=np.float32)
        evidence[1:3, :] = 1.0
        evidence[7:9, :] = 1.0

        config, metrics, predictions = select_ot_event_set_params(
            base_predictions,
            candidates,
            selector_scores,
            evidence=[evidence],
            labels=labels,
            score_thresholds=(0.4,),
            base_keep_scores=(0.1,),
            evidence_thresholds=(0.5,),
            selector_weights=(0.0,),
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(1, 2), (7, 8)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)
        self.assertEqual(metrics["segment"]["fn"], 0)

    def test_fp_budget_prevents_harmful_addition(self):
        labels = [np.array([0, 1, 1, 0, 0, 0, 0], dtype=np.int64)]
        base_predictions = [[(1, 2)]]
        candidates = [[(1, 2), (5, 6)]]
        selector_scores = [np.array([0.1, 0.1], dtype=np.float32)]
        evidence = np.zeros((7, 3), dtype=np.float32)
        evidence[5:7, :] = 1.0

        config, metrics, predictions = select_ot_event_set_params(
            base_predictions,
            candidates,
            selector_scores,
            evidence=[evidence],
            labels=labels,
            score_thresholds=(0.3,),
            base_keep_scores=(1.0,),
            evidence_thresholds=(0.5,),
            selector_weights=(0.0,),
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(predictions, base_predictions)
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
