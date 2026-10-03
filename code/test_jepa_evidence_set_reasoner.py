import unittest

import numpy as np

from jepa_evidence_set_reasoner import (
    evidence_set_reasoner_scores,
    select_evidence_set_reasoner_params,
    select_evidence_set_reasoner_predictions,
)


class JEPAEvidenceSetReasonerTests(unittest.TestCase):
    def test_reasoner_scores_low_selector_candidate_from_dual_jepa_evidence(self):
        candidates = [(5, 6), (8, 8)]
        selector_scores = np.array([0.04, 0.8], dtype=np.float32)
        evidence = np.zeros((10, 3), dtype=np.float32)
        evidence[5:7] = np.array([0.9, 0.85, 0.88], dtype=np.float32)
        evidence[8] = np.array([0.2, 0.1, 0.15], dtype=np.float32)

        scores = evidence_set_reasoner_scores(
            base_segments=[],
            candidates=candidates,
            selector_scores=selector_scores,
            evidence=evidence,
            evidence_threshold=0.7,
            selector_weight=0.1,
            contrast_weight=0.5,
            active_fraction_weight=0.5,
            consensus_weight=0.5,
            support_iou=0.3,
            support_count_weight=0.0,
            base_iou_penalty=0.0,
            length_penalty=0.0,
        )

        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_predictions_add_low_selector_high_evidence_candidate(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6), (8, 8)]]
        selector_scores = [np.array([0.04, 0.8], dtype=np.float32)]
        evidence = [np.zeros((10, 3), dtype=np.float32)]
        evidence[0][5:7] = np.array([0.9, 0.85, 0.88], dtype=np.float32)

        fused = select_evidence_set_reasoner_predictions(
            base,
            candidates,
            selector_scores,
            evidence,
            min_selector_score=0.0,
            evidence_threshold=0.7,
            min_evidence_mean=0.6,
            min_active_fraction=0.8,
            min_contrast=0.2,
            score_threshold=1.0,
            selector_weight=0.1,
            contrast_weight=0.5,
            active_fraction_weight=0.5,
            consensus_weight=0.5,
            support_iou=0.3,
            support_count_weight=0.0,
            max_base_iou=None,
            base_iou_penalty=0.0,
            length_penalty=0.0,
            nms_iou=None,
            max_rescues_per_video=1,
        )

        self.assertEqual(fused, [[(1, 2), (5, 6)]])

    def test_prefilter_keeps_candidate_indices_aligned(self):
        base = [[]]
        candidates = [[(0, 0), (5, 6), (8, 8)]]
        selector_scores = [np.array([0.9, 0.04, 0.2], dtype=np.float32)]
        evidence = [np.zeros((10, 3), dtype=np.float32)]
        evidence[0][5:7] = np.array([0.9, 0.85, 0.88], dtype=np.float32)

        fused = select_evidence_set_reasoner_predictions(
            base,
            candidates,
            selector_scores,
            evidence,
            min_selector_score=0.0,
            evidence_threshold=0.7,
            min_evidence_mean=0.6,
            min_active_fraction=0.8,
            min_contrast=0.2,
            score_threshold=1.0,
            selector_weight=0.1,
            contrast_weight=0.5,
            active_fraction_weight=0.5,
            consensus_weight=0.5,
            support_iou=0.3,
            support_count_weight=0.0,
            max_base_iou=None,
            base_iou_penalty=0.0,
            length_penalty=0.0,
            nms_iou=None,
            max_rescues_per_video=1,
            prefilter_top_k=1,
        )

        self.assertEqual(fused, [[(5, 6)]])

    def test_select_params_enables_when_evidence_rescue_improves_f1(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6), (8, 8)]]
        selector_scores = [np.array([0.04, 0.8], dtype=np.float32)]
        evidence = [np.zeros((10, 3), dtype=np.float32)]
        evidence[0][5:7] = np.array([0.9, 0.85, 0.88], dtype=np.float32)
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=np.int64)]

        config, metrics, fused = select_evidence_set_reasoner_params(
            base,
            candidates,
            selector_scores,
            evidence,
            labels,
            min_selector_scores=[0.0],
            evidence_thresholds=[0.7],
            min_evidence_means=[0.6],
            min_active_fractions=[0.8],
            min_contrasts=[0.2],
            score_thresholds=[1.0],
            selector_weights=[0.1],
            contrast_weights=[0.5],
            active_fraction_weights=[0.5],
            consensus_weights=[0.5],
            support_ious=[0.3],
            support_count_weights=[0.0],
            max_base_ious=[None],
            base_iou_penalties=[0.0],
            length_penalties=[0.0],
            nms_ious=[None],
            max_rescues_per_videos=[1],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(fused, [[(1, 2), (5, 6)]])
        self.assertEqual(metrics["segment"]["tp"], 2)


if __name__ == "__main__":
    unittest.main()
