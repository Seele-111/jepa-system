import unittest

import numpy as np

from jepa_scale_completeness_rescue import (
    select_scale_completeness_rescue_params,
    select_scale_completeness_rescue_predictions,
)


class JEPAScaleCompletenessRescueTests(unittest.TestCase):
    def test_rescues_low_selector_score_candidate_with_complete_jepa_evidence(self):
        base = [[(1, 2)]]
        candidates = [[(7, 9), (13, 15)]]
        scores = [np.array([0.42, 0.8], dtype=np.float32)]
        evidence = [np.array([0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.85, 0.9, 0.86, 0.1, 0.1, 0.1, 0.25, 0.25, 0.2], dtype=np.float32)]

        predictions = select_scale_completeness_rescue_predictions(
            base,
            candidates,
            scores,
            evidence,
            prob_threshold=0.4,
            evidence_threshold=0.7,
            min_active_fraction=0.8,
            min_mean=0.75,
            min_contrast=0.4,
            max_base_iou=0.0,
            length_penalty=0.0,
            nms_iou=None,
            max_rescues_per_video=2,
        )

        self.assertEqual(predictions, [[(1, 2), (7, 9)]])

    def test_param_search_enables_evidence_complete_rescue_when_it_improves_f1(self):
        base = [[(1, 2)]]
        candidates = [[(7, 9), (13, 15)]]
        scores = [np.array([0.42, 0.8], dtype=np.float32)]
        evidence = [np.array([0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.85, 0.9, 0.86, 0.1, 0.1, 0.1, 0.25, 0.25, 0.2], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)]

        config, metrics, predictions = select_scale_completeness_rescue_params(
            base,
            candidates,
            scores,
            evidence,
            labels,
            prob_thresholds=[0.4],
            evidence_thresholds=[0.7],
            min_active_fractions=[0.8],
            min_means=[0.75],
            min_contrasts=[0.4],
            max_base_ious=[0.0],
            length_penalties=[0.0],
            nms_ious=[None],
            max_rescues_per_videos=[2],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(1, 2), (7, 9)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_does_not_rescue_high_score_candidate_with_weak_jepa_completeness(self):
        base = [[(1, 2)]]
        candidates = [[(7, 9)]]
        scores = [np.array([0.9], dtype=np.float32)]
        evidence = [np.array([0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.78, 0.2, 0.18, 0.1], dtype=np.float32)]

        predictions = select_scale_completeness_rescue_predictions(
            base,
            candidates,
            scores,
            evidence,
            prob_threshold=0.4,
            evidence_threshold=0.7,
            min_active_fraction=0.8,
            min_mean=0.75,
            min_contrast=0.4,
            max_base_iou=0.0,
            length_penalty=0.0,
            nms_iou=None,
            max_rescues_per_video=1,
        )

        self.assertEqual(predictions, [[(1, 2)]])


if __name__ == "__main__":
    unittest.main()
