import unittest

import numpy as np

from jepa_boundary_distribution_refine import (
    refine_boundary_distribution_predictions,
    select_boundary_distribution_refine_params,
)


class JEPABoundaryDistributionRefineTests(unittest.TestCase):
    def test_refines_near_miss_boundary_using_high_score_jepa_distribution(self):
        base = [[(16, 18)]]
        candidates = [[(8, 18), (40, 42)]]
        scores = [np.array([0.9, 0.95], dtype=np.float32)]

        refined = refine_boundary_distribution_predictions(
            base,
            candidates,
            scores,
            prob_threshold=0.5,
            min_base_coverage=0.8,
            max_length_ratio=5.0,
            start_quantile=0.5,
            end_quantile=0.5,
            base_boundary_weight=0.0,
            max_shift_ratio=4.0,
        )

        self.assertEqual(refined, [[(8, 18)]])

    def test_rejects_unrelated_high_score_jepa_boundary(self):
        base = [[(16, 18)]]
        candidates = [[(40, 42)]]
        scores = [np.array([0.99], dtype=np.float32)]

        refined = refine_boundary_distribution_predictions(
            base,
            candidates,
            scores,
            prob_threshold=0.5,
            min_base_coverage=0.8,
            max_length_ratio=5.0,
            start_quantile=0.5,
            end_quantile=0.5,
            base_boundary_weight=0.0,
            max_shift_ratio=4.0,
        )

        self.assertEqual(refined, [[(16, 18)]])

    def test_param_search_enables_boundary_refine_when_it_improves_segment_f1(self):
        base = [[(16, 18)]]
        candidates = [[(8, 18)]]
        scores = [np.array([0.9], dtype=np.float32)]
        labels = [np.array([0] * 8 + [1] * 11 + [0] * 5, dtype=np.int64)]

        config, metrics, refined = select_boundary_distribution_refine_params(
            base,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.5],
            min_base_coverages=[0.8],
            max_length_ratios=[5.0],
            start_quantiles=[0.5],
            end_quantiles=[0.5],
            base_boundary_weights=[0.0],
            max_shift_ratios=[4.0],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(refined, [[(8, 18)]])
        self.assertEqual(metrics["segment"]["tp"], 1)
        self.assertEqual(metrics["segment"]["fn"], 0)

    def test_param_search_disables_boundary_refine_when_only_fp_grows(self):
        base = [[(10, 14)]]
        candidates = [[(5, 24)]]
        scores = [np.array([0.9], dtype=np.float32)]
        labels = [np.array([0] * 10 + [1] * 5 + [0] * 15, dtype=np.int64)]

        config, metrics, refined = select_boundary_distribution_refine_params(
            base,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.5],
            min_base_coverages=[0.8],
            max_length_ratios=[5.0],
            start_quantiles=[0.5],
            end_quantiles=[0.5],
            base_boundary_weights=[0.0],
            max_shift_ratios=[4.0],
            iou_threshold=0.5,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(refined, base)
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
