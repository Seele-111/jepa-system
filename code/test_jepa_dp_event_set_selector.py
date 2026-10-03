import unittest

import numpy as np

from jepa_dp_event_set_selector import (
    run_fold_heldout_dp_event_set,
    select_dp_event_set_params,
    select_dp_event_set_predictions,
)


class JEPADPEventSetSelectorTests(unittest.TestCase):
    def test_dp_selector_replaces_wide_parent_with_two_high_score_children(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14)]]
        scores = [np.array([0.9, 0.85], dtype=np.float32)]

        selected = select_dp_event_set_predictions(
            base,
            candidates,
            scores,
            base_keep_score=0.5,
            candidate_score_weight=1.0,
            min_candidate_score=0.1,
            length_penalty=0.0,
            base_overlap_penalty=0.0,
        )

        self.assertEqual(selected, [[(1, 4), (11, 14)]])

    def test_dp_selector_keeps_base_when_children_are_low_score(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14)]]
        scores = [np.array([0.1, 0.1], dtype=np.float32)]

        selected = select_dp_event_set_predictions(
            base,
            candidates,
            scores,
            base_keep_score=0.5,
            candidate_score_weight=1.0,
            min_candidate_score=0.0,
            length_penalty=0.0,
            base_overlap_penalty=0.0,
        )

        self.assertEqual(selected, [[(0, 20)]])

    def test_param_search_enables_replacement_when_segment_f1_improves(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14)]]
        scores = [np.array([0.9, 0.85], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 8, dtype=np.int64)]

        config, metrics, selected = select_dp_event_set_params(
            base,
            candidates,
            scores,
            labels,
            base_keep_scores=[0.5],
            candidate_score_weights=[1.0],
            min_candidate_scores=[0.1],
            length_penalties=[0.0],
            base_overlap_penalties=[0.0],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(selected, [[(1, 4), (11, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_fold_heldout_dp_does_not_use_current_fold_labels_for_selection(self):
        good_split_labels = np.array(
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 8,
            dtype=np.int64,
        )
        stable_base_labels = np.array([1, 1, 1, 1, 1] + [0] * 18, dtype=np.int64)
        folds = [
            {
                "fold": 0,
                "base": [[(0, 4)]],
                "candidates": [[(8, 10)]],
                "scores": [np.array([0.95], dtype=np.float32)],
                "labels": [stable_base_labels],
            },
            {
                "fold": 1,
                "base": [[(0, 4)]],
                "candidates": [[(12, 14)]],
                "scores": [np.array([0.95], dtype=np.float32)],
                "labels": [stable_base_labels],
            },
            {
                "fold": 2,
                "base": [[(0, 20)]],
                "candidates": [[(1, 4), (11, 14)]],
                "scores": [np.array([0.95, 0.9], dtype=np.float32)],
                "labels": [good_split_labels],
            },
        ]

        result = run_fold_heldout_dp_event_set(
            folds,
            base_keep_scores=[0.5],
            candidate_score_weights=[1.0],
            min_candidate_scores=[0.1],
            length_penalties=[0.0],
            base_overlap_penalties=[0.0],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        fold_two = [fold for fold in result["folds"] if fold["fold"] == 2][0]
        self.assertFalse(fold_two["config"]["enabled"])
        self.assertEqual(fold_two["predictions"], [[[0, 20]]])


if __name__ == "__main__":
    unittest.main()
