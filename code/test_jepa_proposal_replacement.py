import unittest

import numpy as np

from jepa_proposal_replacement import replace_merged_predictions, select_proposal_replacement_params


class JEPAProposalReplacementTests(unittest.TestCase):
    def test_replaces_long_parent_with_multiple_high_score_candidates(self):
        mainline = [[(0, 20)]]
        candidates = [[(1, 4), (10, 14), (0, 20)]]
        scores = [np.array([0.9, 0.85, 0.95], dtype=np.float32)]

        replaced = replace_merged_predictions(
            mainline,
            candidates,
            scores,
            prob_threshold=0.8,
            parent_min_length=12,
            min_replacements=2,
            max_replacements_per_parent=3,
            min_candidate_parent_coverage=0.8,
            max_candidate_parent_ratio=0.6,
            length_penalty=0.0,
        )

        self.assertEqual(replaced, [[(1, 4), (10, 14)]])

    def test_keeps_parent_when_only_one_candidate_survives(self):
        mainline = [[(0, 20)]]
        candidates = [[(1, 4), (10, 14)]]
        scores = [np.array([0.9, 0.2], dtype=np.float32)]

        replaced = replace_merged_predictions(
            mainline,
            candidates,
            scores,
            prob_threshold=0.8,
            parent_min_length=12,
            min_replacements=2,
            max_replacements_per_parent=3,
            min_candidate_parent_coverage=0.8,
            max_candidate_parent_ratio=0.6,
            length_penalty=0.0,
        )

        self.assertEqual(replaced, mainline)

    def test_select_proposal_replacement_params_enables_improving_replacement(self):
        mainline = [[(0, 20)]]
        candidates = [[(1, 4), (10, 14), (0, 20)]]
        scores = [np.array([0.9, 0.85, 0.95], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)]

        config, metrics, replaced = select_proposal_replacement_params(
            mainline,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.8],
            parent_min_lengths=[12],
            min_replacements_list=[2],
            max_replacements_per_parents=[3],
            min_candidate_parent_coverages=[0.8],
            max_candidate_parent_ratios=[0.6],
            length_penalties=[0.0],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(replaced, [[(1, 4), (10, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)

    def test_select_proposal_replacement_params_disables_harmful_replacement(self):
        mainline = [[(0, 20)]]
        candidates = [[(1, 4), (10, 14), (0, 20)]]
        scores = [np.array([0.9, 0.85, 0.95], dtype=np.float32)]
        labels = [np.array([1] * 21, dtype=np.int64)]

        config, metrics, replaced = select_proposal_replacement_params(
            mainline,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.8],
            parent_min_lengths=[12],
            min_replacements_list=[2],
            max_replacements_per_parents=[3],
            min_candidate_parent_coverages=[0.8],
            max_candidate_parent_ratios=[0.6],
            length_penalties=[0.0],
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(replaced, mainline)
        self.assertEqual(metrics["segment"]["tp"], 1)

    def test_select_proposal_replacement_params_can_guard_against_extra_fp(self):
        mainline = [[(1, 14)]]
        candidates = [[(1, 4), (6, 8), (10, 14)]]
        scores = [np.array([0.9, 0.85, 0.84], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)]

        config, metrics, replaced = select_proposal_replacement_params(
            mainline,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.8],
            parent_min_lengths=[12],
            min_replacements_list=[2],
            max_replacements_per_parents=[3],
            min_candidate_parent_coverages=[0.8],
            max_candidate_parent_ratios=[0.6],
            length_penalties=[0.0],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(replaced, mainline)
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
