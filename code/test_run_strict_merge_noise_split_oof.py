import unittest
from types import SimpleNamespace

import numpy as np

from run_strict_merge_noise_split_oof import label_merge_noise_records, run_strict_merge_noise_split_oof


class StrictMergeNoiseSplitOOFTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            model="prototype",
            island_thresholds=[0.7],
            island_min_lengths=[2],
            island_min_gaps=[1],
            min_islands=[2],
            min_candidate_scores=[0.0],
            max_candidate_ranks=[None],
            min_island_coverages=[0.75],
            max_child_parent_ratios=[0.7],
            min_child_parent_coverages=[0.5],
            child_nms_ious=[0.3],
            max_children_per_parent=[3],
            thresholds=[0.0],
            max_replaced_parents_per_video=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
            seed=5,
            device="cpu",
            epochs=10,
            batch_size=16,
        )

    def _fold(self, fold_id: int):
        labels = np.zeros(24, dtype=np.int64)
        labels[1:5] = 1
        labels[11:15] = 1
        evidence = np.zeros((24, 3), dtype=np.float32)
        evidence[1:5] = 0.9
        evidence[11:15] = 0.85
        return {
            "fold": fold_id,
            "base": [[(0, 20)]],
            "candidates": [[(1, 4), (11, 14), (0, 20)]],
            "scores": [np.array([0.6, 0.55, 0.95], dtype=np.float32)],
            "evidence": [evidence],
            "labels": [labels],
        }

    def test_label_merge_noise_records_marks_helpful_safe_replacement(self):
        fold = self._fold(0)
        records = label_merge_noise_records(
            fold["base"],
            fold["candidates"],
            fold["scores"],
            fold["evidence"],
            fold["labels"],
            config={
                "island_threshold": 0.7,
                "island_min_length": 2,
                "island_min_gap": 1,
                "min_islands": 2,
                "min_candidate_score": 0.0,
                "max_candidate_rank": None,
                "min_island_coverage": 0.75,
                "max_child_parent_ratio": 0.7,
                "min_child_parent_coverage": 0.5,
                "child_nms_iou": 0.3,
                "max_children_per_parent": 3,
            },
            iou_threshold=0.3,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].label, 1)
        self.assertGreater(records[0].tp_delta, 0)
        self.assertLessEqual(records[0].fp_delta, 0)

    def test_strict_oof_applies_merge_noise_split_from_other_folds(self):
        result = run_strict_merge_noise_split_oof([self._fold(0), self._fold(1), self._fold(2)], self._args())

        self.assertEqual(result["aggregate"]["segment"]["tp"], 6)
        self.assertEqual(result["aggregate"]["segment"]["fp"], 0)
        self.assertEqual(result["aggregate"]["segment"]["fn"], 0)
        self.assertTrue(all(fold["config"]["enabled"] for fold in result["folds"]))


if __name__ == "__main__":
    unittest.main()
