import unittest

import numpy as np

from jepa_merge_noise_selective_splitter import (
    extract_merge_noise_parent_records,
    split_merge_noise_predictions,
)


class JEPAMergeNoiseSelectiveSplitterTests(unittest.TestCase):
    def test_extract_parent_records_detects_multi_island_child_set(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20)]]
        scores = [np.array([0.6, 0.55, 0.95], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = 0.9
        evidence[0][11:15] = 0.85

        records = extract_merge_noise_parent_records(
            base,
            candidates,
            scores,
            evidence,
            island_threshold=0.7,
            island_min_length=2,
            island_min_gap=1,
            min_islands=2,
            min_candidate_score=0.0,
            max_candidate_rank=None,
            min_island_coverage=0.75,
            max_child_parent_ratio=0.7,
            min_child_parent_coverage=0.5,
            child_nms_iou=0.3,
            max_children_per_parent=3,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].children, [(1, 4), (11, 14)])
        self.assertGreater(records[0].features[0], 1.5)
        self.assertGreater(records[0].features[-1], 2.0)

    def test_split_merge_noise_predictions_uses_parent_scores(self):
        base = [[(0, 20), (30, 40)]]
        candidates = [[(1, 4), (11, 14), (0, 20)]]
        scores = [np.array([0.6, 0.55, 0.95], dtype=np.float32)]
        evidence = [np.zeros((45, 3), dtype=np.float32)]
        evidence[0][1:5] = 0.9
        evidence[0][11:15] = 0.85

        split = split_merge_noise_predictions(
            base,
            candidates,
            scores,
            evidence,
            parent_scores=[np.array([0.9], dtype=np.float32)],
            threshold=0.5,
            island_threshold=0.7,
            island_min_length=2,
            island_min_gap=1,
            min_islands=2,
            min_candidate_score=0.0,
            max_candidate_rank=None,
            min_island_coverage=0.75,
            max_child_parent_ratio=0.7,
            min_child_parent_coverage=0.5,
            child_nms_iou=0.3,
            max_children_per_parent=3,
            max_replaced_parents_per_video=1,
        )

        self.assertEqual(split, [[(1, 4), (11, 14), (30, 40)]])

    def test_split_merge_noise_predictions_rejects_low_parent_score(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14)]]
        scores = [np.array([0.6, 0.55], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = 0.9
        evidence[0][11:15] = 0.85

        split = split_merge_noise_predictions(
            base,
            candidates,
            scores,
            evidence,
            parent_scores=[np.array([0.1], dtype=np.float32)],
            threshold=0.5,
            island_threshold=0.7,
            island_min_length=2,
            island_min_gap=1,
            min_islands=2,
            min_candidate_score=0.0,
            max_candidate_rank=None,
            min_island_coverage=0.75,
            max_child_parent_ratio=0.7,
            min_child_parent_coverage=0.5,
            child_nms_iou=0.3,
            max_children_per_parent=3,
            max_replaced_parents_per_video=1,
        )

        self.assertEqual(split, [[(0, 20)]])


if __name__ == "__main__":
    unittest.main()
