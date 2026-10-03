import unittest

import numpy as np

from jepa_evidence_island_splitter import (
    evidence_islands,
    split_evidence_island_predictions,
)


class JEPAEvidenceIslandSplitterTests(unittest.TestCase):
    def test_evidence_islands_detects_separated_high_regions(self):
        curve = np.array([0.1, 0.8, 0.85, 0.1, 0.2, 0.9, 0.88, 0.1], dtype=np.float32)

        islands = evidence_islands(curve, threshold=0.7, min_length=2, min_gap=0)

        self.assertEqual(islands, [(1, 2), (5, 6)])

    def test_splitter_replaces_parent_when_children_cover_evidence_islands(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20)]]
        scores = [np.array([0.12, 0.11, 0.9], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = 0.9
        evidence[0][11:15] = 0.85

        split = split_evidence_island_predictions(
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
            max_replaced_parents_per_video=1,
        )

        self.assertEqual(split, [[(1, 4), (11, 14)]])

    def test_splitter_allows_child_inside_larger_evidence_island(self):
        base = [[(0, 20)]]
        candidates = [[(2, 3), (12, 13), (0, 20)]]
        scores = [np.array([0.12, 0.11, 0.9], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:6] = 0.9
        evidence[0][11:16] = 0.85

        split = split_evidence_island_predictions(
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
            min_island_coverage=0.3,
            max_child_parent_ratio=0.7,
            min_child_parent_coverage=0.5,
            child_nms_iou=0.3,
            max_children_per_parent=3,
            max_replaced_parents_per_video=1,
        )

        self.assertEqual(split, [[(2, 3), (12, 13)]])


if __name__ == "__main__":
    unittest.main()
