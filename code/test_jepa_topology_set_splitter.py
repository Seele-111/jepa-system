import unittest

import numpy as np

from jepa_topology_set_splitter import (
    select_topology_set_split_params,
    split_topology_set_predictions,
)


class JEPATopologySetSplitterTests(unittest.TestCase):
    def test_set_splitter_replaces_parent_with_low_threshold_child_set(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20)]]
        scores = [np.array([0.08, 0.12, 0.8], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = np.array([0.42, 0.39, 0.41], dtype=np.float32)
        evidence[0][11:15] = np.array([0.45, 0.43, 0.44], dtype=np.float32)

        split = split_topology_set_predictions(
            base,
            candidates,
            scores,
            evidence,
            evidence_threshold=0.4,
            min_selector_score=0.0,
            max_selector_rank=None,
            min_evidence_mean=0.2,
            min_active_fraction=0.0,
            min_parent_contrast=-0.3,
            parent_min_length=12,
            max_child_parent_ratio=0.5,
            min_child_parent_coverage=0.5,
            min_gap_between_children=0,
            child_score_threshold=0.0,
            set_score_threshold=0.1,
            selector_weight=1.0,
            evidence_weight=1.0,
            active_fraction_weight=0.25,
            contrast_weight=0.5,
            length_penalty=0.0,
            nms_iou=0.3,
            min_children_per_parent=2,
            max_children_per_parent=3,
            max_replaced_parents_per_video=1,
        )

        self.assertEqual(split, [[(1, 4), (11, 14)]])

    def test_select_params_enables_when_set_split_improves_f1(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20)]]
        scores = [np.array([0.08, 0.12, 0.8], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = np.array([0.42, 0.39, 0.41], dtype=np.float32)
        evidence[0][11:15] = np.array([0.45, 0.43, 0.44], dtype=np.float32)
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)]

        config, metrics, split = select_topology_set_split_params(
            base,
            candidates,
            scores,
            evidence,
            labels,
            evidence_thresholds=[0.4],
            min_selector_scores=[0.0],
            max_selector_ranks=[None],
            min_evidence_means=[0.2],
            min_active_fractions=[0.0],
            min_parent_contrasts=[-0.3],
            parent_min_lengths=[12],
            max_child_parent_ratios=[0.5],
            min_child_parent_coverages=[0.5],
            min_gap_between_children_values=[0],
            child_score_thresholds=[0.0],
            set_score_thresholds=[0.1],
            selector_weights=[1.0],
            evidence_weights=[1.0],
            active_fraction_weights=[0.25],
            contrast_weights=[0.5],
            length_penalties=[0.0],
            nms_ious=[0.3],
            min_children_per_parents=[2],
            max_children_per_parents=[3],
            max_replaced_parents_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(split, [[(1, 4), (11, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
