import unittest

import numpy as np

from jepa_event_topology_splitter import (
    diagnose_swallowed_events,
    select_event_topology_split_params,
    split_event_topology_predictions,
)


class JEPAEventTopologySplitterTests(unittest.TestCase):
    def test_splitter_replaces_wide_parent_with_supported_children(self):
        base = [[(0, 20)]]
        candidates = [[(0, 20), (1, 4), (11, 14), (1, 5), (10, 14)]]
        scores = [np.array([0.8, 0.72, 0.71, 0.68, 0.67], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = np.array([0.9, 0.86, 0.88], dtype=np.float32)
        evidence[0][11:15] = np.array([0.88, 0.84, 0.86], dtype=np.float32)

        split = split_event_topology_predictions(
            base,
            candidates,
            scores,
            evidence,
            min_selector_score=0.6,
            evidence_threshold=0.7,
            min_evidence_mean=0.65,
            min_active_fraction=0.75,
            min_parent_contrast=0.2,
            parent_min_length=12,
            max_child_parent_ratio=0.35,
            min_child_parent_coverage=0.8,
            min_gap_between_children=2,
            support_iou=0.3,
            support_count_weight=0.05,
            length_penalty=0.0,
            nms_iou=0.3,
            min_children_per_parent=2,
            max_children_per_parent=3,
            max_replaced_parents_per_video=1,
        )

        self.assertEqual(split, [[(1, 4), (11, 14)]])

    def test_select_params_enables_only_when_split_improves_segment_f1(self):
        base = [[(0, 20)]]
        candidates = [[(0, 20), (1, 4), (11, 14), (1, 5), (10, 14)]]
        scores = [np.array([0.8, 0.72, 0.71, 0.68, 0.67], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = np.array([0.9, 0.86, 0.88], dtype=np.float32)
        evidence[0][11:15] = np.array([0.88, 0.84, 0.86], dtype=np.float32)
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)]

        config, metrics, split = select_event_topology_split_params(
            base,
            candidates,
            scores,
            evidence,
            labels,
            min_selector_scores=[0.6],
            evidence_thresholds=[0.7],
            min_evidence_means=[0.65],
            min_active_fractions=[0.75],
            min_parent_contrasts=[0.2],
            parent_min_lengths=[12],
            max_child_parent_ratios=[0.35],
            min_child_parent_coverages=[0.8],
            min_gap_between_children_values=[2],
            support_ious=[0.3],
            support_count_weights=[0.05],
            length_penalties=[0.0],
            nms_ious=[0.3],
            min_children_per_parents=[2],
            max_children_per_parents=[3],
            max_replaced_parents_per_videos=[1],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(split, [[(1, 4), (11, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fn"], 0)

    def test_swallowed_event_diagnostic_counts_unmatched_gt_inside_matched_parent(self):
        predictions = [[(0, 12)]]
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)]

        summary = diagnose_swallowed_events(predictions, labels, iou_threshold=0.3)

        self.assertEqual(summary["segment"]["fn"], 1)
        self.assertEqual(summary["swallowed_fn"], 1)
        self.assertEqual(summary["videos_with_swallowed_fn"], 1)

    def test_zero_support_weight_does_not_require_candidate_graph_count(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14)]]
        scores = [np.array([0.72, 0.71], dtype=np.float32)]
        evidence = [np.zeros((24, 3), dtype=np.float32)]
        evidence[0][1:5] = np.array([0.9, 0.86, 0.88], dtype=np.float32)
        evidence[0][11:15] = np.array([0.88, 0.84, 0.86], dtype=np.float32)

        split = split_event_topology_predictions(
            base,
            candidates,
            scores,
            evidence,
            min_selector_score=0.6,
            evidence_threshold=0.7,
            min_evidence_mean=0.65,
            min_active_fraction=0.75,
            min_parent_contrast=0.2,
            parent_min_length=12,
            max_child_parent_ratio=0.35,
            min_child_parent_coverage=0.8,
            min_gap_between_children=2,
            support_iou=0.3,
            support_count_weight=0.0,
            length_penalty=0.0,
            nms_iou=0.3,
            min_children_per_parent=2,
            max_children_per_parent=3,
            max_replaced_parents_per_video=1,
        )

        self.assertEqual(split, [[(1, 4), (11, 14)]])


if __name__ == "__main__":
    unittest.main()
