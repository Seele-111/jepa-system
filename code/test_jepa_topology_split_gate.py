import unittest

import numpy as np

from jepa_topology_split_gate import (
    build_split_gate_records,
    build_synthetic_split_gate_records,
    select_topology_split_gate_params,
)


class JEPATopologySplitGateTests(unittest.TestCase):
    def test_records_label_parent_split_when_children_cover_multiple_gt(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (30, 32)]]
        scores = [np.array([0.1, 0.2, 0.9], dtype=np.float32)]
        evidence = [np.zeros((40, 3), dtype=np.float32)]
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 25, dtype=np.int64)]

        records = build_split_gate_records(
            base,
            candidates,
            scores,
            evidence,
            labels,
            iou_threshold=0.3,
            parent_min_length=12,
            min_child_parent_coverage=0.5,
            max_child_parent_ratio=0.5,
            max_children_per_parent=3,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].label, 1)
        self.assertEqual(records[0].children, [(1, 4), (11, 14)])

    def test_select_gate_params_applies_learned_split_when_it_improves(self):
        train_base = [[(0, 20)], [(0, 20)]]
        train_candidates = [[(1, 4), (11, 14)], [(1, 4), (11, 14)]]
        train_scores = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.9, 0.8], dtype=np.float32)]
        train_evidence = [np.zeros((40, 3), dtype=np.float32), np.zeros((40, 3), dtype=np.float32)]
        train_labels = [
            np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 25, dtype=np.int64),
            np.array([0] * 40, dtype=np.int64),
        ]
        val_base = [[(0, 20)]]
        val_candidates = [[(1, 4), (11, 14)]]
        val_scores = [np.array([0.12, 0.18], dtype=np.float32)]
        val_evidence = [np.zeros((40, 3), dtype=np.float32)]
        val_labels = [train_labels[0]]

        config, metrics, predictions = select_topology_split_gate_params(
            train_base,
            train_candidates,
            train_scores,
            train_evidence,
            train_labels,
            val_base,
            val_candidates,
            val_scores,
            val_evidence,
            val_labels,
            model_name="logreg",
            seed=7,
            thresholds=[0.1, 0.3, 0.5],
            parent_min_lengths=[12],
            min_child_parent_coverages=[0.5],
            max_child_parent_ratios=[0.5],
            max_children_per_parents=[3],
            max_replaced_parents_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(1, 4), (11, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_synthetic_records_create_positive_merged_parent_examples(self):
        candidates = [[(1, 4), (11, 14), (30, 34)]]
        scores = [np.array([0.1, 0.2, 0.9], dtype=np.float32)]
        evidence = [np.zeros((40, 3), dtype=np.float32)]
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 25, dtype=np.int64)]

        records = build_synthetic_split_gate_records(
            candidates,
            scores,
            evidence,
            labels,
            iou_threshold=0.3,
            max_gt_gap=12,
            parent_pad=0,
            min_child_parent_coverage=0.5,
            max_child_parent_ratio=0.5,
            max_children_per_parent=3,
        )

        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0].label, 1)
        self.assertEqual(records[0].children, [(1, 4), (11, 14)])


if __name__ == "__main__":
    unittest.main()
