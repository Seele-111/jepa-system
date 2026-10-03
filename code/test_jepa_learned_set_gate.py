import unittest

import numpy as np

from jepa_learned_set_gate import (
    build_gate_examples_from_folds,
    label_switch_preference,
    run_fold_heldout_learned_set_gate,
    set_gate_feature_vector,
    select_threshold_on_gate_scores,
)
from train_segment_locator import VideoRecord


class JEPALearnedSetGateTests(unittest.TestCase):
    def test_set_gate_feature_vector_reflects_recall_evidence_gain(self):
        signals = np.zeros((12, 3), dtype=np.float32)
        signals[7:10, :] = 1.0
        record = VideoRecord("v0", signals, np.zeros(12, dtype=np.int64))

        features, names = set_gate_feature_vector(
            record,
            feature_names=["true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"],
            base=[(0, 2)],
            recall=[(7, 9)],
            evidence_names=["true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"],
        )

        name_to_value = dict(zip(names, features))
        self.assertGreater(name_to_value["delta_evidence_mean"], 0.5)
        self.assertIn("length_ratio", name_to_value)
        self.assertEqual(len(features), len(names))

    def test_label_switch_preference_marks_recall_better_when_it_recovers_event(self):
        labels = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int64)

        label, stats = label_switch_preference(
            base=[(0, 2)],
            recall=[(0, 2), (6, 8)],
            labels=labels,
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertEqual(label, 1)
        self.assertEqual(stats["base_tp"], 1)
        self.assertEqual(stats["recall_tp"], 2)

    def test_select_threshold_on_gate_scores_respects_fp_budget(self):
        base = [[(0, 2)], [(0, 2)]]
        recall = [[(0, 2), (6, 8)], [(0, 2), (6, 8)]]
        labels = [
            np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int64),
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        ]
        scores = np.array([0.9, 0.8], dtype=np.float32)

        config, metrics, predictions = select_threshold_on_gate_scores(
            base,
            recall,
            labels,
            scores,
            thresholds=[0.85, 0.75],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(0, 2), (6, 8)], [(0, 2)]])
        self.assertEqual(metrics["segment"]["tp"], 3)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_run_fold_heldout_learned_set_gate_returns_oof_summary(self):
        records = []
        folds = []
        for idx in range(4):
            signals = np.zeros((10, 3), dtype=np.float32)
            labels = np.zeros(10, dtype=np.int64)
            labels[0:3] = 1
            if idx % 2 == 0:
                labels[6:9] = 1
                signals[6:9, :] = 1.0
                recall = [[(0, 2), (6, 8)]]
            else:
                signals[6:9, :] = 0.1
                recall = [[(0, 2), (6, 8)]]
            records.append(VideoRecord(f"v{idx}", signals, labels))
            folds.append(
                {
                    "fold": idx,
                    "val_idx": [idx],
                    "base": [[(0, 2)]],
                    "recall": recall,
                    "labels": [labels],
                }
            )

        result = run_fold_heldout_learned_set_gate(
            folds,
            records,
            feature_names=["true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"],
            evidence_names=["true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"],
            model_name="logreg",
            thresholds=[0.5],
            max_fp_increase=0,
            iou_threshold=0.3,
            seed=7,
        )

        self.assertEqual(len(result["folds"]), 4)
        self.assertIn("aggregate", result)
        self.assertIn("segment", result["aggregate"])


if __name__ == "__main__":
    unittest.main()
