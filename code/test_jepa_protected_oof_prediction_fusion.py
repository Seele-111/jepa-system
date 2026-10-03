import unittest

import numpy as np

from jepa_protected_oof_prediction_fusion import (
    apply_protected_addition_config,
    candidate_additions,
    run_fold_heldout_fusion,
    select_config_on_folds,
)
from train_proposal_calibrator import evaluate_segment_predictions


class JEPAProtectedOOFPredictionFusionTests(unittest.TestCase):
    def test_candidate_additions_keeps_recall_segments_that_do_not_overlap_base(self):
        additions = candidate_additions(
            base=[(0, 4)],
            recall=[(0, 4), (10, 14), (20, 22)],
            max_base_iou=0.0,
        )

        self.assertEqual(additions, [(10, 14), (20, 22)])

    def test_fusion_rejects_candidate_when_fp_budget_exceeded(self):
        base = [[(0, 4)]]
        recall = [[(0, 4), (10, 14)]]
        labels = [np.array([1, 1, 1, 1, 1] + [0] * 15, dtype=np.int64)]

        selected = apply_protected_addition_config(
            base,
            recall,
            max_base_iou=0.0,
            max_additions_per_video=1,
            nms_iou=None,
        )
        selected_metrics = evaluate_segment_predictions(selected, labels, iou_threshold=0.3)
        base_metrics = evaluate_segment_predictions(base, labels, iou_threshold=0.3)

        self.assertEqual(selected_metrics["fp"], base_metrics["fp"] + 1)
        config, metrics, _ = select_config_on_folds(
            [{"base": base, "recall": recall, "labels": labels}],
            max_fp_increase=0,
            max_base_ious=[0.0],
            max_additions_per_videos=[1],
            nms_ious=[None],
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(metrics["segment"]["fp"], base_metrics["fp"])

    def test_select_config_accepts_candidate_when_tp_improves_without_fp(self):
        base = [[(0, 4)]]
        recall = [[(0, 4), (10, 14)]]
        labels = [np.array([1, 1, 1, 1, 1] + [0] * 5 + [1, 1, 1, 1, 1], dtype=np.int64)]

        config, metrics, predictions = select_config_on_folds(
            [{"base": base, "recall": recall, "labels": labels}],
            max_fp_increase=0,
            max_base_ious=[0.0],
            max_additions_per_videos=[1],
            nms_ious=[None],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(0, 4), (10, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_fold_heldout_selection_does_not_use_current_fold_labels(self):
        folds = [
            {
                "base": [[(0, 4)]],
                "recall": [[(0, 4), (10, 14)]],
                "labels": [np.array([1, 1, 1, 1, 1] + [0] * 15, dtype=np.int64)],
            },
            {
                "base": [[(0, 4)]],
                "recall": [[(0, 4), (10, 14)]],
                "labels": [np.array([1, 1, 1, 1, 1] + [0] * 5 + [1, 1, 1, 1, 1], dtype=np.int64)],
            },
        ]

        result = run_fold_heldout_fusion(
            folds,
            max_fp_increase=0,
            max_base_ious=[0.0],
            max_additions_per_videos=[1],
            nms_ious=[None],
            iou_threshold=0.3,
        )

        self.assertTrue(result["folds"][0]["config"]["enabled"])
        self.assertFalse(result["folds"][1]["config"]["enabled"])
        self.assertEqual(result["aggregate"]["segment"]["tp"], 2)
        self.assertEqual(result["aggregate"]["segment"]["fp"], 1)


if __name__ == "__main__":
    unittest.main()
