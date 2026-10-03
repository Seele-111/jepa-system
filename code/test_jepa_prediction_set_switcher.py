import unittest

import numpy as np

from jepa_prediction_set_switcher import (
    apply_set_switch_config,
    build_prediction_export_from_switch_result,
    prediction_set_features,
    run_fold_heldout_set_switching,
    select_set_switch_config,
)


class JEPAPredictionSetSwitcherTests(unittest.TestCase):
    def test_prediction_set_features_capture_overlap_and_length_change(self):
        features = prediction_set_features(base=[(0, 9)], recall=[(1, 8), (20, 24)])

        self.assertEqual(features["base_count"], 1)
        self.assertEqual(features["recall_count"], 2)
        self.assertEqual(features["count_delta"], 1)
        self.assertGreater(features["length_ratio"], 1.0)
        self.assertGreater(features["mean_recall_to_base_iou"], 0.0)

    def test_apply_set_switch_config_switches_only_matching_videos(self):
        switched = apply_set_switch_config(
            base_predictions=[[(0, 9)], [(0, 9)]],
            recall_predictions=[[(1, 8)], [(1, 8), (20, 24)]],
            config={
                "enabled": True,
                "min_count_delta": -1,
                "max_count_delta": 0,
                "min_length_ratio": 0.1,
                "max_length_ratio": 1.0,
                "min_mutual_iou": 0.5,
                "max_extra_segments": 0,
                "max_dropped_segments": 1,
            },
        )

        self.assertEqual(switched, [[(1, 8)], [(0, 9)]])

    def test_select_set_switch_config_uses_recall_when_it_improves_without_fp(self):
        folds = [
            {
                "base": [[(0, 12)]],
                "recall": [[(0, 4), (8, 12)]],
                "labels": [np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)],
            }
        ]

        config, metrics, predictions = select_set_switch_config(
            folds,
            max_fp_increase=0,
            max_count_deltas=[1],
            min_length_ratios=[0.1],
            max_length_ratios=[1.0],
            min_mutual_ious=[0.0],
            max_extra_segments_list=[2],
            max_dropped_segments_list=[1],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(0, 4), (8, 12)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_fold_heldout_set_switching_selects_rule_on_other_folds(self):
        folds = [
            {
                "base": [[(0, 12)]],
                "recall": [[(0, 4), (8, 12)]],
                "labels": [np.array([1, 1, 1, 1, 1] + [0] * 8, dtype=np.int64)],
            },
            {
                "base": [[(0, 12)]],
                "recall": [[(0, 4), (8, 12)]],
                "labels": [np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)],
            },
        ]

        result = run_fold_heldout_set_switching(
            folds,
            max_fp_increase=0,
            max_count_deltas=[1],
            min_length_ratios=[0.1],
            max_length_ratios=[1.0],
            min_mutual_ious=[0.0],
            max_extra_segments_list=[2],
            max_dropped_segments_list=[1],
            iou_threshold=0.3,
        )

        self.assertTrue(result["folds"][0]["config"]["enabled"])
        self.assertFalse(result["folds"][1]["config"]["enabled"])
        self.assertEqual(result["aggregate"]["segment"]["fp"], 1)

    def test_build_prediction_export_preserves_fold_indices_and_labels(self):
        source_folds = [
            {
                "fold": 3,
                "train_idx": [1, 2],
                "val_idx": [0],
                "val_names": ["v0"],
                "labels": [np.array([1, 0, 0], dtype=np.int64)],
            }
        ]
        result = {
            "folds": [
                {
                    "fold": 3,
                    "validation": {"segment": {"f1": 1.0}},
                    "predictions": [[[0, 0]]],
                }
            ]
        }

        export = build_prediction_export_from_switch_result(
            result,
            source_folds,
            include_labels=True,
        )

        self.assertEqual(export["folds"][0]["train_idx"], [1, 2])
        self.assertEqual(export["folds"][0]["val_idx"], [0])
        self.assertEqual(export["folds"][0]["val_names"], ["v0"])
        self.assertEqual(export["folds"][0]["predictions"], [[[0, 0]]])
        self.assertEqual(export["folds"][0]["labels"], [[1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
