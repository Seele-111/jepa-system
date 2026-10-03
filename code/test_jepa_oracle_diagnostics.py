import unittest

import numpy as np

from jepa_oracle_diagnostics import (
    build_oracle_rescue_predictions,
    evaluate_budgeted_oracle,
)


class JEPAOracleDiagnosticsTests(unittest.TestCase):
    def test_oracle_rescue_only_targets_unmatched_ground_truth(self):
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)]
        base = [[(1, 2)]]
        candidates = [[(1, 2), (5, 6), (7, 7)]]

        rescued = build_oracle_rescue_predictions(
            base,
            candidates,
            labels,
            iou_threshold=0.3,
            max_rescues_per_video=2,
            max_base_iou=0.0,
            selector_nms_iou=None,
        )

        self.assertEqual(rescued, [[(1, 2), (5, 6)]])

    def test_budgeted_oracle_reports_best_fp_constrained_recall_gain(self):
        labels = [
            np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64),
            np.array([0, 0, 1, 1, 0, 0], dtype=np.int64),
        ]
        base = [[(1, 2)], []]
        candidates = [[(5, 6), (0, 7)], [(2, 3), (4, 5)]]

        result = evaluate_budgeted_oracle(
            base,
            candidates,
            labels,
            iou_threshold=0.3,
            max_rescues_per_video_options=[1, 2],
            max_base_iou_options=[0.0],
            selector_nms_iou_options=[None],
            fp_budget_options=[0],
        )

        best = result["best_by_fp_budget"]["0"]
        self.assertEqual(best["metrics"]["segment"]["tp"], 3)
        self.assertEqual(best["metrics"]["segment"]["fp"], 0)
        self.assertEqual(best["rescued_tp"], 2)
        self.assertGreater(best["metrics"]["segment"]["f1"], result["base_metrics"]["segment"]["f1"])


if __name__ == "__main__":
    unittest.main()
