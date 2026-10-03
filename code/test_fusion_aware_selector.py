import unittest

import numpy as np

from fusion_aware_selector import select_fusion_aware_selector_params_from_scores


class FusionAwareSelectorTests(unittest.TestCase):
    def test_selects_selector_threshold_that_improves_final_protected_fusion(self):
        mainline = [[(1, 2)]]
        candidates = [[(5, 6)]]
        scores = [np.array([0.6], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)]

        params, metrics, selector_predictions = select_fusion_aware_selector_params_from_scores(
            mainline,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.7, 0.5],
            length_penalties=[0.0],
            protected_mainline_iou_candidates=[0.0],
            selector_nms_iou_candidates=[None],
            iou_threshold=0.3,
        )

        self.assertEqual(params["prob_threshold"], 0.5)
        self.assertEqual(selector_predictions, [[(5, 6)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
