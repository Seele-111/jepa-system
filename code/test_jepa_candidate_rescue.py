import unittest

import numpy as np

from jepa_candidate_rescue import select_candidate_rescue_params, select_candidate_rescue_predictions


class JEPACandidateRescueTests(unittest.TestCase):
    def test_candidate_rescue_adds_uncovered_high_score_candidate(self):
        mainline = [[(1, 2)]]
        candidates = [[(1, 2), (7, 8), (10, 11)]]
        scores = [np.array([0.95, 0.9, 0.2], dtype=np.float32)]

        fused = select_candidate_rescue_predictions(
            mainline,
            candidates,
            scores,
            prob_threshold=0.5,
            max_mainline_iou=0.0,
            candidate_nms_iou=0.3,
            length_penalty=0.0,
            max_rescues_per_video=2,
        )

        self.assertEqual(fused, [[(1, 2), (7, 8)]])

    def test_candidate_rescue_limits_per_video_by_score(self):
        mainline = [[]]
        candidates = [[(1, 1), (3, 3), (5, 5)]]
        scores = [np.array([0.6, 0.9, 0.8], dtype=np.float32)]

        fused = select_candidate_rescue_predictions(
            mainline,
            candidates,
            scores,
            prob_threshold=0.5,
            max_mainline_iou=0.0,
            candidate_nms_iou=0.3,
            length_penalty=0.0,
            max_rescues_per_video=2,
        )

        self.assertEqual(fused, [[(3, 3), (5, 5)]])

    def test_candidate_rescue_can_allow_overlap_when_unbounded(self):
        mainline = [[(0, 10)]]
        candidates = [[(5, 8)]]
        scores = [np.array([0.9], dtype=np.float32)]

        fused = select_candidate_rescue_predictions(
            mainline,
            candidates,
            scores,
            prob_threshold=0.5,
            max_mainline_iou=None,
            candidate_nms_iou=None,
            length_penalty=0.0,
            max_rescues_per_video=1,
        )

        self.assertEqual(fused, [[(0, 10), (5, 8)]])

    def test_select_candidate_rescue_params_enables_improving_rescue(self):
        mainline = [[(1, 2)]]
        candidates = [[(7, 8)]]
        scores = [np.array([0.9], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int64)]

        config, metrics, fused = select_candidate_rescue_params(
            mainline,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.5],
            max_mainline_ious=[0.0],
            candidate_nms_ious=[0.3],
            length_penalties=[0.0],
            max_rescues_per_videos=[1],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(fused, [[(1, 2), (7, 8)]])
        self.assertEqual(metrics["segment"]["tp"], 2)

    def test_select_candidate_rescue_params_disables_fp_only_rescue(self):
        mainline = [[(1, 2)]]
        candidates = [[(7, 8)]]
        scores = [np.array([0.9], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)]

        config, metrics, fused = select_candidate_rescue_params(
            mainline,
            candidates,
            scores,
            labels,
            prob_thresholds=[0.5],
            max_mainline_ious=[0.0],
            candidate_nms_ious=[0.3],
            length_penalties=[0.0],
            max_rescues_per_videos=[1],
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(fused, mainline)
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
