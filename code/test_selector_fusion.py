import unittest

import numpy as np

from selector_fusion import (
    evaluate_fused_predictions,
    fuse_segment_predictions,
    fuse_protected_segment_predictions,
    select_fusion_nms_iou,
    select_protected_fusion_params,
    short_first_nms_segments,
)


class SelectorFusionTests(unittest.TestCase):
    def test_short_first_nms_keeps_specific_event_over_merged_span(self):
        segments = [(1, 8), (1, 2), (6, 8)]

        kept = short_first_nms_segments(segments, iou_threshold=0.25)

        self.assertEqual(kept, [(1, 2), (6, 8)])

    def test_fuse_segment_predictions_adds_selector_recall_and_suppresses_duplicates(self):
        mainline = [[(1, 3)]]
        selector = [[(1, 3), (8, 9)]]

        fused = fuse_segment_predictions(mainline, selector, nms_iou=0.3)

        self.assertEqual(fused, [[(1, 3), (8, 9)]])

    def test_select_fusion_nms_iou_optimizes_segment_f1(self):
        mainline = [[(1, 2)]]
        selector = [[(7, 8), (10, 11)]]
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.int64)]

        nms_iou, metrics = select_fusion_nms_iou(
            mainline,
            selector,
            labels,
            candidates=[None, 0.3],
            iou_threshold=0.3,
        )

        self.assertIsNone(nms_iou)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)

    def test_evaluate_fused_predictions_reports_frame_and_segment_metrics(self):
        labels = [np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)]
        predictions = [[(1, 2), (4, 4)]]

        metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=0.3)

        self.assertEqual(metrics["frame"]["tp"], 2)
        self.assertEqual(metrics["frame"]["fp"], 1)
        self.assertEqual(metrics["frame"]["fn"], 1)
        self.assertEqual(metrics["segment"]["tp"], 1)
        self.assertEqual(metrics["segment"]["fp"], 1)
        self.assertEqual(metrics["segment"]["fn"], 1)

    def test_protected_fusion_never_replaces_overlapping_mainline_segment(self):
        mainline = [[(1, 6)]]
        selector = [[(1, 2), (8, 9)]]

        fused = fuse_protected_segment_predictions(
            mainline,
            selector,
            max_mainline_iou=0.3,
            selector_nms_iou=0.3,
        )

        self.assertEqual(fused, [[(1, 6), (8, 9)]])

    def test_select_protected_fusion_can_choose_rescue_when_it_improves_recall(self):
        mainline = [[(1, 2)]]
        selector = [[(7, 8)]]
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int64)]

        config, metrics = select_protected_fusion_params(
            mainline,
            selector,
            labels,
            max_mainline_iou_candidates=[None, 0.1],
            selector_nms_iou_candidates=[0.3],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fn"], 0)

    def test_select_protected_fusion_can_disable_rescue_when_it_only_adds_false_positive(self):
        mainline = [[(1, 2)]]
        selector = [[(7, 8)]]
        labels = [np.array([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)]

        config, metrics = select_protected_fusion_params(
            mainline,
            selector,
            labels,
            max_mainline_iou_candidates=[None, 0.1],
            selector_nms_iou_candidates=[0.3],
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
