import unittest

import numpy as np

from jepa_residual_sub_event_expansion import (
    select_residual_sub_event_expansion_params,
    select_residual_sub_event_predictions,
)


class JEPAResidualSubEventExpansionTests(unittest.TestCase):
    def test_adds_low_selector_child_inside_long_parent_when_jepa_evidence_is_local(self):
        base = [[(0, 119)]]
        candidates = [[(0, 119), (70, 82), (20, 95)]]
        scores = [np.array([0.9, 0.08, 0.6], dtype=np.float32)]
        evidence = np.zeros(120, dtype=np.float32)
        evidence[70:83] = 0.95

        fused = select_residual_sub_event_predictions(
            base,
            candidates,
            scores,
            [evidence],
            min_selector_score=0.05,
            evidence_threshold=0.8,
            min_evidence_mean=0.75,
            min_active_fraction=0.8,
            min_parent_contrast=0.5,
            parent_min_length=48,
            max_child_parent_ratio=0.25,
            min_child_parent_coverage=0.9,
            length_penalty=0.0,
            nms_iou=0.3,
            max_sub_events_per_parent=2,
            max_sub_events_per_video=3,
        )

        self.assertEqual(fused, [[(0, 119), (70, 82)]])

    def test_rejects_child_without_residual_jepa_contrast(self):
        base = [[(0, 119)]]
        candidates = [[(70, 82)]]
        scores = [np.array([0.95], dtype=np.float32)]
        evidence = np.full(120, 0.7, dtype=np.float32)

        fused = select_residual_sub_event_predictions(
            base,
            candidates,
            scores,
            [evidence],
            min_selector_score=0.05,
            evidence_threshold=0.6,
            min_evidence_mean=0.6,
            min_active_fraction=0.8,
            min_parent_contrast=0.2,
            parent_min_length=48,
            max_child_parent_ratio=0.25,
            min_child_parent_coverage=0.9,
            length_penalty=0.0,
            nms_iou=0.3,
            max_sub_events_per_parent=2,
            max_sub_events_per_video=3,
        )

        self.assertEqual(fused, [[(0, 119)]])

    def test_replace_mode_decomposes_long_parent_into_multiple_jepa_children(self):
        base = [[(0, 119)]]
        candidates = [[(0, 49), (70, 82)]]
        scores = [np.array([0.2, 0.1], dtype=np.float32)]
        evidence = np.zeros(120, dtype=np.float32)
        evidence[0:50] = 0.9
        evidence[70:83] = 0.95

        fused = select_residual_sub_event_predictions(
            base,
            candidates,
            scores,
            [evidence],
            min_selector_score=0.05,
            evidence_threshold=0.8,
            min_evidence_mean=0.75,
            min_active_fraction=0.8,
            min_parent_contrast=0.5,
            parent_min_length=48,
            max_child_parent_ratio=0.5,
            min_child_parent_coverage=0.9,
            length_penalty=0.0,
            nms_iou=0.3,
            max_sub_events_per_parent=3,
            max_sub_events_per_video=3,
            mode="replace",
            min_sub_events_per_parent=2,
        )

        self.assertEqual(fused, [[(0, 49), (70, 82)]])

    def test_select_params_enables_only_when_segment_f1_improves(self):
        base = [[(0, 119)]]
        candidates = [[(70, 82)]]
        scores = [np.array([0.05], dtype=np.float32)]
        evidence = np.zeros(120, dtype=np.float32)
        evidence[70:83] = 0.95
        labels = [np.zeros(120, dtype=np.int64)]
        labels[0][0:50] = 1
        labels[0][70:83] = 1

        config, metrics, fused = select_residual_sub_event_expansion_params(
            base,
            candidates,
            scores,
            [evidence],
            labels,
            min_selector_scores=[0.01],
            evidence_thresholds=[0.8],
            min_evidence_means=[0.75],
            min_active_fractions=[0.8],
            min_parent_contrasts=[0.5],
            parent_min_lengths=[48],
            max_child_parent_ratios=[0.25],
            min_child_parent_coverages=[0.9],
            length_penalties=[0.0],
            nms_ious=[0.3],
            max_sub_events_per_parents=[2],
            max_sub_events_per_videos=[3],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(fused, [[(0, 119), (70, 82)]])
        self.assertEqual(metrics["segment"]["tp"], 2)

    def test_select_params_can_choose_replace_mode_for_over_merged_parent(self):
        base = [[(0, 119)]]
        candidates = [[(0, 49), (70, 82)]]
        scores = [np.array([0.2, 0.1], dtype=np.float32)]
        evidence = np.zeros(120, dtype=np.float32)
        evidence[0:50] = 0.9
        evidence[70:83] = 0.95
        labels = [np.zeros(120, dtype=np.int64)]
        labels[0][0:50] = 1
        labels[0][70:83] = 1

        config, metrics, fused = select_residual_sub_event_expansion_params(
            base,
            candidates,
            scores,
            [evidence],
            labels,
            min_selector_scores=[0.05],
            evidence_thresholds=[0.8],
            min_evidence_means=[0.75],
            min_active_fractions=[0.8],
            min_parent_contrasts=[0.5],
            parent_min_lengths=[48],
            max_child_parent_ratios=[0.5],
            min_child_parent_coverages=[0.9],
            length_penalties=[0.0],
            nms_ious=[0.3],
            max_sub_events_per_parents=[3],
            max_sub_events_per_videos=[3],
            modes=["replace"],
            min_sub_events_per_parents=[2],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(config["mode"], "replace")
        self.assertEqual(fused, [[(0, 49), (70, 82)]])
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
