import unittest

import numpy as np

from jepa_proposal_set_distillation import (
    build_distillation_candidate_records,
    fit_soft_quality_distillation_student,
    label_distillation_candidates,
    score_distillation_candidates,
    select_distilled_proposal_set_params,
    select_distilled_topology_replacement_params,
)
from train_segment_locator import VideoRecord


class ConstantModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict_proba(self, x):
        positive = np.resize(self.scores, len(x)).astype(np.float32)
        return np.stack([1.0 - positive, positive], axis=1)


class JEPAProposalSetDistillationTests(unittest.TestCase):
    def test_teacher_labels_oracle_rescue_set_not_all_overlapping_gt_candidates(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        base = [(1, 2)]
        candidates = [(1, 2), (5, 6), (8, 8), (9, 10)]

        y, best_iou, teacher = label_distillation_candidates(
            candidates,
            labels,
            base_segments=base,
            iou_threshold=0.3,
            max_teacher_rescues=1,
            max_base_iou=0.0,
            teacher_nms_iou=0.3,
        )

        np.testing.assert_array_equal(y, np.array([0, 1, 0, 0], dtype=np.int64))
        self.assertEqual(teacher, [(5, 6)])
        self.assertGreater(float(best_iou[1]), 0.99)

    def test_overlap_teacher_can_label_child_candidate_inside_wide_base(self):
        labels = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0], dtype=np.int64)
        base = [(0, 10)]
        candidates = [(1, 2), (8, 9), (4, 5)]

        y, _, teacher = label_distillation_candidates(
            candidates,
            labels,
            base_segments=base,
            iou_threshold=0.3,
            max_teacher_rescues=2,
            max_base_iou=0.0,
            teacher_nms_iou=0.3,
            teacher_mode="oracle_or_overlap",
        )

        np.testing.assert_array_equal(y, np.array([1, 1, 0], dtype=np.int64))
        self.assertEqual(teacher, [(1, 2), (8, 9)])

    def test_topology_teacher_labels_matched_and_swallowed_children_inside_parent(self):
        labels = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
        base = [(0, 8)]
        candidates = [(1, 4), (7, 8), (0, 8), (5, 5)]

        y, _, teacher = label_distillation_candidates(
            candidates,
            labels,
            base_segments=base,
            iou_threshold=0.3,
            max_teacher_rescues=3,
            max_base_iou=0.0,
            teacher_nms_iou=0.3,
            teacher_mode="oracle_or_topology",
        )

        np.testing.assert_array_equal(y, np.array([1, 1, 0, 0], dtype=np.int64))
        self.assertEqual(teacher, [(7, 8), (1, 4)])

    def test_build_records_keeps_low_selector_teacher_positive_under_prefilter(self):
        record = VideoRecord(
            name="toy",
            signals=np.array(
                [
                    [0.1],
                    [0.9],
                    [0.9],
                    [0.1],
                    [0.2],
                    [0.3],
                    [0.2],
                    [0.1],
                    [0.9],
                    [0.9],
                ],
                dtype=np.float32,
            ),
            labels=np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64),
        )

        rows = build_distillation_candidate_records(
            records=[record],
            indices=[0],
            candidate_predictions_by_idx={0: [(1, 2), (3, 3), (4, 4), (5, 5), (8, 9)]},
            candidate_scores_by_idx={0: np.array([0.95, 0.9, 0.8, 0.7, 0.01], dtype=np.float32)},
            base_segments_by_idx={0: [(1, 2)]},
            feature_names=["jepa"],
            channel_names=["jepa"],
            iou_threshold=0.3,
            max_teacher_rescues=1,
            max_base_iou=0.0,
            teacher_nms_iou=0.3,
            max_records_per_video=2,
            hard_negative_top_k=2,
        )

        self.assertEqual(len(rows), 3)
        self.assertIn((8, 9), [row.segment for row in rows])
        self.assertEqual(sum(row.label for row in rows), 1)

    def test_scores_use_student_features_and_match_candidate_count(self):
        record = VideoRecord(
            name="toy",
            signals=np.array([[0.1], [0.9], [0.8], [0.2]], dtype=np.float32),
            labels=np.zeros(4, dtype=np.int64),
        )

        scores = score_distillation_candidates(
            ConstantModel([0.2, 0.8]),
            record,
            candidates=[(0, 0), (1, 2)],
            selector_scores=np.array([0.4, 0.6], dtype=np.float32),
            feature_names=["jepa"],
            channel_names=["jepa"],
            base_segments=[(0, 0)],
        )

        np.testing.assert_allclose(scores, np.array([0.2, 0.8], dtype=np.float32), rtol=1e-5)

    def test_soft_quality_student_scores_high_iou_candidate_above_low_iou_candidate(self):
        x = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.8, 0.9],
                [0.1, 0.2],
            ],
            dtype=np.float32,
        )
        best_iou = np.array([0.0, 1.0, 0.8, 0.1], dtype=np.float32)

        student = fit_soft_quality_distillation_student(x, best_iou, seed=0, model_name="logreg")
        scores = student.predict_proba(x)[:, 1]

        self.assertGreater(float(scores[1]), float(scores[0]))
        self.assertGreater(float(scores[2]), float(scores[3]))

    def test_select_distilled_params_adds_student_selected_rescue_with_protection(self):
        base = [[(1, 2)]]
        candidates = [[(1, 2), (5, 6), (8, 8)]]
        student_scores = [np.array([0.95, 0.9, 0.7], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)]

        config, metrics, fused = select_distilled_proposal_set_params(
            base,
            candidates,
            student_scores,
            labels,
            thresholds=[0.5, 0.8],
            max_base_ious=[0.0],
            nms_ious=[0.3],
            length_penalties=[0.0],
            max_rescues_per_videos=[2],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(fused, [[(1, 2), (5, 6)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_distilled_topology_replacement_uses_student_scores_to_split_parent(self):
        base = [[(0, 20)]]
        candidates = [[(1, 4), (11, 14), (0, 20), (30, 32)]]
        student_scores = [np.array([0.9, 0.85, 0.05, 0.1], dtype=np.float32)]
        evidence = [np.zeros((40, 3), dtype=np.float32)]
        evidence[0][1:5] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        evidence[0][11:15] = np.array([0.52, 0.52, 0.52], dtype=np.float32)
        labels = [np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 25, dtype=np.int64)]

        config, metrics, replaced = select_distilled_topology_replacement_params(
            base,
            candidates,
            student_scores,
            evidence,
            labels,
            evidence_thresholds=[0.4],
            min_student_scores=[0.5],
            max_student_ranks=[None],
            min_evidence_means=[0.2],
            min_active_fractions=[0.0],
            min_parent_contrasts=[-0.3],
            parent_min_lengths=[12],
            max_child_parent_ratios=[0.5],
            min_child_parent_coverages=[0.5],
            min_gap_between_children_values=[0],
            child_score_thresholds=[0.2],
            set_score_thresholds=[0.2],
            student_weights=[1.0],
            evidence_weights=[0.0],
            active_fraction_weights=[0.0],
            contrast_weights=[0.0],
            length_penalties=[0.0],
            nms_ious=[0.3],
            min_children_per_parents=[2],
            max_children_per_parents=[3],
            max_replaced_parents_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(replaced, [[(1, 4), (11, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
