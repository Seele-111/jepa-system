import unittest

import numpy as np

from jepa_fn_aware_reranker import (
    apply_fn_aware_reranker_config,
    build_fn_acceptance_candidate_records,
    build_fn_aware_candidate_records,
    extract_fn_aware_features,
    label_fn_acceptance_oracle,
    label_fn_aware_candidates,
    score_fn_acceptance_candidates,
    select_fn_aware_reranker_predictions,
    select_fn_aware_reranker_params,
    select_strict_fn_aware_reranker_params,
    split_indices_for_inner_calibration,
)
from train_segment_locator import VideoRecord


class JEPAFNAwareRerankerTests(unittest.TestCase):
    def test_label_fn_aware_candidates_targets_current_base_misses(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
        candidates = [(1, 2), (5, 6), (0, 0)]

        y, best_iou = label_fn_aware_candidates(
            candidates,
            labels,
            base_segments=[(1, 2)],
            iou_threshold=0.3,
        )

        np.testing.assert_array_equal(y, np.array([0, 1, 0], dtype=np.int64))
        self.assertGreater(float(best_iou[1]), 0.99)

    def test_extract_fn_aware_features_includes_rank_and_base_context(self):
        record = VideoRecord(
            name="toy",
            signals=np.array(
                [
                    [0.1],
                    [0.9],
                    [0.8],
                    [0.1],
                    [0.7],
                    [0.6],
                ],
                dtype=np.float32,
            ),
            labels=np.zeros(6, dtype=np.int64),
        )
        candidates = [(1, 2), (4, 5)]
        scores = np.array([0.4, 0.8], dtype=np.float32)

        features = extract_fn_aware_features(
            record,
            segment=(1, 2),
            candidates=candidates,
            scores=scores,
            candidate_rank=2,
            channel_indices=[0],
            base_segments=[(4, 5)],
        )

        self.assertTrue(np.all(np.isfinite(features)))
        self.assertGreater(features.shape[0], 0)
        self.assertAlmostEqual(float(features[-22]), 0.4, places=5)
        self.assertAlmostEqual(float(features[-21]), 1.0, places=5)

    def test_extract_fn_aware_features_adds_strata_evidence_context(self):
        record = VideoRecord(
            name="toy",
            signals=np.array(
                [
                    [0.05],
                    [0.9],
                    [0.85],
                    [0.05],
                    [0.1],
                    [0.88],
                    [0.86],
                    [0.05],
                    [0.1],
                ],
                dtype=np.float32,
            ),
            labels=np.zeros(9, dtype=np.int64),
        )
        candidates = [(1, 2), (5, 6), (5, 7), (0, 8)]
        scores = np.array([0.95, 0.65, 0.5, 0.9], dtype=np.float32)

        features = extract_fn_aware_features(
            record,
            segment=(5, 6),
            candidates=candidates,
            scores=scores,
            candidate_rank=3,
            channel_indices=[0],
            base_segments=[(1, 2)],
        )

        self.assertAlmostEqual(float(features[-22]), 0.65, places=5)
        self.assertLess(float(features[-18]), 0.01)
        self.assertGreaterEqual(float(features[-13]), 2.0)
        self.assertGreaterEqual(float(features[-10]), 1.0)
        self.assertGreater(float(features[-2]), 0.8)
        self.assertGreater(float(features[-1]), 1.5)

    def test_build_fn_aware_candidate_records_uses_current_base_segments(self):
        record = VideoRecord(
            name="toy",
            signals=np.array([[0.1], [0.9], [0.9], [0.1], [0.9], [0.9]], dtype=np.float32),
            labels=np.array([0, 1, 1, 0, 1, 1], dtype=np.int64),
        )

        rows = build_fn_aware_candidate_records(
            records=[record],
            indices=[0],
            candidate_predictions_by_idx={0: [(1, 2), (4, 5)]},
            candidate_scores_by_idx={0: np.array([0.95, 0.05], dtype=np.float32)},
            base_segments_by_idx={0: [(1, 2)]},
            feature_names=["jepa"],
            channel_names=["jepa"],
            iou_threshold=0.3,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual([row.label for row in rows], [0, 1])

    def test_build_fn_aware_candidate_records_prefilter_keeps_low_score_positive(self):
        record = VideoRecord(
            name="toy",
            signals=np.array(
                [[0.1], [0.9], [0.9], [0.2], [0.3], [0.4], [0.2], [0.1], [0.9], [0.9]],
                dtype=np.float32,
            ),
            labels=np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64),
        )

        rows = build_fn_aware_candidate_records(
            records=[record],
            indices=[0],
            candidate_predictions_by_idx={0: [(1, 2), (3, 3), (4, 4), (5, 5), (8, 9)]},
            candidate_scores_by_idx={0: np.array([0.95, 0.9, 0.8, 0.7, 0.01], dtype=np.float32)},
            base_segments_by_idx={0: [(1, 2)]},
            feature_names=["jepa"],
            channel_names=["jepa"],
            iou_threshold=0.3,
            max_records_per_video=2,
        )

        self.assertEqual(len(rows), 3)
        self.assertIn((8, 9), [row.segment for row in rows])
        self.assertEqual(sum(row.label for row in rows), 1)

    def test_select_fn_aware_reranker_params_adds_helpful_low_selector_candidate(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6), (8, 8)]]
        scores = [np.array([0.9, 0.05], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)]

        config, metrics, fused = select_fn_aware_reranker_params(
            base,
            candidates,
            scores,
            labels,
            thresholds=[0.5],
            max_base_ious=[0.0],
            nms_ious=[0.3],
            length_penalties=[0.0],
            max_rescues_per_videos=[1],
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(fused, [[(1, 2), (5, 6)]])
        self.assertEqual(metrics["segment"]["tp"], 2)

    def test_select_fn_aware_reranker_params_respects_fp_budget(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6), (8, 8)]]
        scores = [np.array([0.9, 0.95], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)]

        config, metrics, fused = select_fn_aware_reranker_params(
            base,
            candidates,
            scores,
            labels,
            thresholds=[0.5],
            max_base_ious=[None],
            nms_ious=[None],
            length_penalties=[0.0],
            max_rescues_per_videos=[2],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(fused, [[(1, 2), (5, 6)]])
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_label_fn_acceptance_oracle_marks_budget_accepted_rescue_only(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6), (8, 8)]]
        scores = [np.array([0.9, 0.95], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)]

        accepted = label_fn_acceptance_oracle(
            base,
            candidates,
            scores,
            labels,
            threshold=0.5,
            max_base_iou=None,
            nms_iou=None,
            length_penalty=0.0,
            max_rescues_per_video=2,
            max_fp_increase=0,
            max_candidates_per_video=None,
            iou_threshold=0.3,
        )

        np.testing.assert_array_equal(accepted[0], np.array([1, 0], dtype=np.int64))

    def test_build_fn_acceptance_candidate_records_appends_first_stage_context(self):
        record = VideoRecord(
            name="toy",
            signals=np.array([[0.1], [0.9], [0.9], [0.1], [0.9], [0.9], [0.1]], dtype=np.float32),
            labels=np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.int64),
        )
        rows = build_fn_acceptance_candidate_records(
            records=[record],
            indices=[0],
            candidate_predictions_by_idx={0: [(1, 2), (4, 5)]},
            raw_scores_by_idx={0: np.array([0.2, 0.7], dtype=np.float32)},
            first_stage_scores_by_idx={0: np.array([0.1, 0.9], dtype=np.float32)},
            base_segments_by_idx={0: [(1, 2)]},
            acceptance_labels_by_idx={0: np.array([0, 1], dtype=np.int64)},
            feature_names=["jepa"],
            channel_names=["jepa"],
            max_records_per_video=2,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual([row.label for row in rows], [0, 1])
        self.assertAlmostEqual(float(rows[1].features[-3]), 0.9, places=5)
        self.assertAlmostEqual(float(rows[1].features[-2]), 0.5, places=5)

    def test_build_fn_acceptance_candidate_records_can_use_auxiliary_rescue_positives(self):
        record = VideoRecord(
            name="toy",
            signals=np.array([[0.1], [0.9], [0.9], [0.1], [0.9], [0.9], [0.1]], dtype=np.float32),
            labels=np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.int64),
        )
        rows = build_fn_acceptance_candidate_records(
            records=[record],
            indices=[0],
            candidate_predictions_by_idx={0: [(1, 2), (4, 5)]},
            raw_scores_by_idx={0: np.array([0.2, 0.7], dtype=np.float32)},
            first_stage_scores_by_idx={0: np.array([0.1, 0.9], dtype=np.float32)},
            base_segments_by_idx={0: [(1, 2)]},
            acceptance_labels_by_idx={0: np.array([0, 0], dtype=np.int64)},
            aux_positive_labels_by_idx={0: np.array([0, 1], dtype=np.int64)},
            feature_names=["jepa"],
            channel_names=["jepa"],
            max_records_per_video=2,
        )

        self.assertEqual([row.label for row in rows], [0, 1])

    def test_score_fn_acceptance_candidates_returns_scores_aligned_to_original_candidates(self):
        class FirstStageEchoModel:
            def predict_proba(self, x):
                p = np.asarray(x[:, -3], dtype=np.float32)
                return np.stack([1.0 - p, p], axis=1)

        record = VideoRecord(
            name="toy",
            signals=np.array([[0.1], [0.9], [0.9], [0.1], [0.9], [0.9], [0.1]], dtype=np.float32),
            labels=np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.int64),
        )

        scores = score_fn_acceptance_candidates(
            FirstStageEchoModel(),
            records=[record],
            indices=[0],
            candidate_predictions_by_idx={0: [(1, 2), (4, 5)]},
            raw_scores_by_idx={0: np.array([0.2, 0.7], dtype=np.float32)},
            first_stage_scores_by_idx={0: np.array([0.25, 0.75], dtype=np.float32)},
            base_segments_by_idx={0: [(1, 2)]},
            feature_names=["jepa"],
            channel_names=["jepa"],
        )

        self.assertEqual(len(scores), 1)
        np.testing.assert_allclose(scores[0], np.array([0.25, 0.75], dtype=np.float32), atol=1e-6)

    def test_budgeted_selection_prefilters_top_candidates_per_video(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6), (8, 8), (0, 0)]]
        scores = [np.array([0.8, 0.95, 0.1], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)]

        config, metrics, fused = select_fn_aware_reranker_params(
            base,
            candidates,
            scores,
            labels,
            thresholds=[0.5],
            max_base_ious=[None],
            nms_ious=[None],
            length_penalties=[0.0],
            max_rescues_per_videos=[2],
            max_fp_increase=0,
            max_candidates_per_video=1,
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(fused, [[(1, 2)]])
        self.assertEqual(metrics["segment"]["tp"], 1)
        self.assertEqual(config["diagnostics"]["max_candidates_per_video"], 1)

    def test_non_budgeted_predictions_prefilter_top_candidates_per_video(self):
        fused = select_fn_aware_reranker_predictions(
            base_predictions=[[(1, 2)]],
            candidate_predictions=[[(5, 6), (8, 8)]],
            rescue_scores=[np.array([0.8, 0.95], dtype=np.float32)],
            threshold=0.5,
            max_base_iou=None,
            nms_iou=None,
            length_penalty=0.0,
            max_rescues_per_video=2,
            max_candidates_per_video=1,
        )

        self.assertEqual(fused, [[(1, 2), (8, 8)]])

    def test_fn_aware_predictions_can_allow_overlapping_base_candidate(self):
        fused = select_fn_aware_reranker_predictions(
            base_predictions=[[(0, 10)]],
            candidate_predictions=[[(5, 8)]],
            rescue_scores=[np.array([0.9], dtype=np.float32)],
            threshold=0.5,
            max_base_iou=None,
            nms_iou=None,
            length_penalty=0.0,
            max_rescues_per_video=1,
        )

        self.assertEqual(fused, [[(0, 10), (5, 8)]])

    def test_strict_param_selection_filters_fp_budget_without_label_aware_candidate_picking(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6), (8, 8)]]
        scores = [np.array([0.8, 0.95], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)]

        config, metrics, fused = select_strict_fn_aware_reranker_params(
            base,
            candidates,
            scores,
            labels,
            thresholds=[0.5],
            max_base_ious=[None],
            nms_ious=[None],
            length_penalties=[0.0],
            max_rescues_per_videos=[2],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(fused, [[(1, 2)]])
        self.assertEqual(metrics["segment"]["fp"], 0)
        self.assertEqual(config["diagnostics"]["fp_rejected_configs"], 1)

    def test_strict_param_selection_enables_label_free_config_within_fp_budget(self):
        base = [[(1, 2)]]
        candidates = [[(5, 6)]]
        scores = [np.array([0.8], dtype=np.float32)]
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)]

        config, metrics, fused = select_strict_fn_aware_reranker_params(
            base,
            candidates,
            scores,
            labels,
            thresholds=[0.5],
            max_base_ious=[None],
            nms_ious=[None],
            length_penalties=[0.0],
            max_rescues_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(fused, [[(1, 2), (5, 6)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_apply_fn_aware_reranker_config_uses_frozen_config_without_labels(self):
        config = {
            "enabled": True,
            "threshold": 0.5,
            "max_base_iou": None,
            "nms_iou": None,
            "length_penalty": 0.0,
            "max_rescues_per_video": 1,
            "max_candidates_per_video": None,
        }

        fused = apply_fn_aware_reranker_config(
            base_predictions=[[(1, 2)]],
            candidate_predictions=[[(5, 6), (8, 8)]],
            rescue_scores=[np.array([0.9, 0.8], dtype=np.float32)],
            config=config,
        )

        self.assertEqual(fused, [[(1, 2), (5, 6)]])

    def test_split_indices_for_inner_calibration_is_deterministic_and_stratified(self):
        labels_by_idx = {
            0: np.array([0, 0], dtype=np.int64),
            1: np.array([1, 0], dtype=np.int64),
            2: np.array([0, 0], dtype=np.int64),
            3: np.array([1, 1], dtype=np.int64),
            4: np.array([0, 0], dtype=np.int64),
            5: np.array([1, 0], dtype=np.int64),
        }

        inner_a, calib_a = split_indices_for_inner_calibration(
            [0, 1, 2, 3, 4, 5],
            labels_by_idx,
            calibration_fraction=0.34,
            seed=123,
        )
        inner_b, calib_b = split_indices_for_inner_calibration(
            [0, 1, 2, 3, 4, 5],
            labels_by_idx,
            calibration_fraction=0.34,
            seed=123,
        )

        self.assertEqual(inner_a, inner_b)
        self.assertEqual(calib_a, calib_b)
        self.assertEqual(set(inner_a) | set(calib_a), {0, 1, 2, 3, 4, 5})
        self.assertTrue(any(labels_by_idx[idx].sum() > 0 for idx in calib_a))
        self.assertTrue(any(labels_by_idx[idx].sum() == 0 for idx in calib_a))


if __name__ == "__main__":
    unittest.main()
