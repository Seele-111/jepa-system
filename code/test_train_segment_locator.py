import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_segment_locator import (
    TemporalSegmentLocator,
    VideoRecord,
    apply_recall_repair_to_records,
    apply_valley_split_to_records,
    blend_prediction_records,
    boundary_transition_loss,
    evaluate_records,
    event_refine_probabilities,
    gaussian_smooth_probabilities,
    hysteresis_probabilities_to_segments,
    load_signal_dataset,
    make_stratified_folds,
    probabilities_to_segments,
    select_event_refined_postprocess_params,
    select_blended_postprocess_params,
    select_postprocess_params,
    select_recall_repair_params,
    select_valley_split_params,
    soft_dice_loss,
    train_val_split,
)


class TrainSegmentLocatorTests(unittest.TestCase):
    def test_load_signal_dataset_is_binary_only_and_aligns_lengths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            np.savez_compressed(
                root / "signals.npz",
                np.array(
                    [
                        [0.0, 0.1, 0.2],
                        [0.3, 0.4, 0.5],
                        [0.6, 0.7, 0.8],
                    ],
                    dtype=np.float32,
                ),
            )
            np.savez_compressed(root / "labels.npz", np.array([0, 1, 1, 0], dtype=np.int64))

            records = load_signal_dataset(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].signals.shape, (3, 3))
            np.testing.assert_array_equal(records[0].labels, np.array([0, 1, 1]))

    def test_load_signal_dataset_accepts_event_features_with_more_than_three_channels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            np.savez_compressed(root / "signals.npz", np.zeros((4, 7), dtype=np.float32))
            np.savez_compressed(root / "labels.npz", np.array([0, 1, 0, 1], dtype=np.int64))

            records = load_signal_dataset(root)

            self.assertEqual(records[0].signals.shape, (4, 7))

    def test_probabilities_to_segments_merges_short_gaps_and_filters_short_segments(self):
        probs = np.array([0.0, 0.7, 0.8, 0.1, 0.75, 0.0, 0.0, 0.9])

        segments = probabilities_to_segments(
            probs,
            threshold=0.5,
            smooth_window=1,
            min_gap=1,
            min_length=2,
        )

        self.assertEqual(segments, [(1, 4)])

    def test_hysteresis_probabilities_to_segments_requires_high_confidence_seed(self):
        probs = np.array([0.1, 0.45, 0.8, 0.45, 0.1, 0.55, 0.1], dtype=np.float32)

        segments = hysteresis_probabilities_to_segments(
            probs,
            high_threshold=0.7,
            low_threshold=0.4,
            smooth_window=1,
            min_gap=0,
            min_length=1,
        )

        self.assertEqual(segments, [(1, 3)])

    def test_gaussian_smooth_probabilities_preserves_length_and_peak(self):
        probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        smoothed = gaussian_smooth_probabilities(probs, sigma=1.0)

        self.assertEqual(smoothed.shape, probs.shape)
        self.assertEqual(int(np.argmax(smoothed)), 2)
        self.assertTrue(np.all(smoothed >= 0.0))

    def test_event_refine_probabilities_fills_local_event_context(self):
        probs = np.array([0.0, 0.2, 1.0, 0.2, 0.0], dtype=np.float32)

        refined = event_refine_probabilities(probs, sigmas=[1.0], raw_weight=0.5)

        self.assertGreater(float(refined[1]), float(probs[1]))
        self.assertGreater(float(refined[3]), float(probs[3]))
        self.assertGreaterEqual(float(refined[2]), float(refined[1]))

    def test_select_postprocess_params_optimizes_segment_f1(self):
        records = [
            {
                "labels": np.array([0, 1, 1, 0, 0, 1], dtype=np.int64),
                "probs": np.array([0.1, 0.8, 0.7, 0.2, 0.1, 0.9], dtype=np.float32),
            }
        ]

        params, metrics = select_postprocess_params(
            records,
            thresholds=[0.5],
            smooth_windows=[1],
            min_gaps=[0],
            min_lengths=[1],
        )

        self.assertEqual(params["threshold"], 0.5)
        self.assertGreater(metrics["segment"]["f1"], 0.99)
        self.assertGreater(metrics["frame"]["f1"], 0.99)

    def test_evaluate_records_counts_each_ground_truth_segment_once(self):
        records = [
            {
                "labels": np.array([1, 1, 1, 0, 1, 1], dtype=np.int64),
                "probs": np.array([0.9, 0.0, 0.9, 0.0, 0.9, 0.8], dtype=np.float32),
            }
        ]

        metrics = evaluate_records(
            records,
            {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1},
        )

        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 1)
        self.assertEqual(metrics["segment"]["fn"], 0)

    def test_temporal_segment_locator_outputs_one_logit_per_frame(self):
        model = TemporalSegmentLocator(in_channels=3, hidden=8, dropout=0.0)
        x = torch.randn(2, 5, 3)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 5))

    def test_temporal_segment_locator_attention_architecture_outputs_per_frame_logits(self):
        model = TemporalSegmentLocator(in_channels=3, hidden=8, dropout=0.0, architecture="attn_tcn")
        x = torch.randn(2, 5, 3)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 5))

    def test_temporal_segment_locator_bilstm_architecture_outputs_per_frame_logits(self):
        model = TemporalSegmentLocator(in_channels=3, hidden=8, dropout=0.0, architecture="bilstm")
        x = torch.randn(2, 5, 3)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 5))

    def test_temporal_segment_locator_rejects_unknown_architecture(self):
        with self.assertRaises(ValueError):
            TemporalSegmentLocator(in_channels=3, hidden=8, dropout=0.0, architecture="unknown")

    def test_soft_dice_loss_rewards_segment_overlap(self):
        labels = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        good = torch.tensor([[0.05, 0.95, 0.90, 0.05]])
        bad = torch.tensor([[0.90, 0.05, 0.05, 0.90]])

        self.assertLess(float(soft_dice_loss(good, labels)), float(soft_dice_loss(bad, labels)))
        self.assertLess(float(soft_dice_loss(good, labels)), 0.1)

    def test_boundary_transition_loss_rewards_correct_label_changes(self):
        labels = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        sharp = torch.tensor([[0.05, 0.95, 0.90, 0.05]])
        flat = torch.tensor([[0.50, 0.50, 0.50, 0.50]])

        self.assertLess(float(boundary_transition_loss(sharp, labels)), float(boundary_transition_loss(flat, labels)))

    def test_locator_loss_downweights_low_quality_frames(self):
        from train_segment_locator import locator_loss

        logits = torch.tensor([[0.0, -3.0]], dtype=torch.float32)
        labels = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        full_weight = torch.ones_like(labels)
        low_positive_weight = torch.tensor([[1.0, 0.1]], dtype=torch.float32)

        full_loss = locator_loss(logits, labels, pos_weight=1.0, frame_weights=full_weight, lambda_segment=0.0, lambda_smooth=0.0)
        weighted_loss = locator_loss(
            logits,
            labels,
            pos_weight=1.0,
            frame_weights=low_positive_weight,
            lambda_segment=0.0,
            lambda_smooth=0.0,
        )

        self.assertLess(float(weighted_loss), float(full_loss))

    def test_train_val_split_keeps_singleton_class_in_training(self):
        records = [
            VideoRecord("pos_a", np.zeros((4, 3), dtype=np.float32), np.array([1, 1, 0, 0])),
            VideoRecord("pos_b", np.zeros((4, 3), dtype=np.float32), np.array([0, 1, 0, 0])),
            VideoRecord("pos_c", np.zeros((4, 3), dtype=np.float32), np.array([0, 0, 1, 0])),
            VideoRecord("neg", np.zeros((4, 3), dtype=np.float32), np.array([0, 0, 0, 0])),
        ]

        train_idx, val_idx = train_val_split(records, val_ratio=0.25, seed=0)

        self.assertIn(3, train_idx)
        self.assertNotIn(3, val_idx)

    def test_train_val_split_samples_positive_ratio_bins(self):
        records = []
        specs = [
            ("very_low_a", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ("very_low_b", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ("low_mid_a", [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ("low_mid_b", [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            ("mid_a", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            ("mid_b", [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
            ("high_a", [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
            ("high_b", [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
        ]
        for name, labels in specs:
            records.append(VideoRecord(name, np.zeros((10, 3), dtype=np.float32), np.array(labels)))

        _, val_idx = train_val_split(records, val_ratio=0.5, seed=1)
        val_names = {records[idx].name.rsplit("_", 1)[0] for idx in val_idx}

        self.assertEqual(val_names, {"very_low", "low_mid", "mid", "high"})

    def test_make_stratified_folds_covers_each_video_once(self):
        records = []
        for idx, labels in enumerate(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 0, 1],
            ]
        ):
            records.append(VideoRecord(f"clip_{idx}", np.zeros((4, 3), dtype=np.float32), np.array(labels)))

        folds = make_stratified_folds(records, n_folds=3, seed=7)

        self.assertEqual(len(folds), 3)
        validation_items = sorted(idx for fold in folds for idx in fold)
        self.assertEqual(validation_items, list(range(len(records))))
        for fold in folds:
            train = set(range(len(records))) - set(fold)
            self.assertTrue(train.isdisjoint(fold))
            self.assertGreater(len(fold), 0)

    def test_blend_prediction_records_combines_model_and_auxiliary_signal(self):
        records = [
            VideoRecord(
                "clip",
                np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.8]], dtype=np.float32),
                np.array([0, 1], dtype=np.int64),
            )
        ]
        model_outputs = [{"name": "clip", "labels": records[0].labels, "probs": np.array([0.6, 0.4], dtype=np.float32)}]

        blended = blend_prediction_records(model_outputs, records, [0], aux_channel=2, alpha_model=0.75)

        np.testing.assert_allclose(blended[0]["probs"], np.array([0.5, 0.5], dtype=np.float32), atol=1e-6)

    def test_select_blended_postprocess_params_can_choose_auxiliary_blend(self):
        records = [
            VideoRecord(
                "clip",
                np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32),
                np.array([0, 1, 1], dtype=np.int64),
            )
        ]
        model_outputs = [{"name": "clip", "labels": records[0].labels, "probs": np.array([0.9, 0.2, 0.2], dtype=np.float32)}]

        alpha, params, metrics = select_blended_postprocess_params(
            model_outputs,
            records,
            [0],
            aux_channel=2,
            alphas=[1.0, 0.0],
            thresholds=[0.5],
            smooth_windows=[1],
            min_gaps=[0],
            min_lengths=[1],
        )

        self.assertEqual(alpha, 0.0)
        self.assertGreater(metrics["segment"]["f1"], 0.99)
        self.assertEqual(params["threshold"], 0.5)

    def test_select_event_refined_postprocess_params_can_choose_refinement(self):
        model_outputs = [
            {
                "name": "clip",
                "labels": np.array([0, 1, 1, 1, 0], dtype=np.int64),
                "probs": np.array([0.0, 0.25, 1.0, 0.25, 0.0], dtype=np.float32),
            }
        ]

        refine_config, params, metrics = select_event_refined_postprocess_params(
            model_outputs,
            raw_weights=[1.0, 0.0],
            sigma_sets=[[1.0]],
            thresholds=[0.3],
            smooth_windows=[1],
            min_gaps=[0],
            min_lengths=[1],
            low_thresholds=[None],
            iou_threshold=0.7,
        )

        self.assertEqual(refine_config["raw_weight"], 0.0)
        self.assertGreater(metrics["segment"]["f1"], 0.99)
        self.assertEqual(params["threshold"], 0.3)

    def test_recall_repair_adds_uncovered_high_confidence_auxiliary_segment(self):
        records = [
            VideoRecord(
                "clip",
                np.array(
                    [
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.2],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.95],
                        [0.0, 0.0, 0.2],
                        [0.0, 0.0, 0.1],
                    ],
                    dtype=np.float32,
                ),
                np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.array([0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}

        repaired = apply_recall_repair_to_records(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            aux_threshold=0.8,
            min_contrast=0.2,
            rescue_value=1.0,
        )

        self.assertEqual(repaired[0]["rescue_segments"], [(4, 5)])
        self.assertGreaterEqual(float(repaired[0]["probs"][4]), 1.0)
        metrics = evaluate_records(repaired, base_params)
        self.assertEqual(metrics["segment"]["tp"], 1)

    def test_recall_repair_does_not_duplicate_existing_covered_prediction(self):
        records = [
            VideoRecord(
                "clip",
                np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.9], [0.0, 0.0, 0.9], [0.0, 0.0, 0.1]], dtype=np.float32),
                np.array([0, 1, 1, 0], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.array([0.1, 0.8, 0.8, 0.1], dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}

        repaired = apply_recall_repair_to_records(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            aux_threshold=0.8,
            max_existing_coverage=0.5,
        )

        self.assertEqual(repaired[0]["rescue_segments"], [])
        np.testing.assert_allclose(repaired[0]["probs"], model_outputs[0]["probs"], atol=1e-6)

    def test_recall_repair_rejects_low_contrast_auxiliary_plateau(self):
        records = [
            VideoRecord(
                "clip",
                np.full((6, 3), 0.85, dtype=np.float32),
                np.array([0, 0, 1, 1, 0, 0], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.full(6, 0.1, dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}

        repaired = apply_recall_repair_to_records(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            aux_threshold=0.8,
            min_contrast=0.2,
        )

        self.assertEqual(repaired[0]["rescue_segments"], [])
        self.assertTrue(np.all(repaired[0]["probs"] < 0.5))

    def test_select_recall_repair_params_enables_rescue_only_when_segment_f1_improves(self):
        records = [
            VideoRecord(
                "clip",
                np.array(
                    [
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.1],
                    ],
                    dtype=np.float32,
                ),
                np.array([0, 0, 0, 1, 1, 0], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.full(6, 0.1, dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}

        config, repaired, metrics = select_recall_repair_params(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            aux_thresholds=[0.8],
            aux_smooth_windows=[1],
            aux_min_gaps=[0],
            aux_min_lengths=[1],
            pads=[0],
            min_contrasts=[0.2],
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(repaired[0]["rescue_segments"], [(3, 4)])
        self.assertGreater(metrics["segment"]["f1"], 0.99)

    def test_valley_split_separates_merged_prediction_using_jepa_evidence_valley(self):
        records = [
            VideoRecord(
                "clip",
                np.array(
                    [
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.2],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.2],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.1],
                    ],
                    dtype=np.float32,
                ),
                np.array([0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.array([0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}

        split = apply_valley_split_to_records(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            evidence_threshold=0.6,
            parent_min_length=4,
        )

        self.assertEqual(split[0]["valley_split_segments"], [(1, 2), (6, 7)])
        metrics = evaluate_records(split, base_params)
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fn"], 0)

    def test_valley_split_leaves_single_event_prediction_unchanged(self):
        records = [
            VideoRecord(
                "clip",
                np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.9], [0.0, 0.0, 0.9], [0.0, 0.0, 0.8]], dtype=np.float32),
                np.array([0, 1, 1, 1], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.array([0.1, 0.7, 0.7, 0.7], dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}

        split = apply_valley_split_to_records(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            evidence_threshold=0.6,
            parent_min_length=2,
        )

        self.assertEqual(split[0]["valley_split_segments"], [])
        np.testing.assert_allclose(split[0]["probs"], model_outputs[0]["probs"], atol=1e-6)

    def test_select_valley_split_params_enables_split_only_when_segment_f1_improves(self):
        records = [
            VideoRecord(
                "clip",
                np.array(
                    [
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.8],
                        [0.0, 0.0, 0.8],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.9],
                    ],
                    dtype=np.float32,
                ),
                np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.array([0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}

        config, split, metrics = select_valley_split_params(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            evidence_thresholds=[0.6],
            evidence_smooth_windows=[1],
            island_min_gaps=[0],
            island_min_lengths=[1],
            pads=[0],
            parent_min_lengths=[4],
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(split[0]["valley_split_segments"], [(1, 2), (5, 6)])
        self.assertGreater(metrics["segment"]["f1"], 0.99)

    def test_select_valley_split_params_reselects_postprocess_after_splitting(self):
        records = [
            VideoRecord(
                "clip",
                np.array(
                    [
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.9],
                        [0.0, 0.0, 0.1],
                    ],
                    dtype=np.float32,
                ),
                np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64),
            )
        ]
        model_outputs = [
            {"name": "clip", "labels": records[0].labels, "probs": np.array([0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], dtype=np.float32)}
        ]
        base_params = {"threshold": 0.5, "smooth_window": 5, "min_gap": 0, "min_length": 1}

        config, _, metrics = select_valley_split_params(
            model_outputs,
            records,
            [0],
            base_params=base_params,
            aux_channels=[2],
            evidence_thresholds=[0.6],
            evidence_smooth_windows=[1],
            island_min_gaps=[0],
            island_min_lengths=[1],
            pads=[0],
            parent_min_lengths=[4],
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(config["params"]["smooth_window"], 1)
        self.assertGreater(metrics["segment"]["f1"], 0.99)


if __name__ == "__main__":
    unittest.main()
