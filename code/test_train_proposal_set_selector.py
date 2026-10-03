import unittest

import numpy as np

from train_proposal_set_selector import (
    candidate_quality_targets,
    evaluate_segment_predictions,
    extract_event_topology_features,
    fit_classifier,
    fit_quality_model,
    generate_multichannel_candidates,
    select_weighted_proposal_set,
)


class ProposalSetSelectorTests(unittest.TestCase):
    def test_generate_multichannel_candidates_uses_complementary_jepa_channels(self):
        signals = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1],
                [0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.1],
            ],
            dtype=np.float32,
        )

        candidates = generate_multichannel_candidates(
            signals,
            feature_names=["vjepa", "ijepa", "dual"],
            channel_names=["vjepa", "ijepa"],
            thresholds=[0.5],
            min_gaps=[0],
            min_lengths=[1],
        )

        self.assertIn((1, 2), candidates)
        self.assertIn((5, 6), candidates)

    def test_select_weighted_proposal_set_prefers_two_events_over_one_merged_event(self):
        candidates = [(1, 6), (1, 2), (5, 6)]
        scores = [0.85, 0.7, 0.7]

        selected = select_weighted_proposal_set(candidates, scores, overlap_iou=0.0, length_penalty=0.0)

        self.assertEqual(selected, [(1, 2), (5, 6)])

    def test_evaluate_segment_predictions_counts_split_events(self):
        labels = [np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)]
        predictions = [[(1, 2), (5, 6)]]

        metrics = evaluate_segment_predictions(predictions, labels, iou_threshold=0.3)

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fn"], 0)

    def test_candidate_quality_targets_keep_iou_order(self):
        best_iou = np.array([0.25, 1.0, 0.8], dtype=np.float32)

        targets = candidate_quality_targets(best_iou)

        self.assertLess(float(targets[0]), float(targets[1]))
        self.assertLessEqual(float(targets[2]), 1.0)

    def test_event_topology_features_distinguish_compact_from_merged_evidence(self):
        compact = extract_event_topology_features(np.array([0.1, 0.8, 0.9, 0.8, 0.1], dtype=np.float32))
        merged = extract_event_topology_features(np.array([0.8, 0.9, 0.1, 0.1, 0.85, 0.9], dtype=np.float32))

        self.assertEqual(compact["island_count"], 1)
        self.assertEqual(merged["island_count"], 2)
        self.assertGreater(merged["valley_depth"], compact["valley_depth"])
        self.assertGreater(compact["compactness"], merged["compactness"])

    def test_fit_quality_model_scores_tight_candidate_above_merged_candidate(self):
        x = np.array(
            [
                [6.0, 0.55],
                [2.0, 0.90],
                [2.0, 0.85],
                [5.0, 0.20],
                [1.0, 0.10],
            ],
            dtype=np.float32,
        )
        best_iou = np.array([0.33, 1.0, 0.9, 0.0, 0.0], dtype=np.float32)

        model = fit_quality_model(x, best_iou, seed=0, model_name="logreg")
        scores = model.predict_proba(x)[:, 1]

        self.assertGreater(float(scores[1]), float(scores[0]))
        self.assertGreater(float(scores[2]), float(scores[3]))

    def test_fit_classifier_mlp_exposes_predict_proba(self):
        x = np.array(
            [
                [0.0, 0.0, 0.1],
                [0.2, 0.1, 0.0],
                [1.0, 0.9, 0.8],
                [0.8, 1.0, 0.9],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1], dtype=np.int64)

        model = fit_classifier(x, y, seed=0, model_name="mlp", device="cpu", epochs=20)
        probs = model.predict_proba(x)

        self.assertEqual(probs.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(probs)))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
