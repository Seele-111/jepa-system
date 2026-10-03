import unittest

import numpy as np

from train_proposal_rescuer import (
    extract_rescue_features,
    fit_rescuer,
    label_rescue_candidates,
    matched_ground_truth_indices,
    select_rescuer_params,
)
from train_segment_locator import VideoRecord
from jepa_prototype_rescue import PrototypeRescuer


class ProposalRescuerTests(unittest.TestCase):
    def test_matched_ground_truth_indices_tracks_mainline_coverage(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)

        matched = matched_ground_truth_indices([(1, 2)], labels, iou_threshold=0.3)

        self.assertEqual(matched, {0})

    def test_label_rescue_candidates_only_marks_missed_ground_truth_positive(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
        candidates = [(1, 2), (5, 6), (0, 0)]

        y, best_iou = label_rescue_candidates(
            candidates,
            labels,
            mainline_segments=[(1, 2)],
            iou_threshold=0.3,
        )

        np.testing.assert_array_equal(y, np.array([0, 1, 0], dtype=np.int64))
        self.assertGreater(float(best_iou[1]), 0.99)

    def test_extract_rescue_features_includes_selector_and_mainline_context(self):
        signals = np.array(
            [
                [0.1, 0.2],
                [0.8, 0.7],
                [0.9, 0.8],
                [0.1, 0.2],
            ],
            dtype=np.float32,
        )
        mainline_probs = np.array([0.9, 0.1, 0.1, 0.9], dtype=np.float32)

        features = extract_rescue_features(
            signals,
            segment=(1, 2),
            channel_indices=[0, 1],
            selector_score=0.75,
            mainline_probs=mainline_probs,
            mainline_segments=[(0, 0)],
        )

        self.assertTrue(np.all(np.isfinite(features)))
        self.assertGreater(features.shape[0], 0)
        self.assertAlmostEqual(float(features[-9]), 0.75, places=5)
        self.assertLess(float(features[-7]), 0.5)

    def test_fit_rescuer_mlp_exposes_predict_proba(self):
        x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [1.0, 1.0],
                [1.2, 0.9],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1], dtype=np.int64)

        model = fit_rescuer(x, y, seed=0, model_name="mlp", device="cpu", epochs=20)
        probs = model.predict_proba(x)

        self.assertEqual(probs.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(probs)))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-5)

    def test_fit_rescuer_prototype_uses_prototype_memory(self):
        x = np.array(
            [
                [1.0, 0.9, 0.8],
                [0.9, 1.0, 0.8],
                [-1.0, -0.9, -0.8],
                [-0.9, -1.0, -0.8],
            ],
            dtype=np.float32,
        )
        y = np.array([1, 1, 0, 0], dtype=np.int64)

        model = fit_rescuer(x, y, seed=0, model_name="prototype")
        probs = model.predict_proba(x)

        self.assertIsInstance(model, PrototypeRescuer)
        self.assertEqual(probs.shape, (4, 2))
        self.assertGreater(float(probs[0, 1]), float(probs[2, 1]))

    def test_select_rescuer_params_uses_active_base_segments(self):
        class AlwaysHigh:
            def predict_proba(self, x):
                return np.tile(np.array([[0.1, 0.9]], dtype=np.float32), (len(x), 1))

        record = VideoRecord(
            name="toy",
            signals=np.array(
                [
                    [0.1],
                    [0.9],
                    [0.9],
                    [0.1],
                    [0.1],
                    [0.9],
                    [0.9],
                    [0.1],
                ],
                dtype=np.float32,
            ),
            labels=np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64),
        )

        config, metrics, predictions = select_rescuer_params(
            selector=AlwaysHigh(),
            rescuer=AlwaysHigh(),
            records=[record],
            val_idx=[0],
            feature_names=["jepa"],
            channel_names=["jepa"],
            thresholds=[0.5],
            min_gaps=[0],
            min_lengths=[1],
            mainline_probs_by_idx={0: np.zeros(8, dtype=np.float32)},
            mainline_segments_by_idx={0: [(1, 2)]},
            iou_threshold=0.3,
            active_base_segments_by_idx={0: [(1, 2), (5, 6)]},
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(predictions, [[(1, 2), (5, 6)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
