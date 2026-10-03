import unittest

import numpy as np
import torch

from train_jepa_temporal_set_predictor import (
    JEPATemporalSetPredictor,
    decode_set_predictions,
    labels_to_normalized_segments,
    set_prediction_loss,
)


class JEPATemporalSetPredictorTests(unittest.TestCase):
    def test_labels_to_normalized_segments_converts_binary_events(self):
        labels = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int64)

        segments = labels_to_normalized_segments(labels)

        self.assertEqual(tuple(segments.shape), (2, 2))
        np.testing.assert_allclose(segments[0], np.array([0.25, 0.25], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(segments[1], np.array([0.6875, 0.375], dtype=np.float32), atol=1e-6)

    def test_model_forward_shapes(self):
        model = JEPATemporalSetPredictor(in_features=6, hidden=16, num_queries=4, num_layers=1, num_heads=4)
        x = torch.randn(2, 7, 6)
        mask = torch.ones(2, 7, dtype=torch.bool)

        out = model(x, mask)

        self.assertEqual(tuple(out["logits"].shape), (2, 4))
        self.assertEqual(tuple(out["segments"].shape), (2, 4, 2))
        self.assertEqual(tuple(out["frame_logits"].shape), (2, 7))
        self.assertTrue(torch.all(out["segments"] >= 0.0))
        self.assertTrue(torch.all(out["segments"] <= 1.0))

    def test_loss_prefers_prediction_that_matches_gt_segment(self):
        good = {
            "logits": torch.tensor([[5.0, -5.0]], dtype=torch.float32),
            "segments": torch.tensor([[[0.5, 0.25], [0.1, 0.1]]], dtype=torch.float32),
        }
        bad = {
            "logits": torch.tensor([[5.0, -5.0]], dtype=torch.float32),
            "segments": torch.tensor([[[0.1, 0.1], [0.9, 0.1]]], dtype=torch.float32),
        }
        targets = [torch.tensor([[0.5, 0.25]], dtype=torch.float32)]

        good_loss, _ = set_prediction_loss(good, targets)
        bad_loss, _ = set_prediction_loss(bad, targets)

        self.assertLess(float(good_loss), float(bad_loss))

    def test_decode_set_predictions_returns_valid_nms_segments(self):
        outputs = {
            "logits": torch.tensor([[3.0, 2.0, -2.0]], dtype=torch.float32),
            "segments": torch.tensor([[[0.5, 0.4], [0.52, 0.38], [0.1, 0.1]]], dtype=torch.float32),
        }

        decoded = decode_set_predictions(outputs, lengths=[20], score_threshold=0.5, nms_iou=0.3)

        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(decoded[0]), 1)
        start, end = decoded[0][0]
        self.assertLessEqual(0, start)
        self.assertLessEqual(start, end)
        self.assertLess(end, 20)

    def test_decode_can_fuse_frame_head_segments(self):
        outputs = {
            "logits": torch.tensor([[-3.0, -3.0]], dtype=torch.float32),
            "segments": torch.tensor([[[0.2, 0.1], [0.8, 0.1]]], dtype=torch.float32),
            "frame_logits": torch.tensor([[-5.0, -5.0, 4.0, 4.0, 4.0, -5.0]], dtype=torch.float32),
        }

        decoded = decode_set_predictions(
            outputs,
            lengths=[6],
            score_threshold=0.5,
            frame_threshold=0.5,
            frame_min_length=2,
            nms_iou=0.3,
        )

        self.assertEqual(decoded, [[(2, 4)]])


if __name__ == "__main__":
    unittest.main()
