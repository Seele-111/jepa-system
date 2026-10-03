import unittest

import numpy as np
import torch

from train_jepa_tridet_lite import (
    JEPATriDetLite,
    build_tridet_targets,
    decode_dense_predictions,
    distance_to_distribution_target,
    distribution_logits_to_distance,
)


class JEPATriDetLiteTests(unittest.TestCase):
    def test_distance_to_distribution_target_uses_adjacent_bins(self):
        target = distance_to_distribution_target(2.25, max_bin=5)

        self.assertEqual(tuple(target.shape), (6,))
        self.assertAlmostEqual(float(target.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(target[2]), 0.75, places=6)
        self.assertAlmostEqual(float(target[3]), 0.25, places=6)

    def test_distribution_logits_to_distance_returns_expected_value(self):
        logits = torch.full((1, 6), -10.0)
        logits[0, 4] = 10.0

        distance = distribution_logits_to_distance(logits)

        self.assertAlmostEqual(float(distance.item()), 4.0, places=3)

    def test_build_tridet_targets_marks_event_centers_and_distances(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)

        levels = build_tridet_targets(labels, strides=[1], max_regression_bin=6, center_sampling_radius=2.0)

        self.assertEqual(len(levels), 1)
        level = levels[0]
        np.testing.assert_array_equal(level["center"], np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.float32))
        self.assertAlmostEqual(float(level["left_distribution"][3, 1]), 1.0)
        self.assertAlmostEqual(float(level["right_distribution"][3, 1]), 1.0)

    def test_build_tridet_targets_can_use_sparse_event_centers(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)

        levels = build_tridet_targets(
            labels,
            strides=[1],
            max_regression_bin=6,
            center_target_mode="event_center",
        )

        level = levels[0]
        np.testing.assert_array_equal(level["center"], np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32))
        self.assertAlmostEqual(float(level["left_distribution"][3, 1]), 1.0)
        self.assertAlmostEqual(float(level["right_distribution"][3, 1]), 1.0)

    def test_decode_dense_predictions_uses_boundary_distributions_and_nms(self):
        center_logits = torch.full((1, 8), -6.0)
        center_logits[0, 3] = 6.0
        left_logits = torch.full((1, 8, 7), -8.0)
        right_logits = torch.full((1, 8, 7), -8.0)
        left_logits[0, 3, 1] = 8.0
        right_logits[0, 3, 2] = 8.0
        outputs = [
            {
                "stride": 1,
                "center_logits": center_logits,
                "left_logits": left_logits,
                "right_logits": right_logits,
            }
        ]

        decoded = decode_dense_predictions(outputs, lengths=[8], score_threshold=0.5, nms_iou=0.3)

        self.assertEqual(decoded, [[(2, 5)]])

    def test_model_forward_returns_multiscale_boundary_logits(self):
        model = JEPATriDetLite(in_features=10, hidden=16, strides=(1, 2), max_regression_bin=5, dropout=0.0)
        x = torch.randn(2, 9, 10)
        mask = torch.ones(2, 9, dtype=torch.bool)

        outputs = model(x, mask)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0]["center_logits"].shape), (2, 9))
        self.assertEqual(tuple(outputs[0]["left_logits"].shape), (2, 9, 6))
        self.assertEqual(tuple(outputs[0]["right_logits"].shape), (2, 9, 6))
        self.assertEqual(tuple(outputs[1]["center_logits"].shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
