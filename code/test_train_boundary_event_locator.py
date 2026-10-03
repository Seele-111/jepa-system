import unittest

import numpy as np
import torch

from train_boundary_event_locator import (
    TemporalBoundaryLocator,
    build_boundary_targets,
    refine_segments_with_boundaries,
)


class BoundaryEventLocatorTests(unittest.TestCase):
    def test_build_boundary_targets_marks_object_and_soft_boundaries(self):
        labels = np.array([0, 1, 1, 1, 0, 0], dtype=np.int64)

        targets = build_boundary_targets(labels, boundary_radius=1)

        np.testing.assert_array_equal(targets["object"], np.array([0, 1, 1, 1, 0, 0], dtype=np.float32))
        self.assertAlmostEqual(float(targets["start"][1]), 1.0)
        self.assertAlmostEqual(float(targets["end"][3]), 1.0)
        self.assertGreater(float(targets["start"][0]), 0.0)
        self.assertGreater(float(targets["end"][4]), 0.0)

    def test_refine_segments_with_boundaries_uses_boundary_peaks(self):
        coarse = [(2, 8)]
        start_probs = np.zeros(12, dtype=np.float32)
        end_probs = np.zeros(12, dtype=np.float32)
        start_probs[1] = 0.9
        end_probs[9] = 0.8

        refined = refine_segments_with_boundaries(coarse, start_probs, end_probs, radius=2)

        self.assertEqual(refined, [(1, 9)])

    def test_temporal_boundary_locator_outputs_three_logits_per_frame(self):
        model = TemporalBoundaryLocator(in_channels=7, hidden=8, dropout=0.0)
        x = torch.randn(2, 5, 7)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 5, 3))


if __name__ == "__main__":
    unittest.main()
