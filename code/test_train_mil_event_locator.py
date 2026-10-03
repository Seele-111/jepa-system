import unittest

import numpy as np
import torch

from train_mil_event_locator import build_soft_frame_targets, segment_topk_mil_loss


class MILEventLocatorTests(unittest.TestCase):
    def test_build_soft_frame_targets_downweights_uncertain_boundaries(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.int64)

        target, weight = build_soft_frame_targets(labels, margin=1)

        np.testing.assert_allclose(target, np.array([0.25, 1.0, 1.0, 1.0, 0.25], dtype=np.float32))
        self.assertLess(float(weight[1]), float(weight[2]))
        self.assertLess(float(weight[0]), 1.0)

    def test_segment_topk_mil_loss_is_low_for_good_positive_and_negative_segments(self):
        logits = torch.tensor([[[-3.0, 3.0, 4.0, 3.0, -2.0]]])
        labels = torch.tensor([[0, 1, 1, 1, 0]])

        loss = segment_topk_mil_loss(logits.squeeze(0), labels.squeeze(0), topk_fraction=0.5)

        self.assertLess(float(loss), 0.2)


if __name__ == "__main__":
    unittest.main()
