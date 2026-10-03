import unittest

import torch

from cross_validate_rgb_actionformer import RGBActionFormer, auxiliary_targets


class RGBActionFormerTests(unittest.TestCase):
    def test_model_preserves_timeline_and_masks_padding(self):
        model = RGBActionFormer(in_features=12, hidden=16, heads=4, window=5, dropout=0.0)
        x = torch.randn(2, 17, 12)
        valid = torch.ones(2, 17, dtype=torch.bool)
        valid[1, 13:] = False
        logits = model(x, valid)
        self.assertEqual(tuple(logits.shape), (2, 17))
        self.assertTrue(torch.isfinite(logits).all())

    def test_auxiliary_heads_and_targets(self):
        model = RGBActionFormer(in_features=12, hidden=16, heads=4, window=5, dropout=0.0)
        x = torch.randn(3, 8, 12)
        valid = torch.ones(3, 8, dtype=torch.bool)
        valid[2, 6:] = False
        labels = torch.tensor(
            [[0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        logits, occupancy, event_count_logits = model(x, valid, return_aux=True)
        target_occupancy, target_event_count = auxiliary_targets(labels, valid)
        self.assertEqual(tuple(logits.shape), (3, 8))
        self.assertEqual(tuple(occupancy.shape), (3,))
        self.assertEqual(tuple(event_count_logits.shape), (3, 3))
        self.assertTrue(torch.isfinite(occupancy).all())
        self.assertTrue(torch.isfinite(event_count_logits).all())
        self.assertTrue(torch.allclose(target_occupancy, torch.tensor([0.25, 1.0, 1.0 / 3.0])))
        self.assertEqual(target_event_count.tolist(), [1, 1, 2])
        loss = logits[valid].square().mean() + occupancy.mean() + event_count_logits.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))


if __name__ == "__main__":
    unittest.main()
