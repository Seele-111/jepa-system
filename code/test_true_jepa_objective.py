import unittest

import torch

from true_jepa_objective import (
    MaskPair,
    block_target_context_masks,
    compute_jepa_prediction_errors,
)


class RecordingPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, context_tokens, context_masks, target_masks):
        self.called = True
        batch = context_tokens.shape[0]
        n_target = target_masks[0].shape[1]
        dim = context_tokens.shape[-1]
        return torch.zeros(batch, n_target, dim)


class TrueJepaObjectiveTests(unittest.TestCase):
    def test_compute_errors_calls_predictor_and_returns_per_target_loss(self):
        target_tokens = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [3.0, 0.0],
                    [0.0, 4.0],
                ]
            ]
        )
        context_tokens = target_tokens[:, :2, :]
        masks = MaskPair(
            context=[torch.tensor([[0, 1]])],
            target=[torch.tensor([[2, 3]])],
        )
        predictor = RecordingPredictor()

        errors = compute_jepa_prediction_errors(
            predictor=predictor,
            context_tokens=context_tokens,
            target_tokens=target_tokens,
            masks=masks,
        )

        self.assertTrue(predictor.called)
        self.assertEqual(tuple(errors.shape), (1, 2))
        self.assertTrue(torch.all(errors > 0))

    def test_block_mask_policy_keeps_context_and_target_disjoint(self):
        masks = block_target_context_masks(
            num_frames=4,
            grid_size=3,
            target_frame=1,
            target_top=0,
            target_left=1,
            target_height=2,
            target_width=2,
        )

        context = set(masks.context[0][0].tolist())
        target = set(masks.target[0][0].tolist())
        self.assertEqual(len(target), 4)
        self.assertTrue(context.isdisjoint(target))
        self.assertEqual(len(context) + len(target), 4 * 3 * 3)


if __name__ == "__main__":
    unittest.main()
