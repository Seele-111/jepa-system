import unittest

import torch

from true_vjepa_scorer import TrueVJEPAScoreConfig, TrueVJEPAScorer


class ZeroPredictor(torch.nn.Module):
    def forward(self, context_tokens, context_masks, target_masks):
        batch = context_tokens.shape[0]
        n_target = target_masks[0].shape[1]
        dim = context_tokens.shape[-1]
        return torch.zeros(batch, n_target, dim)


class RecordingAdapter:
    def __init__(self, total_tokens, dim):
        self.total_tokens = total_tokens
        self.dim = dim
        self.context_calls = 0
        self.target_calls = 0

    def encode_target(self, video):
        self.target_calls += 1
        return torch.ones(1, self.total_tokens, self.dim)

    def encode_context(self, video, context_masks):
        self.context_calls += 1
        n_context = context_masks[0].shape[1]
        return torch.ones(1, n_context, self.dim)


class TrueVJEPAScorerTests(unittest.TestCase):
    def test_score_tokens_returns_one_score_per_frame(self):
        num_frames = 3
        grid_size = 2
        total_tokens = num_frames * grid_size * grid_size
        dim = 4
        adapter = RecordingAdapter(total_tokens=total_tokens, dim=dim)
        scorer = TrueVJEPAScorer(
            predictor=ZeroPredictor(),
            config=TrueVJEPAScoreConfig(grid_size=grid_size, target_height=1, target_width=1),
        )

        scores = scorer.score_video_tokens(adapter=adapter, video=torch.zeros(1), num_frames=num_frames)

        self.assertEqual(tuple(scores.shape), (num_frames,))
        self.assertTrue(torch.all(scores >= 0))
        self.assertEqual(adapter.target_calls, 1)
        self.assertEqual(adapter.context_calls, num_frames)


if __name__ == "__main__":
    unittest.main()
