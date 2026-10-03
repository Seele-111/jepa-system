#!/usr/bin/env python3
"""
True V-JEPA scorer scaffold.

This module is the integration boundary for top-conference-grade experiments:
it must call a JEPA predictor with masks and compare against target encoder
features. The testable JEPA objective core lives in true_jepa_objective.py.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from true_jepa_objective import block_target_context_masks, compute_jepa_prediction_errors


@dataclass(frozen=True)
class TrueVJEPAScoreConfig:
    grid_size: int = 24
    target_height: int = 6
    target_width: int = 6


class TrueVJEPAScorer:
    """
    Compute masked-prediction JEPA errors from already-extracted token tensors.

    A repository-specific adapter should provide:
    - context_encoder(video, masks_context) -> [B, N_context, D]
    - target_encoder(video) -> [B, N_total, D]
    - predictor(context_tokens, masks_context, masks_target) -> [B, N_target, D]
    """

    def __init__(self, predictor: torch.nn.Module, config: TrueVJEPAScoreConfig | None = None):
        self.predictor = predictor
        self.config = config or TrueVJEPAScoreConfig()

    def score_video_tokens(
        self,
        adapter,
        video,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Return one JEPA prediction-error score per frame.

        The adapter is intentionally explicit so this scorer mirrors JEPA
        training: target_encoder(video) runs once, while context_encoder(video,
        masks_context) runs for each target mask.
        """
        frame_scores = []
        target_tokens = adapter.encode_target(video)
        top = max(0, (self.config.grid_size - self.config.target_height) // 2)
        left = max(0, (self.config.grid_size - self.config.target_width) // 2)

        for frame_idx in range(num_frames):
            masks = block_target_context_masks(
                num_frames=num_frames,
                grid_size=self.config.grid_size,
                target_frame=frame_idx,
                target_top=top,
                target_left=left,
                target_height=self.config.target_height,
                target_width=self.config.target_width,
                batch_size=target_tokens.shape[0],
            )
            context_tokens = adapter.encode_context(video, masks.context)
            errors = compute_jepa_prediction_errors(
                predictor=self.predictor,
                context_tokens=context_tokens,
                target_tokens=target_tokens,
                masks=masks,
            )
            frame_scores.append(errors.mean())
        return torch.stack(frame_scores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    missing = []
    if not Path(args.video).exists():
        missing.append(f"video not found: {args.video}")
    if not Path(args.checkpoint).exists():
        missing.append(f"checkpoint not found: {args.checkpoint}")
    if missing:
        for item in missing:
            print(item)
        return 2

    print(
        "True V-JEPA runtime adapter is not wired in this lightweight workspace. "
        "Use TrueVJEPAScorer.score_tokens() with an official V-JEPA predictor adapter "
        "in the WSL/GPU environment."
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
