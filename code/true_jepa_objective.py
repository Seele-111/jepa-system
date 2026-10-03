#!/usr/bin/env python3
"""
Small, testable core for true JEPA objective scoring.

The important distinction from the old proxy pipeline:
context tokens go through the predictor, target tokens come from a target
encoder, and the anomaly signal is prediction error on masked targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MaskPair:
    context: list[torch.Tensor]
    target: list[torch.Tensor]


def _gather_tokens(tokens: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    gathered = []
    for mask in masks:
        index = mask.to(tokens.device).unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        gathered.append(torch.gather(tokens, dim=1, index=index))
    return torch.cat(gathered, dim=0)


def compute_jepa_prediction_errors(
    predictor: torch.nn.Module,
    context_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    masks: MaskPair,
    normalize_target: bool = True,
) -> torch.Tensor:
    """
    Return per-target-token JEPA prediction error.

    Args:
        predictor: JEPA predictor called as predictor(context_tokens, context_masks, target_masks).
        context_tokens: encoder output for context tokens, shape [B, N_context, D].
        target_tokens: target encoder output for all tokens, shape [B, N_total, D].
        masks: context and target token indices in JEPA list-of-mask format.
        normalize_target: match I-JEPA training, which layer-normalizes target features.

    Returns:
        Tensor [B * n_context_masks, N_target] with Smooth L1 error per target token.
    """
    targets = target_tokens
    if normalize_target:
        targets = F.layer_norm(targets, (targets.shape[-1],))
    target_masked = _gather_tokens(targets, masks.target)

    predicted = predictor(context_tokens, masks.context, masks.target)
    if predicted.shape != target_masked.shape:
        raise ValueError(
            f"Predictor output shape {tuple(predicted.shape)} does not match "
            f"masked target shape {tuple(target_masked.shape)}"
        )

    return F.smooth_l1_loss(predicted, target_masked, reduction="none").mean(dim=-1)


def block_target_context_masks(
    num_frames: int,
    grid_size: int,
    target_frame: int,
    target_top: int,
    target_left: int,
    target_height: int,
    target_width: int,
    batch_size: int = 1,
) -> MaskPair:
    """Create one spatiotemporal target block and its disjoint context complement."""
    if not (0 <= target_frame < num_frames):
        raise ValueError("target_frame is outside the video token range")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target block must have positive size")
    if target_top < 0 or target_left < 0:
        raise ValueError("target block must start inside the grid")
    if target_top + target_height > grid_size or target_left + target_width > grid_size:
        raise ValueError("target block exceeds the spatial grid")

    tokens_per_frame = grid_size * grid_size
    total_tokens = num_frames * tokens_per_frame
    target_indices = []
    frame_offset = target_frame * tokens_per_frame
    for row in range(target_top, target_top + target_height):
        for col in range(target_left, target_left + target_width):
            target_indices.append(frame_offset + row * grid_size + col)

    target_set = set(target_indices)
    context_indices = [idx for idx in range(total_tokens) if idx not in target_set]

    context = torch.tensor([context_indices] * batch_size, dtype=torch.long)
    target = torch.tensor([target_indices] * batch_size, dtype=torch.long)
    return MaskPair(context=[context], target=[target])
