"""Exact offline binary temporal decoding with NumPy as the sole dependency.

For x[-1] = 0, maximize
    sum((logit(clip(p[t])) - logit(threshold)) * x[t] / fps)
    - transition_seconds * sum(abs(x[t] - x[t - 1])).
There is no terminal transition charge. Minimum duration is a post-decode
filter, not a constraint on the dynamic program.
"""

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

__all__ = ["decode_duration"]

_KIND = "duration-logit-v4"
_REQUIRED_FIELDS = frozenset(
    ("kind", "threshold", "transition_seconds", "min_seconds")
)
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"video_threshold"}
_PROBABILITY_EPSILON = 1e-6
_DURATION_EPSILON = 1e-9


def _finite_real(value, name):
    """Reject coercible strings, booleans, arrays, and non-finite scalars."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_config(config):
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    fields = set(config)
    unknown = fields - _ALLOWED_FIELDS
    if unknown:
        names = ", ".join(sorted(map(repr, unknown)))
        raise ValueError(f"unknown config fields: {names}")
    missing = _REQUIRED_FIELDS - fields
    if missing:
        raise ValueError(f"missing config fields: {', '.join(sorted(missing))}")
    if not isinstance(config["kind"], str) or config["kind"] != _KIND:
        raise ValueError(f"config kind must be {_KIND!r}")

    threshold = _finite_real(config["threshold"], "threshold")
    transition_seconds = _finite_real(
        config["transition_seconds"], "transition_seconds"
    )
    min_seconds = _finite_real(config["min_seconds"], "min_seconds")
    video_threshold = _finite_real(
        config.get("video_threshold", 0.0), "video_threshold"
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if transition_seconds < 0.0:
        raise ValueError("transition_seconds must be nonnegative")
    if min_seconds < 0.0:
        raise ValueError("min_seconds must be nonnegative")
    if not 0.0 <= video_threshold <= 1.0:
        raise ValueError("video_threshold must be between 0 and 1 inclusive")
    return threshold, transition_seconds, min_seconds, video_threshold


def _validate_probabilities(probabilities):
    try:
        values = np.asarray(probabilities)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("probabilities must be a 1D real numeric array") from error
    if values.ndim != 1:
        raise ValueError("probabilities must be 1D (an empty 1D array is allowed)")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError("probabilities must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must be between 0 and 1 inclusive")
    return values.astype(np.float64, copy=False)


def _prefer_one(score_zero, count_zero, score_one, count_one):
    """Rank energy first, fewer transitions second, state 0 last on ties."""
    return score_one > score_zero or (
        score_one == score_zero and count_one < count_zero
    )


def decode_duration(probabilities, fps, video_probability, config):
    """Return ordered inclusive frame spans as ``list[tuple[int, int]]``.

    ``config`` requires ``kind='duration-logit-v4'``, ``threshold`` in (0, 1),
    and nonnegative ``transition_seconds`` and ``min_seconds``. The only
    optional field is ``video_threshold`` in [0, 1], defaulting to 0. Unknown
    kinds/fields and invalid or non-finite values raise ``ValueError``.

    Probabilities must be a real numeric 1D sequence in [0, 1]; empty is valid.
    They are clipped to [1e-6, 1-1e-6] only after validation. The threshold
    itself is not clipped. FPS must be finite and positive. Video probability
    must be finite and in [0, 1], and is used ONLY for the hard gate:
    ``video_probability < video_threshold`` returns []. All inputs are
    validated even when the gate is closed or the sequence is empty.

    The two-state DP has O(T) time and O(T) backpointer storage. Exact score
    ties prefer fewer transitions, then predecessor/final state 0 (equivalent
    to preferring 0 at the latest differing frame when both earlier criteria
    tie). No approximate score tolerance, seed, labels, forced event, video
    multiplication, or duration-ratio heuristic is used. After backtracking,
    spans shorter than ceil(min_seconds * fps - 1e-9) frames are discarded;
    surviving spans are not reoptimized, expanded, or merged.
    """
    threshold, transition_seconds, min_seconds, video_threshold = (
        _validate_config(config)
    )
    fps = _finite_real(fps, "fps")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    video_probability = _finite_real(video_probability, "video_probability")
    if not 0.0 <= video_probability <= 1.0:
        raise ValueError("video_probability must be between 0 and 1 inclusive")
    values = _validate_probabilities(probabilities)
    frame_count = values.size
    if video_probability < video_threshold or frame_count == 0:
        return []

    # Multiplication of the entire objective by positive fps is equivalent
    # and avoids reciprocal overflow for very small, but valid, frame rates.
    clipped = np.clip(values, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    evidence = (
        np.log(clipped)
        - np.log1p(-clipped)
        - (np.log(threshold) - np.log1p(-threshold))
    )
    transition_cost = transition_seconds * fps
    if math.isinf(transition_cost):
        # Both operands are finite and nonnegative. An overflow-sized entry
        # cost exceeds the bounded total evidence of any representable array.
        return []

    duration_frames = min_seconds * fps - _DURATION_EPSILON
    # Saturation preserves the filter while avoiding ceil(inf) on valid input.
    minimum_frames = (
        frame_count + 1
        if duration_frames > frame_count
        else math.ceil(duration_frames)
    )

    parents = np.empty((frame_count, 2), dtype=np.uint8)
    score_zero, score_one = 0.0, -math.inf
    count_zero, count_one = 0, frame_count + 1
    for frame, reward in enumerate(evidence):
        zero_from_one = _prefer_one(
            score_zero, count_zero, score_one - transition_cost, count_one + 1
        )
        one_from_one = _prefer_one(
            score_zero - transition_cost, count_zero + 1, score_one, count_one
        )
        parents[frame, 0] = zero_from_one
        parents[frame, 1] = one_from_one
        next_zero = score_one - transition_cost if zero_from_one else score_zero
        next_one = float(reward) + (
            score_one if one_from_one else score_zero - transition_cost
        )
        next_count_zero = count_one + 1 if zero_from_one else count_zero
        next_count_one = count_one if one_from_one else count_zero + 1
        score_zero, score_one = next_zero, next_one
        count_zero, count_one = next_count_zero, next_count_one

    # Compare the final states directly: ending in 1 incurs no exit charge.
    state = int(_prefer_one(score_zero, count_zero, score_one, count_one))
    spans = []
    end = None
    for frame in range(frame_count - 1, -1, -1):
        predecessor = int(parents[frame, state])
        if state == 1:
            if end is None:
                end = frame
            if predecessor == 0:
                if end - frame + 1 >= minimum_frames:
                    spans.append((frame, end))
                end = None
        state = predecessor
    spans.reverse()
    return spans
