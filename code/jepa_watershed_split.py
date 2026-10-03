#!/usr/bin/env python3
"""JEPA evidence watershed splitting for merged error segments."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(0, end - start + 1)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    window = int(window)
    if window <= 1 or len(values) <= 1:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _contiguous_segments(binary: np.ndarray) -> list[Segment]:
    segments: list[Segment] = []
    start: int | None = None
    for idx, value in enumerate(np.asarray(binary).astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(binary) - 1))
    return segments


def _merge_short_gaps(binary: np.ndarray, min_gap: int) -> np.ndarray:
    binary = np.asarray(binary, dtype=np.int64).reshape(-1)
    min_gap = int(min_gap)
    if min_gap <= 0:
        return binary.copy()
    merged = binary.copy()
    segments = _contiguous_segments(merged)
    for left, right in zip(segments, segments[1:]):
        gap_start = left[1] + 1
        gap_end = right[0] - 1
        if 0 <= gap_end - gap_start + 1 <= min_gap:
            merged[gap_start : gap_end + 1] = 1
    return merged


def _split_one_parent(
    parent: tuple[int, int],
    evidence: np.ndarray,
    evidence_threshold: float,
    smooth_window: int,
    min_gap: int,
    min_length: int,
    pad: int,
    parent_min_length: int,
) -> list[Segment]:
    parent = _normalise_segment(parent)
    n = len(evidence)
    if n <= 0 or _segment_length(parent) < int(parent_min_length):
        return [parent]
    start = max(0, parent[0])
    end = min(n - 1, parent[1])
    if end < start:
        return [parent]

    local = _smooth(evidence[start : end + 1], smooth_window)
    binary = (local >= float(evidence_threshold)).astype(np.int64)
    binary = _merge_short_gaps(binary, int(min_gap))
    islands: list[Segment] = []
    for island_start, island_end in _contiguous_segments(binary):
        if island_end - island_start + 1 < int(min_length):
            continue
        padded_start = max(start, start + island_start - int(pad))
        padded_end = min(end, start + island_end + int(pad))
        if padded_end - padded_start + 1 >= int(min_length):
            islands.append((padded_start, padded_end))
    if len(islands) < 2:
        return [parent]
    return islands


def watershed_split_predictions(
    predictions: list[list[tuple[int, int]]],
    evidence: list[np.ndarray],
    evidence_threshold: float,
    smooth_window: int,
    min_gap: int,
    min_length: int,
    pad: int,
    parent_min_length: int,
) -> list[list[Segment]]:
    """Replace long merged predictions with multiple JEPA evidence islands."""
    if len(predictions) != len(evidence):
        raise ValueError(f"prediction/evidence count mismatch: {len(predictions)} vs {len(evidence)}")
    split_videos: list[list[Segment]] = []
    for video_predictions, video_evidence in zip(predictions, evidence):
        video_evidence = np.asarray(video_evidence, dtype=np.float32).reshape(-1)
        split_segments: list[Segment] = []
        for segment in video_predictions:
            split_segments.extend(
                _split_one_parent(
                    segment,
                    video_evidence,
                    evidence_threshold=evidence_threshold,
                    smooth_window=smooth_window,
                    min_gap=min_gap,
                    min_length=min_length,
                    pad=pad,
                    parent_min_length=parent_min_length,
                )
            )
        split_videos.append(sorted(set(split_segments)))
    return split_videos


def select_watershed_split_params(
    predictions: list[list[tuple[int, int]]],
    evidence: list[np.ndarray],
    labels: list[np.ndarray],
    evidence_thresholds: Iterable[float] = (0.45, 0.55, 0.65, 0.75),
    smooth_windows: Iterable[int] = (1, 3),
    min_gaps: Iterable[int] = (0, 1),
    min_lengths: Iterable[int] = (1, 2, 4),
    pads: Iterable[int] = (0, 1),
    parent_min_lengths: Iterable[int] = (24, 48),
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    """Tune conservative split params on the current validation fold."""
    base_predictions = [[_normalise_segment(segment) for segment in video] for video in predictions]
    base_metrics = evaluate_fused_predictions(base_predictions, labels, iou_threshold=iou_threshold)
    best_config = {
        "enabled": False,
        "base_metrics": base_metrics,
    }
    best_metrics = base_metrics
    best_predictions = base_predictions
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for evidence_threshold in evidence_thresholds:
        for smooth_window in smooth_windows:
            for min_gap in min_gaps:
                for min_length in min_lengths:
                    for pad in pads:
                        for parent_min_length in parent_min_lengths:
                            split = watershed_split_predictions(
                                base_predictions,
                                evidence,
                                evidence_threshold=float(evidence_threshold),
                                smooth_window=int(smooth_window),
                                min_gap=int(min_gap),
                                min_length=int(min_length),
                                pad=int(pad),
                                parent_min_length=int(parent_min_length),
                            )
                            metrics = evaluate_fused_predictions(split, labels, iou_threshold=iou_threshold)
                            changed = sum(
                                int(sorted(before) != sorted(after))
                                for before, after in zip(base_predictions, split)
                            )
                            key = (
                                float(metrics["segment"]["f1"]),
                                float(metrics["segment"]["precision"]),
                                float(metrics["segment"]["recall"]),
                                float(metrics["frame"]["f1"]),
                                -float(changed),
                            )
                            if key > best_key:
                                best_key = key
                                best_metrics = metrics
                                best_predictions = split
                                best_config = {
                                    "enabled": True,
                                    "evidence_threshold": float(evidence_threshold),
                                    "smooth_window": int(smooth_window),
                                    "min_gap": int(min_gap),
                                    "min_length": int(min_length),
                                    "pad": int(pad),
                                    "parent_min_length": int(parent_min_length),
                                    "changed_videos": int(changed),
                                    "base_metrics": base_metrics,
                                }
    return best_config, best_metrics, best_predictions
