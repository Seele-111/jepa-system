#!/usr/bin/env python3
"""JEPA optimal-transport style event-set matching.

The module treats high JEPA evidence islands as event demand and candidate
segments as explanations. It then selects a non-overlapping event set from
base predictions plus evidence-matched proposals.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    a = _normalise_segment(a)
    b = _normalise_segment(b)
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _max_iou(segment: tuple[int, int], segments: Iterable[tuple[int, int]]) -> float:
    norm = _normalise_segment(segment)
    return max((float(interval_iou(norm, _normalise_segment(other))) for other in segments), default=0.0)


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


def _evidence_curve(evidence: np.ndarray, smooth_window: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim == 1:
        curve = values.reshape(-1)
    elif values.ndim == 2:
        curve = values.mean(axis=1)
    else:
        raise ValueError(f"evidence must have shape [T] or [T, C], got {values.shape}")
    return np.clip(_smooth(curve, int(smooth_window)), 0.0, 1.0).astype(np.float32)


def _evidence_islands(curve: np.ndarray, threshold: float, min_length: int) -> list[Segment]:
    active = (np.asarray(curve, dtype=np.float32) >= float(threshold)).astype(np.int64)
    return [
        (int(start), int(end))
        for start, end in _contiguous_segments(active)
        if end - start + 1 >= int(min_length)
    ]


def _context_values(curve: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    start, end = _normalise_segment(segment)
    length = _segment_length((start, end))
    left = curve[max(0, start - length) : max(0, start)]
    right = curve[min(len(curve), end + 1) : min(len(curve), end + 1 + length)]
    if len(left) == 0 and len(right) == 0:
        return np.asarray([], dtype=np.float32)
    return np.concatenate([left, right]).astype(np.float32)


def _channel_agreement(evidence: np.ndarray, segment: tuple[int, int]) -> float:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim != 2 or values.shape[1] <= 1:
        return 0.0
    start, end = _normalise_segment(segment)
    if len(values) == 0:
        return 0.0
    start = max(0, min(start, len(values) - 1))
    end = max(start, min(end, len(values) - 1))
    means = np.clip(values[start : end + 1].mean(axis=0), 0.0, 1.0)
    mean_value = float(means.mean())
    if mean_value <= 1e-6:
        return 0.0
    return float(np.clip(means.min() / mean_value, 0.0, 1.0) * mean_value)


def _island_coverage(segment: tuple[int, int], islands: list[Segment]) -> float:
    if not islands:
        return 0.0
    coverages = []
    for island in islands:
        coverages.append(_intersection(segment, island) / float(_segment_length(island)))
    return float(max(coverages, default=0.0))


def score_ot_candidates(
    base_segments: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    selector_scores: np.ndarray,
    evidence: np.ndarray,
    evidence_threshold: float = 0.5,
    selector_weight: float = 0.1,
    contrast_weight: float = 0.5,
    active_fraction_weight: float = 0.5,
    agreement_weight: float = 0.5,
    island_coverage_weight: float = 1.0,
    peak_alignment_weight: float = 0.5,
    base_overlap_penalty: float = 0.0,
    length_penalty: float = 0.0,
    smooth_window: int = 1,
    min_evidence_island_length: int = 1,
) -> np.ndarray:
    """Score candidates by how well they explain JEPA evidence islands."""
    scores = np.nan_to_num(np.asarray(selector_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(candidates) != len(scores):
        raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
    curve = _evidence_curve(evidence, smooth_window=int(smooth_window))
    islands = _evidence_islands(curve, threshold=float(evidence_threshold), min_length=int(min_evidence_island_length))
    out = np.zeros(len(candidates), dtype=np.float32)
    for idx, segment_raw in enumerate(candidates):
        segment = _normalise_segment(segment_raw)
        if len(curve) == 0:
            continue
        start = max(0, min(segment[0], len(curve) - 1))
        end = max(start, min(segment[1], len(curve) - 1))
        inside = curve[start : end + 1]
        context = _context_values(curve, (start, end))
        evidence_mean = float(inside.mean()) if len(inside) else 0.0
        active_fraction = float((inside >= float(evidence_threshold)).mean()) if len(inside) else 0.0
        contrast = evidence_mean - (float(context.mean()) if len(context) else 0.0)
        peak_alignment = float(inside.max()) if len(inside) else 0.0
        agreement = _channel_agreement(evidence, (start, end))
        coverage = _island_coverage((start, end), islands)
        out[idx] = (
            evidence_mean
            + float(selector_weight) * float(scores[idx])
            + float(contrast_weight) * contrast
            + float(active_fraction_weight) * active_fraction
            + float(agreement_weight) * agreement
            + float(island_coverage_weight) * coverage
            + float(peak_alignment_weight) * peak_alignment
            - float(base_overlap_penalty) * _max_iou((start, end), base_segments)
            - float(length_penalty) * math.log1p(_segment_length((start, end)))
        )
    return out.astype(np.float32)


def _weighted_interval_schedule(items: list[tuple[Segment, float, int]]) -> list[Segment]:
    filtered = [
        (_normalise_segment(segment), float(weight), int(tie_index))
        for segment, weight, tie_index in items
        if float(weight) > 0.0
    ]
    if not filtered:
        return []
    best_by_segment: dict[Segment, tuple[float, int]] = {}
    for segment, weight, tie_index in filtered:
        current = best_by_segment.get(segment)
        if current is None or (weight, -tie_index) > (current[0], -current[1]):
            best_by_segment[segment] = (weight, tie_index)
    ordered = sorted(
        [(segment[0], segment[1], weight, tie_index) for segment, (weight, tie_index) in best_by_segment.items()],
        key=lambda item: (item[1], item[0], item[3]),
    )
    ends = [item[1] for item in ordered]
    previous: list[int] = []
    for start, _, _, _ in ordered:
        lo, hi = 0, len(ends)
        while lo < hi:
            mid = (lo + hi) // 2
            if ends[mid] < start:
                lo = mid + 1
            else:
                hi = mid
        previous.append(lo - 1)

    n = len(ordered)
    dp = np.zeros(n + 1, dtype=np.float64)
    take = np.zeros(n, dtype=bool)
    for idx, item in enumerate(ordered, start=1):
        take_score = float(item[2]) + dp[previous[idx - 1] + 1]
        skip_score = dp[idx - 1]
        if take_score > skip_score:
            dp[idx] = take_score
            take[idx - 1] = True
        else:
            dp[idx] = skip_score

    selected: list[Segment] = []
    idx = n
    while idx > 0:
        item = ordered[idx - 1]
        if take[idx - 1] and float(item[2]) + dp[previous[idx - 1] + 1] >= dp[idx - 1]:
            selected.append((int(item[0]), int(item[1])))
            idx = previous[idx - 1] + 1
        else:
            idx -= 1
    return sorted(selected)


def select_ot_event_set_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence: list[np.ndarray],
    score_threshold: float,
    base_keep_score: float,
    evidence_threshold: float,
    selector_weight: float = 0.1,
    contrast_weight: float = 0.5,
    active_fraction_weight: float = 0.5,
    agreement_weight: float = 0.5,
    island_coverage_weight: float = 1.0,
    peak_alignment_weight: float = 0.5,
    base_overlap_penalty: float = 0.0,
    length_penalty: float = 0.0,
    smooth_window: int = 1,
    min_evidence_island_length: int = 1,
) -> list[list[Segment]]:
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence)
    ):
        raise ValueError("base/candidate/score/evidence video counts must match")
    selected_videos: list[list[Segment]] = []
    for base_raw, candidates_raw, scores_raw, evidence_raw in zip(
        base_predictions, candidate_predictions, candidate_scores, evidence
    ):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        ot_scores = score_ot_candidates(
            base,
            candidates,
            scores_raw,
            evidence_raw,
            evidence_threshold=float(evidence_threshold),
            selector_weight=float(selector_weight),
            contrast_weight=float(contrast_weight),
            active_fraction_weight=float(active_fraction_weight),
            agreement_weight=float(agreement_weight),
            island_coverage_weight=float(island_coverage_weight),
            peak_alignment_weight=float(peak_alignment_weight),
            base_overlap_penalty=float(base_overlap_penalty),
            length_penalty=float(length_penalty),
            smooth_window=int(smooth_window),
            min_evidence_island_length=int(min_evidence_island_length),
        )
        items: list[tuple[Segment, float, int]] = []
        tie_index = 0
        for segment in base:
            items.append((segment, float(base_keep_score), tie_index))
            tie_index += 1
        for segment, score in zip(candidates, ot_scores):
            if float(score) < float(score_threshold):
                continue
            items.append((segment, float(score), tie_index))
            tie_index += 1
        selected_videos.append(_weighted_interval_schedule(items))
    return selected_videos


def select_ot_event_set_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence: list[np.ndarray],
    labels: list[np.ndarray],
    score_thresholds: Iterable[float] = (0.4, 0.6, 0.8, 1.0),
    base_keep_scores: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    evidence_thresholds: Iterable[float] = (0.45, 0.55, 0.65),
    selector_weights: Iterable[float] = (0.0, 0.1, 0.25),
    contrast_weights: Iterable[float] = (0.25, 0.5),
    active_fraction_weights: Iterable[float] = (0.25, 0.5),
    agreement_weights: Iterable[float] = (0.25, 0.5),
    island_coverage_weights: Iterable[float] = (0.5, 1.0),
    peak_alignment_weights: Iterable[float] = (0.25, 0.5),
    base_overlap_penalties: Iterable[float] = (0.0, 0.25),
    length_penalties: Iterable[float] = (0.0, 0.005),
    smooth_windows: Iterable[int] = (1, 3),
    min_evidence_island_lengths: Iterable[int] = (1, 2),
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=float(iou_threshold))
    best_config: dict = {"enabled": False, "base_metrics": base_metrics}
    best_metrics = base_metrics
    best_predictions = base
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    base_fp = int(base_metrics["segment"]["fp"])
    for score_threshold in score_thresholds:
        for base_keep_score in base_keep_scores:
            for evidence_threshold in evidence_thresholds:
                for selector_weight in selector_weights:
                    for contrast_weight in contrast_weights:
                        for active_fraction_weight in active_fraction_weights:
                            for agreement_weight in agreement_weights:
                                for island_coverage_weight in island_coverage_weights:
                                    for peak_alignment_weight in peak_alignment_weights:
                                        for base_overlap_penalty in base_overlap_penalties:
                                            for length_penalty in length_penalties:
                                                for smooth_window in smooth_windows:
                                                    for min_island_length in min_evidence_island_lengths:
                                                        predictions = select_ot_event_set_predictions(
                                                            base,
                                                            candidate_predictions,
                                                            candidate_scores,
                                                            evidence,
                                                            score_threshold=float(score_threshold),
                                                            base_keep_score=float(base_keep_score),
                                                            evidence_threshold=float(evidence_threshold),
                                                            selector_weight=float(selector_weight),
                                                            contrast_weight=float(contrast_weight),
                                                            active_fraction_weight=float(active_fraction_weight),
                                                            agreement_weight=float(agreement_weight),
                                                            island_coverage_weight=float(island_coverage_weight),
                                                            peak_alignment_weight=float(peak_alignment_weight),
                                                            base_overlap_penalty=float(base_overlap_penalty),
                                                            length_penalty=float(length_penalty),
                                                            smooth_window=int(smooth_window),
                                                            min_evidence_island_length=int(min_island_length),
                                                        )
                                                        metrics = evaluate_fused_predictions(
                                                            predictions,
                                                            labels,
                                                            iou_threshold=float(iou_threshold),
                                                        )
                                                        if max_fp_increase is not None and int(metrics["segment"]["fp"]) > base_fp + int(max_fp_increase):
                                                            continue
                                                        changed = sum(
                                                            int(sorted(before) != sorted(after))
                                                            for before, after in zip(base, predictions)
                                                        )
                                                        key = (
                                                            float(metrics["segment"]["f1"]),
                                                            float(metrics["segment"]["recall"]),
                                                            float(metrics["segment"]["precision"]),
                                                            float(metrics["frame"]["f1"]),
                                                            -float(changed),
                                                        )
                                                        if key > best_key:
                                                            best_key = key
                                                            best_metrics = metrics
                                                            best_predictions = predictions
                                                            best_config = {
                                                                "enabled": True,
                                                                "score_threshold": float(score_threshold),
                                                                "base_keep_score": float(base_keep_score),
                                                                "evidence_threshold": float(evidence_threshold),
                                                                "selector_weight": float(selector_weight),
                                                                "contrast_weight": float(contrast_weight),
                                                                "active_fraction_weight": float(active_fraction_weight),
                                                                "agreement_weight": float(agreement_weight),
                                                                "island_coverage_weight": float(island_coverage_weight),
                                                                "peak_alignment_weight": float(peak_alignment_weight),
                                                                "base_overlap_penalty": float(base_overlap_penalty),
                                                                "length_penalty": float(length_penalty),
                                                                "smooth_window": int(smooth_window),
                                                                "min_evidence_island_length": int(min_island_length),
                                                                "changed_videos": int(changed),
                                                                "base_metrics": base_metrics,
                                                            }
    return best_config, best_metrics, best_predictions
