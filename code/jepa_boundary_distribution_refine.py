#!/usr/bin/env python3
"""JEPA boundary-distribution refinement for near-miss segments.

The module does not add new detections. It only adjusts boundaries of already
active predictions using overlapping high-confidence JEPA proposals, aiming to
convert near-miss segments into true positives without increasing segment count.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(0, end - start + 1)


def _intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    a_start, a_end = _normalise_segment(a)
    b_start, b_end = _normalise_segment(b)
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _base_coverage(base: tuple[int, int], candidate: tuple[int, int]) -> float:
    return _intersection(base, candidate) / max(1, _length(base))


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if len(values) != len(weights):
        raise ValueError("value/weight count mismatch")
    if not values:
        raise ValueError("cannot compute quantile without values")
    items = sorted((float(value), max(0.0, float(weight))) for value, weight in zip(values, weights))
    total = sum(weight for _, weight in items)
    if total <= 0:
        return float(items[len(items) // 2][0])
    target = min(1.0, max(0.0, float(quantile))) * total
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(items[-1][0])


def _candidate_pool_for_base(
    base: Segment,
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    prob_threshold: float,
    min_base_coverage: float,
    max_length_ratio: float,
) -> tuple[list[Segment], list[float]]:
    base_len = max(1, _length(base))
    pool: list[Segment] = []
    pool_scores: list[float] = []
    for candidate_raw, score_raw in zip(candidates, scores):
        score = float(score_raw)
        if score < float(prob_threshold):
            continue
        candidate = _normalise_segment(candidate_raw)
        if _base_coverage(base, candidate) < float(min_base_coverage):
            continue
        if _length(candidate) / base_len > float(max_length_ratio):
            continue
        pool.append(candidate)
        pool_scores.append(score)
    return pool, pool_scores


def refine_boundary_distribution_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    prob_threshold: float,
    min_base_coverage: float,
    max_length_ratio: float,
    start_quantile: float,
    end_quantile: float,
    base_boundary_weight: float,
    max_shift_ratio: float,
) -> list[list[Segment]]:
    if len(base_predictions) != len(candidate_predictions) or len(base_predictions) != len(candidate_scores):
        raise ValueError("base/candidate/score video counts must match")

    refined_videos: list[list[Segment]] = []
    for base_segments, candidates, scores_raw in zip(base_predictions, candidate_predictions, candidate_scores):
        scores = np.asarray(scores_raw, dtype=np.float32).reshape(-1)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        refined_segments: list[Segment] = []
        for base_raw in base_segments:
            base = _normalise_segment(base_raw)
            base_len = max(1, _length(base))
            pool, pool_scores = _candidate_pool_for_base(
                base,
                candidates,
                scores,
                prob_threshold=float(prob_threshold),
                min_base_coverage=float(min_base_coverage),
                max_length_ratio=float(max_length_ratio),
            )
            if not pool:
                refined_segments.append(base)
                continue

            starts = [float(segment[0]) for segment in pool]
            ends = [float(segment[1]) for segment in pool]
            weights = [max(1e-6, float(score)) for score in pool_scores]
            if float(base_boundary_weight) > 0:
                starts.append(float(base[0]))
                ends.append(float(base[1]))
                weights.append(float(base_boundary_weight))

            new_start = int(round(_weighted_quantile(starts, weights, float(start_quantile))))
            new_end = int(round(_weighted_quantile(ends, weights, float(end_quantile))))
            refined = _normalise_segment((new_start, new_end))
            max_shift = float(max_shift_ratio) * base_len
            if abs(refined[0] - base[0]) > max_shift or abs(refined[1] - base[1]) > max_shift:
                refined_segments.append(base)
            else:
                refined_segments.append(refined)
        refined_videos.append(sorted(set(refined_segments)))
    return refined_videos


def select_boundary_distribution_refine_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    labels: list[np.ndarray],
    prob_thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8),
    min_base_coverages: Iterable[float] = (0.5, 0.75, 0.9),
    max_length_ratios: Iterable[float] = (2.0, 3.0, 4.0),
    start_quantiles: Iterable[float] = (0.25, 0.5),
    end_quantiles: Iterable[float] = (0.5, 0.75),
    base_boundary_weights: Iterable[float] = (0.0, 0.25, 0.5),
    max_shift_ratios: Iterable[float] = (1.0, 2.0, 4.0),
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base_predictions_norm = [[_normalise_segment(segment) for segment in video] for video in base_predictions]
    base_metrics = evaluate_fused_predictions(base_predictions_norm, labels, iou_threshold=iou_threshold)
    best_config: dict = {
        "enabled": False,
        "base_metrics": base_metrics,
    }
    best_metrics = base_metrics
    best_predictions = base_predictions_norm
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )

    for prob_threshold in prob_thresholds:
        for min_base_coverage in min_base_coverages:
            for max_length_ratio in max_length_ratios:
                for start_quantile in start_quantiles:
                    for end_quantile in end_quantiles:
                        for base_boundary_weight in base_boundary_weights:
                            for max_shift_ratio in max_shift_ratios:
                                refined = refine_boundary_distribution_predictions(
                                    base_predictions_norm,
                                    candidate_predictions,
                                    candidate_scores,
                                    prob_threshold=float(prob_threshold),
                                    min_base_coverage=float(min_base_coverage),
                                    max_length_ratio=float(max_length_ratio),
                                    start_quantile=float(start_quantile),
                                    end_quantile=float(end_quantile),
                                    base_boundary_weight=float(base_boundary_weight),
                                    max_shift_ratio=float(max_shift_ratio),
                                )
                                metrics = evaluate_fused_predictions(refined, labels, iou_threshold=iou_threshold)
                                changed = sum(
                                    int(sorted(before) != sorted(after))
                                    for before, after in zip(base_predictions_norm, refined)
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
                                    best_predictions = refined
                                    best_config = {
                                        "enabled": True,
                                        "prob_threshold": float(prob_threshold),
                                        "min_base_coverage": float(min_base_coverage),
                                        "max_length_ratio": float(max_length_ratio),
                                        "start_quantile": float(start_quantile),
                                        "end_quantile": float(end_quantile),
                                        "base_boundary_weight": float(base_boundary_weight),
                                        "max_shift_ratio": float(max_shift_ratio),
                                        "changed_videos": int(changed),
                                        "base_metrics": base_metrics,
                                    }
    return best_config, best_metrics, best_predictions
