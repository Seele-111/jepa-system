#!/usr/bin/env python3
"""DP-style event-set selection for JEPA temporal proposals.

This module selects a complete non-overlapping event set from base predictions
and JEPA candidate proposals. Unlike protected add-only fusion, it can replace
one wide parent segment with several compact child events when their total
evidence is stronger.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from cross_validate_segment_locator import aggregate_fold_metrics
from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _max_iou(segment: tuple[int, int], others: Iterable[tuple[int, int]]) -> float:
    norm = _normalise_segment(segment)
    return max((float(interval_iou(norm, _normalise_segment(other))) for other in others), default=0.0)


def _weighted_interval_schedule(items: list[tuple[Segment, float, int]]) -> list[Segment]:
    """Return max-weight non-overlapping intervals.

    Items are `(segment, weight, tie_index)`. Non-positive weights are ignored.
    """
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


def select_dp_event_set_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    base_keep_score: float,
    candidate_score_weight: float,
    min_candidate_score: float,
    length_penalty: float,
    base_overlap_penalty: float,
) -> list[list[Segment]]:
    if len(base_predictions) != len(candidate_predictions) or len(base_predictions) != len(candidate_scores):
        raise ValueError("base/candidate/score video counts must match")
    selected_videos: list[list[Segment]] = []
    for base_raw, candidates_raw, scores_raw in zip(base_predictions, candidate_predictions, candidate_scores):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        items: list[tuple[Segment, float, int]] = []
        tie_index = 0
        for segment in base:
            weight = float(base_keep_score) - float(length_penalty) * math.log1p(_segment_length(segment))
            items.append((segment, weight, tie_index))
            tie_index += 1
        for segment, score in zip(candidates, scores):
            score_value = float(score)
            if score_value < float(min_candidate_score):
                continue
            weight = (
                float(candidate_score_weight) * score_value
                - float(length_penalty) * math.log1p(_segment_length(segment))
                - float(base_overlap_penalty) * _max_iou(segment, base)
            )
            items.append((segment, weight, tie_index))
            tie_index += 1
        selected_videos.append(_weighted_interval_schedule(items))
    return selected_videos


def select_dp_event_set_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    labels: list[np.ndarray],
    base_keep_scores: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    candidate_score_weights: Iterable[float] = (0.75, 1.0, 1.25, 1.5),
    min_candidate_scores: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01, 0.02),
    base_overlap_penalties: Iterable[float] = (0.0, 0.1, 0.25),
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
    for base_keep_score in base_keep_scores:
        for candidate_score_weight in candidate_score_weights:
            for min_candidate_score in min_candidate_scores:
                for length_penalty in length_penalties:
                    for base_overlap_penalty in base_overlap_penalties:
                        predictions = select_dp_event_set_predictions(
                            base,
                            candidate_predictions,
                            candidate_scores,
                            base_keep_score=float(base_keep_score),
                            candidate_score_weight=float(candidate_score_weight),
                            min_candidate_score=float(min_candidate_score),
                            length_penalty=float(length_penalty),
                            base_overlap_penalty=float(base_overlap_penalty),
                        )
                        metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(iou_threshold))
                        if max_fp_increase is not None and int(metrics["segment"]["fp"]) > base_fp + int(max_fp_increase):
                            continue
                        changed = sum(int(sorted(before) != sorted(after)) for before, after in zip(base, predictions))
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
                                "base_keep_score": float(base_keep_score),
                                "candidate_score_weight": float(candidate_score_weight),
                                "min_candidate_score": float(min_candidate_score),
                                "length_penalty": float(length_penalty),
                                "base_overlap_penalty": float(base_overlap_penalty),
                                "changed_videos": int(changed),
                                "base_metrics": base_metrics,
                            }
    return best_config, best_metrics, best_predictions


def _json_segments(segments: list[Segment]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in segments]


def _flatten_folds(folds: list[dict], key: str) -> list:
    values = []
    for fold in folds:
        values.extend(fold[key])
    return values


def run_fold_heldout_dp_event_set(
    folds: list[dict],
    base_keep_scores: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    candidate_score_weights: Iterable[float] = (0.75, 1.0, 1.25, 1.5),
    min_candidate_scores: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01, 0.02),
    base_overlap_penalties: Iterable[float] = (0.0, 0.1, 0.25),
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> dict:
    """Run strict outer-fold DP event-set selection.

    DP hyperparameters are selected on all folds except the current held-out
    fold, then applied unchanged to the held-out fold.
    """
    results: list[dict] = []
    for heldout_pos, heldout in enumerate(folds):
        calibration = [fold for pos, fold in enumerate(folds) if pos != heldout_pos]
        config, calibration_metrics, _ = select_dp_event_set_params(
            _flatten_folds(calibration, "base"),
            _flatten_folds(calibration, "candidates"),
            _flatten_folds(calibration, "scores"),
            _flatten_folds(calibration, "labels"),
            base_keep_scores=base_keep_scores,
            candidate_score_weights=candidate_score_weights,
            min_candidate_scores=min_candidate_scores,
            length_penalties=length_penalties,
            base_overlap_penalties=base_overlap_penalties,
            max_fp_increase=max_fp_increase,
            iou_threshold=float(iou_threshold),
        )
        if config.get("enabled"):
            predictions = select_dp_event_set_predictions(
                heldout["base"],
                heldout["candidates"],
                heldout["scores"],
                base_keep_score=float(config["base_keep_score"]),
                candidate_score_weight=float(config["candidate_score_weight"]),
                min_candidate_score=float(config["min_candidate_score"]),
                length_penalty=float(config["length_penalty"]),
                base_overlap_penalty=float(config["base_overlap_penalty"]),
            )
        else:
            predictions = [[_normalise_segment(item) for item in video] for video in heldout["base"]]
        metrics = evaluate_fused_predictions(predictions, heldout["labels"], iou_threshold=float(iou_threshold))
        results.append(
            {
                "fold": int(heldout.get("fold", heldout_pos)),
                "config": config,
                "calibration": calibration_metrics,
                "validation": metrics,
                "predictions": [_json_segments(video) for video in predictions],
            }
        )
    return {"folds": results, "aggregate": aggregate_fold_metrics(results)}
