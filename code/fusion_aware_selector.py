#!/usr/bin/env python3
"""Fusion-aware calibration for JEPA proposal selectors."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from selector_fusion import (
    evaluate_fused_predictions,
    fuse_protected_segment_predictions,
    select_protected_fusion_params,
)
from train_proposal_set_selector import select_weighted_proposal_set


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _predict_from_candidate_scores(
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    prob_threshold: float,
    length_penalty: float,
) -> list[list[Segment]]:
    if len(candidate_predictions) != len(candidate_scores):
        raise ValueError(f"candidate/score video counts mismatch: {len(candidate_predictions)} vs {len(candidate_scores)}")
    predictions: list[list[Segment]] = []
    for candidates, scores_raw in zip(candidate_predictions, candidate_scores):
        scores = np.asarray(scores_raw, dtype=np.float32).reshape(-1)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        filtered_candidates: list[Segment] = []
        filtered_scores: list[float] = []
        for segment, score in zip(candidates, scores):
            if float(score) < float(prob_threshold):
                continue
            filtered_candidates.append(_normalise_segment(segment))
            filtered_scores.append(float(score))
        predictions.append(
            select_weighted_proposal_set(
                filtered_candidates,
                filtered_scores,
                length_penalty=float(length_penalty),
            )
        )
    return predictions


def _apply_protected_fusion(
    mainline_predictions: list[list[tuple[int, int]]],
    selector_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    protected_mainline_iou_candidates: Iterable[float | None],
    selector_nms_iou_candidates: Iterable[float | None],
    iou_threshold: float,
) -> tuple[dict, dict, list[list[Segment]]]:
    protected_config, protected_metrics = select_protected_fusion_params(
        mainline_predictions,
        selector_predictions,
        labels,
        max_mainline_iou_candidates=protected_mainline_iou_candidates,
        selector_nms_iou_candidates=selector_nms_iou_candidates,
        iou_threshold=iou_threshold,
    )
    if protected_config.get("enabled"):
        fused = fuse_protected_segment_predictions(
            mainline_predictions,
            selector_predictions,
            max_mainline_iou=protected_config["max_mainline_iou"],
            selector_nms_iou=protected_config["selector_nms_iou"],
        )
        metrics = evaluate_fused_predictions(fused, labels, iou_threshold=iou_threshold)
    else:
        fused = [[_normalise_segment(segment) for segment in video] for video in mainline_predictions]
        metrics = protected_metrics
    return protected_config, metrics, fused


def select_fusion_aware_selector_params_from_scores(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    labels: list[np.ndarray],
    prob_thresholds: Iterable[float],
    length_penalties: Iterable[float],
    protected_mainline_iou_candidates: Iterable[float | None],
    selector_nms_iou_candidates: Iterable[float | None],
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base_metrics = evaluate_fused_predictions(mainline_predictions, labels, iou_threshold=iou_threshold)
    best_params: dict = {
        "prob_threshold": None,
        "length_penalty": None,
        "enabled": False,
        "base_metrics": base_metrics,
    }
    best_metrics = base_metrics
    best_selector_predictions: list[list[Segment]] = [[] for _ in mainline_predictions]
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for prob_threshold in prob_thresholds:
        for length_penalty in length_penalties:
            selector_predictions = _predict_from_candidate_scores(
                candidate_predictions,
                candidate_scores,
                prob_threshold=float(prob_threshold),
                length_penalty=float(length_penalty),
            )
            protected_config, metrics, _ = _apply_protected_fusion(
                mainline_predictions,
                selector_predictions,
                labels,
                protected_mainline_iou_candidates=protected_mainline_iou_candidates,
                selector_nms_iou_candidates=selector_nms_iou_candidates,
                iou_threshold=iou_threshold,
            )
            rescue_count = sum(len(video) for video in selector_predictions)
            key = (
                float(metrics["segment"]["f1"]),
                float(metrics["segment"]["precision"]),
                float(metrics["segment"]["recall"]),
                float(metrics["frame"]["f1"]),
                -float(rescue_count),
            )
            if key > best_key:
                best_key = key
                best_metrics = metrics
                best_selector_predictions = selector_predictions
                best_params = {
                    "enabled": True,
                    "prob_threshold": float(prob_threshold),
                    "length_penalty": float(length_penalty),
                    "protected": protected_config,
                    "base_metrics": base_metrics,
                }
    return best_params, best_metrics, best_selector_predictions
