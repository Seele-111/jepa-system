#!/usr/bin/env python3
"""Candidate-level JEPA rescue for missed error segments."""
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
    return max(0, end - start + 1)


def _max_iou(segment: tuple[int, int], segments: Iterable[tuple[int, int]]) -> float:
    return max((interval_iou(_normalise_segment(segment), _normalise_segment(other)) for other in segments), default=0.0)


def _score_first_nms(
    candidates: list[tuple[int, int]],
    scores: list[float],
    iou_threshold: float | None,
    max_items: int,
) -> list[Segment]:
    items = []
    for segment, score in zip(candidates, scores):
        norm = _normalise_segment(segment)
        items.append((norm, float(score), _segment_length(norm)))
    items.sort(key=lambda item: (item[1], -item[2], -item[0][0]), reverse=True)
    kept: list[tuple[Segment, float]] = []
    for segment, score, _ in items:
        if iou_threshold is not None:
            if any(interval_iou(segment, kept_segment) > float(iou_threshold) for kept_segment, _ in kept):
                continue
        kept.append((segment, score))
        if len(kept) >= int(max_items):
            break
    return sorted({segment for segment, _ in kept})


def select_candidate_rescue_predictions(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    prob_threshold: float,
    max_mainline_iou: float | None,
    candidate_nms_iou: float | None,
    length_penalty: float,
    max_rescues_per_video: int,
) -> list[list[Segment]]:
    if len(mainline_predictions) != len(candidate_predictions) or len(mainline_predictions) != len(candidate_scores):
        raise ValueError("mainline/candidate/score video counts must match")
    fused: list[list[Segment]] = []
    for main_segments, candidates, scores in zip(mainline_predictions, candidate_predictions, candidate_scores):
        main = sorted({_normalise_segment(segment) for segment in main_segments})
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        rescue_candidates: list[Segment] = []
        rescue_scores: list[float] = []
        for segment, score in zip(candidates, scores):
            norm = _normalise_segment(segment)
            adjusted = float(score) - float(length_penalty) * math.log1p(max(1, _segment_length(norm)))
            if adjusted < float(prob_threshold):
                continue
            if max_mainline_iou is not None and _max_iou(norm, main) > float(max_mainline_iou):
                continue
            rescue_candidates.append(norm)
            rescue_scores.append(adjusted)
        selected = _score_first_nms(
            rescue_candidates,
            rescue_scores,
            iou_threshold=candidate_nms_iou,
            max_items=max_rescues_per_video,
        )
        fused.append(sorted(set(main + selected)))
    return fused


def select_candidate_rescue_params(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    labels: list[np.ndarray],
    prob_thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    max_mainline_ious: Iterable[float | None] = (0.0, 0.05, 0.1, 0.25, None),
    candidate_nms_ious: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01, 0.02, 0.04),
    max_rescues_per_videos: Iterable[int] = (1, 2, 3),
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base_metrics = evaluate_fused_predictions(mainline_predictions, labels, iou_threshold=iou_threshold)
    best_config: dict = {
        "enabled": False,
        "base_metrics": base_metrics,
    }
    best_metrics = base_metrics
    best_predictions = [[_normalise_segment(segment) for segment in video] for video in mainline_predictions]
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for prob_threshold in prob_thresholds:
        for max_mainline_iou in max_mainline_ious:
            for candidate_nms_iou in candidate_nms_ious:
                for length_penalty in length_penalties:
                    for max_rescues_per_video in max_rescues_per_videos:
                        fused = select_candidate_rescue_predictions(
                            mainline_predictions,
                            candidate_predictions,
                            candidate_scores,
                            prob_threshold=float(prob_threshold),
                            max_mainline_iou=None if max_mainline_iou is None else float(max_mainline_iou),
                            candidate_nms_iou=candidate_nms_iou,
                            length_penalty=float(length_penalty),
                            max_rescues_per_video=int(max_rescues_per_video),
                        )
                        metrics = evaluate_fused_predictions(fused, labels, iou_threshold=iou_threshold)
                        rescue_count = sum(max(0, len(after) - len(before)) for before, after in zip(mainline_predictions, fused))
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
                            best_predictions = fused
                            best_config = {
                                "enabled": True,
                                "prob_threshold": float(prob_threshold),
                                "max_mainline_iou": None if max_mainline_iou is None else float(max_mainline_iou),
                                "candidate_nms_iou": candidate_nms_iou,
                                "length_penalty": float(length_penalty),
                                "max_rescues_per_video": int(max_rescues_per_video),
                                "rescue_segments": int(rescue_count),
                                "base_metrics": base_metrics,
                            }
    return best_config, best_metrics, best_predictions
