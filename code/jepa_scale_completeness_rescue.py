#!/usr/bin/env python3
"""JEPA evidence-completeness rescue for missed error segments.

This rescuer is deliberately conservative: a candidate must have selector
support, dense JEPA evidence inside the proposal, and a clear contrast against
its local context before it can be added to the current fused prediction set.
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
    return max(0, end - start + 1)


def _max_iou(segment: tuple[int, int], segments: Iterable[tuple[int, int]]) -> float:
    norm = _normalise_segment(segment)
    return max((interval_iou(norm, _normalise_segment(other)) for other in segments), default=0.0)


def _segment_evidence(evidence: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    start, end = _normalise_segment(segment)
    start = max(0, start)
    end = min(len(values) - 1, end)
    if end < start:
        return np.zeros(0, dtype=np.float32)
    return values[start : end + 1]


def _context_evidence(evidence: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    start, end = _normalise_segment(segment)
    length = max(1, end - start + 1)
    left_start = max(0, start - length)
    left_end = max(-1, start - 1)
    right_start = min(len(values), end + 1)
    right_end = min(len(values) - 1, end + length)
    parts: list[np.ndarray] = []
    if left_end >= left_start:
        parts.append(values[left_start : left_end + 1])
    if right_end >= right_start:
        parts.append(values[right_start : right_end + 1])
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def _completeness_features(
    evidence: np.ndarray,
    segment: tuple[int, int],
    evidence_threshold: float,
) -> tuple[float, float, float]:
    inside = _segment_evidence(evidence, segment)
    if len(inside) == 0:
        return 0.0, 0.0, -1.0
    context = _context_evidence(evidence, segment)
    inside_mean = float(inside.mean())
    active_fraction = float((inside >= float(evidence_threshold)).mean())
    context_mean = float(context.mean()) if len(context) else 0.0
    contrast = inside_mean - context_mean
    return inside_mean, active_fraction, contrast


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


def select_scale_completeness_rescue_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    prob_threshold: float,
    evidence_threshold: float,
    min_active_fraction: float,
    min_mean: float,
    min_contrast: float,
    max_base_iou: float,
    length_penalty: float,
    nms_iou: float | None,
    max_rescues_per_video: int,
) -> list[list[Segment]]:
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
    ):
        raise ValueError("base/candidate/score/evidence video counts must match")

    fused: list[list[Segment]] = []
    for base_segments, candidates, scores, evidence in zip(
        base_predictions,
        candidate_predictions,
        candidate_scores,
        evidence_arrays,
    ):
        base = sorted({_normalise_segment(segment) for segment in base_segments})
        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(candidates) != len(scores_arr):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores_arr)}")

        rescue_candidates: list[Segment] = []
        rescue_scores: list[float] = []
        for segment, score in zip(candidates, scores_arr):
            norm = _normalise_segment(segment)
            if float(score) < float(prob_threshold):
                continue
            if _max_iou(norm, base) > float(max_base_iou):
                continue

            inside_mean, active_fraction, contrast = _completeness_features(
                evidence,
                norm,
                evidence_threshold=float(evidence_threshold),
            )
            if active_fraction < float(min_active_fraction):
                continue
            if inside_mean < float(min_mean):
                continue
            if contrast < float(min_contrast):
                continue

            length = max(1, _segment_length(norm))
            adjusted = (
                float(score)
                + inside_mean
                + contrast
                + 0.25 * active_fraction
                - float(length_penalty) * math.log1p(length)
            )
            rescue_candidates.append(norm)
            rescue_scores.append(adjusted)

        selected = _score_first_nms(
            rescue_candidates,
            rescue_scores,
            iou_threshold=nms_iou,
            max_items=max_rescues_per_video,
        )
        fused.append(sorted(set(base + selected)))
    return fused


def select_scale_completeness_rescue_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    prob_thresholds: Iterable[float] = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6),
    evidence_thresholds: Iterable[float] = (0.6, 0.7, 0.8),
    min_active_fractions: Iterable[float] = (0.5, 0.67, 0.8),
    min_means: Iterable[float] = (0.55, 0.65, 0.75),
    min_contrasts: Iterable[float] = (0.1, 0.2, 0.3),
    max_base_ious: Iterable[float] = (0.0, 0.1, 0.25),
    length_penalties: Iterable[float] = (0.0, 0.005),
    nms_ious: Iterable[float | None] = (None, 0.3),
    max_rescues_per_videos: Iterable[int] = (1, 2),
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
        for evidence_threshold in evidence_thresholds:
            for min_active_fraction in min_active_fractions:
                for min_mean in min_means:
                    for min_contrast in min_contrasts:
                        for max_base_iou in max_base_ious:
                            for length_penalty in length_penalties:
                                for nms_iou in nms_ious:
                                    for max_rescues_per_video in max_rescues_per_videos:
                                        fused = select_scale_completeness_rescue_predictions(
                                            base_predictions_norm,
                                            candidate_predictions,
                                            candidate_scores,
                                            evidence_arrays,
                                            prob_threshold=float(prob_threshold),
                                            evidence_threshold=float(evidence_threshold),
                                            min_active_fraction=float(min_active_fraction),
                                            min_mean=float(min_mean),
                                            min_contrast=float(min_contrast),
                                            max_base_iou=float(max_base_iou),
                                            length_penalty=float(length_penalty),
                                            nms_iou=nms_iou,
                                            max_rescues_per_video=int(max_rescues_per_video),
                                        )
                                        metrics = evaluate_fused_predictions(
                                            fused,
                                            labels,
                                            iou_threshold=iou_threshold,
                                        )
                                        rescue_count = sum(
                                            max(0, len(after) - len(before))
                                            for before, after in zip(base_predictions_norm, fused)
                                        )
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
                                                "evidence_threshold": float(evidence_threshold),
                                                "min_active_fraction": float(min_active_fraction),
                                                "min_mean": float(min_mean),
                                                "min_contrast": float(min_contrast),
                                                "max_base_iou": float(max_base_iou),
                                                "length_penalty": float(length_penalty),
                                                "nms_iou": nms_iou,
                                                "max_rescues_per_video": int(max_rescues_per_video),
                                                "rescue_segments": int(rescue_count),
                                                "base_metrics": base_metrics,
                                            }
    return best_config, best_metrics, best_predictions
