#!/usr/bin/env python3
"""Mainline-guided JEPA boundary snapping.

This module uses JEPA proposals as boundary-completion evidence for existing
high-precision mainline segments. It is deliberately conservative: proposals
can expand or align an existing segment, but cannot introduce unrelated events.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou


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


def _mainline_coverage(mainline: tuple[int, int], candidate: tuple[int, int]) -> float:
    return _intersection(mainline, candidate) / max(1, _length(mainline))


def snap_segments_to_candidates(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    min_score: float,
    min_mainline_coverage: float,
    max_length_ratio: float,
) -> list[list[Segment]]:
    if len(mainline_predictions) != len(candidate_predictions) or len(mainline_predictions) != len(candidate_scores):
        raise ValueError("mainline/candidate/score video counts must match")
    snapped_videos: list[list[Segment]] = []
    for main_segments, candidates, scores in zip(mainline_predictions, candidate_predictions, candidate_scores):
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        snapped_segments: list[Segment] = []
        for main_segment_raw in main_segments:
            main_segment = _normalise_segment(main_segment_raw)
            main_len = max(1, _length(main_segment))
            best_segment = main_segment
            best_key = (0.0, 0.0, -float(main_len))
            for candidate_raw, score in zip(candidates, scores):
                candidate = _normalise_segment(candidate_raw)
                if float(score) < float(min_score):
                    continue
                coverage = _mainline_coverage(main_segment, candidate)
                if coverage < float(min_mainline_coverage):
                    continue
                ratio = _length(candidate) / main_len
                if ratio > float(max_length_ratio):
                    continue
                union_segment = (min(main_segment[0], candidate[0]), max(main_segment[1], candidate[1]))
                key = (float(score), interval_iou(main_segment, candidate), -float(_length(union_segment)))
                if key > best_key:
                    best_key = key
                    best_segment = union_segment
            snapped_segments.append(best_segment)
        snapped_videos.append(sorted(set(snapped_segments)))
    return snapped_videos


def select_boundary_snap_params(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    labels: list[np.ndarray],
    min_scores: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    min_mainline_coverages: Iterable[float] = (0.25, 0.5, 0.75, 0.9),
    max_length_ratios: Iterable[float] = (1.5, 2.0, 3.0, 4.0),
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base_metrics = evaluate_fused_predictions(mainline_predictions, labels, iou_threshold=iou_threshold)
    best_config = {
        "enabled": False,
        "base_metrics": base_metrics,
    }
    best_metrics = base_metrics
    best_predictions = [[_normalise_segment(seg) for seg in video] for video in mainline_predictions]
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for min_score in min_scores:
        for min_mainline_coverage in min_mainline_coverages:
            for max_length_ratio in max_length_ratios:
                snapped = snap_segments_to_candidates(
                    mainline_predictions,
                    candidate_predictions,
                    candidate_scores,
                    min_score=float(min_score),
                    min_mainline_coverage=float(min_mainline_coverage),
                    max_length_ratio=float(max_length_ratio),
                )
                metrics = evaluate_fused_predictions(snapped, labels, iou_threshold=iou_threshold)
                changed = sum(
                    int(sorted(map(_normalise_segment, before)) != sorted(map(_normalise_segment, after)))
                    for before, after in zip(mainline_predictions, snapped)
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
                    best_config = {
                        "enabled": True,
                        "min_score": float(min_score),
                        "min_mainline_coverage": float(min_mainline_coverage),
                        "max_length_ratio": float(max_length_ratio),
                        "changed_videos": int(changed),
                        "base_metrics": base_metrics,
                    }
                    best_metrics = metrics
                    best_predictions = snapped
    return best_config, best_metrics, best_predictions
