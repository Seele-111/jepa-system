#!/usr/bin/env python3
"""Proposal-aware replacement for merged mainline JEPA error segments."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_set_selector import select_weighted_proposal_set


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(0, end - start + 1)


def _intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    a_start, a_end = _normalise_segment(a)
    b_start, b_end = _normalise_segment(b)
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _candidate_parent_coverage(candidate: tuple[int, int], parent: tuple[int, int]) -> float:
    return _intersection(candidate, parent) / max(1, _segment_length(candidate))


def _candidates_for_parent(
    parent: Segment,
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    prob_threshold: float,
    min_candidate_parent_coverage: float,
    max_candidate_parent_ratio: float,
    length_penalty: float,
    max_replacements_per_parent: int,
) -> list[Segment]:
    parent_len = max(1, _segment_length(parent))
    filtered_segments: list[Segment] = []
    filtered_scores: list[float] = []
    for candidate_raw, score_raw in zip(candidates, scores):
        candidate = _normalise_segment(candidate_raw)
        candidate_len = max(1, _segment_length(candidate))
        adjusted_score = float(score_raw) - float(length_penalty) * math.log1p(candidate_len)
        if adjusted_score < float(prob_threshold):
            continue
        if _candidate_parent_coverage(candidate, parent) < float(min_candidate_parent_coverage):
            continue
        if candidate_len / parent_len > float(max_candidate_parent_ratio):
            continue
        filtered_segments.append(candidate)
        filtered_scores.append(adjusted_score)
    selected = select_weighted_proposal_set(
        filtered_segments,
        filtered_scores,
        length_penalty=0.0,
    )
    if max_replacements_per_parent > 0:
        selected = sorted(
            selected,
            key=lambda segment: max(
                (score for cand, score in zip(filtered_segments, filtered_scores) if cand == segment),
                default=0.0,
            ),
            reverse=True,
        )[: int(max_replacements_per_parent)]
    return sorted({_normalise_segment(segment) for segment in selected})


def replace_merged_predictions(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    prob_threshold: float,
    parent_min_length: int,
    min_replacements: int,
    max_replacements_per_parent: int,
    min_candidate_parent_coverage: float,
    max_candidate_parent_ratio: float,
    length_penalty: float,
) -> list[list[Segment]]:
    """Replace long merged mainline intervals with a selected JEPA proposal set."""
    if len(mainline_predictions) != len(candidate_predictions) or len(mainline_predictions) != len(candidate_scores):
        raise ValueError("mainline/candidate/score video counts must match")
    replaced_videos: list[list[Segment]] = []
    for main_segments, candidates, scores_raw in zip(mainline_predictions, candidate_predictions, candidate_scores):
        scores = np.asarray(scores_raw, dtype=np.float32).reshape(-1)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        video_segments: list[Segment] = []
        for parent_raw in main_segments:
            parent = _normalise_segment(parent_raw)
            if _segment_length(parent) < int(parent_min_length):
                video_segments.append(parent)
                continue
            replacements = _candidates_for_parent(
                parent,
                candidates,
                scores,
                prob_threshold=prob_threshold,
                min_candidate_parent_coverage=min_candidate_parent_coverage,
                max_candidate_parent_ratio=max_candidate_parent_ratio,
                length_penalty=length_penalty,
                max_replacements_per_parent=max_replacements_per_parent,
            )
            if len(replacements) >= int(min_replacements):
                video_segments.extend(replacements)
            else:
                video_segments.append(parent)
        replaced_videos.append(sorted(set(video_segments)))
    return replaced_videos


def select_proposal_replacement_params(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    labels: list[np.ndarray],
    prob_thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    parent_min_lengths: Iterable[int] = (24, 48, 72),
    min_replacements_list: Iterable[int] = (2, 3),
    max_replacements_per_parents: Iterable[int] = (2, 3, 4),
    min_candidate_parent_coverages: Iterable[float] = (0.8, 0.9, 1.0),
    max_candidate_parent_ratios: Iterable[float] = (0.25, 0.4, 0.6, 0.8),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01, 0.02),
    max_fp_increase: int | None = None,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base_predictions = [[_normalise_segment(segment) for segment in video] for video in mainline_predictions]
    base_metrics = evaluate_fused_predictions(base_predictions, labels, iou_threshold=iou_threshold)
    best_config: dict = {
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
    for prob_threshold in prob_thresholds:
        for parent_min_length in parent_min_lengths:
            for min_replacements in min_replacements_list:
                for max_replacements_per_parent in max_replacements_per_parents:
                    for min_candidate_parent_coverage in min_candidate_parent_coverages:
                        for max_candidate_parent_ratio in max_candidate_parent_ratios:
                            for length_penalty in length_penalties:
                                replaced = replace_merged_predictions(
                                    base_predictions,
                                    candidate_predictions,
                                    candidate_scores,
                                    prob_threshold=float(prob_threshold),
                                    parent_min_length=int(parent_min_length),
                                    min_replacements=int(min_replacements),
                                    max_replacements_per_parent=int(max_replacements_per_parent),
                                    min_candidate_parent_coverage=float(min_candidate_parent_coverage),
                                    max_candidate_parent_ratio=float(max_candidate_parent_ratio),
                                    length_penalty=float(length_penalty),
                                )
                                metrics = evaluate_fused_predictions(replaced, labels, iou_threshold=iou_threshold)
                                if max_fp_increase is not None:
                                    fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                                    if fp_delta > int(max_fp_increase):
                                        continue
                                changed = sum(
                                    int(sorted(before) != sorted(after))
                                    for before, after in zip(base_predictions, replaced)
                                )
                                replacement_count = sum(
                                    max(0, len(after) - len(before))
                                    for before, after in zip(base_predictions, replaced)
                                )
                                key = (
                                    float(metrics["segment"]["f1"]),
                                    float(metrics["segment"]["precision"]),
                                    float(metrics["segment"]["recall"]),
                                    float(metrics["frame"]["f1"]),
                                    -float(changed + 0.01 * replacement_count),
                                )
                                if key > best_key:
                                    best_key = key
                                    best_metrics = metrics
                                    best_predictions = replaced
                                    best_config = {
                                        "enabled": True,
                                        "prob_threshold": float(prob_threshold),
                                        "parent_min_length": int(parent_min_length),
                                        "min_replacements": int(min_replacements),
                                        "max_replacements_per_parent": int(max_replacements_per_parent),
                                        "min_candidate_parent_coverage": float(min_candidate_parent_coverage),
                                        "max_candidate_parent_ratio": float(max_candidate_parent_ratio),
                                        "length_penalty": float(length_penalty),
                                        "changed_videos": int(changed),
                                        "replacement_delta": int(replacement_count),
                                        "max_fp_increase": max_fp_increase,
                                        "base_metrics": base_metrics,
                                    }
    return best_config, best_metrics, best_predictions
