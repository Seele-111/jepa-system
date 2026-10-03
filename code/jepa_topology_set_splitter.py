#!/usr/bin/env python3
"""Set-level topology splitter for broad JEPA-fused predictions.

Unlike per-child threshold rescue, this module evaluates a compact child event
set inside each broad parent. It is designed for the observed failure mode where
real sub-events may have modest individual JEPA/selector evidence, but the set
of non-overlapping child proposals explains the parent topology better than one
merged segment.
"""
from __future__ import annotations

import math
from itertools import product
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]


@dataclass(frozen=True)
class SetChild:
    segment: Segment
    selector_score: float
    selector_rank: int
    evidence_mean: float
    active_fraction: float
    parent_contrast: float
    parent_ratio: float
    parent_coverage: float
    score: float


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


def _coverage(child: tuple[int, int], parent: tuple[int, int]) -> float:
    return float(_intersection(child, parent) / max(1, _segment_length(child)))


def _matrix(evidence: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values.astype(np.float32)


def _segment_matrix(evidence: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    values = _matrix(evidence)
    if len(values) == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    start, end = _normalise_segment(segment)
    start = max(0, min(start, len(values) - 1))
    end = max(start, min(end, len(values) - 1))
    return values[start : end + 1].astype(np.float32)


def _parent_residual_matrix(evidence: np.ndarray, parent: tuple[int, int], child: tuple[int, int]) -> np.ndarray:
    values = _matrix(evidence)
    if len(values) == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    parent_start, parent_end = _normalise_segment(parent)
    child_start, child_end = _normalise_segment(child)
    parent_start = max(0, min(parent_start, len(values) - 1))
    parent_end = max(parent_start, min(parent_end, len(values) - 1))
    left = values[parent_start : max(parent_start, min(child_start, parent_end + 1))]
    right_start = max(parent_start, min(child_end + 1, parent_end + 1))
    right = values[right_start : parent_end + 1]
    if len(left) == 0 and len(right) == 0:
        return values[parent_start : parent_end + 1].astype(np.float32)
    return np.concatenate([left, right], axis=0).astype(np.float32)


def _evidence_features(evidence: np.ndarray, parent: tuple[int, int], child: tuple[int, int], evidence_threshold: float) -> tuple[float, float, float]:
    inside = _segment_matrix(evidence, child)
    if len(inside) == 0:
        return 0.0, 0.0, -1.0
    per_frame = inside.mean(axis=1)
    evidence_mean = float(per_frame.mean())
    active_fraction = float((per_frame >= float(evidence_threshold)).mean())
    residual = _parent_residual_matrix(evidence, parent, child)
    residual_baseline = float(np.median(residual.mean(axis=1))) if len(residual) else 0.0
    return evidence_mean, active_fraction, evidence_mean - residual_baseline


def _rank_desc(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def _score_first_nms(children: list[SetChild], nms_iou: float | None, max_items: int) -> list[SetChild]:
    ordered = sorted(
        children,
        key=lambda item: (item.score, item.evidence_mean, item.selector_score, -_segment_length(item.segment)),
        reverse=True,
    )
    kept: list[SetChild] = []
    for child in ordered:
        if nms_iou is not None and any(interval_iou(child.segment, item.segment) > float(nms_iou) for item in kept):
            continue
        kept.append(child)
        if len(kept) >= int(max_items):
            break
    return sorted(kept, key=lambda item: item.segment)


def _children_are_separated(children: list[SetChild], min_gap_between_children: int) -> bool:
    ordered = sorted(child.segment for child in children)
    for left, right in zip(ordered, ordered[1:]):
        if right[0] - left[1] - 1 < int(min_gap_between_children):
            return False
    return True


def _parent_children(
    parent: Segment,
    candidates: list[Segment],
    scores: np.ndarray,
    ranks: np.ndarray,
    evidence: np.ndarray,
    evidence_threshold: float,
    min_selector_score: float,
    max_selector_rank: int | None,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_parent_contrast: float,
    max_child_parent_ratio: float,
    min_child_parent_coverage: float,
    child_score_threshold: float,
    selector_weight: float,
    evidence_weight: float,
    active_fraction_weight: float,
    contrast_weight: float,
    length_penalty: float,
) -> list[SetChild]:
    parent_length = _segment_length(parent)
    children: list[SetChild] = []
    for candidate, selector_score_raw, selector_rank_raw in zip(candidates, scores, ranks):
        child = _normalise_segment(candidate)
        child_length = _segment_length(child)
        if child == parent or child_length >= parent_length:
            continue
        parent_ratio = float(child_length / max(1, parent_length))
        if parent_ratio > float(max_child_parent_ratio):
            continue
        parent_coverage = _coverage(child, parent)
        if parent_coverage < float(min_child_parent_coverage):
            continue
        selector_score = float(selector_score_raw)
        selector_rank = int(selector_rank_raw)
        if selector_score < float(min_selector_score):
            continue
        if max_selector_rank is not None and selector_rank > int(max_selector_rank):
            continue
        evidence_mean, active_fraction, parent_contrast = _evidence_features(
            evidence,
            parent,
            child,
            evidence_threshold=float(evidence_threshold),
        )
        if evidence_mean < float(min_evidence_mean):
            continue
        if active_fraction < float(min_active_fraction):
            continue
        if parent_contrast < float(min_parent_contrast):
            continue
        score = (
            float(selector_weight) * selector_score
            + float(evidence_weight) * evidence_mean
            + float(active_fraction_weight) * active_fraction
            + float(contrast_weight) * parent_contrast
            - float(length_penalty) * math.log1p(child_length)
        )
        if score < float(child_score_threshold):
            continue
        children.append(
            SetChild(
                segment=child,
                selector_score=selector_score,
                selector_rank=selector_rank,
                evidence_mean=float(evidence_mean),
                active_fraction=float(active_fraction),
                parent_contrast=float(parent_contrast),
                parent_ratio=parent_ratio,
                parent_coverage=parent_coverage,
                score=float(score),
            )
        )
    return children


def split_topology_set_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    evidence_threshold: float,
    min_selector_score: float,
    max_selector_rank: int | None,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_parent_contrast: float,
    parent_min_length: int,
    max_child_parent_ratio: float,
    min_child_parent_coverage: float,
    min_gap_between_children: int,
    child_score_threshold: float,
    set_score_threshold: float,
    selector_weight: float,
    evidence_weight: float,
    active_fraction_weight: float,
    contrast_weight: float,
    length_penalty: float,
    nms_iou: float | None,
    min_children_per_parent: int,
    max_children_per_parent: int,
    max_replaced_parents_per_video: int,
) -> list[list[Segment]]:
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
    ):
        raise ValueError("base/candidate/score/evidence video counts must match")

    out: list[list[Segment]] = []
    for base_raw, candidates_raw, scores_raw, evidence in zip(base_predictions, candidate_predictions, candidate_scores, evidence_arrays):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        ranks = _rank_desc(scores)
        replacements: list[tuple[Segment, list[SetChild], float]] = []
        for parent in base:
            if _segment_length(parent) < int(parent_min_length):
                continue
            children = _parent_children(
                parent,
                candidates,
                scores,
                ranks,
                evidence,
                evidence_threshold=float(evidence_threshold),
                min_selector_score=float(min_selector_score),
                max_selector_rank=max_selector_rank,
                min_evidence_mean=float(min_evidence_mean),
                min_active_fraction=float(min_active_fraction),
                min_parent_contrast=float(min_parent_contrast),
                max_child_parent_ratio=float(max_child_parent_ratio),
                min_child_parent_coverage=float(min_child_parent_coverage),
                child_score_threshold=float(child_score_threshold),
                selector_weight=float(selector_weight),
                evidence_weight=float(evidence_weight),
                active_fraction_weight=float(active_fraction_weight),
                contrast_weight=float(contrast_weight),
                length_penalty=float(length_penalty),
            )
            selected = _score_first_nms(children, nms_iou=nms_iou, max_items=int(max_children_per_parent))
            if len(selected) < int(min_children_per_parent):
                continue
            if not _children_are_separated(selected, min_gap_between_children=int(min_gap_between_children)):
                continue
            set_score = float(sum(child.score for child in selected) / max(1, len(selected)))
            set_score += 0.05 * float(len(selected) - 1)
            if set_score < float(set_score_threshold):
                continue
            replacements.append((parent, selected, set_score))
        replacements.sort(key=lambda item: item[2], reverse=True)
        if max_replaced_parents_per_video > 0:
            replacements = replacements[: int(max_replaced_parents_per_video)]
        replace_parents = {parent for parent, _, _ in replacements}
        children = [child.segment for _, selected, _ in replacements for child in selected]
        kept = [segment for segment in base if segment not in replace_parents]
        out.append(sorted(set(kept + children)))
    return out


def select_topology_set_split_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    evidence_thresholds: Iterable[float] = (0.4, 0.5, 0.6),
    min_selector_scores: Iterable[float] = (0.0, 0.05, 0.1),
    max_selector_ranks: Iterable[int | None] = (None, 80),
    min_evidence_means: Iterable[float] = (0.2, 0.3, 0.4),
    min_active_fractions: Iterable[float] = (0.0, 0.1, 0.25),
    min_parent_contrasts: Iterable[float] = (-0.3, -0.1, 0.0),
    parent_min_lengths: Iterable[int] = (12, 24),
    max_child_parent_ratios: Iterable[float] = (0.5, 0.7),
    min_child_parent_coverages: Iterable[float] = (0.5, 0.7),
    min_gap_between_children_values: Iterable[int] = (0,),
    child_score_thresholds: Iterable[float] = (0.0, 0.2),
    set_score_thresholds: Iterable[float] = (0.1, 0.2, 0.3),
    selector_weights: Iterable[float] = (0.5, 1.0),
    evidence_weights: Iterable[float] = (0.5, 1.0),
    active_fraction_weights: Iterable[float] = (0.0, 0.25),
    contrast_weights: Iterable[float] = (0.0, 0.5),
    length_penalties: Iterable[float] = (0.0, 0.005),
    nms_ious: Iterable[float | None] = (0.3,),
    min_children_per_parents: Iterable[int] = (2,),
    max_children_per_parents: Iterable[int] = (2, 3),
    max_replaced_parents_per_videos: Iterable[int] = (1,),
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=iou_threshold)
    best_config: dict = {"enabled": False, "base_metrics": base_metrics}
    best_metrics = base_metrics
    best_predictions = base
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    search_space = product(
        evidence_thresholds,
        min_selector_scores,
        max_selector_ranks,
        min_evidence_means,
        min_active_fractions,
        min_parent_contrasts,
        parent_min_lengths,
        max_child_parent_ratios,
        min_child_parent_coverages,
        min_gap_between_children_values,
        child_score_thresholds,
        set_score_thresholds,
        selector_weights,
        evidence_weights,
        active_fraction_weights,
        contrast_weights,
        length_penalties,
        nms_ious,
        min_children_per_parents,
        max_children_per_parents,
        max_replaced_parents_per_videos,
    )
    for (
        evidence_threshold,
        min_selector_score,
        max_selector_rank,
        min_evidence_mean,
        min_active_fraction,
        min_parent_contrast,
        parent_min_length,
        max_child_parent_ratio,
        min_child_parent_coverage,
        min_gap_between_children,
        child_score_threshold,
        set_score_threshold,
        selector_weight,
        evidence_weight,
        active_fraction_weight,
        contrast_weight,
        length_penalty,
        nms_iou,
        min_children_per_parent,
        max_children_per_parent,
        max_replaced_parents_per_video,
    ) in search_space:
        split = split_topology_set_predictions(
            base,
            candidate_predictions,
            candidate_scores,
            evidence_arrays,
            evidence_threshold=float(evidence_threshold),
            min_selector_score=float(min_selector_score),
            max_selector_rank=max_selector_rank,
            min_evidence_mean=float(min_evidence_mean),
            min_active_fraction=float(min_active_fraction),
            min_parent_contrast=float(min_parent_contrast),
            parent_min_length=int(parent_min_length),
            max_child_parent_ratio=float(max_child_parent_ratio),
            min_child_parent_coverage=float(min_child_parent_coverage),
            min_gap_between_children=int(min_gap_between_children),
            child_score_threshold=float(child_score_threshold),
            set_score_threshold=float(set_score_threshold),
            selector_weight=float(selector_weight),
            evidence_weight=float(evidence_weight),
            active_fraction_weight=float(active_fraction_weight),
            contrast_weight=float(contrast_weight),
            length_penalty=float(length_penalty),
            nms_iou=nms_iou,
            min_children_per_parent=int(min_children_per_parent),
            max_children_per_parent=int(max_children_per_parent),
            max_replaced_parents_per_video=int(max_replaced_parents_per_video),
        )
        metrics = evaluate_fused_predictions(split, labels, iou_threshold=iou_threshold)
        if max_fp_increase is not None:
            fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
            if fp_delta > int(max_fp_increase):
                continue
        changed = sum(int(sorted(a) != sorted(b)) for a, b in zip(base, split))
        segment_delta = sum(len(b) - len(a) for a, b in zip(base, split))
        key = (
            float(metrics["segment"]["f1"]),
            float(metrics["segment"]["precision"]),
            float(metrics["segment"]["recall"]),
            float(metrics["frame"]["f1"]),
            -float(changed + 0.01 * max(0, segment_delta)),
        )
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_predictions = split
            best_config = {
                "enabled": True,
                "evidence_threshold": float(evidence_threshold),
                "min_selector_score": float(min_selector_score),
                "max_selector_rank": max_selector_rank,
                "min_evidence_mean": float(min_evidence_mean),
                "min_active_fraction": float(min_active_fraction),
                "min_parent_contrast": float(min_parent_contrast),
                "parent_min_length": int(parent_min_length),
                "max_child_parent_ratio": float(max_child_parent_ratio),
                "min_child_parent_coverage": float(min_child_parent_coverage),
                "min_gap_between_children": int(min_gap_between_children),
                "child_score_threshold": float(child_score_threshold),
                "set_score_threshold": float(set_score_threshold),
                "selector_weight": float(selector_weight),
                "evidence_weight": float(evidence_weight),
                "active_fraction_weight": float(active_fraction_weight),
                "contrast_weight": float(contrast_weight),
                "length_penalty": float(length_penalty),
                "nms_iou": nms_iou,
                "min_children_per_parent": int(min_children_per_parent),
                "max_children_per_parent": int(max_children_per_parent),
                "max_replaced_parents_per_video": int(max_replaced_parents_per_video),
                "max_fp_increase": max_fp_increase,
                "changed_videos": int(changed),
                "segment_delta": int(segment_delta),
                "base_metrics": base_metrics,
            }
    return best_config, best_metrics, best_predictions
