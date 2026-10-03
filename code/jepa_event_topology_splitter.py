#!/usr/bin/env python3
"""JEPA event-topology splitter for wide merged error predictions.

The mainline temporal model can produce one broad segment that matches one
ground-truth event while swallowing nearby error events. This module treats the
V/I/dual-JEPA candidate set as an event graph and replaces such broad parents
with multiple compact, mutually separated JEPA-supported child events only when
the split improves validation segment localization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import evaluate_segment_predictions, interval_iou
from train_segment_locator import contiguous_segments


Segment = tuple[int, int]


@dataclass(frozen=True)
class TopologyChild:
    parent: Segment
    child: Segment
    selector_score: float
    topology_score: float
    child_length: int
    child_parent_ratio: float
    child_parent_coverage: float
    evidence_mean: float
    active_fraction: float
    parent_contrast: float
    support_count: int


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


def _evidence_features(
    evidence: np.ndarray,
    parent: tuple[int, int],
    child: tuple[int, int],
    evidence_threshold: float,
) -> tuple[float, float, float]:
    inside = _segment_matrix(evidence, child)
    if len(inside) == 0:
        return 0.0, 0.0, -1.0
    per_frame = inside.mean(axis=1)
    evidence_mean = float(per_frame.mean())
    active_fraction = float((per_frame >= float(evidence_threshold)).mean())
    residual = _parent_residual_matrix(evidence, parent, child)
    residual_baseline = float(np.median(residual.mean(axis=1))) if len(residual) else 0.0
    return evidence_mean, active_fraction, evidence_mean - residual_baseline


def _support_count(
    child: tuple[int, int],
    candidates: list[Segment],
    support_iou: float,
) -> int:
    norm = _normalise_segment(child)
    return int(
        sum(
            1
            for other in candidates
            if other != norm and interval_iou(norm, other) >= float(support_iou)
        )
    )


def _children_are_separated(children: list[Segment], min_gap_between_children: int) -> bool:
    ordered = sorted({_normalise_segment(child) for child in children})
    for left, right in zip(ordered, ordered[1:]):
        if right[0] - left[1] - 1 < int(min_gap_between_children):
            return False
    return True


def _score_first_nms(
    children: list[TopologyChild],
    nms_iou: float | None,
    max_items: int,
) -> list[TopologyChild]:
    ordered = sorted(
        children,
        key=lambda item: (item.topology_score, -item.child_length, -item.child[0], -item.child[1]),
        reverse=True,
    )
    kept: list[TopologyChild] = []
    for item in ordered:
        if nms_iou is not None and any(interval_iou(item.child, kept_item.child) > float(nms_iou) for kept_item in kept):
            continue
        kept.append(item)
        if len(kept) >= int(max_items):
            break
    return sorted(kept, key=lambda item: item.child)


def _candidate_children_for_parent(
    parent: Segment,
    candidates: list[Segment],
    scores: np.ndarray,
    evidence: np.ndarray,
    min_selector_score: float,
    evidence_threshold: float,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_parent_contrast: float,
    max_child_parent_ratio: float,
    min_child_parent_coverage: float,
    support_iou: float,
    support_count_weight: float,
    length_penalty: float,
) -> list[TopologyChild]:
    parent_length = _segment_length(parent)
    children: list[TopologyChild] = []
    for child_raw, selector_score_raw in zip(candidates, scores):
        child = _normalise_segment(child_raw)
        if child == parent:
            continue
        child_length = _segment_length(child)
        if child_length >= parent_length:
            continue
        child_parent_ratio = float(child_length / max(1, parent_length))
        if child_parent_ratio > float(max_child_parent_ratio):
            continue
        child_parent_coverage = _coverage(child, parent)
        if child_parent_coverage < float(min_child_parent_coverage):
            continue
        selector_score = float(selector_score_raw)
        if selector_score < float(min_selector_score):
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
        support_count = (
            _support_count(child, candidates, support_iou=float(support_iou))
            if float(support_count_weight) != 0.0
            else 0
        )
        topology_score = (
            selector_score
            + evidence_mean
            + active_fraction
            + parent_contrast
            + float(support_count_weight) * math.log1p(support_count)
            - float(length_penalty) * math.log1p(child_length)
        )
        children.append(
            TopologyChild(
                parent=parent,
                child=child,
                selector_score=selector_score,
                topology_score=float(topology_score),
                child_length=int(child_length),
                child_parent_ratio=child_parent_ratio,
                child_parent_coverage=child_parent_coverage,
                evidence_mean=float(evidence_mean),
                active_fraction=float(active_fraction),
                parent_contrast=float(parent_contrast),
                support_count=int(support_count),
            )
        )
    return children


def split_event_topology_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    min_selector_score: float,
    evidence_threshold: float,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_parent_contrast: float,
    parent_min_length: int,
    max_child_parent_ratio: float,
    min_child_parent_coverage: float,
    min_gap_between_children: int,
    support_iou: float,
    support_count_weight: float,
    length_penalty: float,
    nms_iou: float | None,
    min_children_per_parent: int,
    max_children_per_parent: int,
    max_replaced_parents_per_video: int,
) -> list[list[Segment]]:
    """Replace broad parents with multiple compact JEPA-supported child events."""
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
    ):
        raise ValueError("base/candidate/score/evidence video counts must match")

    split_videos: list[list[Segment]] = []
    for base_raw, candidates_raw, scores_raw, evidence in zip(
        base_predictions,
        candidate_predictions,
        candidate_scores,
        evidence_arrays,
    ):
        base = sorted({_normalise_segment(segment) for segment in base_raw})
        candidates = [_normalise_segment(segment) for segment in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        parent_replacements: list[tuple[Segment, list[TopologyChild], float]] = []
        for parent in base:
            if _segment_length(parent) < int(parent_min_length):
                continue
            children = _candidate_children_for_parent(
                parent,
                candidates,
                scores,
                evidence,
                min_selector_score=float(min_selector_score),
                evidence_threshold=float(evidence_threshold),
                min_evidence_mean=float(min_evidence_mean),
                min_active_fraction=float(min_active_fraction),
                min_parent_contrast=float(min_parent_contrast),
                max_child_parent_ratio=float(max_child_parent_ratio),
                min_child_parent_coverage=float(min_child_parent_coverage),
                support_iou=float(support_iou),
                support_count_weight=float(support_count_weight),
                length_penalty=float(length_penalty),
            )
            selected_children = _score_first_nms(
                children,
                nms_iou=nms_iou,
                max_items=int(max_children_per_parent),
            )
            child_segments = [item.child for item in selected_children]
            if len(child_segments) < int(min_children_per_parent):
                continue
            if not _children_are_separated(child_segments, min_gap_between_children=int(min_gap_between_children)):
                continue
            replacement_score = float(sum(item.topology_score for item in selected_children))
            parent_replacements.append((parent, selected_children, replacement_score))

        parent_replacements.sort(key=lambda item: item[2], reverse=True)
        if max_replaced_parents_per_video > 0:
            parent_replacements = parent_replacements[: int(max_replaced_parents_per_video)]
        parents_to_replace = {parent for parent, _, _ in parent_replacements}
        child_segments = [child.child for _, children, _ in parent_replacements for child in children]
        kept = [segment for segment in base if segment not in parents_to_replace]
        split_videos.append(sorted(set(kept + child_segments)))
    return split_videos


def select_event_topology_split_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    min_selector_scores: Iterable[float] = (0.4, 0.5, 0.6),
    evidence_thresholds: Iterable[float] = (0.6, 0.7, 0.8),
    min_evidence_means: Iterable[float] = (0.55, 0.65, 0.75),
    min_active_fractions: Iterable[float] = (0.5, 0.67, 0.8),
    min_parent_contrasts: Iterable[float] = (0.05, 0.1, 0.2),
    parent_min_lengths: Iterable[int] = (24, 48, 72),
    max_child_parent_ratios: Iterable[float] = (0.25, 0.4, 0.6),
    min_child_parent_coverages: Iterable[float] = (0.8, 0.9),
    min_gap_between_children_values: Iterable[int] = (0, 2, 4),
    support_ious: Iterable[float] = (0.2, 0.3),
    support_count_weights: Iterable[float] = (0.0, 0.05, 0.1),
    length_penalties: Iterable[float] = (0.0, 0.005),
    nms_ious: Iterable[float | None] = (0.3,),
    min_children_per_parents: Iterable[int] = (2,),
    max_children_per_parents: Iterable[int] = (2, 3),
    max_replaced_parents_per_videos: Iterable[int] = (1, 2),
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    """Tune topology split parameters by validation F1 with FP protection."""
    base = [[_normalise_segment(segment) for segment in video] for video in base_predictions]
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

    for min_selector_score in min_selector_scores:
        for evidence_threshold in evidence_thresholds:
            for min_evidence_mean in min_evidence_means:
                for min_active_fraction in min_active_fractions:
                    for min_parent_contrast in min_parent_contrasts:
                        for parent_min_length in parent_min_lengths:
                            for max_child_parent_ratio in max_child_parent_ratios:
                                for min_child_parent_coverage in min_child_parent_coverages:
                                    for min_gap_between_children in min_gap_between_children_values:
                                        for support_iou in support_ious:
                                            for support_count_weight in support_count_weights:
                                                for length_penalty in length_penalties:
                                                    for nms_iou in nms_ious:
                                                        for min_children_per_parent in min_children_per_parents:
                                                            for max_children_per_parent in max_children_per_parents:
                                                                for max_replaced_parents_per_video in max_replaced_parents_per_videos:
                                                                    split = split_event_topology_predictions(
                                                                        base,
                                                                        candidate_predictions,
                                                                        candidate_scores,
                                                                        evidence_arrays,
                                                                        min_selector_score=float(min_selector_score),
                                                                        evidence_threshold=float(evidence_threshold),
                                                                        min_evidence_mean=float(min_evidence_mean),
                                                                        min_active_fraction=float(min_active_fraction),
                                                                        min_parent_contrast=float(min_parent_contrast),
                                                                        parent_min_length=int(parent_min_length),
                                                                        max_child_parent_ratio=float(max_child_parent_ratio),
                                                                        min_child_parent_coverage=float(min_child_parent_coverage),
                                                                        min_gap_between_children=int(min_gap_between_children),
                                                                        support_iou=float(support_iou),
                                                                        support_count_weight=float(support_count_weight),
                                                                        length_penalty=float(length_penalty),
                                                                        nms_iou=nms_iou,
                                                                        min_children_per_parent=int(min_children_per_parent),
                                                                        max_children_per_parent=int(max_children_per_parent),
                                                                        max_replaced_parents_per_video=int(max_replaced_parents_per_video),
                                                                    )
                                                                    metrics = evaluate_fused_predictions(
                                                                        split,
                                                                        labels,
                                                                        iou_threshold=iou_threshold,
                                                                    )
                                                                    if max_fp_increase is not None:
                                                                        fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                                                                        if fp_delta > int(max_fp_increase):
                                                                            continue
                                                                    changed = sum(
                                                                        int(sorted(before) != sorted(after))
                                                                        for before, after in zip(base, split)
                                                                    )
                                                                    replaced_delta = sum(len(after) - len(before) for before, after in zip(base, split))
                                                                    key = (
                                                                        float(metrics["segment"]["f1"]),
                                                                        float(metrics["segment"]["precision"]),
                                                                        float(metrics["segment"]["recall"]),
                                                                        float(metrics["frame"]["f1"]),
                                                                        -float(changed + 0.01 * max(0, replaced_delta)),
                                                                    )
                                                                    if key > best_key:
                                                                        best_key = key
                                                                        best_metrics = metrics
                                                                        best_predictions = split
                                                                        best_config = {
                                                                            "enabled": True,
                                                                            "min_selector_score": float(min_selector_score),
                                                                            "evidence_threshold": float(evidence_threshold),
                                                                            "min_evidence_mean": float(min_evidence_mean),
                                                                            "min_active_fraction": float(min_active_fraction),
                                                                            "min_parent_contrast": float(min_parent_contrast),
                                                                            "parent_min_length": int(parent_min_length),
                                                                            "max_child_parent_ratio": float(max_child_parent_ratio),
                                                                            "min_child_parent_coverage": float(min_child_parent_coverage),
                                                                            "min_gap_between_children": int(min_gap_between_children),
                                                                            "support_iou": float(support_iou),
                                                                            "support_count_weight": float(support_count_weight),
                                                                            "length_penalty": float(length_penalty),
                                                                            "nms_iou": nms_iou,
                                                                            "min_children_per_parent": int(min_children_per_parent),
                                                                            "max_children_per_parent": int(max_children_per_parent),
                                                                            "max_replaced_parents_per_video": int(max_replaced_parents_per_video),
                                                                            "max_fp_increase": max_fp_increase,
                                                                            "changed_videos": int(changed),
                                                                            "segment_delta": int(replaced_delta),
                                                                            "base_metrics": base_metrics,
                                                                        }
    return best_config, best_metrics, best_predictions


def diagnose_swallowed_events(
    predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    iou_threshold: float = 0.3,
) -> dict:
    """Count FNs that sit inside a broad prediction already matched to another GT."""
    segment_metrics = evaluate_segment_predictions(predictions, labels, iou_threshold=iou_threshold)
    swallowed_fn = 0
    videos_with_swallowed_fn = 0
    total_gt = 0
    matched_gt = 0
    for video_predictions, label_array in zip(predictions, labels):
        pred_segments = [_normalise_segment(segment) for segment in video_predictions]
        gt_segments = [_normalise_segment(segment) for segment in contiguous_segments(np.asarray(label_array, dtype=np.int64))]
        total_gt += len(gt_segments)
        used_gt: set[int] = set()
        matched_preds: list[Segment] = []
        for pred in pred_segments:
            best_idx = -1
            best_iou = 0.0
            for gt_idx, gt in enumerate(gt_segments):
                if gt_idx in used_gt:
                    continue
                iou = interval_iou(pred, gt)
                if iou > best_iou:
                    best_iou = float(iou)
                    best_idx = gt_idx
            if best_idx >= 0 and best_iou >= float(iou_threshold):
                used_gt.add(best_idx)
                matched_preds.append(pred)
        matched_gt += len(used_gt)
        local_swallowed = 0
        for gt_idx, gt in enumerate(gt_segments):
            if gt_idx in used_gt:
                continue
            if any(_coverage(gt, pred) >= 0.8 for pred in matched_preds):
                local_swallowed += 1
        if local_swallowed:
            videos_with_swallowed_fn += 1
            swallowed_fn += local_swallowed
    return {
        "segment": segment_metrics,
        "total_gt": int(total_gt),
        "matched_gt": int(matched_gt),
        "swallowed_fn": int(swallowed_fn),
        "videos_with_swallowed_fn": int(videos_with_swallowed_fn),
        "swallowed_fn_fraction_of_fn": float(swallowed_fn / max(1, int(segment_metrics["fn"]))),
    }
