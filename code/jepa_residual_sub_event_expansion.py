#!/usr/bin/env python3
"""Residual JEPA sub-event expansion inside broad fused predictions.

The current system often predicts a broad anomalous span that covers one
ground-truth event but swallows additional local error events. This module keeps
the broad span and adds compact JEPA-supported sub-events inside it when the
local residual evidence is strong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]


@dataclass(frozen=True)
class ResidualSubEventRecord:
    video_idx: int
    parent: Segment
    child: Segment
    selector_score: float
    parent_length: int
    child_length: int
    child_parent_ratio: float
    child_parent_coverage: float
    evidence_mean: float
    parent_contrast: float
    active_fractions: dict[float, float]


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


def _coverage_of_child_by_parent(child: tuple[int, int], parent: tuple[int, int]) -> float:
    return float(_intersection(child, parent) / max(1, _segment_length(child)))


def _segment_values(values: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    evidence = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(evidence) == 0:
        return np.zeros(0, dtype=np.float32)
    start, end = _normalise_segment(segment)
    start = max(0, min(start, len(evidence) - 1))
    end = max(start, min(end, len(evidence) - 1))
    return evidence[start : end + 1].astype(np.float32)


def _parent_residual_values(values: np.ndarray, parent: tuple[int, int], child: tuple[int, int]) -> np.ndarray:
    evidence = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(evidence) == 0:
        return np.zeros(0, dtype=np.float32)
    parent_start, parent_end = _normalise_segment(parent)
    child_start, child_end = _normalise_segment(child)
    parent_start = max(0, min(parent_start, len(evidence) - 1))
    parent_end = max(parent_start, min(parent_end, len(evidence) - 1))
    left = evidence[parent_start : max(parent_start, min(child_start, parent_end + 1))]
    right_start = max(parent_start, min(child_end + 1, parent_end + 1))
    right = evidence[right_start : parent_end + 1]
    if len(left) == 0 and len(right) == 0:
        return evidence[parent_start : parent_end + 1].astype(np.float32)
    return np.concatenate([left, right]).astype(np.float32)


def _score_first_nms(
    candidates: list[Segment],
    scores: list[float],
    iou_threshold: float | None,
    max_items: int,
) -> list[tuple[Segment, float]]:
    items = sorted(
        [(segment, float(score), _segment_length(segment)) for segment, score in zip(candidates, scores)],
        key=lambda item: (item[1], -item[2], -item[0][0]),
        reverse=True,
    )
    kept: list[tuple[Segment, float]] = []
    for segment, score, _ in items:
        if iou_threshold is not None and any(interval_iou(segment, existing) > float(iou_threshold) for existing, _ in kept):
            continue
        kept.append((segment, score))
        if len(kept) >= int(max_items):
            break
    return kept


def _candidate_residual_score(
    parent: tuple[int, int],
    child: tuple[int, int],
    selector_score: float,
    evidence: np.ndarray,
    evidence_threshold: float,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_parent_contrast: float,
    length_penalty: float,
) -> float | None:
    inside = _segment_values(evidence, child)
    if len(inside) == 0:
        return None
    evidence_mean = float(inside.mean())
    active_fraction = float((inside >= float(evidence_threshold)).mean())
    residual = _parent_residual_values(evidence, parent, child)
    residual_baseline = float(np.median(residual)) if len(residual) else 0.0
    parent_contrast = evidence_mean - residual_baseline
    if evidence_mean < float(min_evidence_mean):
        return None
    if active_fraction < float(min_active_fraction):
        return None
    if parent_contrast < float(min_parent_contrast):
        return None
    return float(selector_score) + evidence_mean + parent_contrast + 0.25 * active_fraction - float(length_penalty) * math.log1p(_segment_length(child))


def build_residual_sub_event_records(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    evidence_thresholds: Iterable[float],
) -> list[list[ResidualSubEventRecord]]:
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
    ):
        raise ValueError("base/candidate/score/evidence video counts must match")
    threshold_values = [float(value) for value in evidence_thresholds]
    records_by_video: list[list[ResidualSubEventRecord]] = []
    for video_idx, (base_segments, candidates_raw, scores_raw, evidence) in enumerate(
        zip(base_predictions, candidate_predictions, candidate_scores, evidence_arrays)
    ):
        base = sorted({_normalise_segment(segment) for segment in base_segments})
        candidates = [_normalise_segment(segment) for segment in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        video_records: list[ResidualSubEventRecord] = []
        for parent in base:
            parent_length = _segment_length(parent)
            for child, selector_score in zip(candidates, scores):
                if child in base:
                    continue
                child_length = _segment_length(child)
                if child_length >= parent_length:
                    continue
                coverage = _coverage_of_child_by_parent(child, parent)
                if coverage <= 0.0:
                    continue
                inside = _segment_values(evidence, child)
                if len(inside) == 0:
                    continue
                evidence_mean = float(inside.mean())
                residual = _parent_residual_values(evidence, parent, child)
                residual_baseline = float(np.median(residual)) if len(residual) else 0.0
                active_fractions = {
                    threshold: float((inside >= threshold).mean())
                    for threshold in threshold_values
                }
                video_records.append(
                    ResidualSubEventRecord(
                        video_idx=int(video_idx),
                        parent=parent,
                        child=child,
                        selector_score=float(selector_score),
                        parent_length=int(parent_length),
                        child_length=int(child_length),
                        child_parent_ratio=float(child_length / max(1, parent_length)),
                        child_parent_coverage=float(coverage),
                        evidence_mean=evidence_mean,
                        parent_contrast=float(evidence_mean - residual_baseline),
                        active_fractions=active_fractions,
                    )
                )
        records_by_video.append(video_records)
    return records_by_video


def select_residual_sub_event_predictions_from_records(
    base_predictions: list[list[tuple[int, int]]],
    records_by_video: list[list[ResidualSubEventRecord]],
    min_selector_score: float,
    evidence_threshold: float,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_parent_contrast: float,
    parent_min_length: int,
    max_child_parent_ratio: float,
    min_child_parent_coverage: float,
    length_penalty: float,
    nms_iou: float | None,
    max_sub_events_per_parent: int,
    max_sub_events_per_video: int,
    mode: str = "add",
    min_sub_events_per_parent: int = 2,
) -> list[list[Segment]]:
    if len(base_predictions) != len(records_by_video):
        raise ValueError("base/record video counts must match")
    if mode not in {"add", "replace", "hybrid"}:
        raise ValueError(f"unsupported residual sub-event mode: {mode}")
    fused: list[list[Segment]] = []
    threshold = float(evidence_threshold)
    for base_segments, video_records in zip(base_predictions, records_by_video):
        base = sorted({_normalise_segment(segment) for segment in base_segments})
        video_child_scores: dict[Segment, float] = {}
        by_parent: dict[Segment, list[tuple[Segment, float]]] = {}
        for record in video_records:
            if record.parent_length < int(parent_min_length):
                continue
            if record.child_parent_ratio > float(max_child_parent_ratio):
                continue
            if record.child_parent_coverage < float(min_child_parent_coverage):
                continue
            if record.selector_score < float(min_selector_score):
                continue
            if record.evidence_mean < float(min_evidence_mean):
                continue
            if record.parent_contrast < float(min_parent_contrast):
                continue
            active_fraction = record.active_fractions.get(threshold)
            if active_fraction is None:
                active_fraction = 0.0
            if active_fraction < float(min_active_fraction):
                continue
            score = (
                record.selector_score
                + record.evidence_mean
                + record.parent_contrast
                + 0.25 * active_fraction
                - float(length_penalty) * math.log1p(record.child_length)
            )
            by_parent.setdefault(record.parent, []).append((record.child, float(score)))
        parents_to_remove: set[Segment] = set()
        for parent, parent_items in by_parent.items():
            parent_candidates = [child for child, _ in parent_items]
            parent_scores = [score for _, score in parent_items]
            selected_for_parent = _score_first_nms(
                parent_candidates,
                parent_scores,
                iou_threshold=nms_iou,
                max_items=max_sub_events_per_parent,
            )
            should_replace = mode in {"replace", "hybrid"} and len(selected_for_parent) >= int(min_sub_events_per_parent)
            if should_replace:
                parents_to_remove.add(parent)
            if mode == "replace" and not should_replace:
                continue
            for child, score in selected_for_parent:
                video_child_scores[child] = max(float(score), video_child_scores.get(child, -1e9))

        selected_children = _score_first_nms(
            list(video_child_scores.keys()),
            list(video_child_scores.values()),
            iou_threshold=nms_iou,
            max_items=max_sub_events_per_video,
        )
        selected_child_segments = [child for child, _ in selected_children]
        if mode in {"replace", "hybrid"}:
            selected_set = set(selected_child_segments)
            for parent in list(parents_to_remove):
                parent_has_selected_child = any(
                    _coverage_of_child_by_parent(child, parent) >= float(min_child_parent_coverage)
                    for child in selected_set
                )
                if not parent_has_selected_child:
                    parents_to_remove.discard(parent)
        kept_base = [segment for segment in base if segment not in parents_to_remove]
        fused.append(sorted(set(kept_base + selected_child_segments)))
    return fused


def select_residual_sub_event_predictions(
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
    length_penalty: float,
    nms_iou: float | None,
    max_sub_events_per_parent: int,
    max_sub_events_per_video: int,
    mode: str = "add",
    min_sub_events_per_parent: int = 2,
) -> list[list[Segment]]:
    records_by_video = build_residual_sub_event_records(
        base_predictions,
        candidate_predictions,
        candidate_scores,
        evidence_arrays,
        evidence_thresholds=[float(evidence_threshold)],
    )
    return select_residual_sub_event_predictions_from_records(
        base_predictions,
        records_by_video,
        min_selector_score=min_selector_score,
        evidence_threshold=evidence_threshold,
        min_evidence_mean=min_evidence_mean,
        min_active_fraction=min_active_fraction,
        min_parent_contrast=min_parent_contrast,
        parent_min_length=parent_min_length,
        max_child_parent_ratio=max_child_parent_ratio,
        min_child_parent_coverage=min_child_parent_coverage,
        length_penalty=length_penalty,
        nms_iou=nms_iou,
        max_sub_events_per_parent=max_sub_events_per_parent,
        max_sub_events_per_video=max_sub_events_per_video,
        mode=mode,
        min_sub_events_per_parent=min_sub_events_per_parent,
    )



def select_residual_sub_event_expansion_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    min_selector_scores: Iterable[float] = (0.0, 0.05, 0.1, 0.2),
    evidence_thresholds: Iterable[float] = (0.65, 0.75, 0.85),
    min_evidence_means: Iterable[float] = (0.55, 0.65, 0.75),
    min_active_fractions: Iterable[float] = (0.5, 0.67, 0.8),
    min_parent_contrasts: Iterable[float] = (0.05, 0.1, 0.2, 0.3),
    parent_min_lengths: Iterable[int] = (48, 72),
    max_child_parent_ratios: Iterable[float] = (0.15, 0.25, 0.4),
    min_child_parent_coverages: Iterable[float] = (0.8, 0.9, 1.0),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01),
    nms_ious: Iterable[float | None] = (None, 0.3),
    max_sub_events_per_parents: Iterable[int] = (1, 2),
    max_sub_events_per_videos: Iterable[int] = (1, 2, 3),
    modes: Iterable[str] = ("add",),
    min_sub_events_per_parents: Iterable[int] = (2,),
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
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
        0.0,
    )
    records_by_video = build_residual_sub_event_records(
        base,
        candidate_predictions,
        candidate_scores,
        evidence_arrays,
        evidence_thresholds=evidence_thresholds,
    )

    for min_selector_score in min_selector_scores:
        for evidence_threshold in evidence_thresholds:
            for min_evidence_mean in min_evidence_means:
                for min_active_fraction in min_active_fractions:
                    for min_parent_contrast in min_parent_contrasts:
                        for parent_min_length in parent_min_lengths:
                            for max_child_parent_ratio in max_child_parent_ratios:
                                for min_child_parent_coverage in min_child_parent_coverages:
                                    for length_penalty in length_penalties:
                                        for nms_iou in nms_ious:
                                            for max_sub_events_per_parent in max_sub_events_per_parents:
                                                for max_sub_events_per_video in max_sub_events_per_videos:
                                                    for mode in modes:
                                                        for min_sub_events_per_parent in min_sub_events_per_parents:
                                                            fused = select_residual_sub_event_predictions_from_records(
                                                                base,
                                                                records_by_video,
                                                                min_selector_score=float(min_selector_score),
                                                                evidence_threshold=float(evidence_threshold),
                                                                min_evidence_mean=float(min_evidence_mean),
                                                                min_active_fraction=float(min_active_fraction),
                                                                min_parent_contrast=float(min_parent_contrast),
                                                                parent_min_length=int(parent_min_length),
                                                                max_child_parent_ratio=float(max_child_parent_ratio),
                                                                min_child_parent_coverage=float(min_child_parent_coverage),
                                                                length_penalty=float(length_penalty),
                                                                nms_iou=nms_iou,
                                                                max_sub_events_per_parent=int(max_sub_events_per_parent),
                                                                max_sub_events_per_video=int(max_sub_events_per_video),
                                                                mode=str(mode),
                                                                min_sub_events_per_parent=int(min_sub_events_per_parent),
                                                            )
                                                            metrics = evaluate_fused_predictions(
                                                                fused,
                                                                labels,
                                                                iou_threshold=iou_threshold,
                                                            )
                                                            added = sum(max(0, len(after) - len(before)) for before, after in zip(base, fused))
                                                            changed = sum(1 for before, after in zip(base, fused) if set(before) != set(after))
                                                            key = (
                                                                float(metrics["segment"]["f1"]),
                                                                float(metrics["segment"]["precision"]),
                                                                float(metrics["segment"]["recall"]),
                                                                float(metrics["frame"]["f1"]),
                                                                -float(changed),
                                                                -float(added),
                                                            )
                                                            if key > best_key:
                                                                best_key = key
                                                                best_metrics = metrics
                                                                best_predictions = fused
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
                                                                    "length_penalty": float(length_penalty),
                                                                    "nms_iou": nms_iou,
                                                                    "max_sub_events_per_parent": int(max_sub_events_per_parent),
                                                                    "max_sub_events_per_video": int(max_sub_events_per_video),
                                                                    "mode": str(mode),
                                                                    "min_sub_events_per_parent": int(min_sub_events_per_parent),
                                                                    "added_sub_events": int(added),
                                                                    "changed_videos": int(changed),
                                                                    "base_metrics": base_metrics,
                                                                }
    return best_config, best_metrics, best_predictions
