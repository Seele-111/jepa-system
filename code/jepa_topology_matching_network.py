#!/usr/bin/env python3
"""Structured JEPA topology matching for swallowed error events.

This module scores a parent-to-child-set replacement, not isolated child
segments. Training folds build teacher child sets from GT events inside broad
parents; validation folds only see JEPA candidates, scores, and evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou
from train_proposal_rescuer import fit_rescuer
from train_segment_locator import contiguous_segments


Segment = tuple[int, int]


TOPOLOGY_MATCHING_FEATURE_NAMES = [
    "parent_length",
    "log_parent_length",
    "child_count",
    "child_total_parent_ratio",
    "child_mean_parent_ratio",
    "child_max_parent_ratio",
    "coverage_union",
    "mean_child_parent_coverage",
    "child_score_mean",
    "child_score_min",
    "child_score_max",
    "child_score_std",
    "child_rank_mean",
    "child_rank_min",
    "parent_evidence_mean",
    "parent_evidence_std",
    "child_evidence_mean",
    "child_evidence_min",
    "child_evidence_max",
    "child_evidence_std_mean",
    "child_parent_evidence_delta",
    "child_residual_evidence_delta",
    "residual_evidence_mean",
    "residual_evidence_max",
    "gap_mean_parent_ratio",
    "gap_max_parent_ratio",
    "candidate_count",
    "evidence_island_count",
    "evidence_island_coverage",
    "child_evidence_mass_fraction",
    "residual_active_fraction",
    "topology_count_alignment",
    "set_evidence_demand_score",
]
TOPOLOGY_MATCHING_FEATURE_INDEX = {name: idx for idx, name in enumerate(TOPOLOGY_MATCHING_FEATURE_NAMES)}


@dataclass(frozen=True)
class ChildSetProposal:
    video_idx: int
    parent: Segment
    children: list[Segment]
    features: np.ndarray
    base_tp: int = 0
    split_tp: int = 0
    base_fp: int = 0
    split_fp: int = 0
    label: int = 0
    gain_label: int = 0
    safe_label: int = 0
    teacher: bool = False


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


def _coverage(segment: tuple[int, int], parent: tuple[int, int]) -> float:
    return float(_intersection(segment, parent) / max(1, _segment_length(segment)))


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


def _residual_parent_matrix(evidence: np.ndarray, parent: tuple[int, int], children: list[Segment]) -> np.ndarray:
    values = _matrix(evidence)
    if len(values) == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    parent_start, parent_end = _normalise_segment(parent)
    parent_start = max(0, min(parent_start, len(values) - 1))
    parent_end = max(parent_start, min(parent_end, len(values) - 1))
    mask = np.ones(parent_end - parent_start + 1, dtype=bool)
    for child_raw in children:
        child_start, child_end = _normalise_segment(child_raw)
        left = max(parent_start, child_start) - parent_start
        right = min(parent_end, child_end) - parent_start
        if right >= left:
            mask[left : right + 1] = False
    parent_values = values[parent_start : parent_end + 1]
    if mask.any():
        return parent_values[mask].astype(np.float32)
    return parent_values.astype(np.float32)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-clean, kind="mergesort")
    ranks = np.empty(len(clean), dtype=np.int64)
    ranks[order] = np.arange(1, len(clean) + 1, dtype=np.int64)
    return ranks


def _contiguous_bool_segments(values: np.ndarray) -> list[Segment]:
    segments: list[Segment] = []
    start: int | None = None
    for idx, active in enumerate(np.asarray(values, dtype=bool)):
        if bool(active) and start is None:
            start = int(idx)
        elif not bool(active) and start is not None:
            segments.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        segments.append((int(start), int(len(values) - 1)))
    return segments


def _evidence_curve(evidence: np.ndarray) -> np.ndarray:
    values = _matrix(evidence)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.clip(values.mean(axis=1), 0.0, 1.0).astype(np.float32)


def _evidence_islands_for_parent(evidence: np.ndarray, parent: Segment) -> list[Segment]:
    curve = _evidence_curve(evidence)
    if len(curve) == 0:
        return []
    parent_start, parent_end = _normalise_segment(parent)
    parent_start = max(0, min(parent_start, len(curve) - 1))
    parent_end = max(parent_start, min(parent_end, len(curve) - 1))
    parent_curve = curve[parent_start : parent_end + 1]
    if len(parent_curve) == 0:
        return []
    threshold = max(0.5, float(parent_curve.mean() + 0.25 * parent_curve.std()))
    local_segments = _contiguous_bool_segments(parent_curve >= threshold)
    return [(int(start + parent_start), int(end + parent_start)) for start, end in local_segments]


def _union_mask(length: int, segments: list[Segment], offset: int = 0) -> np.ndarray:
    mask = np.zeros(max(0, int(length)), dtype=bool)
    if len(mask) == 0:
        return mask
    for segment_raw in segments:
        start, end = _normalise_segment(segment_raw)
        left = max(0, int(start - offset))
        right = min(len(mask) - 1, int(end - offset))
        if right >= left:
            mask[left : right + 1] = True
    return mask


def _v2_evidence_demand_features(parent: Segment, children: list[Segment], evidence: np.ndarray) -> list[float]:
    curve = _evidence_curve(evidence)
    if len(curve) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    parent_start, parent_end = _normalise_segment(parent)
    parent_start = max(0, min(parent_start, len(curve) - 1))
    parent_end = max(parent_start, min(parent_end, len(curve) - 1))
    parent_curve = curve[parent_start : parent_end + 1]
    if len(parent_curve) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    islands = _evidence_islands_for_parent(evidence, (parent_start, parent_end))
    child_mask = _union_mask(len(parent_curve), children, offset=parent_start)
    residual_mask = ~child_mask
    parent_mass = float(parent_curve.sum())
    child_mass = float(parent_curve[child_mask].sum()) if child_mask.any() else 0.0
    child_evidence_mass_fraction = child_mass / max(parent_mass, 1e-6)

    if islands:
        island_coverages = []
        for island in islands:
            island_start, island_end = _normalise_segment(island)
            local_start = max(0, island_start - parent_start)
            local_end = min(len(parent_curve) - 1, island_end - parent_start)
            if local_end < local_start:
                island_coverages.append(0.0)
            else:
                island_coverages.append(float(child_mask[local_start : local_end + 1].mean()))
        evidence_island_coverage = float(np.mean(island_coverages))
        topology_count_alignment = min(len(children), len(islands)) / max(len(children), len(islands), 1)
    else:
        evidence_island_coverage = 0.0
        topology_count_alignment = 0.0

    threshold = max(0.5, float(parent_curve.mean() + 0.25 * parent_curve.std()))
    residual_active_fraction = (
        float((parent_curve[residual_mask] >= threshold).mean()) if residual_mask.any() else 0.0
    )
    set_evidence_demand_score = (
        evidence_island_coverage
        + child_evidence_mass_fraction
        + topology_count_alignment
        + max(0.0, child_evidence_mass_fraction - residual_active_fraction)
    )
    return [
        float(len(islands)),
        float(evidence_island_coverage),
        float(child_evidence_mass_fraction),
        float(residual_active_fraction),
        float(topology_count_alignment),
        float(set_evidence_demand_score),
    ]


def _greedy_segment_metrics(predictions: list[Segment], labels: np.ndarray, iou_threshold: float) -> tuple[int, int, int]:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched: set[int] = set()
    tp = fp = 0
    for pred_raw in predictions:
        pred = _normalise_segment(pred_raw)
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_segments):
            if idx in matched:
                continue
            iou = float(interval_iou(pred, gt))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt_segments) - len(matched)


def _children_are_compatible(children: list[Segment], nms_iou: float | None) -> bool:
    ordered = sorted({_normalise_segment(item) for item in children})
    if len(ordered) != len(children):
        return False
    for left, right in combinations(ordered, 2):
        if nms_iou is not None and interval_iou(left, right) > float(nms_iou):
            return False
    return True


def _candidate_feature_summary(
    parent: Segment,
    children: list[Segment],
    candidates: list[Segment],
    scores: np.ndarray,
    ranks: np.ndarray,
    evidence: np.ndarray,
) -> np.ndarray:
    parent_len = float(_segment_length(parent))
    score_by_segment = {candidate: float(score) for candidate, score in zip(candidates, scores)}
    rank_by_segment = {candidate: float(rank) for candidate, rank in zip(candidates, ranks)}
    parent_evidence = _segment_matrix(evidence, parent)
    parent_frame_values = parent_evidence.mean(axis=1) if len(parent_evidence) else np.zeros(0, dtype=np.float32)
    parent_mean = float(parent_frame_values.mean()) if len(parent_frame_values) else 0.0
    parent_std = float(parent_frame_values.std()) if len(parent_frame_values) else 0.0

    child_scores = np.asarray([score_by_segment.get(child, 0.0) for child in children], dtype=np.float32)
    child_ranks = np.asarray([rank_by_segment.get(child, len(candidates) + 1.0) for child in children], dtype=np.float32)
    child_lengths = np.asarray([_segment_length(child) for child in children], dtype=np.float32)
    child_coverages = np.asarray([_coverage(child, parent) for child in children], dtype=np.float32)
    child_evidence_means: list[float] = []
    child_evidence_maxes: list[float] = []
    child_evidence_stds: list[float] = []
    for child in children:
        values = _segment_matrix(evidence, child)
        per_frame = values.mean(axis=1) if len(values) else np.zeros(0, dtype=np.float32)
        child_evidence_means.append(float(per_frame.mean()) if len(per_frame) else 0.0)
        child_evidence_maxes.append(float(per_frame.max()) if len(per_frame) else 0.0)
        child_evidence_stds.append(float(per_frame.std()) if len(per_frame) else 0.0)

    residual = _residual_parent_matrix(evidence, parent, children)
    residual_frame_values = residual.mean(axis=1) if len(residual) else np.zeros(0, dtype=np.float32)
    residual_mean = float(residual_frame_values.mean()) if len(residual_frame_values) else 0.0
    residual_max = float(residual_frame_values.max()) if len(residual_frame_values) else 0.0

    child_evidence = np.asarray(child_evidence_means, dtype=np.float32)
    child_count = float(len(children))
    coverage_union = sum(_intersection(child, parent) for child in children) / max(1.0, parent_len)
    gaps = np.asarray([max(0, right[0] - left[1] - 1) for left, right in zip(children, children[1:])], dtype=np.float32)
    base_features = [
            parent_len,
            float(np.log1p(parent_len)),
            child_count,
            float(child_lengths.sum() / max(1.0, parent_len)),
            float(child_lengths.mean() / max(1.0, parent_len)) if len(child_lengths) else 0.0,
            float(child_lengths.max() / max(1.0, parent_len)) if len(child_lengths) else 0.0,
            float(coverage_union),
            float(child_coverages.mean()) if len(child_coverages) else 0.0,
            float(child_scores.mean()) if len(child_scores) else 0.0,
            float(child_scores.min()) if len(child_scores) else 0.0,
            float(child_scores.max()) if len(child_scores) else 0.0,
            float(child_scores.std()) if len(child_scores) else 0.0,
            float(child_ranks.mean()) if len(child_ranks) else 0.0,
            float(child_ranks.min()) if len(child_ranks) else 0.0,
            float(parent_mean),
            float(parent_std),
            float(child_evidence.mean()) if len(child_evidence) else 0.0,
            float(child_evidence.min()) if len(child_evidence) else 0.0,
            float(max(child_evidence_maxes) if child_evidence_maxes else 0.0),
            float(np.mean(child_evidence_stds)) if child_evidence_stds else 0.0,
            float((child_evidence.mean() - parent_mean) if len(child_evidence) else 0.0),
            float((child_evidence.mean() - residual_mean) if len(child_evidence) else 0.0),
            float(residual_mean),
            float(residual_max),
            float(gaps.mean() / max(1.0, parent_len)) if len(gaps) else 0.0,
            float(gaps.max() / max(1.0, parent_len)) if len(gaps) else 0.0,
            float(len(candidates)),
    ]
    features = np.asarray(
        base_features + _v2_evidence_demand_features(parent, children, evidence),
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _candidate_pool_for_parent(
    parent: Segment,
    candidates: list[Segment],
    candidate_scores: np.ndarray,
    evidence: np.ndarray,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
) -> list[Segment]:
    parent_len = _segment_length(parent)
    scored: list[tuple[Segment, tuple[float, float, float, float]]] = []
    ranks = _rank_desc(candidate_scores)
    for candidate_raw, score, rank in zip(candidates, candidate_scores, ranks):
        child = _normalise_segment(candidate_raw)
        child_len = _segment_length(child)
        if child == parent or child_len >= parent_len:
            continue
        if _coverage(child, parent) < float(min_child_parent_coverage):
            continue
        if child_len / max(1, parent_len) > float(max_child_parent_ratio):
            continue
        evidence_values = _segment_matrix(evidence, child)
        evidence_mean = float(evidence_values.mean()) if len(evidence_values) else 0.0
        key = (float(score), evidence_mean, -float(rank), -float(child_len))
        scored.append((child, key))
    scored.sort(key=lambda item: item[1], reverse=True)
    out: list[Segment] = []
    seen: set[Segment] = set()
    for child, _ in scored:
        if child in seen:
            continue
        seen.add(child)
        out.append(child)
    return out


def enumerate_child_set_proposals(
    parent: tuple[int, int],
    candidates: list[tuple[int, int]],
    candidate_scores: np.ndarray,
    evidence: np.ndarray,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_children_per_parent: int,
    max_set_proposals: int,
    nms_iou: float | None = 0.3,
    max_child_pool: int = 24,
    video_idx: int = 0,
) -> list[ChildSetProposal]:
    """Enumerate compact child sets that can replace one broad parent."""
    parent_norm = _normalise_segment(parent)
    norm_candidates = [_normalise_segment(item) for item in candidates]
    scores = np.nan_to_num(np.asarray(candidate_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(norm_candidates) != len(scores):
        raise ValueError(f"candidate/score count mismatch: {len(norm_candidates)} vs {len(scores)}")
    ranks = _rank_desc(scores)
    pool = _candidate_pool_for_parent(
        parent_norm,
        norm_candidates,
        scores,
        evidence,
        min_child_parent_coverage=float(min_child_parent_coverage),
        max_child_parent_ratio=float(max_child_parent_ratio),
    )
    pool = pool[: max(2, int(max_child_pool))]
    proposals: list[ChildSetProposal] = []
    max_children = max(2, int(max_children_per_parent))
    for child_count in range(2, max_children + 1):
        for child_tuple in combinations(pool, child_count):
            children = sorted(child_tuple)
            if not _children_are_compatible(children, nms_iou=nms_iou):
                continue
            features = _candidate_feature_summary(parent_norm, children, norm_candidates, scores, ranks, evidence)
            proposals.append(
                ChildSetProposal(
                    video_idx=int(video_idx),
                    parent=parent_norm,
                    children=children,
                    features=features,
                )
            )
    proposals.sort(
        key=lambda item: (
            float(item.features[10]),
            float(item.features[16]),
            float(item.features[6]),
            -float(item.features[4]),
        ),
        reverse=True,
    )
    return proposals[: max(1, int(max_set_proposals))]


def _teacher_child_set(
    parent: Segment,
    candidates: list[Segment],
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_children_per_parent: int,
) -> list[Segment]:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    inside_gts = [gt for gt in gt_segments if _coverage(gt, parent) >= 0.8]
    if len(inside_gts) < 2:
        return []
    parent_len = _segment_length(parent)
    selected: list[Segment] = []
    used: set[Segment] = set()
    for gt in inside_gts:
        best_child: Segment | None = None
        best_key: tuple[float, float, float, float] | None = None
        for candidate, score in zip(candidates, scores):
            child = _normalise_segment(candidate)
            if child in used or child == parent:
                continue
            child_len = _segment_length(child)
            if child_len >= parent_len:
                continue
            if _coverage(child, parent) < float(min_child_parent_coverage):
                continue
            if child_len / max(1, parent_len) > float(max_child_parent_ratio):
                continue
            iou = float(interval_iou(child, gt))
            if iou < float(iou_threshold):
                continue
            length_ratio = min(child_len, _segment_length(gt)) / max(child_len, _segment_length(gt))
            center_gap = abs((child[0] + child[1]) - (gt[0] + gt[1])) / 2.0
            key = (iou, length_ratio, float(score), -center_gap)
            if best_key is None or key > best_key:
                best_key = key
                best_child = child
        if best_child is not None:
            used.add(best_child)
            selected.append(best_child)
        if len(selected) >= int(max_children_per_parent):
            break
    return sorted(selected) if len(selected) >= 2 else []


def _proposal_with_metrics(
    proposal: ChildSetProposal,
    labels: np.ndarray,
    iou_threshold: float,
    teacher: bool = False,
) -> ChildSetProposal:
    base_tp, base_fp, _ = _greedy_segment_metrics([proposal.parent], labels, iou_threshold=float(iou_threshold))
    split_tp, split_fp, _ = _greedy_segment_metrics(proposal.children, labels, iou_threshold=float(iou_threshold))
    gain_label = int(split_tp > base_tp)
    safe_label = int((split_fp - base_fp) <= 0)
    label = int(gain_label and split_fp == 0)
    return ChildSetProposal(
        video_idx=proposal.video_idx,
        parent=proposal.parent,
        children=proposal.children,
        features=proposal.features,
        base_tp=int(base_tp),
        split_tp=int(split_tp),
        base_fp=int(base_fp),
        split_fp=int(split_fp),
        label=label,
        gain_label=gain_label,
        safe_label=safe_label,
        teacher=bool(teacher),
    )


def build_topology_matching_records(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    iou_threshold: float,
    parent_min_length: int,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_children_per_parent: int,
    max_set_proposals_per_parent: int,
    include_teacher: bool,
    max_child_pool_per_parent: int = 24,
    nms_iou: float | None = 0.3,
) -> list[ChildSetProposal]:
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
        == len(labels)
    ):
        raise ValueError("base/candidate/score/evidence/label video counts must match")
    records: list[ChildSetProposal] = []
    seen: set[tuple[int, Segment, tuple[Segment, ...]]] = set()
    for video_idx, (base_raw, candidates_raw, scores_raw, evidence, label_array) in enumerate(
        zip(base_predictions, candidate_predictions, candidate_scores, evidence_arrays, labels)
    ):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        ranks = _rank_desc(scores)
        for parent in base:
            if _segment_length(parent) < int(parent_min_length):
                continue
            parent_records: list[ChildSetProposal] = []
            if include_teacher:
                teacher_children = _teacher_child_set(
                    parent,
                    candidates,
                    scores,
                    label_array,
                    iou_threshold=float(iou_threshold),
                    min_child_parent_coverage=float(min_child_parent_coverage),
                    max_child_parent_ratio=float(max_child_parent_ratio),
                    max_children_per_parent=int(max_children_per_parent),
                )
                if teacher_children:
                    parent_records.append(
                        ChildSetProposal(
                            video_idx=int(video_idx),
                            parent=parent,
                            children=teacher_children,
                            features=_candidate_feature_summary(parent, teacher_children, candidates, scores, ranks, evidence),
                            teacher=True,
                        )
                    )
            parent_records.extend(
                enumerate_child_set_proposals(
                    parent,
                    candidates,
                    scores,
                    evidence,
                    min_child_parent_coverage=float(min_child_parent_coverage),
                    max_child_parent_ratio=float(max_child_parent_ratio),
                    max_children_per_parent=int(max_children_per_parent),
                    max_set_proposals=int(max_set_proposals_per_parent),
                    nms_iou=nms_iou,
                    max_child_pool=int(max_child_pool_per_parent),
                    video_idx=int(video_idx),
                )
            )
            for proposal in parent_records:
                key = (int(video_idx), proposal.parent, tuple(proposal.children))
                if key in seen:
                    continue
                seen.add(key)
                records.append(_proposal_with_metrics(proposal, label_array, iou_threshold=float(iou_threshold), teacher=proposal.teacher))
    return records


def build_topology_inference_records(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    parent_min_length: int,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_children_per_parent: int,
    max_set_proposals_per_parent: int,
    max_child_pool_per_parent: int = 24,
    nms_iou: float | None = 0.3,
) -> list[ChildSetProposal]:
    """Enumerate parent-to-child-set proposals without reading labels.

    This is the inference-side companion to ``build_topology_matching_records``.
    It deliberately leaves teacher/label fields at their dataclass defaults so
    strict OOF runners can prove that held-out proposal construction is
    label-free.
    """
    if not (len(base_predictions) == len(candidate_predictions) == len(candidate_scores) == len(evidence_arrays)):
        raise ValueError("base/candidate/score/evidence video counts must match")
    records: list[ChildSetProposal] = []
    seen: set[tuple[int, Segment, tuple[Segment, ...]]] = set()
    for video_idx, (base_raw, candidates_raw, scores_raw, evidence) in enumerate(
        zip(base_predictions, candidate_predictions, candidate_scores, evidence_arrays)
    ):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        for parent in base:
            if _segment_length(parent) < int(parent_min_length):
                continue
            parent_records = enumerate_child_set_proposals(
                parent,
                candidates,
                scores,
                evidence,
                min_child_parent_coverage=float(min_child_parent_coverage),
                max_child_parent_ratio=float(max_child_parent_ratio),
                max_children_per_parent=int(max_children_per_parent),
                max_set_proposals=int(max_set_proposals_per_parent),
                nms_iou=nms_iou,
                max_child_pool=int(max_child_pool_per_parent),
                video_idx=int(video_idx),
            )
            for proposal in parent_records:
                key = (int(video_idx), proposal.parent, tuple(proposal.children))
                if key in seen:
                    continue
                seen.add(key)
                records.append(proposal)
    return records


def fit_topology_matching_model(
    records: list[ChildSetProposal],
    model_name: str,
    seed: int,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    x = np.stack([record.features for record in records])
    y = np.asarray([record.label for record in records], dtype=np.int64)
    return fit_rescuer(
        x,
        y,
        seed=int(seed),
        model_name=str(model_name),
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
    )


def _score_records(model, records: list[ChildSetProposal]) -> np.ndarray:
    if not records:
        return np.zeros(0, dtype=np.float32)
    x = np.stack([record.features for record in records])
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)


class _ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x)
        n = int(values.shape[0]) if values.ndim > 1 else 1
        positive = np.full(n, self.probability, dtype=np.float32)
        return np.stack([1.0 - positive, positive], axis=1)


def _fit_binary_model_or_constant(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seed: int,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if len(labels) == 0:
        return _ConstantProbabilityModel(0.0)
    if len(np.unique(labels)) < 2:
        return _ConstantProbabilityModel(float(labels.mean()))
    return fit_rescuer(
        x,
        labels,
        seed=int(seed),
        model_name=str(model_name),
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
    )


def fit_topology_matching_risk_models(
    records: list[ChildSetProposal],
    model_name: str,
    seed: int,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    """Fit separate completeness-gain and no-FP-risk heads for child sets."""
    x = np.stack([record.features for record in records])
    gain_y = np.asarray([record.gain_label for record in records], dtype=np.int64)
    safe_y = np.asarray([record.safe_label for record in records], dtype=np.int64)
    gain_model = _fit_binary_model_or_constant(
        x,
        gain_y,
        model_name=model_name,
        seed=int(seed),
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
    )
    safety_model = _fit_binary_model_or_constant(
        x,
        safe_y,
        model_name=model_name,
        seed=int(seed) + 137,
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
    )
    return gain_model, safety_model


def score_topology_matching_risk_records(
    gain_model,
    safety_model,
    records: list[ChildSetProposal],
    risk_penalty: float = 0.5,
    safety_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return combined, gain, and safety scores for structured replacement."""
    gain_scores = _score_records(gain_model, records)
    safety_scores = _score_records(safety_model, records)
    fp_risk = 1.0 - safety_scores
    combined = gain_scores + float(safety_weight) * safety_scores - float(risk_penalty) * fp_risk
    return (
        np.asarray(combined, dtype=np.float32),
        np.asarray(gain_scores, dtype=np.float32),
        np.asarray(safety_scores, dtype=np.float32),
    )


def apply_topology_matching_predictions(
    base_predictions: list[list[tuple[int, int]]],
    proposals: list[ChildSetProposal],
    proposal_scores: np.ndarray,
    threshold: float,
    max_replaced_parents_per_video: int,
    safety_scores: np.ndarray | None = None,
    safety_threshold: float | None = None,
    min_evidence_island_coverage: float | None = None,
    max_residual_active_fraction: float | None = None,
    min_topology_count_alignment: float | None = None,
    min_child_evidence_mass_fraction: float | None = None,
) -> list[list[Segment]]:
    out = [[_normalise_segment(item) for item in video] for video in base_predictions]
    scores = np.nan_to_num(np.asarray(proposal_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(proposals) != len(scores):
        raise ValueError(f"proposal/score count mismatch: {len(proposals)} vs {len(scores)}")
    safe = None
    if safety_scores is not None:
        safe = np.nan_to_num(np.asarray(safety_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(proposals) != len(safe):
            raise ValueError(f"proposal/safety score count mismatch: {len(proposals)} vs {len(safe)}")
    by_video: dict[int, list[tuple[ChildSetProposal, float]]] = {}
    for idx, (proposal, score) in enumerate(zip(proposals, scores)):
        if float(score) < float(threshold):
            continue
        if safe is not None and safety_threshold is not None and float(safe[idx]) < float(safety_threshold):
            continue
        features = np.asarray(proposal.features, dtype=np.float32).reshape(-1)
        if min_evidence_island_coverage is not None:
            value = float(features[TOPOLOGY_MATCHING_FEATURE_INDEX["evidence_island_coverage"]])
            if value < float(min_evidence_island_coverage):
                continue
        if max_residual_active_fraction is not None:
            value = float(features[TOPOLOGY_MATCHING_FEATURE_INDEX["residual_active_fraction"]])
            if value > float(max_residual_active_fraction):
                continue
        if min_topology_count_alignment is not None:
            value = float(features[TOPOLOGY_MATCHING_FEATURE_INDEX["topology_count_alignment"]])
            if value < float(min_topology_count_alignment):
                continue
        if min_child_evidence_mass_fraction is not None:
            value = float(features[TOPOLOGY_MATCHING_FEATURE_INDEX["child_evidence_mass_fraction"]])
            if value < float(min_child_evidence_mass_fraction):
                continue
        by_video.setdefault(proposal.video_idx, []).append((proposal, float(score)))
    for video_idx, items in by_video.items():
        items.sort(
            key=lambda item: (
                item[1],
                len(item[0].children),
                float(item[0].features[16]),
                -_segment_length(item[0].parent),
            ),
            reverse=True,
        )
        selected: list[tuple[ChildSetProposal, float]] = []
        used_parents: set[Segment] = set()
        for proposal, score in items:
            if proposal.parent in used_parents:
                continue
            used_parents.add(proposal.parent)
            selected.append((proposal, score))
            if max_replaced_parents_per_video > 0 and len(selected) >= int(max_replaced_parents_per_video):
                break
        replace_parents = {proposal.parent for proposal, _ in selected}
        children = [child for proposal, _ in selected for child in proposal.children]
        out[int(video_idx)] = sorted(set([segment for segment in out[int(video_idx)] if segment not in replace_parents] + children))
    return out


def select_topology_matching_params(
    train_base_predictions: list[list[tuple[int, int]]],
    train_candidate_predictions: list[list[tuple[int, int]]],
    train_candidate_scores: list[np.ndarray],
    train_evidence_arrays: list[np.ndarray],
    train_labels: list[np.ndarray],
    val_base_predictions: list[list[tuple[int, int]]],
    val_candidate_predictions: list[list[tuple[int, int]]],
    val_candidate_scores: list[np.ndarray],
    val_evidence_arrays: list[np.ndarray],
    val_labels: list[np.ndarray],
    model_name: str,
    seed: int,
    thresholds: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6),
    parent_min_lengths: Iterable[int] = (12, 24),
    min_child_parent_coverages: Iterable[float] = (0.5, 0.7),
    max_child_parent_ratios: Iterable[float] = (0.5, 0.7),
    max_children_per_parents: Iterable[int] = (2, 3),
    max_set_proposals_per_parents: Iterable[int] = (8,),
    max_child_pool_per_parents: Iterable[int] = (24,),
    max_replaced_parents_per_videos: Iterable[int] = (1,),
    max_fp_increase: int | None = 0,
    risk_aware: bool = False,
    risk_penalties: Iterable[float] = (0.5,),
    safety_thresholds: Iterable[float] = (0.0,),
    safety_weight: float = 0.25,
    min_evidence_island_coverages: Iterable[float | None] = (None,),
    max_residual_active_fractions: Iterable[float | None] = (None,),
    min_topology_count_alignments: Iterable[float | None] = (None,),
    min_child_evidence_mass_fractions: Iterable[float | None] = (None,),
    iou_threshold: float = 0.3,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in val_base_predictions]
    base_metrics = evaluate_fused_predictions(base, val_labels, iou_threshold=iou_threshold)
    diagnostics = {
        "configs_considered": 0,
        "no_train_records": 0,
        "train_single_class": 0,
        "no_val_records": 0,
        "evaluated_configs": 0,
        "fp_rejected_configs": 0,
        "max_train_records": 0,
        "max_positive_train_records": 0,
        "max_gain_train_records": 0,
        "max_safe_train_records": 0,
        "max_teacher_train_records": 0,
        "max_val_records": 0,
        "best_observed_metrics": None,
        "best_observed_config": None,
        "best_rejected_metrics": None,
        "best_rejected_config": None,
    }
    best_config: dict = {"enabled": False, "base_metrics": base_metrics, "diagnostics": diagnostics}
    best_metrics = base_metrics
    best_predictions = base
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )

    for parent_min_length in parent_min_lengths:
        for min_child_parent_coverage in min_child_parent_coverages:
            for max_child_parent_ratio in max_child_parent_ratios:
                for max_children_per_parent in max_children_per_parents:
                    for max_set_proposals_per_parent in max_set_proposals_per_parents:
                        for max_child_pool_per_parent in max_child_pool_per_parents:
                            diagnostics["configs_considered"] += 1
                            train_records = build_topology_matching_records(
                                train_base_predictions,
                                train_candidate_predictions,
                                train_candidate_scores,
                                train_evidence_arrays,
                                train_labels,
                                iou_threshold=float(iou_threshold),
                                parent_min_length=int(parent_min_length),
                                min_child_parent_coverage=float(min_child_parent_coverage),
                                max_child_parent_ratio=float(max_child_parent_ratio),
                                max_children_per_parent=int(max_children_per_parent),
                                max_set_proposals_per_parent=int(max_set_proposals_per_parent),
                                max_child_pool_per_parent=int(max_child_pool_per_parent),
                                include_teacher=True,
                            )
                            if not train_records:
                                diagnostics["no_train_records"] += 1
                                continue
                            y_train = np.asarray([record.label for record in train_records], dtype=np.int64)
                            diagnostics["max_train_records"] = max(int(diagnostics["max_train_records"]), int(len(train_records)))
                            diagnostics["max_positive_train_records"] = max(
                                int(diagnostics["max_positive_train_records"]),
                                int(y_train.sum()),
                            )
                            diagnostics["max_gain_train_records"] = max(
                                int(diagnostics["max_gain_train_records"]),
                                int(sum(record.gain_label for record in train_records)),
                            )
                            diagnostics["max_safe_train_records"] = max(
                                int(diagnostics["max_safe_train_records"]),
                                int(sum(record.safe_label for record in train_records)),
                            )
                            diagnostics["max_teacher_train_records"] = max(
                                int(diagnostics["max_teacher_train_records"]),
                                int(sum(record.teacher for record in train_records)),
                            )
                            if not bool(risk_aware) and len(np.unique(y_train)) < 2:
                                diagnostics["train_single_class"] += 1
                                continue
                            val_records = build_topology_matching_records(
                                val_base_predictions,
                                val_candidate_predictions,
                                val_candidate_scores,
                                val_evidence_arrays,
                                val_labels,
                                iou_threshold=float(iou_threshold),
                                parent_min_length=int(parent_min_length),
                                min_child_parent_coverage=float(min_child_parent_coverage),
                                max_child_parent_ratio=float(max_child_parent_ratio),
                                max_children_per_parent=int(max_children_per_parent),
                                max_set_proposals_per_parent=int(max_set_proposals_per_parent),
                                max_child_pool_per_parent=int(max_child_pool_per_parent),
                                include_teacher=False,
                            )
                            if not val_records:
                                diagnostics["no_val_records"] += 1
                                continue
                            diagnostics["max_val_records"] = max(int(diagnostics["max_val_records"]), int(len(val_records)))
                            if bool(risk_aware):
                                gain_model, safety_model = fit_topology_matching_risk_models(
                                    train_records,
                                    model_name=model_name,
                                    seed=int(seed),
                                    device=device,
                                    epochs=int(epochs),
                                    batch_size=int(batch_size),
                                )
                                score_variants = []
                                for risk_penalty in risk_penalties:
                                    combined_scores, gain_scores, safety_scores = score_topology_matching_risk_records(
                                        gain_model,
                                        safety_model,
                                        val_records,
                                        risk_penalty=float(risk_penalty),
                                        safety_weight=float(safety_weight),
                                    )
                                    score_variants.append(
                                        {
                                            "scores": combined_scores,
                                            "gain_scores": gain_scores,
                                            "safety_scores": safety_scores,
                                            "risk_penalty": float(risk_penalty),
                                        }
                                    )
                            else:
                                model = fit_topology_matching_model(
                                    train_records,
                                    model_name=model_name,
                                    seed=int(seed),
                                    device=device,
                                    epochs=int(epochs),
                                    batch_size=int(batch_size),
                                )
                                score_variants = [
                                    {
                                        "scores": _score_records(model, val_records),
                                        "gain_scores": None,
                                        "safety_scores": None,
                                        "risk_penalty": None,
                                    }
                                ]
                            for score_variant in score_variants:
                                active_safety_thresholds = safety_thresholds if bool(risk_aware) else (None,)
                                for threshold in thresholds:
                                    for safety_threshold in active_safety_thresholds:
                                        for min_evidence_island_coverage in min_evidence_island_coverages:
                                            for max_residual_active_fraction in max_residual_active_fractions:
                                                for min_topology_count_alignment in min_topology_count_alignments:
                                                    for min_child_evidence_mass_fraction in min_child_evidence_mass_fractions:
                                                        for max_replaced_parents_per_video in max_replaced_parents_per_videos:
                                                            diagnostics["evaluated_configs"] += 1
                                                            predictions = apply_topology_matching_predictions(
                                                                base,
                                                                val_records,
                                                                score_variant["scores"],
                                                                threshold=float(threshold),
                                                                max_replaced_parents_per_video=int(max_replaced_parents_per_video),
                                                                safety_scores=score_variant["safety_scores"],
                                                                safety_threshold=None if safety_threshold is None else float(safety_threshold),
                                                                min_evidence_island_coverage=min_evidence_island_coverage,
                                                                max_residual_active_fraction=max_residual_active_fraction,
                                                                min_topology_count_alignment=min_topology_count_alignment,
                                                                min_child_evidence_mass_fraction=min_child_evidence_mass_fraction,
                                                            )
                                                            metrics = evaluate_fused_predictions(predictions, val_labels, iou_threshold=iou_threshold)
                                                            changed = sum(int(sorted(a) != sorted(b)) for a, b in zip(base, predictions))
                                                            segment_delta = sum(len(b) - len(a) for a, b in zip(base, predictions))
                                                            observed_config = {
                                                                "model": str(model_name),
                                                                "risk_aware": bool(risk_aware),
                                                                "risk_penalty": score_variant["risk_penalty"],
                                                                "safety_threshold": None if safety_threshold is None else float(safety_threshold),
                                                                "safety_weight": float(safety_weight) if bool(risk_aware) else None,
                                                                "threshold": float(threshold),
                                                                "parent_min_length": int(parent_min_length),
                                                                "min_child_parent_coverage": float(min_child_parent_coverage),
                                                                "max_child_parent_ratio": float(max_child_parent_ratio),
                                                                "max_children_per_parent": int(max_children_per_parent),
                                                                "max_set_proposals_per_parent": int(max_set_proposals_per_parent),
                                                                "max_child_pool_per_parent": int(max_child_pool_per_parent),
                                                                "max_replaced_parents_per_video": int(max_replaced_parents_per_video),
                                                                "min_evidence_island_coverage": None
                                                                if min_evidence_island_coverage is None
                                                                else float(min_evidence_island_coverage),
                                                                "max_residual_active_fraction": None
                                                                if max_residual_active_fraction is None
                                                                else float(max_residual_active_fraction),
                                                                "min_topology_count_alignment": None
                                                                if min_topology_count_alignment is None
                                                                else float(min_topology_count_alignment),
                                                                "min_child_evidence_mass_fraction": None
                                                                if min_child_evidence_mass_fraction is None
                                                                else float(min_child_evidence_mass_fraction),
                                                                "changed_videos": int(changed),
                                                                "segment_delta": int(segment_delta),
                                                            }
                                                            observed_key = (
                                                                float(metrics["segment"]["f1"]),
                                                                float(metrics["segment"]["precision"]),
                                                                float(metrics["segment"]["recall"]),
                                                                float(metrics["frame"]["f1"]),
                                                            )
                                                            if max_fp_increase is not None:
                                                                fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                                                                if fp_delta > int(max_fp_increase):
                                                                    diagnostics["fp_rejected_configs"] += 1
                                                                    rejected = diagnostics.get("best_rejected_metrics")
                                                                    if rejected is None:
                                                                        diagnostics["best_rejected_metrics"] = metrics
                                                                        diagnostics["best_rejected_config"] = {
                                                                            **observed_config,
                                                                            "fp_delta": int(fp_delta),
                                                                        }
                                                                    else:
                                                                        rejected_key = (
                                                                            float(rejected["segment"]["f1"]),
                                                                            float(rejected["segment"]["precision"]),
                                                                            float(rejected["segment"]["recall"]),
                                                                            float(rejected["frame"]["f1"]),
                                                                        )
                                                                        if observed_key > rejected_key:
                                                                            diagnostics["best_rejected_metrics"] = metrics
                                                                            diagnostics["best_rejected_config"] = {
                                                                                **observed_config,
                                                                                "fp_delta": int(fp_delta),
                                                                            }
                                                                    continue
                                                            key = (
                                                                float(metrics["segment"]["f1"]),
                                                                float(metrics["segment"]["precision"]),
                                                                float(metrics["segment"]["recall"]),
                                                                float(metrics["frame"]["f1"]),
                                                                -float(changed + 0.01 * max(0, segment_delta)),
                                                            )
                                                            current_best = diagnostics.get("best_observed_metrics")
                                                            if current_best is None:
                                                                diagnostics["best_observed_metrics"] = metrics
                                                                diagnostics["best_observed_config"] = observed_config
                                                            else:
                                                                current_key = (
                                                                    float(current_best["segment"]["f1"]),
                                                                    float(current_best["segment"]["precision"]),
                                                                    float(current_best["segment"]["recall"]),
                                                                    float(current_best["frame"]["f1"]),
                                                                )
                                                                if key[:4] > current_key:
                                                                    diagnostics["best_observed_metrics"] = metrics
                                                                    diagnostics["best_observed_config"] = observed_config
                                                            if key > best_key:
                                                                best_key = key
                                                                best_metrics = metrics
                                                                best_predictions = predictions
                                                                best_config = {
                                                                    "enabled": True,
                                                                    "model": str(model_name),
                                                                    "threshold": float(threshold),
                                                                    "parent_min_length": int(parent_min_length),
                                                                    "min_child_parent_coverage": float(min_child_parent_coverage),
                                                                    "max_child_parent_ratio": float(max_child_parent_ratio),
                                                                    "max_children_per_parent": int(max_children_per_parent),
                                                                    "max_set_proposals_per_parent": int(max_set_proposals_per_parent),
                                                                    "max_child_pool_per_parent": int(max_child_pool_per_parent),
                                                                    "max_replaced_parents_per_video": int(max_replaced_parents_per_video),
                                                                    "min_evidence_island_coverage": None
                                                                    if min_evidence_island_coverage is None
                                                                    else float(min_evidence_island_coverage),
                                                                    "max_residual_active_fraction": None
                                                                    if max_residual_active_fraction is None
                                                                    else float(max_residual_active_fraction),
                                                                    "min_topology_count_alignment": None
                                                                    if min_topology_count_alignment is None
                                                                    else float(min_topology_count_alignment),
                                                                    "min_child_evidence_mass_fraction": None
                                                                    if min_child_evidence_mass_fraction is None
                                                                    else float(min_child_evidence_mass_fraction),
                                                                    "max_fp_increase": max_fp_increase,
                                                                    "risk_aware": bool(risk_aware),
                                                                    "risk_penalty": score_variant["risk_penalty"],
                                                                    "safety_threshold": None if safety_threshold is None else float(safety_threshold),
                                                                    "safety_weight": float(safety_weight) if bool(risk_aware) else None,
                                                                    "n_train_records": int(len(train_records)),
                                                                    "n_positive_train_records": int(y_train.sum()),
                                                                    "n_gain_train_records": int(sum(record.gain_label for record in train_records)),
                                                                    "n_safe_train_records": int(sum(record.safe_label for record in train_records)),
                                                                    "n_teacher_train_records": int(sum(record.teacher for record in train_records)),
                                                                    "n_val_records": int(len(val_records)),
                                                                    "changed_videos": int(changed),
                                                                    "segment_delta": int(segment_delta),
                                                                    "base_metrics": base_metrics,
                                                                    "diagnostics": diagnostics,
                                                                }
    if not best_config.get("enabled"):
        best_config = {"enabled": False, "base_metrics": base_metrics, "diagnostics": diagnostics}
    return best_config, best_metrics, best_predictions
