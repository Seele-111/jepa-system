#!/usr/bin/env python3
"""Distill JEPA oracle proposal sets into a test-time event selector.

Training folds can see labels, so they can ask an oracle which JEPA proposals
would rescue current false negatives without increasing false positives. This
module distills that set-level teacher into a candidate scorer used on
validation/test folds without labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from jepa_oracle_diagnostics import oracle_rescue_for_video
from jepa_topology_set_splitter import select_topology_set_split_params
from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou
from train_proposal_rescuer import fit_rescuer
from train_proposal_set_selector import (
    candidate_quality_targets,
    extract_set_proposal_features,
    fit_classifier,
    fit_quality_model,
    select_weighted_proposal_set,
)
from train_segment_locator import VideoRecord, contiguous_segments


Segment = tuple[int, int]


@dataclass
class DistillationCandidateRecord:
    video_idx: int
    segment: Segment
    features: np.ndarray
    label: int
    best_iou: float
    selector_score: float
    candidate_rank: int


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _max_iou(segment: tuple[int, int], segments: Iterable[tuple[int, int]]) -> float:
    norm = _normalise_segment(segment)
    return max((float(interval_iou(norm, _normalise_segment(other))) for other in segments), default=0.0)


def _distance_to_segments(segment: tuple[int, int], segments: list[tuple[int, int]], n_frames: int) -> float:
    if not segments:
        return 1.0
    start, end = _normalise_segment(segment)
    distances: list[int] = []
    for other_raw in segments:
        left, right = _normalise_segment(other_raw)
        if end < left:
            distances.append(left - end)
        elif right < start:
            distances.append(start - right)
        else:
            distances.append(0)
    return float(min(distances) / max(1, n_frames))


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def _segment_overlap_fraction(segment: tuple[int, int], other: tuple[int, int]) -> float:
    segment = _normalise_segment(segment)
    other = _normalise_segment(other)
    overlap = max(0, min(segment[1], other[1]) - max(segment[0], other[0]) + 1)
    return float(overlap / max(1, _segment_length(segment)))


def _support_stats(
    segment: tuple[int, int],
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    support_iou: float = 0.2,
) -> tuple[float, float, float, float, float]:
    norm = _normalise_segment(segment)
    values = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    support_scores: list[float] = []
    contains_count = 0
    contained_by_count = 0
    for other_raw, score in zip(candidates, values):
        other = _normalise_segment(other_raw)
        if other == norm:
            continue
        if interval_iou(norm, other) >= float(support_iou):
            support_scores.append(float(score))
        if norm[0] <= other[0] and other[1] <= norm[1] and _segment_length(other) < _segment_length(norm):
            contains_count += 1
        if other[0] <= norm[0] and norm[1] <= other[1] and _segment_length(other) > _segment_length(norm):
            contained_by_count += 1
    return (
        float(len(support_scores)),
        float(np.mean(support_scores)) if support_scores else 0.0,
        float(np.max(support_scores)) if support_scores else 0.0,
        float(contains_count),
        float(contained_by_count),
    )


def _best_iou_to_gt(candidates: list[Segment], labels: np.ndarray) -> np.ndarray:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    best = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        best[idx] = max((float(interval_iou(candidate, gt)) for gt in gt_segments), default=0.0)
    return best


def _matched_gt_indices(
    predictions: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float,
) -> set[int]:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched: set[int] = set()
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
    return matched


def _overlap_teacher_candidates(
    candidates: list[Segment],
    labels: np.ndarray,
    base_segments: list[tuple[int, int]],
    iou_threshold: float,
    max_teacher_rescues: int,
    teacher_nms_iou: float | None,
) -> list[Segment]:
    """Select compact candidates covering GT events, including those inside wide base segments."""
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    if not gt_segments:
        return []
    matched = _matched_gt_indices(base_segments, labels, iou_threshold=float(iou_threshold))
    selected: list[Segment] = []
    used: set[Segment] = set()
    for gt_idx, gt in enumerate(gt_segments):
        if gt_idx in matched:
            continue
        best_candidate: Segment | None = None
        best_key: tuple[float, float, float, float] | None = None
        for candidate in candidates:
            if candidate in used:
                continue
            iou = float(interval_iou(candidate, gt))
            if iou < float(iou_threshold):
                continue
            if teacher_nms_iou is not None and any(
                interval_iou(candidate, existing) > float(teacher_nms_iou) for existing in selected
            ):
                continue
            length_ratio = min(_segment_length(candidate), _segment_length(gt)) / max(
                _segment_length(candidate),
                _segment_length(gt),
            )
            center_gap = abs((candidate[0] + candidate[1]) - (gt[0] + gt[1])) / 2.0
            base_iou = _max_iou(candidate, base_segments)
            key = (iou, length_ratio, base_iou, -center_gap)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate
        if best_candidate is not None:
            used.add(best_candidate)
            selected.append(best_candidate)
        if len(selected) >= int(max_teacher_rescues):
            break
    return sorted(selected, key=lambda item: (_segment_length(item), item[0], item[1]))


def _coverage(segment: tuple[int, int], parent: tuple[int, int]) -> float:
    segment = _normalise_segment(segment)
    parent = _normalise_segment(parent)
    overlap = max(0, min(segment[1], parent[1]) - max(segment[0], parent[0]) + 1)
    return float(overlap / max(1, _segment_length(segment)))


def _topology_teacher_candidates(
    candidates: list[Segment],
    labels: np.ndarray,
    base_segments: list[tuple[int, int]],
    iou_threshold: float,
    max_teacher_rescues: int,
    teacher_nms_iou: float | None,
    parent_min_length: int = 8,
    min_parent_gt_count: int = 2,
    max_child_parent_ratio: float = 0.75,
    min_child_parent_coverage: float = 0.5,
) -> list[Segment]:
    """Select a complete child set for wide parents covering multiple GT events."""
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    if not gt_segments:
        return []
    selected: list[Segment] = []
    used: set[Segment] = set()
    for parent_raw in base_segments:
        parent = _normalise_segment(parent_raw)
        parent_length = _segment_length(parent)
        if parent_length < int(parent_min_length):
            continue
        inside_gts = [gt for gt in gt_segments if _coverage(gt, parent) >= 0.8]
        if len(inside_gts) < int(min_parent_gt_count):
            continue
        parent_children: list[Segment] = []
        for gt in inside_gts:
            best_candidate: Segment | None = None
            best_key: tuple[float, float, float, float] | None = None
            for candidate in candidates:
                if candidate in used or candidate in parent_children:
                    continue
                if candidate == parent:
                    continue
                if _segment_length(candidate) >= parent_length:
                    continue
                if _segment_length(candidate) / max(1, parent_length) > float(max_child_parent_ratio):
                    continue
                if _coverage(candidate, parent) < float(min_child_parent_coverage):
                    continue
                if teacher_nms_iou is not None and any(
                    interval_iou(candidate, existing) > float(teacher_nms_iou)
                    for existing in selected + parent_children
                ):
                    continue
                iou = float(interval_iou(candidate, gt))
                if iou < float(iou_threshold):
                    continue
                length_ratio = min(_segment_length(candidate), _segment_length(gt)) / max(
                    _segment_length(candidate),
                    _segment_length(gt),
                )
                center_gap = abs((candidate[0] + candidate[1]) - (gt[0] + gt[1])) / 2.0
                key = (iou, length_ratio, -center_gap, -float(_segment_length(candidate)))
                if best_key is None or key > best_key:
                    best_key = key
                    best_candidate = candidate
            if best_candidate is not None:
                parent_children.append(best_candidate)
        if len(parent_children) >= int(min_parent_gt_count):
            for child in sorted(parent_children, key=lambda item: (_segment_length(item), item[0], item[1])):
                if child not in used:
                    used.add(child)
                    selected.append(child)
                if len(selected) >= int(max_teacher_rescues):
                    return selected
    return selected


def label_distillation_candidates(
    candidates: list[tuple[int, int]],
    labels: np.ndarray,
    base_segments: list[tuple[int, int]],
    iou_threshold: float,
    max_teacher_rescues: int = 2,
    max_base_iou: float | None = 0.0,
    teacher_nms_iou: float | None = 0.3,
    teacher_mode: str = "oracle",
) -> tuple[np.ndarray, np.ndarray, list[Segment]]:
    """Label exactly the JEPA candidates selected by the train-fold oracle teacher."""
    norm_candidates = [_normalise_segment(item) for item in candidates]
    best_iou = _best_iou_to_gt(norm_candidates, labels)
    teacher = oracle_rescue_for_video(
        base_segments,
        norm_candidates,
        np.asarray(labels, dtype=np.int64),
        iou_threshold=float(iou_threshold),
        max_rescues_per_video=int(max_teacher_rescues),
        max_base_iou=max_base_iou,
        selector_nms_iou=teacher_nms_iou,
    )
    if teacher_mode not in {"oracle", "oracle_or_overlap", "oracle_or_topology"}:
        raise ValueError(f"unsupported distillation teacher_mode: {teacher_mode}")
    if teacher_mode == "oracle_or_overlap":
        overlap_teacher = _overlap_teacher_candidates(
            norm_candidates,
            labels,
            base_segments,
            iou_threshold=float(iou_threshold),
            max_teacher_rescues=int(max_teacher_rescues),
            teacher_nms_iou=teacher_nms_iou,
        )
        teacher = sorted(set(teacher) | set(overlap_teacher), key=lambda item: (_segment_length(item), item[0], item[1]))
        teacher = teacher[: int(max_teacher_rescues)]
    if teacher_mode == "oracle_or_topology":
        topology_teacher = _topology_teacher_candidates(
            norm_candidates,
            labels,
            base_segments,
            iou_threshold=float(iou_threshold),
            max_teacher_rescues=int(max_teacher_rescues),
            teacher_nms_iou=teacher_nms_iou,
        )
        teacher = sorted(set(teacher) | set(topology_teacher), key=lambda item: (_segment_length(item), item[0], item[1]))
        teacher = teacher[: int(max_teacher_rescues)]
    teacher_set = set(teacher)
    y = np.asarray([int(candidate in teacher_set) for candidate in norm_candidates], dtype=np.int64)
    return y, best_iou.astype(np.float32), list(teacher)


def _select_distillation_training_indices(
    labels: np.ndarray,
    scores: np.ndarray,
    best_iou: np.ndarray,
    base_ious: np.ndarray,
    max_records_per_video: int | None,
    hard_negative_top_k: int,
    min_selector_score: float | None,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    score_values = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    best_values = np.nan_to_num(np.asarray(best_iou, dtype=np.float32).reshape(-1), nan=0.0)
    base_values = np.nan_to_num(np.asarray(base_ious, dtype=np.float32).reshape(-1), nan=0.0)
    if not (len(y) == len(score_values) == len(best_values) == len(base_values)):
        raise ValueError("labels/scores/best_iou/base_ious must have the same length")
    if len(y) == 0:
        return np.zeros(0, dtype=np.int64)

    keep: set[int] = {int(idx) for idx in np.flatnonzero(y > 0)}
    if min_selector_score is not None:
        keep.update(int(idx) for idx in np.flatnonzero(score_values >= float(min_selector_score)))
    top_k = int(max_records_per_video or 0)
    if top_k <= 0:
        keep.update(range(len(y)))
    else:
        keep.update(int(idx) for idx in np.argsort(-score_values, kind="mergesort")[:top_k])

    hard_k = max(0, int(hard_negative_top_k))
    if hard_k > 0:
        negative = np.flatnonzero(y <= 0)
        if len(negative):
            hard_score = score_values[negative] + best_values[negative] + base_values[negative]
            hard_order = negative[np.argsort(-hard_score, kind="mergesort")[:hard_k]]
            keep.update(int(idx) for idx in hard_order)
    return np.asarray(sorted(keep), dtype=np.int64)


def extract_distillation_features(
    record: VideoRecord,
    segment: tuple[int, int],
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    candidate_rank: int,
    channel_indices: Iterable[int],
    base_segments: list[tuple[int, int]],
) -> np.ndarray:
    """Proposal features plus set context used to imitate the oracle teacher."""
    norm = _normalise_segment(segment)
    score_values = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    norm_candidates = [_normalise_segment(item) for item in candidates]
    try:
        candidate_idx = norm_candidates.index(norm)
        selector_score = float(score_values[candidate_idx])
    except ValueError:
        selector_score = 0.0
    n_candidates = max(1, len(norm_candidates))
    n_frames = max(1, len(record.signals))
    duration = _segment_length(norm)
    support_count, support_mean, support_max, contains_count, contained_by_count = _support_stats(
        norm,
        norm_candidates,
        score_values,
    )
    base_iou = _max_iou(norm, base_segments)
    base_coverage = max((_segment_overlap_fraction(norm, base) for base in base_segments), default=0.0)
    context = np.asarray(
        [
            selector_score,
            float(candidate_rank) / float(n_candidates),
            math.log1p(float(candidate_rank)),
            math.log1p(float(n_candidates)),
            base_iou,
            base_coverage,
            _distance_to_segments(norm, [_normalise_segment(item) for item in base_segments], n_frames),
            float(duration / n_frames),
            math.log1p(float(duration)),
            support_count,
            support_mean,
            support_max,
            contains_count,
            contained_by_count,
        ],
        dtype=np.float32,
    )
    base_features = extract_set_proposal_features(record.signals, norm, channel_indices)
    return np.nan_to_num(np.concatenate([base_features, context]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_distillation_candidate_records(
    records: list[VideoRecord],
    indices: Iterable[int],
    candidate_predictions_by_idx: dict[int, list[tuple[int, int]]],
    candidate_scores_by_idx: dict[int, np.ndarray],
    base_segments_by_idx: dict[int, list[tuple[int, int]]],
    feature_names: list[str],
    channel_names: list[str],
    iou_threshold: float,
    max_teacher_rescues: int = 2,
    max_base_iou: float | None = 0.0,
    teacher_nms_iou: float | None = 0.3,
    teacher_mode: str = "oracle",
    max_records_per_video: int | None = 120,
    hard_negative_top_k: int = 40,
    min_selector_score: float | None = None,
) -> list[DistillationCandidateRecord]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    out: list[DistillationCandidateRecord] = []
    for video_idx_raw in indices:
        video_idx = int(video_idx_raw)
        record = records[video_idx]
        candidates = [_normalise_segment(item) for item in candidate_predictions_by_idx.get(video_idx, [])]
        scores = np.nan_to_num(
            np.asarray(candidate_scores_by_idx.get(video_idx, np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        if not candidates or len(candidates) != len(scores):
            continue
        base_segments = [_normalise_segment(item) for item in base_segments_by_idx.get(video_idx, [])]
        labels, best_iou, _ = label_distillation_candidates(
            candidates,
            record.labels,
            base_segments,
            iou_threshold=float(iou_threshold),
            max_teacher_rescues=int(max_teacher_rescues),
            max_base_iou=max_base_iou,
            teacher_nms_iou=teacher_nms_iou,
            teacher_mode=teacher_mode,
        )
        base_ious = np.asarray([_max_iou(candidate, base_segments) for candidate in candidates], dtype=np.float32)
        keep_indices = _select_distillation_training_indices(
            labels,
            scores,
            best_iou,
            base_ious,
            max_records_per_video=max_records_per_video,
            hard_negative_top_k=int(hard_negative_top_k),
            min_selector_score=min_selector_score,
        )
        ranks = _rank_desc(scores)
        for keep_idx in keep_indices:
            idx = int(keep_idx)
            segment = candidates[idx]
            out.append(
                DistillationCandidateRecord(
                    video_idx=video_idx,
                    segment=segment,
                    features=extract_distillation_features(
                        record,
                        segment,
                        candidates,
                        scores,
                        candidate_rank=int(ranks[idx]),
                        channel_indices=channel_indices,
                        base_segments=base_segments,
                    ),
                    label=int(labels[idx]),
                    best_iou=float(best_iou[idx]),
                    selector_score=float(scores[idx]),
                    candidate_rank=int(ranks[idx]),
                )
            )
    return out


def fit_distillation_student(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    model_name: str,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    return fit_rescuer(
        x,
        y,
        seed=seed,
        model_name=model_name,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
    )


def fit_soft_quality_distillation_student(
    x: np.ndarray,
    best_iou: np.ndarray,
    seed: int,
    model_name: str,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    """Fit a student from dense soft IoU-quality teacher targets.

    Unlike the sparse oracle-rescue labels, this keeps all near-correct
    candidates informative while still using the base-aware distillation
    features from this module.
    """
    x_arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    iou_arr = np.clip(np.asarray(best_iou, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if len(x_arr) != len(iou_arr):
        raise ValueError(f"x/best_iou count mismatch: {len(x_arr)} vs {len(iou_arr)}")
    if model_name == "mlp":
        return fit_classifier(
            x_arr,
            candidate_quality_targets(iou_arr),
            seed=seed,
            model_name="mlp",
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )
    if model_name == "prototype":
        hard = (iou_arr >= 0.3).astype(np.int64)
        return fit_distillation_student(
            x_arr,
            hard,
            seed=seed,
            model_name="prototype",
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )
    return fit_quality_model(x_arr, iou_arr, seed=seed, model_name=model_name)


def score_distillation_candidates(
    student,
    record: VideoRecord,
    candidates: list[tuple[int, int]],
    selector_scores: np.ndarray,
    feature_names: list[str],
    channel_names: list[str],
    base_segments: list[tuple[int, int]],
) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    scores = np.nan_to_num(np.asarray(selector_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(candidates) != len(scores):
        raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    ranks = _rank_desc(scores)
    features = np.stack(
        [
            extract_distillation_features(
                record,
                segment,
                candidates,
                scores,
                candidate_rank=int(rank),
                channel_indices=channel_indices,
                base_segments=base_segments,
            )
            for segment, rank in zip(candidates, ranks)
        ]
    )
    return student.predict_proba(features)[:, 1].astype(np.float32)


def select_distilled_proposal_set_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    student_scores: list[np.ndarray],
    threshold: float,
    max_base_iou: float | None,
    nms_iou: float | None,
    length_penalty: float,
    max_rescues_per_video: int,
) -> list[list[Segment]]:
    if len(base_predictions) != len(candidate_predictions) or len(base_predictions) != len(student_scores):
        raise ValueError("base/candidate/score video counts must match")
    fused: list[list[Segment]] = []
    for base_raw, candidates_raw, scores_raw in zip(base_predictions, candidate_predictions, student_scores):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        filtered_candidates: list[Segment] = []
        filtered_scores: list[float] = []
        for candidate, score in zip(candidates, scores):
            adjusted = float(score) - float(length_penalty) * math.log1p(_segment_length(candidate))
            if adjusted < float(threshold):
                continue
            if max_base_iou is not None and _max_iou(candidate, base) > float(max_base_iou):
                continue
            filtered_candidates.append(candidate)
            filtered_scores.append(adjusted)
        selected = select_weighted_proposal_set(filtered_candidates, filtered_scores, length_penalty=0.0)
        if nms_iou is not None:
            score_lookup = {candidate: score for candidate, score in zip(filtered_candidates, filtered_scores)}
            kept: list[Segment] = []
            for segment in sorted(selected, key=lambda item: score_lookup.get(item, 0.0), reverse=True):
                if any(interval_iou(segment, existing) > float(nms_iou) for existing in kept):
                    continue
                kept.append(segment)
                if len(kept) >= int(max_rescues_per_video):
                    break
            selected = sorted(kept)
        elif max_rescues_per_video > 0:
            score_lookup = {candidate: score for candidate, score in zip(filtered_candidates, filtered_scores)}
            selected = sorted(selected, key=lambda item: score_lookup.get(item, 0.0), reverse=True)
            selected = sorted(selected[: int(max_rescues_per_video)])
        fused.append(sorted(set(base + selected)))
    return fused


def select_distilled_proposal_set_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    student_scores: list[np.ndarray],
    labels: list[np.ndarray],
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    max_base_ious: Iterable[float | None] = (0.0, 0.05, 0.1, 0.25, None),
    nms_ious: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01, 0.02),
    max_rescues_per_videos: Iterable[int] = (1, 2, 3),
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
    for threshold in thresholds:
        for max_base_iou in max_base_ious:
            for nms_iou in nms_ious:
                for length_penalty in length_penalties:
                    for max_rescues_per_video in max_rescues_per_videos:
                        fused = select_distilled_proposal_set_predictions(
                            base,
                            candidate_predictions,
                            student_scores,
                            threshold=float(threshold),
                            max_base_iou=None if max_base_iou is None else float(max_base_iou),
                            nms_iou=nms_iou,
                            length_penalty=float(length_penalty),
                            max_rescues_per_video=int(max_rescues_per_video),
                        )
                        metrics = evaluate_fused_predictions(fused, labels, iou_threshold=iou_threshold)
                        rescue_count = sum(max(0, len(after) - len(before)) for before, after in zip(base, fused))
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
                                "threshold": float(threshold),
                                "max_base_iou": None if max_base_iou is None else float(max_base_iou),
                                "nms_iou": nms_iou,
                                "length_penalty": float(length_penalty),
                                "max_rescues_per_video": int(max_rescues_per_video),
                                "rescue_segments": int(rescue_count),
                                "base_metrics": base_metrics,
                            }
    return best_config, best_metrics, best_predictions


def select_distilled_topology_replacement_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    student_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    evidence_thresholds: Iterable[float] = (0.4, 0.5, 0.6),
    min_student_scores: Iterable[float] = (0.0, 0.05, 0.1),
    max_student_ranks: Iterable[int | None] = (None, 80),
    min_evidence_means: Iterable[float] = (0.2, 0.3, 0.4),
    min_active_fractions: Iterable[float] = (0.0, 0.1, 0.25),
    min_parent_contrasts: Iterable[float] = (-0.3, -0.1, 0.0),
    parent_min_lengths: Iterable[int] = (12, 24),
    max_child_parent_ratios: Iterable[float] = (0.5, 0.7),
    min_child_parent_coverages: Iterable[float] = (0.5, 0.7),
    min_gap_between_children_values: Iterable[int] = (0,),
    child_score_thresholds: Iterable[float] = (0.0, 0.2),
    set_score_thresholds: Iterable[float] = (0.1, 0.2, 0.3),
    student_weights: Iterable[float] = (0.5, 1.0),
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
    """Use distilled student scores as topology child scores for parent replacement."""
    config, metrics, predictions = select_topology_set_split_params(
        base_predictions,
        candidate_predictions,
        student_scores,
        evidence_arrays,
        labels,
        evidence_thresholds=evidence_thresholds,
        min_selector_scores=min_student_scores,
        max_selector_ranks=max_student_ranks,
        min_evidence_means=min_evidence_means,
        min_active_fractions=min_active_fractions,
        min_parent_contrasts=min_parent_contrasts,
        parent_min_lengths=parent_min_lengths,
        max_child_parent_ratios=max_child_parent_ratios,
        min_child_parent_coverages=min_child_parent_coverages,
        min_gap_between_children_values=min_gap_between_children_values,
        child_score_thresholds=child_score_thresholds,
        set_score_thresholds=set_score_thresholds,
        selector_weights=student_weights,
        evidence_weights=evidence_weights,
        active_fraction_weights=active_fraction_weights,
        contrast_weights=contrast_weights,
        length_penalties=length_penalties,
        nms_ious=nms_ious,
        min_children_per_parents=min_children_per_parents,
        max_children_per_parents=max_children_per_parents,
        max_replaced_parents_per_videos=max_replaced_parents_per_videos,
        max_fp_increase=max_fp_increase,
        iou_threshold=iou_threshold,
    )
    if config.get("enabled"):
        config = {
            **config,
            "distilled_topology_replacement": True,
            "student_score_source": "proposal_set_distillation",
        }
    return config, metrics, predictions
