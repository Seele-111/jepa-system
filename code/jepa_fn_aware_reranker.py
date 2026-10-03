#!/usr/bin/env python3
"""FN-aware JEPA reranking for current fused prediction misses.

This module trains a narrow rescue scorer against the *active* fused prediction
set: positives are JEPA proposals that cover ground-truth events not already
matched by the current system. The goal is recall recovery without changing the
task into scoring or classification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import NumpyLogisticClassifier, interval_iou
from train_proposal_rescuer import fit_rescuer, matched_ground_truth_indices
from train_proposal_set_selector import extract_set_proposal_features, select_weighted_proposal_set
from train_segment_locator import VideoRecord, contiguous_segments


Segment = tuple[int, int]


@dataclass
class FNAwareCandidateRecord:
    video_idx: int
    segment: Segment
    features: np.ndarray
    label: int
    best_iou: float
    selector_score: float
    candidate_rank: int


@dataclass
class FNAcceptanceCandidateRecord:
    video_idx: int
    segment: Segment
    features: np.ndarray
    label: int
    raw_score: float
    first_stage_score: float
    candidate_rank: int


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _max_iou(segment: tuple[int, int], segments: Iterable[tuple[int, int]]) -> float:
    return max((interval_iou(_normalise_segment(segment), _normalise_segment(other)) for other in segments), default=0.0)


def _distance_to_segments(segment: tuple[int, int], segments: list[tuple[int, int]], n_frames: int) -> float:
    if not segments:
        return 1.0
    start, end = _normalise_segment(segment)
    distances = []
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


def _outside_curve_values(values: np.ndarray, start: int, end: int) -> np.ndarray:
    length = end - start + 1
    left = values[max(0, start - length) : start]
    right = values[end + 1 : min(len(values), end + 1 + length)]
    if len(left) == 0 and len(right) == 0:
        return np.asarray([], dtype=np.float32)
    return np.concatenate([left, right]).astype(np.float32)


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


def _intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    a = _normalise_segment(a)
    b = _normalise_segment(b)
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _evidence_context_features(
    record: VideoRecord,
    segment: tuple[int, int],
    channel_indices: Iterable[int],
) -> np.ndarray:
    """JEPA evidence-demand context for oracle-rescue separability."""
    signals = np.asarray(record.signals, dtype=np.float32)
    indices = [int(idx) for idx in channel_indices]
    if len(signals) == 0 or not indices:
        return np.zeros(8, dtype=np.float32)
    values = np.nan_to_num(signals[:, indices], nan=0.0, posinf=1.0, neginf=0.0)
    curve = np.clip(values.mean(axis=1), 0.0, 1.0).astype(np.float32)
    start, end = _normalise_segment(segment)
    start = max(0, min(start, len(curve) - 1))
    end = max(start, min(end, len(curve) - 1))
    inside = curve[start : end + 1]
    outside = _outside_curve_values(curve, start, end)
    outside_mean = float(outside.mean()) if len(outside) else 0.0
    evidence_mean = float(inside.mean()) if len(inside) else 0.0
    evidence_max = float(inside.max()) if len(inside) else 0.0
    evidence_contrast = float(evidence_mean - outside_mean)
    global_threshold = max(0.5, float(curve.mean() + 0.25 * curve.std()))
    active_fraction = float((inside >= global_threshold).mean()) if len(inside) else 0.0
    islands = _contiguous_bool_segments(curve >= global_threshold)
    island_coverage = max(
        (_intersection((start, end), island) / max(1, _segment_length(island)) for island in islands),
        default=0.0,
    )
    total_mass = float(curve.sum())
    candidate_mass_fraction = float(inside.sum()) / max(total_mass, 1e-6)
    if values.ndim == 2 and values.shape[1] > 1:
        channel_means = np.clip(values[start : end + 1].mean(axis=0), 0.0, 1.0)
        channel_mean = float(channel_means.mean())
        channel_consensus = float(np.clip(channel_means.min() / max(channel_mean, 1e-6), 0.0, 1.0) * channel_mean)
    else:
        channel_consensus = evidence_mean
    rescue_evidence_score = (
        evidence_mean
        + evidence_max
        + max(0.0, evidence_contrast)
        + active_fraction
        + island_coverage
        + channel_consensus
        - 0.25 * candidate_mass_fraction
    )
    return np.asarray(
        [
            evidence_mean,
            evidence_max,
            evidence_contrast,
            active_fraction,
            island_coverage,
            candidate_mass_fraction,
            channel_consensus,
            rescue_evidence_score,
        ],
        dtype=np.float32,
    )


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
    if support_scores:
        support_mean = float(np.mean(support_scores))
        support_max = float(np.max(support_scores))
    else:
        support_mean = 0.0
        support_max = 0.0
    return (
        float(len(support_scores)),
        support_mean,
        support_max,
        float(contains_count),
        float(contained_by_count),
    )


def label_fn_aware_candidates(
    candidates: list[tuple[int, int]],
    labels: np.ndarray,
    base_segments: list[tuple[int, int]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Label candidates that cover GT not already matched by active base predictions."""
    gt_segments = contiguous_segments(np.asarray(labels, dtype=np.int64))
    already_matched = matched_ground_truth_indices(base_segments, labels, iou_threshold=iou_threshold)
    y = np.zeros(len(candidates), dtype=np.int64)
    best = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate_raw in enumerate(candidates):
        candidate = _normalise_segment(candidate_raw)
        best_idx = -1
        for gt_idx, gt in enumerate(gt_segments):
            iou = float(interval_iou(candidate, gt))
            if iou > float(best[idx]):
                best[idx] = iou
                best_idx = gt_idx
        y[idx] = int(best_idx >= 0 and best_idx not in already_matched and best[idx] >= float(iou_threshold))
    return y, best


def select_fn_aware_training_indices(
    labels: np.ndarray,
    scores: np.ndarray,
    max_records_per_video: int | None = None,
    min_selector_score: float | None = None,
) -> np.ndarray:
    """Keep all miss-covering positives and a compact set of high-value negatives."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    score_values = np.nan_to_num(
        np.asarray(scores, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if len(y) != len(score_values):
        raise ValueError("labels and scores must have the same length")
    if len(y) == 0:
        return np.zeros(0, dtype=np.int64)

    keep: set[int] = {int(idx) for idx in np.flatnonzero(y > 0)}
    if min_selector_score is not None:
        keep.update(int(idx) for idx in np.flatnonzero(score_values >= float(min_selector_score)))
    if max_records_per_video is None or int(max_records_per_video) <= 0:
        keep.update(range(len(y)))
    else:
        order = np.argsort(-score_values, kind="mergesort")
        keep.update(int(idx) for idx in order[: int(max_records_per_video)])
    return np.asarray(sorted(keep), dtype=np.int64)


def extract_fn_aware_features(
    record: VideoRecord,
    segment: tuple[int, int],
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    candidate_rank: int,
    channel_indices: Iterable[int],
    base_segments: list[tuple[int, int]],
) -> np.ndarray:
    """Proposal features plus rank/support/current-fusion context."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    norm = _normalise_segment(segment)
    try:
        candidate_idx = [_normalise_segment(item) for item in candidates].index(norm)
        selector_score = float(scores[candidate_idx])
    except ValueError:
        selector_score = 0.0
    n_candidates = max(1, len(candidates))
    n_frames = max(1, len(record.signals))
    duration = _segment_length(norm)
    support_count, support_mean, support_max, contains_count, contained_by_count = _support_stats(norm, candidates, scores)
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
            contained_by_count,
            contains_count,
        ],
        dtype=np.float32,
    )
    base_features = extract_set_proposal_features(record.signals, norm, channel_indices)
    evidence_context = _evidence_context_features(record, norm, channel_indices)
    return np.nan_to_num(
        np.concatenate([base_features, context, evidence_context]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def build_fn_aware_candidate_records(
    records: list[VideoRecord],
    indices: Iterable[int],
    candidate_predictions_by_idx: dict[int, list[tuple[int, int]]],
    candidate_scores_by_idx: dict[int, np.ndarray],
    base_segments_by_idx: dict[int, list[tuple[int, int]]],
    feature_names: list[str],
    channel_names: list[str],
    iou_threshold: float,
    max_records_per_video: int | None = None,
    min_selector_score: float | None = None,
) -> list[FNAwareCandidateRecord]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    out: list[FNAwareCandidateRecord] = []
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
        ranks = _rank_desc(scores)
        labels, best_iou = label_fn_aware_candidates(
            candidates,
            record.labels,
            base_segments_by_idx.get(video_idx, []),
            iou_threshold=iou_threshold,
        )
        keep_indices = select_fn_aware_training_indices(
            labels,
            scores,
            max_records_per_video=max_records_per_video,
            min_selector_score=min_selector_score,
        )
        for keep_idx in keep_indices:
            segment = candidates[int(keep_idx)]
            score = scores[int(keep_idx)]
            rank = ranks[int(keep_idx)]
            label = labels[int(keep_idx)]
            iou = best_iou[int(keep_idx)]
            out.append(
                FNAwareCandidateRecord(
                    video_idx=video_idx,
                    segment=segment,
                    features=extract_fn_aware_features(
                        record,
                        segment,
                        candidates,
                        scores,
                        candidate_rank=int(rank),
                        channel_indices=channel_indices,
                        base_segments=base_segments_by_idx.get(video_idx, []),
                    ),
                    label=int(label),
                    best_iou=float(iou),
                    selector_score=float(score),
                    candidate_rank=int(rank),
                )
            )
    return out


def fit_fn_aware_reranker(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    model_name: str,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    if model_name in {"gbdt", "rf", "extratrees", "logreg", "mlp", "prototype"}:
        return fit_rescuer(
            x,
            y,
            seed=seed,
            model_name=model_name,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )
    return NumpyLogisticClassifier(lr=0.03, steps=1200, l2=1e-3).fit(x, y)


def score_fn_aware_candidates(
    reranker,
    record: VideoRecord,
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    feature_names: list[str],
    channel_names: list[str],
    base_segments: list[tuple[int, int]],
) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    score_values = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    ranks = _rank_desc(score_values)
    features = np.stack(
        [
            extract_fn_aware_features(
                record,
                segment,
                candidates,
                score_values,
                candidate_rank=int(rank),
                channel_indices=channel_indices,
                base_segments=base_segments,
            )
            for segment, rank in zip(candidates, ranks)
        ]
    )
    return reranker.predict_proba(features)[:, 1].astype(np.float32)


def select_fn_aware_reranker_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    rescue_scores: list[np.ndarray],
    threshold: float,
    max_base_iou: float | None,
    nms_iou: float | None,
    length_penalty: float,
    max_rescues_per_video: int,
    max_candidates_per_video: int | None = None,
) -> list[list[Segment]]:
    fused: list[list[Segment]] = []
    for base_segments, candidates, scores_raw in zip(base_predictions, candidate_predictions, rescue_scores):
        base = sorted({_normalise_segment(item) for item in base_segments})
        items = _candidate_rescue_items_for_video(
            base,
            candidates,
            scores_raw,
            threshold=float(threshold),
            max_base_iou=max_base_iou,
            length_penalty=float(length_penalty),
        )
        if max_candidates_per_video is not None and int(max_candidates_per_video) > 0:
            items = items[: int(max_candidates_per_video)]
        filtered_candidates = [segment for segment, _ in items]
        filtered_scores = [float(score) for _, score in items]
        selected = select_weighted_proposal_set(filtered_candidates, filtered_scores, length_penalty=0.0)
        if nms_iou is not None:
            kept: list[Segment] = []
            ordered = sorted(selected, key=lambda seg: max((score for cand, score in zip(filtered_candidates, filtered_scores) if cand == seg), default=0.0), reverse=True)
            for segment in ordered:
                if any(interval_iou(segment, existing) > float(nms_iou) for existing in kept):
                    continue
                kept.append(segment)
                if len(kept) >= int(max_rescues_per_video):
                    break
            selected = sorted(kept)
        elif max_rescues_per_video > 0:
            selected = sorted(selected, key=lambda seg: max((score for cand, score in zip(filtered_candidates, filtered_scores) if cand == seg), default=0.0), reverse=True)
            selected = sorted(selected[: int(max_rescues_per_video)])
        fused.append(sorted(set(base + selected)))
    return fused


def _candidate_rescue_items_for_video(
    base_segments: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    scores_raw: np.ndarray,
    threshold: float,
    max_base_iou: float | None,
    length_penalty: float,
) -> list[tuple[Segment, float]]:
    base = sorted({_normalise_segment(item) for item in base_segments})
    scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    items: list[tuple[Segment, float]] = []
    seen = set(base)
    for candidate_raw, score in zip(candidates, scores):
        candidate = _normalise_segment(candidate_raw)
        if candidate in seen:
            continue
        adjusted = float(score) - float(length_penalty) * math.log1p(_segment_length(candidate))
        if adjusted < float(threshold):
            continue
        if max_base_iou is not None and _max_iou(candidate, base) > float(max_base_iou):
            continue
        items.append((candidate, adjusted))
    items.sort(key=lambda item: (item[1], -_segment_length(item[0]), -item[0][0]), reverse=True)
    return items


def _budgeted_fn_aware_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    rescue_scores: list[np.ndarray],
    labels: list[np.ndarray],
    threshold: float,
    max_base_iou: float | None,
    nms_iou: float | None,
    length_penalty: float,
    max_rescues_per_video: int,
    max_fp_increase: int,
    max_candidates_per_video: int | None,
    iou_threshold: float,
) -> list[list[Segment]]:
    predictions = [[_normalise_segment(item) for item in video] for video in base_predictions]
    base_metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=iou_threshold)
    current_fp = int(base_metrics["segment"]["fp"])
    fp_budget = current_fp + int(max_fp_increase)
    items: list[tuple[int, Segment, float]] = []
    selected_counts = [0 for _ in predictions]
    for video_idx, (base, candidates, scores) in enumerate(zip(predictions, candidate_predictions, rescue_scores)):
        video_items = _candidate_rescue_items_for_video(
            base,
            candidates,
            scores,
            threshold=float(threshold),
            max_base_iou=max_base_iou,
            length_penalty=float(length_penalty),
        )
        if max_candidates_per_video is not None and int(max_candidates_per_video) > 0:
            video_items = video_items[: int(max_candidates_per_video)]
        for segment, score in video_items:
            items.append((int(video_idx), segment, float(score)))
    items.sort(key=lambda item: (item[2], -_segment_length(item[1]), -item[1][0]), reverse=True)
    current_metrics = base_metrics
    for video_idx, segment, _ in items:
        if int(max_rescues_per_video) > 0 and selected_counts[video_idx] >= int(max_rescues_per_video):
            continue
        if segment in predictions[video_idx]:
            continue
        if nms_iou is not None and any(interval_iou(segment, existing) > float(nms_iou) for existing in predictions[video_idx]):
            continue
        trial = [list(video) for video in predictions]
        trial[video_idx] = sorted(set(trial[video_idx] + [segment]))
        metrics = evaluate_fused_predictions(trial, labels, iou_threshold=iou_threshold)
        if int(metrics["segment"]["fp"]) > fp_budget:
            continue
        current_key = (
            float(current_metrics["segment"]["f1"]),
            float(current_metrics["segment"]["precision"]),
            float(current_metrics["segment"]["recall"]),
            float(current_metrics["frame"]["f1"]),
        )
        trial_key = (
            float(metrics["segment"]["f1"]),
            float(metrics["segment"]["precision"]),
            float(metrics["segment"]["recall"]),
            float(metrics["frame"]["f1"]),
        )
        if trial_key <= current_key:
            continue
        predictions = trial
        current_metrics = metrics
        selected_counts[video_idx] += 1
    return [sorted(set(video)) for video in predictions]


def label_fn_acceptance_oracle(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    rescue_scores: list[np.ndarray],
    labels: list[np.ndarray],
    threshold: float,
    max_base_iou: float | None,
    nms_iou: float | None,
    length_penalty: float,
    max_rescues_per_video: int,
    max_fp_increase: int,
    max_candidates_per_video: int | None,
    iou_threshold: float,
) -> list[np.ndarray]:
    """Label candidates accepted by the label-aware budget oracle.

    This is for train-fold teacher construction only. Validation/test inference
    must not call this with validation/test labels.
    """
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    budgeted = _budgeted_fn_aware_predictions(
        base,
        candidate_predictions,
        rescue_scores,
        labels,
        threshold=float(threshold),
        max_base_iou=max_base_iou,
        nms_iou=nms_iou,
        length_penalty=float(length_penalty),
        max_rescues_per_video=int(max_rescues_per_video),
        max_fp_increase=int(max_fp_increase),
        max_candidates_per_video=max_candidates_per_video,
        iou_threshold=float(iou_threshold),
    )
    out: list[np.ndarray] = []
    for base_video, fused_video, candidates in zip(base, budgeted, candidate_predictions):
        accepted = set(_normalise_segment(item) for item in fused_video) - set(base_video)
        candidate_labels = np.zeros(len(candidates), dtype=np.int64)
        for idx, candidate_raw in enumerate(candidates):
            if _normalise_segment(candidate_raw) in accepted:
                candidate_labels[idx] = 1
        out.append(candidate_labels)
    return out


def build_fn_acceptance_candidate_records(
    records: list[VideoRecord],
    indices: Iterable[int],
    candidate_predictions_by_idx: dict[int, list[tuple[int, int]]],
    raw_scores_by_idx: dict[int, np.ndarray],
    first_stage_scores_by_idx: dict[int, np.ndarray],
    base_segments_by_idx: dict[int, list[tuple[int, int]]],
    acceptance_labels_by_idx: dict[int, np.ndarray],
    feature_names: list[str],
    channel_names: list[str],
    max_records_per_video: int | None = None,
    aux_positive_labels_by_idx: dict[int, np.ndarray] | None = None,
) -> list[FNAcceptanceCandidateRecord]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    out: list[FNAcceptanceCandidateRecord] = []
    for video_idx_raw in indices:
        video_idx = int(video_idx_raw)
        record = records[video_idx]
        candidates = [_normalise_segment(item) for item in candidate_predictions_by_idx.get(video_idx, [])]
        raw_scores = np.nan_to_num(
            np.asarray(raw_scores_by_idx.get(video_idx, np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        first_stage_scores = np.nan_to_num(
            np.asarray(first_stage_scores_by_idx.get(video_idx, np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        labels = np.asarray(acceptance_labels_by_idx.get(video_idx, np.zeros(0, dtype=np.int64)), dtype=np.int64).reshape(-1)
        if aux_positive_labels_by_idx is not None:
            aux_labels = np.asarray(aux_positive_labels_by_idx.get(video_idx, np.zeros(0, dtype=np.int64)), dtype=np.int64).reshape(-1)
            if len(aux_labels) == len(labels):
                labels = np.maximum(labels, aux_labels).astype(np.int64)
        if not candidates or not (len(candidates) == len(raw_scores) == len(first_stage_scores) == len(labels)):
            continue
        ranks = _rank_desc(first_stage_scores)
        keep: set[int] = {int(idx) for idx in np.flatnonzero(labels > 0)}
        if max_records_per_video is None or int(max_records_per_video) <= 0:
            keep.update(range(len(candidates)))
        else:
            order = np.argsort(-first_stage_scores, kind="mergesort")[: int(max_records_per_video)]
            keep.update(int(idx) for idx in order)
        for idx in sorted(keep):
            segment = candidates[int(idx)]
            first_stage_score = float(first_stage_scores[int(idx)])
            rank_fraction = float(ranks[int(idx)] / max(1, len(candidates)))
            context = np.asarray(
                [
                    first_stage_score,
                    rank_fraction,
                    math.log1p(float(ranks[int(idx)])),
                ],
                dtype=np.float32,
            )
            base_features = extract_fn_aware_features(
                record,
                segment,
                candidates,
                raw_scores,
                candidate_rank=int(ranks[int(idx)]),
                channel_indices=channel_indices,
                base_segments=base_segments_by_idx.get(video_idx, []),
            )
            out.append(
                FNAcceptanceCandidateRecord(
                    video_idx=video_idx,
                    segment=segment,
                    features=np.nan_to_num(
                        np.concatenate([base_features, context]),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).astype(np.float32),
                    label=int(labels[int(idx)]),
                    raw_score=float(raw_scores[int(idx)]),
                    first_stage_score=first_stage_score,
                    candidate_rank=int(ranks[int(idx)]),
                )
            )
    return out


def score_fn_acceptance_candidates(
    acceptance_model,
    records: list[VideoRecord],
    indices: Iterable[int],
    candidate_predictions_by_idx: dict[int, list[tuple[int, int]]],
    raw_scores_by_idx: dict[int, np.ndarray],
    first_stage_scores_by_idx: dict[int, np.ndarray],
    base_segments_by_idx: dict[int, list[tuple[int, int]]],
    feature_names: list[str],
    channel_names: list[str],
) -> list[np.ndarray]:
    """Score acceptance candidates and return arrays aligned to original candidates."""
    ordered_indices = [int(idx) for idx in indices]
    zero_labels = {
        int(idx): np.zeros(len(candidate_predictions_by_idx.get(int(idx), [])), dtype=np.int64)
        for idx in ordered_indices
    }
    records_for_scoring = build_fn_acceptance_candidate_records(
        records,
        ordered_indices,
        candidate_predictions_by_idx=candidate_predictions_by_idx,
        raw_scores_by_idx=raw_scores_by_idx,
        first_stage_scores_by_idx=first_stage_scores_by_idx,
        base_segments_by_idx=base_segments_by_idx,
        acceptance_labels_by_idx=zero_labels,
        feature_names=feature_names,
        channel_names=channel_names,
        max_records_per_video=None,
    )
    score_lookup: dict[tuple[int, Segment], float] = {}
    if records_for_scoring:
        probabilities = acceptance_model.predict_proba(np.stack([item.features for item in records_for_scoring]))[:, 1]
        score_lookup = {
            (int(record.video_idx), record.segment): float(score)
            for record, score in zip(records_for_scoring, probabilities)
        }
    out: list[np.ndarray] = []
    for idx in ordered_indices:
        candidates = candidate_predictions_by_idx.get(int(idx), [])
        out.append(
            np.asarray(
                [score_lookup.get((int(idx), _normalise_segment(candidate)), 0.0) for candidate in candidates],
                dtype=np.float32,
            )
        )
    return out


def select_fn_aware_reranker_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    rescue_scores: list[np.ndarray],
    labels: list[np.ndarray],
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    max_base_ious: Iterable[float | None] = (0.0, 0.05, 0.1, 0.25, None),
    nms_ious: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01, 0.02),
    max_rescues_per_videos: Iterable[int] = (1, 2, 3),
    max_fp_increase: int | None = None,
    max_candidates_per_video: int | None = None,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=iou_threshold)
    diagnostics = {
        "evaluated_configs": 0,
        "fp_rejected_configs": 0,
        "best_rejected_metrics": None,
        "best_rejected_config": None,
        "max_candidates_per_video": None if max_candidates_per_video is None else int(max_candidates_per_video),
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
    for threshold in thresholds:
        for max_base_iou in max_base_ious:
            for nms_iou in nms_ious:
                for length_penalty in length_penalties:
                    for max_rescues_per_video in max_rescues_per_videos:
                        diagnostics["evaluated_configs"] += 1
                        if max_fp_increase is None:
                            fused = select_fn_aware_reranker_predictions(
                                base,
                                candidate_predictions,
                                rescue_scores,
                                threshold=float(threshold),
                                max_base_iou=None if max_base_iou is None else float(max_base_iou),
                                nms_iou=nms_iou,
                                length_penalty=float(length_penalty),
                                max_rescues_per_video=int(max_rescues_per_video),
                                max_candidates_per_video=max_candidates_per_video,
                            )
                        else:
                            fused = _budgeted_fn_aware_predictions(
                                base,
                                candidate_predictions,
                                rescue_scores,
                                labels,
                                threshold=float(threshold),
                                max_base_iou=None if max_base_iou is None else float(max_base_iou),
                                nms_iou=nms_iou,
                                length_penalty=float(length_penalty),
                                max_rescues_per_video=int(max_rescues_per_video),
                                max_fp_increase=int(max_fp_increase),
                                max_candidates_per_video=max_candidates_per_video,
                                iou_threshold=float(iou_threshold),
                            )
                        metrics = evaluate_fused_predictions(fused, labels, iou_threshold=iou_threshold)
                        rescue_count = sum(max(0, len(after) - len(before)) for before, after in zip(base, fused))
                        fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                        observed_config = {
                            "threshold": float(threshold),
                            "max_base_iou": None if max_base_iou is None else float(max_base_iou),
                            "nms_iou": nms_iou,
                            "length_penalty": float(length_penalty),
                            "max_rescues_per_video": int(max_rescues_per_video),
                            "rescue_segments": int(rescue_count),
                            "fp_delta": int(fp_delta),
                        }
                        if max_fp_increase is not None and fp_delta > int(max_fp_increase):
                            diagnostics["fp_rejected_configs"] += 1
                            rejected = diagnostics.get("best_rejected_metrics")
                            rejected_key = (
                                -1.0,
                                -1.0,
                                -1.0,
                                -1.0,
                            )
                            if rejected is not None:
                                rejected_key = (
                                    float(rejected["segment"]["f1"]),
                                    float(rejected["segment"]["precision"]),
                                    float(rejected["segment"]["recall"]),
                                    float(rejected["frame"]["f1"]),
                                )
                            current_rejected_key = (
                                float(metrics["segment"]["f1"]),
                                float(metrics["segment"]["precision"]),
                                float(metrics["segment"]["recall"]),
                                float(metrics["frame"]["f1"]),
                            )
                            if current_rejected_key > rejected_key:
                                diagnostics["best_rejected_metrics"] = metrics
                                diagnostics["best_rejected_config"] = observed_config
                            continue
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
                                **observed_config,
                                "max_fp_increase": max_fp_increase,
                                "max_candidates_per_video": None
                                if max_candidates_per_video is None
                                else int(max_candidates_per_video),
                                "base_metrics": base_metrics,
                                "diagnostics": diagnostics,
                            }
    if not best_config.get("enabled"):
        best_config = {
            "enabled": False,
            "base_metrics": base_metrics,
            "max_fp_increase": max_fp_increase,
            "max_candidates_per_video": None if max_candidates_per_video is None else int(max_candidates_per_video),
            "diagnostics": diagnostics,
        }
    return best_config, best_metrics, best_predictions


def select_strict_fn_aware_reranker_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    rescue_scores: list[np.ndarray],
    labels: list[np.ndarray],
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    max_base_ious: Iterable[float | None] = (0.0, 0.05, 0.1, 0.25, None),
    nms_ious: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01, 0.02),
    max_rescues_per_videos: Iterable[int] = (1, 2, 3),
    max_fp_increase: int | None = None,
    max_candidates_per_video: int | None = None,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    """Select FN-aware params using labels only for config evaluation.

    Unlike the optimistic budgeted oracle path in ``select_fn_aware_reranker_params``,
    this function never uses labels to accept or reject individual candidates.
    Labels only evaluate a fully label-free prediction set, which makes it suitable
    for inner calibration before applying the frozen config to an outer fold.
    """
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=iou_threshold)
    diagnostics = {
        "evaluated_configs": 0,
        "fp_rejected_configs": 0,
        "best_rejected_metrics": None,
        "best_rejected_config": None,
        "max_candidates_per_video": None if max_candidates_per_video is None else int(max_candidates_per_video),
        "strict_label_free": True,
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
    for threshold in thresholds:
        for max_base_iou in max_base_ious:
            for nms_iou in nms_ious:
                for length_penalty in length_penalties:
                    for max_rescues_per_video in max_rescues_per_videos:
                        diagnostics["evaluated_configs"] += 1
                        fused = select_fn_aware_reranker_predictions(
                            base,
                            candidate_predictions,
                            rescue_scores,
                            threshold=float(threshold),
                            max_base_iou=None if max_base_iou is None else float(max_base_iou),
                            nms_iou=nms_iou,
                            length_penalty=float(length_penalty),
                            max_rescues_per_video=int(max_rescues_per_video),
                            max_candidates_per_video=max_candidates_per_video,
                        )
                        metrics = evaluate_fused_predictions(fused, labels, iou_threshold=iou_threshold)
                        rescue_count = sum(max(0, len(after) - len(before)) for before, after in zip(base, fused))
                        fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                        observed_config = {
                            "threshold": float(threshold),
                            "max_base_iou": None if max_base_iou is None else float(max_base_iou),
                            "nms_iou": nms_iou,
                            "length_penalty": float(length_penalty),
                            "max_rescues_per_video": int(max_rescues_per_video),
                            "rescue_segments": int(rescue_count),
                            "fp_delta": int(fp_delta),
                            "strict_label_free": True,
                        }
                        if max_fp_increase is not None and fp_delta > int(max_fp_increase):
                            diagnostics["fp_rejected_configs"] += 1
                            rejected = diagnostics.get("best_rejected_metrics")
                            rejected_key = (-1.0, -1.0, -1.0, -1.0)
                            if rejected is not None:
                                rejected_key = (
                                    float(rejected["segment"]["f1"]),
                                    float(rejected["segment"]["precision"]),
                                    float(rejected["segment"]["recall"]),
                                    float(rejected["frame"]["f1"]),
                                )
                            current_rejected_key = (
                                float(metrics["segment"]["f1"]),
                                float(metrics["segment"]["precision"]),
                                float(metrics["segment"]["recall"]),
                                float(metrics["frame"]["f1"]),
                            )
                            if current_rejected_key > rejected_key:
                                diagnostics["best_rejected_metrics"] = metrics
                                diagnostics["best_rejected_config"] = observed_config
                            continue
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
                                **observed_config,
                                "max_fp_increase": max_fp_increase,
                                "max_candidates_per_video": None
                                if max_candidates_per_video is None
                                else int(max_candidates_per_video),
                                "base_metrics": base_metrics,
                                "diagnostics": diagnostics,
                            }
    if not best_config.get("enabled"):
        best_config = {
            "enabled": False,
            "base_metrics": base_metrics,
            "max_fp_increase": max_fp_increase,
            "max_candidates_per_video": None if max_candidates_per_video is None else int(max_candidates_per_video),
            "diagnostics": diagnostics,
        }
    return best_config, best_metrics, best_predictions


def apply_fn_aware_reranker_config(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    rescue_scores: list[np.ndarray],
    config: dict,
) -> list[list[Segment]]:
    """Apply a frozen strict-calibration config without labels."""
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    if not config.get("enabled"):
        return base
    return select_fn_aware_reranker_predictions(
        base,
        candidate_predictions,
        rescue_scores,
        threshold=float(config["threshold"]),
        max_base_iou=None if config.get("max_base_iou") is None else float(config["max_base_iou"]),
        nms_iou=config.get("nms_iou"),
        length_penalty=float(config.get("length_penalty", 0.0)),
        max_rescues_per_video=int(config.get("max_rescues_per_video", 1)),
        max_candidates_per_video=config.get("max_candidates_per_video"),
    )


def split_indices_for_inner_calibration(
    indices: Iterable[int],
    labels_by_idx: dict[int, np.ndarray],
    calibration_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministically split outer-train videos into inner-train/calibration."""
    values = [int(idx) for idx in indices]
    if len(values) < 3:
        return values, []
    fraction = min(0.8, max(0.05, float(calibration_fraction)))
    rng = np.random.default_rng(int(seed))
    positive = [idx for idx in values if int(np.asarray(labels_by_idx.get(idx, np.zeros(0))).sum()) > 0]
    negative = [idx for idx in values if idx not in set(positive)]

    def choose(group: list[int]) -> set[int]:
        if not group:
            return set()
        shuffled = list(group)
        rng.shuffle(shuffled)
        count = int(round(len(shuffled) * fraction))
        if len(shuffled) >= 2:
            count = min(len(shuffled) - 1, max(1, count))
        else:
            count = 1
        return set(int(idx) for idx in shuffled[:count])

    calib = choose(positive) | choose(negative)
    if not calib:
        shuffled = list(values)
        rng.shuffle(shuffled)
        calib.add(int(shuffled[0]))
    inner = [idx for idx in values if idx not in calib]
    if not inner:
        moved = sorted(calib)[0]
        calib.remove(moved)
        inner = [moved]
    return sorted(inner), sorted(calib)
