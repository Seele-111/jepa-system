#!/usr/bin/env python3
"""JEPA evidence-set reasoner for low-score missed event proposals.

The selector can suppress real error fragments when their learned proposal
score is low. This module treats V-JEPA, I-JEPA, and dual-JEPA evidence as the
primary signal, then performs conservative event-set selection against the
current fused prediction set.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]
ReasonerStats = dict[str, np.ndarray]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _max_iou(segment: tuple[int, int], segments: Iterable[tuple[int, int]]) -> float:
    norm = _normalise_segment(segment)
    return max((float(interval_iou(norm, _normalise_segment(other))) for other in segments), default=0.0)


def _segment_matrix(evidence: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(values) == 0:
        return np.zeros((0, values.shape[1] if values.ndim == 2 else 1), dtype=np.float32)
    start, end = _normalise_segment(segment)
    start = max(0, min(start, len(values) - 1))
    end = max(start, min(end, len(values) - 1))
    return values[start : end + 1].astype(np.float32)


def _context_matrix(evidence: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(values) == 0:
        return np.zeros((0, values.shape[1] if values.ndim == 2 else 1), dtype=np.float32)
    start, end = _normalise_segment(segment)
    length = _segment_length((start, end))
    left = values[max(0, start - length) : max(0, start)]
    right = values[min(len(values), end + 1) : min(len(values), end + 1 + length)]
    if len(left) == 0 and len(right) == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    return np.concatenate([left, right], axis=0).astype(np.float32)


def _evidence_stats(
    evidence: np.ndarray,
    segment: tuple[int, int],
    evidence_threshold: float,
) -> tuple[float, float, float, float]:
    inside = _segment_matrix(evidence, segment)
    if len(inside) == 0:
        return 0.0, 0.0, -1.0, 0.0
    per_frame = inside.mean(axis=1)
    evidence_mean = float(per_frame.mean())
    active_fraction = float((per_frame >= float(evidence_threshold)).mean())
    context = _context_matrix(evidence, segment)
    context_mean = float(context.mean(axis=1).mean()) if len(context) else 0.0
    contrast = evidence_mean - context_mean
    if inside.shape[1] >= 2:
        channel_means = inside.mean(axis=0)
        consensus = float(np.min(channel_means[: min(3, len(channel_means))]))
    else:
        consensus = evidence_mean
    return evidence_mean, active_fraction, contrast, consensus


def _support_count(
    segment: tuple[int, int],
    candidates: list[tuple[int, int]],
    support_iou: float,
) -> int:
    norm = _normalise_segment(segment)
    return int(
        sum(
            1
            for other_raw in candidates
            if _normalise_segment(other_raw) != norm
            and interval_iou(norm, _normalise_segment(other_raw)) >= float(support_iou)
        )
    )


def precompute_reasoner_stats(
    base_segments: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    selector_scores: np.ndarray,
    evidence: np.ndarray,
    evidence_threshold: float,
    support_iou: float,
    prefilter_top_k: int | None = None,
) -> ReasonerStats:
    """Precompute reusable per-candidate evidence stats for one video."""
    norm_candidates = [_normalise_segment(segment) for segment in candidates]
    scores = np.nan_to_num(
        np.asarray(selector_scores, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if len(norm_candidates) != len(scores):
        raise ValueError(f"candidate/score count mismatch: {len(norm_candidates)} vs {len(scores)}")
    candidate_indices = list(range(len(norm_candidates)))
    if prefilter_top_k is not None and int(prefilter_top_k) > 0 and len(norm_candidates) > int(prefilter_top_k):
        quick_scores = []
        for candidate in norm_candidates:
            evidence_mean, active_fraction, contrast, consensus = _evidence_stats(
                evidence,
                candidate,
                evidence_threshold=float(evidence_threshold),
            )
            quick_scores.append(evidence_mean + active_fraction + contrast + consensus)
        keep = np.argsort(-np.asarray(quick_scores, dtype=np.float32), kind="mergesort")[: int(prefilter_top_k)]
        keep_set = {int(idx) for idx in keep}
        norm_candidates = [candidate for idx, candidate in enumerate(norm_candidates) if idx in keep_set]
        scores = np.asarray([score for idx, score in enumerate(scores) if idx in keep_set], dtype=np.float32)
        candidate_indices = [idx for idx in candidate_indices if idx in keep_set]
    evidence_means: list[float] = []
    active_fractions: list[float] = []
    contrasts: list[float] = []
    consensuses: list[float] = []
    base_ious: list[float] = []
    lengths: list[float] = []
    support_counts: list[float] = []
    for candidate in norm_candidates:
        evidence_mean, active_fraction, contrast, consensus = _evidence_stats(
            evidence,
            candidate,
            evidence_threshold=float(evidence_threshold),
        )
        evidence_means.append(evidence_mean)
        active_fractions.append(active_fraction)
        contrasts.append(contrast)
        consensuses.append(consensus)
        base_ious.append(_max_iou(candidate, base_segments))
        lengths.append(float(_segment_length(candidate)))
        support_counts.append(float(_support_count(candidate, norm_candidates, support_iou=float(support_iou))))
    return {
        "selector_scores": scores.astype(np.float32),
        "evidence_mean": np.asarray(evidence_means, dtype=np.float32),
        "active_fraction": np.asarray(active_fractions, dtype=np.float32),
        "contrast": np.asarray(contrasts, dtype=np.float32),
        "consensus": np.asarray(consensuses, dtype=np.float32),
        "base_iou": np.asarray(base_ious, dtype=np.float32),
        "length": np.asarray(lengths, dtype=np.float32),
        "support_count": np.asarray(support_counts, dtype=np.float32),
        "candidate_indices": np.asarray(candidate_indices, dtype=np.int64),
    }


def reasoner_scores_from_stats(
    stats: ReasonerStats,
    selector_weight: float,
    contrast_weight: float,
    active_fraction_weight: float,
    consensus_weight: float,
    support_count_weight: float,
    base_iou_penalty: float,
    length_penalty: float,
) -> np.ndarray:
    """Score candidates from precomputed reasoner stats."""
    return (
        stats["evidence_mean"]
        + float(selector_weight) * stats["selector_scores"]
        + float(contrast_weight) * stats["contrast"]
        + float(active_fraction_weight) * stats["active_fraction"]
        + float(consensus_weight) * stats["consensus"]
        + float(support_count_weight) * np.log1p(stats["support_count"])
        - float(base_iou_penalty) * stats["base_iou"]
        - float(length_penalty) * np.log1p(stats["length"])
    ).astype(np.float32)


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
        if iou_threshold is not None and any(
            interval_iou(segment, kept_segment) > float(iou_threshold) for kept_segment, _ in kept
        ):
            continue
        kept.append((segment, score))
        if len(kept) >= int(max_items):
            break
    return sorted({segment for segment, _ in kept})


def evidence_set_reasoner_scores(
    base_segments: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    selector_scores: np.ndarray,
    evidence: np.ndarray,
    evidence_threshold: float,
    selector_weight: float,
    contrast_weight: float,
    active_fraction_weight: float,
    consensus_weight: float,
    support_iou: float,
    support_count_weight: float,
    base_iou_penalty: float,
    length_penalty: float,
) -> np.ndarray:
    """Score candidates from JEPA evidence and video-local candidate support."""
    scores = np.nan_to_num(
        np.asarray(selector_scores, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if len(candidates) != len(scores):
        raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
    out = np.zeros(len(candidates), dtype=np.float32)
    stats = precompute_reasoner_stats(
        base_segments,
        candidates,
        scores,
        evidence,
        evidence_threshold=float(evidence_threshold),
        support_iou=float(support_iou),
    )
    out = reasoner_scores_from_stats(
        stats,
        selector_weight=selector_weight,
        contrast_weight=contrast_weight,
        active_fraction_weight=active_fraction_weight,
        consensus_weight=consensus_weight,
        support_count_weight=support_count_weight,
        base_iou_penalty=base_iou_penalty,
        length_penalty=length_penalty,
    )
    return out.astype(np.float32)


def select_evidence_set_reasoner_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    min_selector_score: float,
    evidence_threshold: float,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_contrast: float,
    score_threshold: float,
    selector_weight: float,
    contrast_weight: float,
    active_fraction_weight: float,
    consensus_weight: float,
    support_iou: float,
    support_count_weight: float,
    max_base_iou: float | None,
    base_iou_penalty: float,
    length_penalty: float,
    nms_iou: float | None,
    max_rescues_per_video: int,
    prefilter_top_k: int | None = None,
) -> list[list[Segment]]:
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
    ):
        raise ValueError("base/candidate/score/evidence video counts must match")

    fused: list[list[Segment]] = []
    for base_segments, candidates_raw, scores_raw, evidence in zip(
        base_predictions,
        candidate_predictions,
        candidate_scores,
        evidence_arrays,
    ):
        base = sorted({_normalise_segment(segment) for segment in base_segments})
        candidates = [_normalise_segment(segment) for segment in candidates_raw]
        selector_scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0)
        stats = precompute_reasoner_stats(
            base,
            candidates,
            selector_scores,
            evidence,
            evidence_threshold=float(evidence_threshold),
            support_iou=float(support_iou),
            prefilter_top_k=prefilter_top_k,
        )
        reasoner_scores = reasoner_scores_from_stats(
            stats,
            selector_weight=float(selector_weight),
            contrast_weight=float(contrast_weight),
            active_fraction_weight=float(active_fraction_weight),
            consensus_weight=float(consensus_weight),
            support_count_weight=float(support_count_weight),
            base_iou_penalty=float(base_iou_penalty),
            length_penalty=float(length_penalty),
        )

        rescue_candidates: list[Segment] = []
        rescue_scores: list[float] = []
        candidate_indices = np.asarray(stats.get("candidate_indices", np.arange(len(candidates))), dtype=np.int64)
        for idx, reasoner_score in enumerate(reasoner_scores):
            candidate = candidates[int(candidate_indices[idx])]
            selector_score = float(stats["selector_scores"][idx])
            if float(selector_score) < float(min_selector_score):
                continue
            if max_base_iou is not None and float(stats["base_iou"][idx]) > float(max_base_iou):
                continue
            evidence_mean = float(stats["evidence_mean"][idx])
            active_fraction = float(stats["active_fraction"][idx])
            contrast = float(stats["contrast"][idx])
            if evidence_mean < float(min_evidence_mean):
                continue
            if active_fraction < float(min_active_fraction):
                continue
            if contrast < float(min_contrast):
                continue
            if float(reasoner_score) < float(score_threshold):
                continue
            rescue_candidates.append(candidate)
            rescue_scores.append(float(reasoner_score))
        selected = _score_first_nms(
            rescue_candidates,
            rescue_scores,
            iou_threshold=nms_iou,
            max_items=int(max_rescues_per_video),
        )
        fused.append(sorted(set(base + selected)))
    return fused


def select_evidence_set_reasoner_predictions_from_stats(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    stats_by_video: list[ReasonerStats],
    min_selector_score: float,
    min_evidence_mean: float,
    min_active_fraction: float,
    min_contrast: float,
    score_threshold: float,
    selector_weight: float,
    contrast_weight: float,
    active_fraction_weight: float,
    consensus_weight: float,
    support_count_weight: float,
    max_base_iou: float | None,
    base_iou_penalty: float,
    length_penalty: float,
    nms_iou: float | None,
    max_rescues_per_video: int,
) -> list[list[Segment]]:
    """Select rescues using cached per-candidate JEPA evidence statistics."""
    if not (len(base_predictions) == len(candidate_predictions) == len(stats_by_video)):
        raise ValueError("base/candidate/stats video counts must match")
    fused: list[list[Segment]] = []
    for base_segments, candidates_raw, stats in zip(base_predictions, candidate_predictions, stats_by_video):
        base = sorted({_normalise_segment(segment) for segment in base_segments})
        candidates = [_normalise_segment(segment) for segment in candidates_raw]
        reasoner_scores = reasoner_scores_from_stats(
            stats,
            selector_weight=float(selector_weight),
            contrast_weight=float(contrast_weight),
            active_fraction_weight=float(active_fraction_weight),
            consensus_weight=float(consensus_weight),
            support_count_weight=float(support_count_weight),
            base_iou_penalty=float(base_iou_penalty),
            length_penalty=float(length_penalty),
        )
        rescue_candidates: list[Segment] = []
        rescue_scores: list[float] = []
        candidate_indices = np.asarray(stats.get("candidate_indices", np.arange(len(candidates))), dtype=np.int64)
        for idx, reasoner_score in enumerate(reasoner_scores):
            candidate = candidates[int(candidate_indices[idx])]
            if float(stats["selector_scores"][idx]) < float(min_selector_score):
                continue
            if max_base_iou is not None and float(stats["base_iou"][idx]) > float(max_base_iou):
                continue
            if float(stats["evidence_mean"][idx]) < float(min_evidence_mean):
                continue
            if float(stats["active_fraction"][idx]) < float(min_active_fraction):
                continue
            if float(stats["contrast"][idx]) < float(min_contrast):
                continue
            if float(reasoner_score) < float(score_threshold):
                continue
            rescue_candidates.append(candidate)
            rescue_scores.append(float(reasoner_score))
        selected = _score_first_nms(
            rescue_candidates,
            rescue_scores,
            iou_threshold=nms_iou,
            max_items=int(max_rescues_per_video),
        )
        fused.append(sorted(set(base + selected)))
    return fused


def select_evidence_set_reasoner_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    min_selector_scores: Iterable[float] = (0.0, 0.05, 0.1),
    evidence_thresholds: Iterable[float] = (0.55, 0.65, 0.75),
    min_evidence_means: Iterable[float] = (0.45, 0.55, 0.65),
    min_active_fractions: Iterable[float] = (0.25, 0.5, 0.67),
    min_contrasts: Iterable[float] = (0.0, 0.05, 0.1, 0.2),
    score_thresholds: Iterable[float] = (0.8, 1.0, 1.2, 1.4),
    selector_weights: Iterable[float] = (0.0, 0.1, 0.25),
    contrast_weights: Iterable[float] = (0.25, 0.5, 1.0),
    active_fraction_weights: Iterable[float] = (0.25, 0.5, 1.0),
    consensus_weights: Iterable[float] = (0.25, 0.5, 1.0),
    support_ious: Iterable[float] = (0.2, 0.3),
    support_count_weights: Iterable[float] = (0.0, 0.05, 0.1),
    max_base_ious: Iterable[float | None] = (0.0, 0.05, 0.1, 0.25, None),
    base_iou_penalties: Iterable[float] = (0.0, 0.2, 0.5),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01),
    nms_ious: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    max_rescues_per_videos: Iterable[int] = (1, 2, 3),
    prefilter_top_k: int | None = 80,
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
    for evidence_threshold in evidence_thresholds:
        for support_iou in support_ious:
            stats_by_video = [
                precompute_reasoner_stats(
                    base_segments,
                    candidates,
                    scores,
                    evidence,
                    evidence_threshold=float(evidence_threshold),
                    support_iou=float(support_iou),
                    prefilter_top_k=prefilter_top_k,
                )
                for base_segments, candidates, scores, evidence in zip(
                    base_predictions_norm,
                    candidate_predictions,
                    candidate_scores,
                    evidence_arrays,
                )
            ]
            for min_selector_score in min_selector_scores:
                for min_evidence_mean in min_evidence_means:
                    for min_active_fraction in min_active_fractions:
                        for min_contrast in min_contrasts:
                            for score_threshold in score_thresholds:
                                for selector_weight in selector_weights:
                                    for contrast_weight in contrast_weights:
                                        for active_fraction_weight in active_fraction_weights:
                                            for consensus_weight in consensus_weights:
                                                for support_count_weight in support_count_weights:
                                                    for max_base_iou in max_base_ious:
                                                        for base_iou_penalty in base_iou_penalties:
                                                            for length_penalty in length_penalties:
                                                                for nms_iou in nms_ious:
                                                                    for max_rescues_per_video in max_rescues_per_videos:
                                                                        fused = select_evidence_set_reasoner_predictions_from_stats(
                                                                            base_predictions_norm,
                                                                            candidate_predictions,
                                                                            stats_by_video,
                                                                            min_selector_score=float(min_selector_score),
                                                                            min_evidence_mean=float(min_evidence_mean),
                                                                            min_active_fraction=float(min_active_fraction),
                                                                            min_contrast=float(min_contrast),
                                                                            score_threshold=float(score_threshold),
                                                                            selector_weight=float(selector_weight),
                                                                            contrast_weight=float(contrast_weight),
                                                                            active_fraction_weight=float(active_fraction_weight),
                                                                            consensus_weight=float(consensus_weight),
                                                                            support_count_weight=float(support_count_weight),
                                                                            max_base_iou=max_base_iou,
                                                                            base_iou_penalty=float(base_iou_penalty),
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
                                                                            for before, after in zip(
                                                                                base_predictions_norm,
                                                                                fused,
                                                                            )
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
                                                                                "min_selector_score": float(min_selector_score),
                                                                                "evidence_threshold": float(evidence_threshold),
                                                                                "min_evidence_mean": float(min_evidence_mean),
                                                                                "min_active_fraction": float(min_active_fraction),
                                                                                "min_contrast": float(min_contrast),
                                                                                "score_threshold": float(score_threshold),
                                                                                "selector_weight": float(selector_weight),
                                                                                "contrast_weight": float(contrast_weight),
                                                                                "active_fraction_weight": float(
                                                                                    active_fraction_weight
                                                                                ),
                                                                                "consensus_weight": float(consensus_weight),
                                                                                "support_iou": float(support_iou),
                                                                                "support_count_weight": float(
                                                                                    support_count_weight
                                                                                ),
                                                                                "max_base_iou": max_base_iou,
                                                                                "base_iou_penalty": float(base_iou_penalty),
                                                                                "length_penalty": float(length_penalty),
                                                                                "nms_iou": nms_iou,
                                                                                "max_rescues_per_video": int(
                                                                                    max_rescues_per_video
                                                                                ),
                                                                                "rescue_segments": int(rescue_count),
                                                                                "base_metrics": base_metrics,
                                                                            }
    return best_config, best_metrics, best_predictions
