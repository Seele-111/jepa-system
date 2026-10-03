#!/usr/bin/env python3
"""Graph-style reranking for JEPA proposal sets.

The selector already scores individual V/I-JEPA proposals. This module adds a
video-local relation layer: candidates supported by nearby/overlapping JEPA
neighbors are boosted, broad candidates that swallow several compact proposals
are penalized, and final selection is calibrated against protected fusion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from selector_fusion import (
    evaluate_fused_predictions,
    fuse_protected_segment_predictions,
    select_protected_fusion_params,
)
from train_proposal_calibrator import interval_iou
from train_proposal_set_selector import select_weighted_proposal_set


Segment = tuple[int, int]


@dataclass(frozen=True)
class ProposalGraphNode:
    """A proposal node in the video-local topology graph."""

    index: int
    segment: Segment
    score: float


@dataclass(frozen=True)
class ProposalGraphEdge:
    """A directed relation used for topology diagnostics and scoring."""

    source: int
    target: int
    relation: str
    weight: float


def build_proposal_graph(
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    support_iou: float,
) -> tuple[list[ProposalGraphNode], list[ProposalGraphEdge]]:
    """Construct the explicit proposal graph used by the reranker.

    Nodes are temporal proposals. ``support`` edges connect proposals whose
    IoU is at least ``support_iou`` and carry the IoU as weight. A ``contains``
    edge points from a broad parent to a compact child and carries the child /
    parent duration ratio. The current reranker remains backward compatible;
    this representation makes its relations inspectable and reproducible.
    """
    norm_candidates = [_normalise_segment(segment) for segment in candidates]
    values = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(norm_candidates) != len(values):
        raise ValueError(f"candidate/score count mismatch: {len(norm_candidates)} vs {len(values)}")
    nodes = [ProposalGraphNode(i, segment, float(values[i])) for i, segment in enumerate(norm_candidates)]
    edges: list[ProposalGraphEdge] = []
    for i, left in enumerate(norm_candidates):
        for j in range(i + 1, len(norm_candidates)):
            right = norm_candidates[j]
            overlap = float(interval_iou(left, right))
            if overlap >= float(support_iou):
                edges.append(ProposalGraphEdge(i, j, "support", overlap))
                edges.append(ProposalGraphEdge(j, i, "support", overlap))
            if _contains(left, right) and left != right:
                edges.append(ProposalGraphEdge(i, j, "contains", _segment_length(right) / _segment_length(left)))
            elif _contains(right, left) and left != right:
                edges.append(ProposalGraphEdge(j, i, "contains", _segment_length(left) / _segment_length(right)))
    return nodes, edges


def proposal_graph_to_dict(
    candidates: list[tuple[int, int]], scores: np.ndarray, support_iou: float
) -> dict[str, object]:
    """Return a JSON-safe graph record for paper audit packets."""
    nodes, edges = build_proposal_graph(candidates, scores, support_iou)
    return {
        "support_iou": float(support_iou),
        "nodes": [
            {"index": n.index, "segment": list(n.segment), "score": n.score} for n in nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target, "relation": e.relation, "weight": e.weight}
            for e in edges
        ],
    }


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _contains(parent: Segment, child: Segment) -> bool:
    return parent[0] <= child[0] and child[1] <= parent[1]


def _max_iou(segment: Segment, others: Iterable[tuple[int, int]]) -> float:
    return max((interval_iou(segment, _normalise_segment(other)) for other in others), default=0.0)


def _contained_compact_count(segment: Segment, candidates: list[Segment]) -> int:
    length = _segment_length(segment)
    count = 0
    for other in candidates:
        if other == segment:
            continue
        if not _contains(segment, other):
            continue
        if _segment_length(other) >= length:
            continue
        count += 1
    return count


def graph_rerank_candidate_scores(
    mainline_segments: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    scores: np.ndarray,
    support_iou: float,
    support_weight: float,
    support_count_weight: float,
    split_penalty: float,
    mainline_overlap_penalty: float,
) -> np.ndarray:
    """Adjust proposal scores using video-local JEPA proposal relations."""
    norm_candidates = [_normalise_segment(segment) for segment in candidates]
    values = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(norm_candidates) != len(values):
        raise ValueError(f"candidate/score count mismatch: {len(norm_candidates)} vs {len(values)}")
    if not norm_candidates:
        return np.zeros(0, dtype=np.float32)

    adjusted = values.astype(np.float32).copy()
    for idx, segment in enumerate(norm_candidates):
        support_scores: list[float] = []
        for other_idx, other in enumerate(norm_candidates):
            if other_idx == idx:
                continue
            if interval_iou(segment, other) >= float(support_iou):
                support_scores.append(float(values[other_idx]))
        if support_scores:
            adjusted[idx] += float(support_weight) * float(np.mean(support_scores))
            adjusted[idx] += float(support_count_weight) * math.log1p(len(support_scores))

        contained_count = _contained_compact_count(segment, norm_candidates)
        if contained_count > 1:
            adjusted[idx] -= float(split_penalty) * float(contained_count - 1)

        if mainline_segments:
            adjusted[idx] -= float(mainline_overlap_penalty) * _max_iou(segment, mainline_segments)

    return adjusted.astype(np.float32)


def select_graph_reranked_predictions(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    prob_threshold: float,
    support_iou: float,
    support_weight: float,
    support_count_weight: float,
    split_penalty: float,
    mainline_overlap_penalty: float,
    length_penalty: float,
) -> list[list[Segment]]:
    """Select one graph-reranked JEPA proposal set per video."""
    if len(mainline_predictions) != len(candidate_predictions) or len(mainline_predictions) != len(candidate_scores):
        raise ValueError("mainline/candidate/score video counts must match")

    predictions: list[list[Segment]] = []
    for main_segments, candidates, scores in zip(mainline_predictions, candidate_predictions, candidate_scores):
        adjusted = graph_rerank_candidate_scores(
            list(main_segments),
            list(candidates),
            np.asarray(scores, dtype=np.float32),
            support_iou=float(support_iou),
            support_weight=float(support_weight),
            support_count_weight=float(support_count_weight),
            split_penalty=float(split_penalty),
            mainline_overlap_penalty=float(mainline_overlap_penalty),
        )
        filtered_candidates: list[Segment] = []
        filtered_scores: list[float] = []
        for segment, score in zip(candidates, adjusted):
            if float(score) < float(prob_threshold):
                continue
            filtered_candidates.append(_normalise_segment(segment))
            filtered_scores.append(float(score))
        predictions.append(
            select_weighted_proposal_set(
                filtered_candidates,
                filtered_scores,
                length_penalty=float(length_penalty),
            )
        )
    return predictions


def is_graph_reranker_improvement(graph_metrics: dict, current_metrics: dict) -> bool:
    """Return True only when graph reranking improves the active fusion metric."""
    graph_key = (
        float(graph_metrics["segment"]["f1"]),
        float(graph_metrics["segment"]["precision"]),
        float(graph_metrics["segment"]["recall"]),
        float(graph_metrics["frame"]["f1"]),
    )
    current_key = (
        float(current_metrics["segment"]["f1"]),
        float(current_metrics["segment"]["precision"]),
        float(current_metrics["segment"]["recall"]),
        float(current_metrics["frame"]["f1"]),
    )
    return graph_key > current_key


def select_graph_reranker_params(
    mainline_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    labels: list[np.ndarray],
    prob_thresholds: Iterable[float],
    support_ious: Iterable[float],
    support_weights: Iterable[float],
    support_count_weights: Iterable[float],
    split_penalties: Iterable[float],
    mainline_overlap_penalties: Iterable[float],
    length_penalties: Iterable[float],
    protected_mainline_iou_candidates: Iterable[float | None],
    selector_nms_iou_candidates: Iterable[float | None],
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    """Tune graph-reranker parameters by final protected-fusion F1."""
    base_predictions = [[_normalise_segment(segment) for segment in video] for video in mainline_predictions]
    base_metrics = evaluate_fused_predictions(base_predictions, labels, iou_threshold=iou_threshold)
    best_config: dict = {
        "enabled": False,
        "base_metrics": base_metrics,
    }
    best_metrics = base_metrics
    best_selector_predictions: list[list[Segment]] = [[] for _ in base_predictions]
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )

    for prob_threshold in prob_thresholds:
        for support_iou in support_ious:
            for support_weight in support_weights:
                for support_count_weight in support_count_weights:
                    for split_penalty in split_penalties:
                        for mainline_overlap_penalty in mainline_overlap_penalties:
                            for length_penalty in length_penalties:
                                selector_predictions = select_graph_reranked_predictions(
                                    base_predictions,
                                    candidate_predictions,
                                    candidate_scores,
                                    prob_threshold=float(prob_threshold),
                                    support_iou=float(support_iou),
                                    support_weight=float(support_weight),
                                    support_count_weight=float(support_count_weight),
                                    split_penalty=float(split_penalty),
                                    mainline_overlap_penalty=float(mainline_overlap_penalty),
                                    length_penalty=float(length_penalty),
                                )
                                protected_config, protected_metrics = select_protected_fusion_params(
                                    base_predictions,
                                    selector_predictions,
                                    labels,
                                    max_mainline_iou_candidates=protected_mainline_iou_candidates,
                                    selector_nms_iou_candidates=selector_nms_iou_candidates,
                                    iou_threshold=iou_threshold,
                                )
                                if protected_config.get("enabled"):
                                    fused = fuse_protected_segment_predictions(
                                        base_predictions,
                                        selector_predictions,
                                        max_mainline_iou=protected_config["max_mainline_iou"],
                                        selector_nms_iou=protected_config["selector_nms_iou"],
                                    )
                                    metrics = evaluate_fused_predictions(fused, labels, iou_threshold=iou_threshold)
                                else:
                                    metrics = protected_metrics
                                selected_count = sum(len(video) for video in selector_predictions)
                                key = (
                                    float(metrics["segment"]["f1"]),
                                    float(metrics["segment"]["precision"]),
                                    float(metrics["segment"]["recall"]),
                                    float(metrics["frame"]["f1"]),
                                    -float(selected_count),
                                )
                                if key > best_key:
                                    best_key = key
                                    best_metrics = metrics
                                    best_selector_predictions = selector_predictions
                                    best_config = {
                                        "enabled": True,
                                        "prob_threshold": float(prob_threshold),
                                        "support_iou": float(support_iou),
                                        "support_weight": float(support_weight),
                                        "support_count_weight": float(support_count_weight),
                                        "split_penalty": float(split_penalty),
                                        "mainline_overlap_penalty": float(mainline_overlap_penalty),
                                        "length_penalty": float(length_penalty),
                                        "selected_segments": int(selected_count),
                                        "protected": protected_config,
                                        "base_metrics": base_metrics,
                                    }
    return best_config, best_metrics, best_selector_predictions
