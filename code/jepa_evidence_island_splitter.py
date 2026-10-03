#!/usr/bin/env python3
"""Unsupervised JEPA evidence-island splitter for broad predictions."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]


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


def _coverage(segment: tuple[int, int], container: tuple[int, int]) -> float:
    return float(_intersection(segment, container) / max(1, _segment_length(segment)))


def _evidence_curve(evidence: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim == 1:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    return np.clip(values.mean(axis=1), 0.0, 1.0).astype(np.float32)


def evidence_islands(curve: np.ndarray, threshold: float, min_length: int, min_gap: int) -> list[Segment]:
    active = (np.asarray(curve, dtype=np.float32).reshape(-1) >= float(threshold)).astype(np.int64)
    islands: list[Segment] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if int(value) and start is None:
            start = int(idx)
        elif not int(value) and start is not None:
            islands.append((start, int(idx - 1)))
            start = None
    if start is not None:
        islands.append((start, int(len(active) - 1)))
    if int(min_gap) > 0 and len(islands) > 1:
        merged = [islands[0]]
        for start, end in islands[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end - 1 <= int(min_gap):
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        islands = merged
    return [(int(s), int(e)) for s, e in islands if e - s + 1 >= int(min_length)]


def _rank_desc(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-clean, kind="mergesort")
    ranks = np.empty(len(clean), dtype=np.int64)
    ranks[order] = np.arange(1, len(clean) + 1, dtype=np.int64)
    return ranks


def _candidate_for_island(
    island: Segment,
    parent: Segment,
    candidates: list[Segment],
    scores: np.ndarray,
    ranks: np.ndarray,
    used: set[Segment],
    min_candidate_score: float,
    max_candidate_rank: int | None,
    min_island_coverage: float,
    max_child_parent_ratio: float,
    min_child_parent_coverage: float,
) -> Segment | None:
    parent_len = _segment_length(parent)
    best: Segment | None = None
    best_key: tuple[float, float, float, float] | None = None
    for candidate, score, rank in zip(candidates, scores, ranks):
        child = _normalise_segment(candidate)
        if child in used or child == parent:
            continue
        child_len = _segment_length(child)
        if child_len >= parent_len:
            continue
        if child_len / max(1, parent_len) > float(max_child_parent_ratio):
            continue
        if _coverage(child, parent) < float(min_child_parent_coverage):
            continue
        if _coverage(island, child) < float(min_island_coverage):
            continue
        if float(score) < float(min_candidate_score):
            continue
        if max_candidate_rank is not None and int(rank) > int(max_candidate_rank):
            continue
        center_gap = abs((child[0] + child[1]) - (island[0] + island[1])) / 2.0
        key = (_coverage(island, child), float(score), -center_gap, -float(child_len))
        if best_key is None or key > best_key:
            best_key = key
            best = child
    return best


def split_evidence_island_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    island_threshold: float,
    island_min_length: int,
    island_min_gap: int,
    min_islands: int,
    min_candidate_score: float,
    max_candidate_rank: int | None,
    min_island_coverage: float,
    max_child_parent_ratio: float,
    min_child_parent_coverage: float,
    child_nms_iou: float | None,
    max_children_per_parent: int,
    max_replaced_parents_per_video: int,
) -> list[list[Segment]]:
    out: list[list[Segment]] = []
    for base_raw, candidates_raw, scores_raw, evidence in zip(base_predictions, candidate_predictions, candidate_scores, evidence_arrays):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        ranks = _rank_desc(scores)
        curve = _evidence_curve(evidence)
        replacements: list[tuple[Segment, list[Segment], float]] = []
        for parent in base:
            if len(curve) == 0:
                continue
            p0, p1 = parent
            p0 = max(0, min(p0, len(curve) - 1))
            p1 = max(p0, min(p1, len(curve) - 1))
            local_islands = evidence_islands(
                curve[p0 : p1 + 1],
                threshold=float(island_threshold),
                min_length=int(island_min_length),
                min_gap=int(island_min_gap),
            )
            islands = [(s + p0, e + p0) for s, e in local_islands]
            if len(islands) < int(min_islands):
                continue
            used: set[Segment] = set()
            children: list[Segment] = []
            for island in islands:
                child = _candidate_for_island(
                    island,
                    parent,
                    candidates,
                    scores,
                    ranks,
                    used,
                    min_candidate_score=float(min_candidate_score),
                    max_candidate_rank=max_candidate_rank,
                    min_island_coverage=float(min_island_coverage),
                    max_child_parent_ratio=float(max_child_parent_ratio),
                    min_child_parent_coverage=float(min_child_parent_coverage),
                )
                if child is None:
                    continue
                if child_nms_iou is not None and any(interval_iou(child, other) > float(child_nms_iou) for other in children):
                    continue
                used.add(child)
                children.append(child)
                if len(children) >= int(max_children_per_parent):
                    break
            if len(children) >= int(min_islands):
                score = float(len(children)) + float(np.mean([scores[candidates.index(child)] for child in children if child in candidates]))
                replacements.append((parent, sorted(children), score))
        replacements.sort(key=lambda item: item[2], reverse=True)
        if int(max_replaced_parents_per_video) > 0:
            replacements = replacements[: int(max_replaced_parents_per_video)]
        replace_parents = {parent for parent, _, _ in replacements}
        new_children = [child for _, children, _ in replacements for child in children]
        kept = [segment for segment in base if segment not in replace_parents]
        out.append(sorted(set(kept + new_children)))
    return out

