#!/usr/bin/env python3
"""Selective splitter for JEPA merge-noise parent predictions.

This module targets the specific failure mode where one broad prediction
swallows multiple neighboring error events. JEPA evidence islands propose the
child topology; a parent-level score decides whether the replacement is safe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jepa_evidence_island_splitter import evidence_islands
from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]


@dataclass(frozen=True)
class MergeNoiseParentRecord:
    video_idx: int
    parent: Segment
    children: list[Segment]
    features: np.ndarray


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


def _evidence_matrix(evidence: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _evidence_curve(evidence: np.ndarray) -> np.ndarray:
    values = _evidence_matrix(evidence)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    return values.mean(axis=1).astype(np.float32)


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
    for child, score, rank in zip(candidates, scores, ranks):
        child = _normalise_segment(child)
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


def _union_mask(length: int, segments: list[Segment], offset: int) -> np.ndarray:
    mask = np.zeros(max(0, int(length)), dtype=bool)
    for segment in segments:
        start, end = _normalise_segment(segment)
        left = max(0, start - int(offset))
        right = min(len(mask) - 1, end - int(offset))
        if right >= left:
            mask[left : right + 1] = True
    return mask


def _record_features(parent: Segment, children: list[Segment], islands: list[Segment], scores: np.ndarray, candidates: list[Segment], evidence: np.ndarray) -> np.ndarray:
    curve = _evidence_curve(evidence)
    parent_start, parent_end = _normalise_segment(parent)
    parent_start = max(0, min(parent_start, len(curve) - 1)) if len(curve) else 0
    parent_end = max(parent_start, min(parent_end, len(curve) - 1)) if len(curve) else 0
    parent_curve = curve[parent_start : parent_end + 1] if len(curve) else np.zeros(0, dtype=np.float32)
    child_scores = []
    for child in children:
        try:
            child_scores.append(float(scores[candidates.index(child)]))
        except ValueError:
            child_scores.append(0.0)
    child_mask = _union_mask(len(parent_curve), children, offset=parent_start)
    island_mask = _union_mask(len(parent_curve), islands, offset=parent_start)
    child_mass = float(parent_curve[child_mask].sum()) if child_mask.any() and len(parent_curve) else 0.0
    island_mass = float(parent_curve[island_mask].sum()) if island_mask.any() and len(parent_curve) else 0.0
    parent_mass = float(parent_curve.sum()) if len(parent_curve) else 0.0
    residual = parent_curve[~child_mask] if len(parent_curve) and child_mask.any() else parent_curve
    gaps = [max(0, right[0] - left[1] - 1) for left, right in zip(sorted(children), sorted(children)[1:])]
    parent_len = _segment_length(parent)
    child_lengths = [_segment_length(child) for child in children]
    features = np.asarray(
        [
            float(len(islands)),
            float(len(children)),
            float(sum(child_lengths) / max(1, parent_len)),
            float(max(child_lengths) / max(1, parent_len)),
            float(np.mean(child_scores) if child_scores else 0.0),
            float(np.min(child_scores) if child_scores else 0.0),
            float(np.max(child_scores) if child_scores else 0.0),
            float(child_mass / max(parent_mass, 1e-6)),
            float(child_mass / max(island_mass, 1e-6)),
            float(residual.mean() if len(residual) else 0.0),
            float(parent_curve.mean() if len(parent_curve) else 0.0),
            float(parent_curve.std() if len(parent_curve) else 0.0),
            float(np.mean(gaps) / max(1, parent_len) if gaps else 0.0),
            float(np.max(gaps) / max(1, parent_len) if gaps else 0.0),
            float(len(islands) + len(children) + child_mass / max(parent_mass, 1e-6)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_merge_noise_parent_records(
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
) -> list[MergeNoiseParentRecord]:
    if not (len(base_predictions) == len(candidate_predictions) == len(candidate_scores) == len(evidence_arrays)):
        raise ValueError("base/candidate/score/evidence video counts must match")
    records: list[MergeNoiseParentRecord] = []
    for video_idx, (base_raw, candidates_raw, scores_raw, evidence) in enumerate(
        zip(base_predictions, candidate_predictions, candidate_scores, evidence_arrays)
    ):
        base = sorted({_normalise_segment(item) for item in base_raw})
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        ranks = _rank_desc(scores)
        curve = _evidence_curve(evidence)
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
            islands = [(int(s + p0), int(e + p0)) for s, e in local_islands]
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
                if child_nms_iou is not None and any(interval_iou(child, existing) > float(child_nms_iou) for existing in children):
                    continue
                used.add(child)
                children.append(child)
                if len(children) >= int(max_children_per_parent):
                    break
            if len(children) >= int(min_islands):
                records.append(
                    MergeNoiseParentRecord(
                        video_idx=int(video_idx),
                        parent=parent,
                        children=sorted(children),
                        features=_record_features(parent, sorted(children), islands, scores, candidates, evidence),
                    )
                )
    return records


def split_merge_noise_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    parent_scores: list[np.ndarray],
    threshold: float,
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
    records = extract_merge_noise_parent_records(
        base_predictions,
        candidate_predictions,
        candidate_scores,
        evidence_arrays,
        island_threshold=float(island_threshold),
        island_min_length=int(island_min_length),
        island_min_gap=int(island_min_gap),
        min_islands=int(min_islands),
        min_candidate_score=float(min_candidate_score),
        max_candidate_rank=max_candidate_rank,
        min_island_coverage=float(min_island_coverage),
        max_child_parent_ratio=float(max_child_parent_ratio),
        min_child_parent_coverage=float(min_child_parent_coverage),
        child_nms_iou=child_nms_iou,
        max_children_per_parent=int(max_children_per_parent),
    )
    out = [sorted({_normalise_segment(item) for item in video}) for video in base_predictions]
    by_video: dict[int, list[tuple[MergeNoiseParentRecord, float]]] = {}
    for record, scores in zip(records, parent_scores):
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
        score = float(score_values[0]) if len(score_values) else 0.0
        if score >= float(threshold):
            by_video.setdefault(int(record.video_idx), []).append((record, score))
    for video_idx, items in by_video.items():
        items.sort(key=lambda item: (item[1], item[0].features[-1]), reverse=True)
        selected = []
        used = set()
        for record, score in items:
            if record.parent in used:
                continue
            used.add(record.parent)
            selected.append((record, score))
            if int(max_replaced_parents_per_video) > 0 and len(selected) >= int(max_replaced_parents_per_video):
                break
        replace = {record.parent for record, _ in selected}
        children = [child for record, _ in selected for child in record.children]
        out[video_idx] = sorted(set([segment for segment in out[video_idx] if segment not in replace] + children))
    return out
