#!/usr/bin/env python3
"""Candidate strata diagnostics for JEPA event-set selection.

This module is diagnostic only. It separates proposal candidates into groups
that explain whether a selector is failing because useful rescue candidates are
scored too low or because they are not separable from false positives.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from train_proposal_calibrator import interval_iou
from train_segment_locator import contiguous_segments


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _matched_gt_indices(
    predictions: Iterable[tuple[int, int]],
    gt_segments: list[Segment],
    iou_threshold: float,
) -> set[int]:
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


def _best_iou_with_index(segment: tuple[int, int], gt_segments: list[Segment]) -> tuple[float, int | None]:
    candidate = _normalise_segment(segment)
    best_iou = 0.0
    best_idx: int | None = None
    for idx, gt in enumerate(gt_segments):
        iou = float(interval_iou(candidate, gt))
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_iou, best_idx


def _max_iou(segment: tuple[int, int], others: Iterable[tuple[int, int]]) -> float:
    candidate = _normalise_segment(segment)
    return max((float(interval_iou(candidate, _normalise_segment(other))) for other in others), default=0.0)


def classify_candidate_strata(
    base_predictions: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float = 0.3,
    base_overlap_iou: float = 0.5,
) -> list[dict]:
    """Classify candidates by their relation to base predictions and GT events."""
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    base = [_normalise_segment(item) for item in base_predictions]
    base_matched_gt = _matched_gt_indices(base, gt_segments, iou_threshold=float(iou_threshold))
    rows: list[dict] = []
    for candidate_raw in candidates:
        candidate = _normalise_segment(candidate_raw)
        best_iou, best_gt_idx = _best_iou_with_index(candidate, gt_segments)
        max_base_iou = _max_iou(candidate, base)
        is_positive = best_iou >= float(iou_threshold) and best_gt_idx is not None
        base_overlaps = max_base_iou >= float(base_overlap_iou)
        if base_overlaps:
            stratum = "base_prediction" if is_positive else "base_overlap_negative"
        elif is_positive and best_gt_idx not in base_matched_gt:
            stratum = "oracle_rescue"
        elif is_positive:
            stratum = "covered_positive"
        else:
            stratum = "false_positive"
        rows.append(
            {
                "candidate": [int(candidate[0]), int(candidate[1])],
                "stratum": stratum,
                "best_iou": float(best_iou),
                "best_gt_index": None if best_gt_idx is None else int(best_gt_idx),
                "max_base_iou": float(max_base_iou),
                "is_positive": bool(is_positive),
                "base_matched_gt": bool(best_gt_idx in base_matched_gt) if best_gt_idx is not None else False,
            }
        )
    return rows


def _numeric_summary(values: list[float]) -> dict:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "min": 0.0,
        }
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def summarize_strata_scores(rows: list[dict], score_keys: Iterable[str]) -> dict:
    """Summarize score distributions for each stratum."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stratum", "unknown"))].append(row)
    summary: dict[str, dict] = {}
    for stratum, items in sorted(grouped.items()):
        item_summary: dict = {"count": int(len(items))}
        for key in score_keys:
            values = [
                float(row[key])
                for row in items
                if key in row and row[key] is not None and np.isfinite(float(row[key]))
            ]
            item_summary[str(key)] = _numeric_summary(values)
        summary[stratum] = item_summary
    return summary
