#!/usr/bin/env python3
"""Oracle diagnostics for JEPA candidate-set recall headroom.

This module is diagnostic only: it uses validation labels to estimate how much
the current JEPA candidate pool could recover under simple FP budgets. It must
not be used for held-out test-time inference.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions, short_first_nms_segments
from train_proposal_calibrator import interval_iou
from train_segment_locator import contiguous_segments


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _matched_gt_indices(
    predictions: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float,
) -> set[int]:
    gt_segments = contiguous_segments(np.asarray(labels, dtype=np.int64))
    matched: set[int] = set()
    for pred_raw in predictions:
        pred = _normalise_segment(pred_raw)
        best_idx = -1
        best_iou = 0.0
        for idx, gt_raw in enumerate(gt_segments):
            if idx in matched:
                continue
            gt = _normalise_segment(gt_raw)
            iou = float(interval_iou(pred, gt))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
    return matched


def _max_iou(segment: tuple[int, int], others: Iterable[tuple[int, int]]) -> float:
    norm = _normalise_segment(segment)
    return max((float(interval_iou(norm, _normalise_segment(other))) for other in others), default=0.0)


def _candidate_key(candidate: Segment, gt: Segment) -> tuple[float, float, float, float]:
    iou = float(interval_iou(candidate, gt))
    length_ratio = min(_segment_length(candidate), _segment_length(gt)) / max(
        _segment_length(candidate),
        _segment_length(gt),
    )
    center_gap = abs((candidate[0] + candidate[1]) - (gt[0] + gt[1])) / 2.0
    return (iou, length_ratio, -center_gap, -float(_segment_length(candidate)))


def oracle_rescue_for_video(
    base_segments: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float = 0.3,
    max_rescues_per_video: int = 1,
    max_base_iou: float | None = 0.0,
    selector_nms_iou: float | None = None,
) -> list[Segment]:
    """Return oracle rescue candidates for GT segments missed by base."""
    base = sorted({_normalise_segment(item) for item in base_segments})
    candidate_pool = sorted({_normalise_segment(item) for item in candidates})
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched = _matched_gt_indices(base, labels, iou_threshold=float(iou_threshold))
    rescues: list[Segment] = []
    used: set[Segment] = set()
    for gt_idx, gt in enumerate(gt_segments):
        if gt_idx in matched:
            continue
        best_candidate: Segment | None = None
        best_key: tuple[float, float, float, float] | None = None
        for candidate in candidate_pool:
            if candidate in used:
                continue
            if max_base_iou is not None and _max_iou(candidate, base) > float(max_base_iou):
                continue
            if selector_nms_iou is not None and any(
                interval_iou(candidate, existing) > float(selector_nms_iou) for existing in rescues
            ):
                continue
            if interval_iou(candidate, gt) < float(iou_threshold):
                continue
            key = _candidate_key(candidate, gt)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate
        if best_candidate is not None:
            used.add(best_candidate)
            rescues.append(best_candidate)
    if selector_nms_iou is not None:
        rescues = short_first_nms_segments(rescues, iou_threshold=float(selector_nms_iou))
    return sorted(rescues, key=lambda item: (_segment_length(item), item[0], item[1]))[: int(max_rescues_per_video)]


def build_oracle_rescue_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    iou_threshold: float = 0.3,
    max_rescues_per_video: int = 1,
    max_base_iou: float | None = 0.0,
    selector_nms_iou: float | None = None,
) -> list[list[Segment]]:
    """Fuse base predictions with label-oracle JEPA rescues."""
    if len(base_predictions) != len(candidate_predictions) or len(base_predictions) != len(labels):
        raise ValueError("base/candidate/label video counts must match")
    fused: list[list[Segment]] = []
    for base_segments, candidates, label_array in zip(base_predictions, candidate_predictions, labels):
        base = sorted({_normalise_segment(item) for item in base_segments})
        rescues = oracle_rescue_for_video(
            base,
            list(candidates),
            np.asarray(label_array, dtype=np.int64),
            iou_threshold=float(iou_threshold),
            max_rescues_per_video=int(max_rescues_per_video),
            max_base_iou=max_base_iou,
            selector_nms_iou=selector_nms_iou,
        )
        fused.append(sorted(set(base + rescues)))
    return fused


def _fp_increase(metrics: dict, base_metrics: dict) -> int:
    return int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])


def _tp_increase(metrics: dict, base_metrics: dict) -> int:
    return int(metrics["segment"]["tp"]) - int(base_metrics["segment"]["tp"])


def evaluate_budgeted_oracle(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    iou_threshold: float = 0.3,
    max_rescues_per_video_options: Iterable[int] = (1, 2, 3),
    max_base_iou_options: Iterable[float | None] = (0.0, 0.05, 0.1, 0.25, None),
    selector_nms_iou_options: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    fp_budget_options: Iterable[int] = (0, 2, 5, 10),
) -> dict:
    """Grid-search oracle rescue settings and summarize FP-budget upper bounds."""
    base_metrics = evaluate_fused_predictions(base_predictions, labels, iou_threshold=iou_threshold)
    grid: list[dict] = []
    for max_rescues in max_rescues_per_video_options:
        for max_base_iou in max_base_iou_options:
            for selector_nms_iou in selector_nms_iou_options:
                predictions = build_oracle_rescue_predictions(
                    base_predictions,
                    candidate_predictions,
                    labels,
                    iou_threshold=iou_threshold,
                    max_rescues_per_video=int(max_rescues),
                    max_base_iou=max_base_iou,
                    selector_nms_iou=selector_nms_iou,
                )
                metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=iou_threshold)
                grid.append(
                    {
                        "config": {
                            "max_rescues_per_video": int(max_rescues),
                            "max_base_iou": max_base_iou,
                            "selector_nms_iou": selector_nms_iou,
                        },
                        "metrics": metrics,
                        "rescued_tp": _tp_increase(metrics, base_metrics),
                        "fp_increase": _fp_increase(metrics, base_metrics),
                        "rescued_segments": int(
                            sum(max(0, len(pred) - len(base)) for pred, base in zip(predictions, base_predictions))
                        ),
                    }
                )

    best_by_budget: dict[str, dict] = {}
    for budget in fp_budget_options:
        eligible = [item for item in grid if int(item["fp_increase"]) <= int(budget)]
        if not eligible:
            continue
        best = max(
            eligible,
            key=lambda item: (
                float(item["metrics"]["segment"]["f1"]),
                float(item["metrics"]["segment"]["recall"]),
                float(item["metrics"]["segment"]["precision"]),
                -float(item["rescued_segments"]),
            ),
        )
        best_by_budget[str(int(budget))] = best

    reason = Counter()
    for candidates, label_array, base in zip(candidate_predictions, labels, base_predictions):
        gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(label_array, dtype=np.int64))]
        matched = _matched_gt_indices(base, label_array, iou_threshold=iou_threshold)
        for gt_idx, gt in enumerate(gt_segments):
            if gt_idx in matched:
                continue
            best_iou = max((float(interval_iou(_normalise_segment(candidate), gt)) for candidate in candidates), default=0.0)
            reason["candidate_covered_miss" if best_iou >= float(iou_threshold) else "candidate_generation_miss"] += 1

    return {
        "base_metrics": base_metrics,
        "grid": grid,
        "best_by_fp_budget": best_by_budget,
        "miss_upper_bound_counts": dict(reason),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose JEPA candidate oracle headroom from cached JSON arrays.")
    parser.add_argument("--base-json", required=True, help="JSON file with base_predictions, candidate_predictions, labels.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    source = json.loads(Path(args.base_json).read_text(encoding="utf-8"))
    base_predictions = [[tuple(item) for item in video] for video in source["base_predictions"]]
    candidate_predictions = [[tuple(item) for item in video] for video in source["candidate_predictions"]]
    labels = [np.asarray(item, dtype=np.int64) for item in source["labels"]]
    result = evaluate_budgeted_oracle(
        base_predictions,
        candidate_predictions,
        labels,
        iou_threshold=float(args.iou_threshold),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "best_by_fp_budget": result["best_by_fp_budget"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
