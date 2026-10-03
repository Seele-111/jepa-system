#!/usr/bin/env python3
"""Fusion helpers for mainline temporal CNN and JEPA proposal selectors."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from train_proposal_calibrator import evaluate_segment_predictions, interval_iou


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    if end < start:
        start, end = end, start
    return start, end


def short_first_nms_segments(segments: Iterable[tuple[int, int]], iou_threshold: float) -> list[Segment]:
    """Keep compact non-duplicate intervals before broad merged intervals."""
    unique = sorted({_normalise_segment(segment) for segment in segments})
    ordered = sorted(unique, key=lambda seg: (seg[1] - seg[0] + 1, seg[0], seg[1]))
    kept: list[Segment] = []
    for segment in ordered:
        if all(interval_iou(segment, existing) <= float(iou_threshold) for existing in kept):
            kept.append(segment)
    return sorted(kept)


def fuse_segment_predictions(
    mainline_predictions: list[list[tuple[int, int]]],
    selector_predictions: list[list[tuple[int, int]]],
    nms_iou: float | None,
) -> list[list[Segment]]:
    if len(mainline_predictions) != len(selector_predictions):
        raise ValueError(f"prediction count mismatch: {len(mainline_predictions)} vs {len(selector_predictions)}")
    fused: list[list[Segment]] = []
    for main_segments, selector_segments in zip(mainline_predictions, selector_predictions):
        merged = [_normalise_segment(seg) for seg in main_segments] + [_normalise_segment(seg) for seg in selector_segments]
        if nms_iou is None:
            fused.append(sorted(set(merged)))
        else:
            fused.append(short_first_nms_segments(merged, iou_threshold=float(nms_iou)))
    return fused


def fuse_protected_segment_predictions(
    mainline_predictions: list[list[tuple[int, int]]],
    selector_predictions: list[list[tuple[int, int]]],
    max_mainline_iou: float | None,
    selector_nms_iou: float | None,
) -> list[list[Segment]]:
    if len(mainline_predictions) != len(selector_predictions):
        raise ValueError(f"prediction count mismatch: {len(mainline_predictions)} vs {len(selector_predictions)}")
    fused: list[list[Segment]] = []
    for main_segments, selector_segments in zip(mainline_predictions, selector_predictions):
        main = sorted({_normalise_segment(seg) for seg in main_segments})
        rescue_candidates: list[Segment] = []
        for selector_segment in sorted({_normalise_segment(seg) for seg in selector_segments}):
            if max_mainline_iou is not None:
                if any(interval_iou(selector_segment, main_segment) > float(max_mainline_iou) for main_segment in main):
                    continue
            rescue_candidates.append(selector_segment)
        if selector_nms_iou is not None:
            rescue_candidates = short_first_nms_segments(rescue_candidates, iou_threshold=float(selector_nms_iou))
        fused.append(sorted(set(main + rescue_candidates)))
    return fused


def select_fusion_nms_iou(
    mainline_predictions: list[list[tuple[int, int]]],
    selector_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    candidates: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    iou_threshold: float = 0.3,
) -> tuple[float | None, dict]:
    best_candidate: float | None = None
    best_metrics: dict | None = None
    best_key: tuple[float, float, float, float] | None = None
    for candidate in candidates:
        fused = fuse_segment_predictions(mainline_predictions, selector_predictions, nms_iou=candidate)
        metrics = evaluate_segment_predictions(fused, labels, iou_threshold=iou_threshold)
        key = (
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
            -float(metrics["fp"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate
            best_metrics = metrics
    assert best_metrics is not None
    return best_candidate, best_metrics


def select_protected_fusion_params(
    mainline_predictions: list[list[tuple[int, int]]],
    selector_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    max_mainline_iou_candidates: Iterable[float | None] = (None, 0.0, 0.1, 0.25, 0.5),
    selector_nms_iou_candidates: Iterable[float | None] = (None, 0.1, 0.3, 0.5),
    iou_threshold: float = 0.3,
) -> tuple[dict, dict]:
    base_metrics = evaluate_fused_predictions(mainline_predictions, labels, iou_threshold=iou_threshold)
    best_config: dict = {
        "enabled": False,
        "max_mainline_iou": None,
        "selector_nms_iou": None,
        "base_metrics": base_metrics,
    }
    best_metrics = base_metrics
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for max_mainline_iou in max_mainline_iou_candidates:
        if max_mainline_iou is None:
            continue
        for selector_nms_iou in selector_nms_iou_candidates:
            fused = fuse_protected_segment_predictions(
                mainline_predictions,
                selector_predictions,
                max_mainline_iou=max_mainline_iou,
                selector_nms_iou=selector_nms_iou,
            )
            metrics = evaluate_fused_predictions(fused, labels, iou_threshold=iou_threshold)
            rescue_count = sum(max(0, len(items) - len(base)) for items, base in zip(fused, mainline_predictions))
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
                best_config = {
                    "enabled": True,
                    "max_mainline_iou": float(max_mainline_iou),
                    "selector_nms_iou": selector_nms_iou,
                    "rescue_segments": int(rescue_count),
                    "base_metrics": base_metrics,
                }
    return best_config, best_metrics


def _binary_frame_metrics(labels: np.ndarray, pred: np.ndarray) -> dict:
    truth = np.asarray(labels, dtype=np.int64).astype(bool)
    predicted = np.asarray(pred, dtype=np.int64).astype(bool)
    n = min(len(truth), len(predicted))
    truth = truth[:n]
    predicted = predicted[:n]
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_fused_predictions(
    predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    iou_threshold: float = 0.3,
) -> dict:
    frame_tp = frame_fp = frame_fn = 0
    for segments, label_array in zip(predictions, labels):
        labels_arr = np.asarray(label_array, dtype=np.int64).reshape(-1)
        binary = np.zeros_like(labels_arr, dtype=np.int64)
        for segment in segments:
            start, end = _normalise_segment(segment)
            if end < 0 or start >= len(binary):
                continue
            start = max(0, start)
            end = min(len(binary) - 1, end)
            binary[start : end + 1] = 1
        frame = _binary_frame_metrics(labels_arr, binary)
        frame_tp += int(frame["tp"])
        frame_fp += int(frame["fp"])
        frame_fn += int(frame["fn"])

    frame_precision = frame_tp / (frame_tp + frame_fp + 1e-6)
    frame_recall = frame_tp / (frame_tp + frame_fn + 1e-6)
    frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall + 1e-6)
    return {
        "frame": {
            "precision": frame_precision,
            "recall": frame_recall,
            "f1": frame_f1,
            "tp": frame_tp,
            "fp": frame_fp,
            "fn": frame_fn,
        },
        "segment": evaluate_segment_predictions(predictions, labels, iou_threshold=iou_threshold),
    }
