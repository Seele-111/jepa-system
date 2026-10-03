#!/usr/bin/env python3
"""Attribute current-best OOF false negatives without retraining the mainline."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from jepa_prediction_set_switcher import _as_segments
from run_strict_dp_event_set_oof import build_fold_local_raw_candidates
from train_proposal_calibrator import interval_iou
from train_segment_locator import (
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    contiguous_segments,
    load_signal_dataset,
)
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES
from selector_fusion import evaluate_fused_predictions


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-clean, kind="mergesort")
    ranks = np.empty(len(clean), dtype=np.int64)
    ranks[order] = np.arange(1, len(clean) + 1, dtype=np.int64)
    return ranks


def _best_iou(segment: Segment, others: Iterable[Segment]) -> tuple[float, Segment | None]:
    best_iou = 0.0
    best_segment = None
    for other in others:
        value = float(interval_iou(segment, other))
        if value > best_iou:
            best_iou = value
            best_segment = other
    return best_iou, best_segment


def _match_predictions(predictions: list[Segment], labels: np.ndarray, iou_threshold: float) -> dict:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched: set[int] = set()
    false_positive_predictions: list[Segment] = []
    for pred_raw in predictions:
        pred = _normalise_segment(pred_raw)
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_segments):
            if idx in matched:
                continue
            value = float(interval_iou(pred, gt))
            if value > best_iou:
                best_iou = value
                best_idx = idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
        else:
            false_positive_predictions.append(pred)
    return {
        "gt_segments": gt_segments,
        "unmatched_gt": [gt for idx, gt in enumerate(gt_segments) if idx not in matched],
        "false_positive_predictions": false_positive_predictions,
    }


def classify_unmatched_gt_reason(
    gt: tuple[int, int],
    predictions: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    candidate_scores: np.ndarray,
    iou_threshold: float,
    active_threshold: float = 0.5,
) -> dict:
    gt_norm = _normalise_segment(gt)
    preds = [_normalise_segment(item) for item in predictions]
    cand = [_normalise_segment(item) for item in candidates]
    scores = np.nan_to_num(np.asarray(candidate_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    pred_iou, pred_segment = _best_iou(gt_norm, preds)
    ranks = _rank_desc(scores) if len(scores) else np.zeros(0, dtype=np.int64)
    best_idx = -1
    best_iou = 0.0
    best_key = None
    covering = 0
    for idx, candidate in enumerate(cand):
        value = float(interval_iou(gt_norm, candidate))
        if value >= float(iou_threshold):
            covering += 1
        key = (value, float(scores[idx]) if idx < len(scores) else 0.0)
        if best_key is None or key > best_key:
            best_key = key
            best_iou = value
            best_idx = idx
    best_score = float(scores[best_idx]) if best_idx >= 0 and best_idx < len(scores) else 0.0
    best_rank = int(ranks[best_idx]) if best_idx >= 0 and best_idx < len(ranks) else None
    if pred_iou > 0.0:
        reason = "wide_prediction_matching_conflict"
    elif best_iou < float(iou_threshold) or best_rank is None:
        reason = "candidate_generation_miss"
    elif best_score < float(active_threshold) or int(best_rank) > 60:
        reason = "candidate_scored_too_low"
    else:
        reason = "event_set_selection_conflict"
    return {
        "reason": reason,
        "best_prediction_iou": float(pred_iou),
        "best_prediction": None if pred_segment is None else list(pred_segment),
        "best_candidate_iou": float(best_iou),
        "best_candidate": None if best_idx < 0 else list(cand[best_idx]),
        "best_candidate_score": float(best_score),
        "best_candidate_rank": best_rank,
        "covering_candidates": int(covering),
    }


def _load_prediction_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--predictions", default="/home/zzy/jepa_data/segment_train_full_event_v2_prediction_set_switcher.predictions.json")
    parser.add_argument("--out", default="/home/zzy/jepa_data/segment_train_full_event_v2_switcher_fn_attribution.json")
    parser.add_argument("--selector-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--selector-target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--selector-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selector-min-gaps", default="0,1,2,4,8")
    parser.add_argument("--selector-min-lengths", default="1,2,4,8")
    parser.add_argument("--selector-epochs", type=int, default=120)
    parser.add_argument("--selector-batch-size", type=int, default=512)
    parser.add_argument("--active-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    export = _load_prediction_export(args.predictions)
    raw_by_video, selector_diagnostics = build_fold_local_raw_candidates(
        records,
        export,
        feature_names,
        channel_names,
        _parse_float_list(args.selector_thresholds),
        _parse_int_list(args.selector_min_gaps),
        _parse_int_list(args.selector_min_lengths),
        selector_model=args.selector_model,
        selector_target=args.selector_target,
        iou_threshold=float(args.iou_threshold),
        seed=int(args.seed),
        device=args.device,
        selector_epochs=int(args.selector_epochs),
        selector_batch_size=int(args.selector_batch_size),
    )
    fn_rows = []
    fp_rows = []
    fold_summaries = []
    for fold in export.get("folds", []):
        val_idx = [int(idx) for idx in fold.get("val_idx", [])]
        predictions = fold.get("predictions", [])
        labels = [np.asarray(item, dtype=np.int64) for item in fold.get("labels", [records[idx].labels for idx in val_idx])]
        metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(args.iou_threshold))
        fold_fn = 0
        fold_fp = 0
        for local_idx, video_idx in enumerate(val_idx):
            match = _match_predictions(predictions[local_idx], labels[local_idx], iou_threshold=float(args.iou_threshold))
            fold_fp += len(match["false_positive_predictions"])
            for pred in match["false_positive_predictions"]:
                fp_rows.append({"fold": int(fold.get("fold", 0)), "video_idx": int(video_idx), "prediction": list(pred)})
            candidates, scores = raw_by_video.get(int(video_idx), ([], np.zeros(0, dtype=np.float32)))
            for gt in match["unmatched_gt"]:
                info = classify_unmatched_gt_reason(
                    gt,
                    predictions[local_idx],
                    candidates,
                    scores,
                    iou_threshold=float(args.iou_threshold),
                    active_threshold=float(args.active_threshold),
                )
                fn_rows.append(
                    {
                        "fold": int(fold.get("fold", 0)),
                        "video_idx": int(video_idx),
                        "video": records[int(video_idx)].name,
                        "gt": list(gt),
                        **info,
                    }
                )
                fold_fn += 1
        fold_summaries.append({"fold": int(fold.get("fold", 0)), "validation": metrics, "fn": fold_fn, "fp": fold_fp})
    reason_counts = Counter(row["reason"] for row in fn_rows)
    covered = sum(int(row["best_candidate_iou"] >= float(args.iou_threshold)) for row in fn_rows)
    high_rank = sum(int(row["best_candidate_rank"] is not None and int(row["best_candidate_rank"]) <= 60) for row in fn_rows)
    result = {
        "data_dir": args.data_dir,
        "predictions": args.predictions,
        "iou_threshold": float(args.iou_threshold),
        "active_threshold": float(args.active_threshold),
        "selector_model": args.selector_model,
        "selector_target": args.selector_target,
        "selector_diagnostics": selector_diagnostics,
        "folds": fold_summaries,
        "fn_summary": {
            "total_fn": len(fn_rows),
            "total_fp": len(fp_rows),
            "candidate_covered_fn": int(covered),
            "candidate_covered_rate": float(covered / max(1, len(fn_rows))),
            "top60_candidate_fn": int(high_rank),
            "top60_candidate_rate": float(high_rank / max(1, len(fn_rows))),
            "reason_counts": dict(reason_counts),
        },
        "fn_rows": fn_rows,
        "fp_rows": fp_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"fn_summary": result["fn_summary"], "out": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
