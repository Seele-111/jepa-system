#!/usr/bin/env python3
"""Expanded, video-bootstrap evaluation for Round-2 OOF prediction bundles."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_prediction_bundle import _flatten
from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import contiguous_segments, interval_iou


def _normalise_predictions(predictions: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    output = []
    for start, end in predictions:
        start, end = sorted((int(start), int(end)))
        if length <= 0 or end < 0 or start >= length:
            continue
        output.append((max(0, start), min(length - 1, end)))
    return output


def _duration_group(segment: tuple[int, int], video_length: int, cutoffs: tuple[float, float]) -> str:
    ratio = (segment[1] - segment[0] + 1) / max(1, video_length)
    if ratio <= cutoffs[0]:
        return "short"
    if ratio <= cutoffs[1]:
        return "medium"
    return "long"


def _safe_prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _match(
    predictions: list[tuple[int, int]], gt_segments: list[tuple[int, int]], iou_threshold: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Mirror the project's prediction-order greedy one-to-one matching."""
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    unmatched_pred: list[int] = []
    for pred_idx, pred in enumerate(predictions):
        best_idx, best_iou = -1, 0.0
        for gt_idx, gt in enumerate(gt_segments):
            if gt_idx in matched_gt:
                continue
            iou = interval_iou(pred, gt)
            if iou > best_iou:
                best_idx, best_iou = gt_idx, iou
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            matches.append((pred_idx, best_idx, float(best_iou)))
        else:
            unmatched_pred.append(pred_idx)
    unmatched_gt = [idx for idx in range(len(gt_segments)) if idx not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


def expanded_metrics(
    rows: list[tuple[str, list[tuple[int, int]], np.ndarray]],
    iou_threshold: float,
    duration_cutoffs: tuple[float, float],
) -> dict[str, Any]:
    predictions, labels = [], []
    duration_counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("short", "medium", "long")}
    boundary_start, boundary_end, matched_ious = [], [], []
    video_groups: dict[str, list[tuple[list[tuple[int, int]], np.ndarray]]] = defaultdict(list)
    prediction_counts, normal_prediction_counts = [], []

    for _name, raw_predictions, raw_label in rows:
        label = np.asarray(raw_label, dtype=np.int64).reshape(-1)
        pred = _normalise_predictions(raw_predictions, len(label))
        gt = contiguous_segments(label)
        matches, unmatched_pred, unmatched_gt = _match(pred, gt, iou_threshold)
        predictions.append(pred)
        labels.append(label)
        prediction_counts.append(len(pred))

        event_group = "normal" if not gt else "single_event" if len(gt) == 1 else "multi_event"
        occupancy_group = "normal" if not gt else "all_positive" if bool(np.all(label > 0)) else "partially_positive"
        for group in {event_group, occupancy_group}:
            video_groups[group].append((pred, label))
        if not gt:
            normal_prediction_counts.append(len(pred))

        for pred_idx, gt_idx, iou in matches:
            group = _duration_group(gt[gt_idx], len(label), duration_cutoffs)
            duration_counts[group]["tp"] += 1
            boundary_start.append(abs(pred[pred_idx][0] - gt[gt_idx][0]))
            boundary_end.append(abs(pred[pred_idx][1] - gt[gt_idx][1]))
            matched_ious.append(iou)
        for pred_idx in unmatched_pred:
            duration_counts[_duration_group(pred[pred_idx], len(label), duration_cutoffs)]["fp"] += 1
        for gt_idx in unmatched_gt:
            duration_counts[_duration_group(gt[gt_idx], len(label), duration_cutoffs)]["fn"] += 1

    core = evaluate_fused_predictions(predictions, labels, iou_threshold=iou_threshold)

    def distribution(values: list[float | int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "p90": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)),
        }

    group_reports = {}
    for group, values in sorted(video_groups.items()):
        metric = evaluate_fused_predictions([item[0] for item in values], [item[1] for item in values], iou_threshold)
        group_reports[group] = {"videos": len(values), "frame": metric["frame"], "segment": metric["segment"]}

    normal_videos = len(normal_prediction_counts)
    return {
        "videos": len(rows),
        "iou_threshold": float(iou_threshold),
        "overall": core,
        "matched_event_iou": distribution(matched_ious),
        "boundary_error_frames": {
            "start": distribution(boundary_start),
            "end": distribution(boundary_end),
            "mean_of_start_and_end_mae": (
                float((np.mean(boundary_start) + np.mean(boundary_end)) / 2) if boundary_start else None
            ),
        },
        "duration_strata": {
            "definition": {
                "unit": "fraction_of_video_frames",
                "short": f"duration_ratio <= {duration_cutoffs[0]}",
                "medium": f"{duration_cutoffs[0]} < duration_ratio <= {duration_cutoffs[1]}",
                "long": f"duration_ratio > {duration_cutoffs[1]}",
                "attribution": "TP/FN use ground-truth duration; unmatched FP uses predicted duration",
            },
            "reports": {group: _safe_prf(**counts) for group, counts in duration_counts.items()},
        },
        "video_strata": group_reports,
        "prediction_count_per_video": distribution(prediction_counts),
        "normal_video_false_positive_rate": {
            "normal_videos": normal_videos,
            "videos_with_prediction": int(sum(value > 0 for value in normal_prediction_counts)),
            "rate": (float(np.mean(np.asarray(normal_prediction_counts) > 0)) if normal_videos else None),
            "note": None if normal_videos else "not estimable: this bundle contains no normal videos",
        },
    }


def _bootstrap(
    rows: list[tuple[str, list[tuple[int, int]], np.ndarray]],
    iou_threshold: float,
    duration_cutoffs: tuple[float, float],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled = [rows[idx] for idx in rng.integers(0, len(rows), size=len(rows))]
        report = expanded_metrics(sampled, iou_threshold, duration_cutoffs)
        values["segment_f1"].append(float(report["overall"]["segment"]["f1"]))
        values["frame_f1"].append(float(report["overall"]["frame"]["f1"]))
        for key, value in (
            ("matched_event_iou_mean", report["matched_event_iou"]["mean"]),
            ("boundary_mae_frames", report["boundary_error_frames"]["mean_of_start_and_end_mae"]),
            ("predictions_per_video", report["prediction_count_per_video"]["mean"]),
            ("normal_video_fpr", report["normal_video_false_positive_rate"]["rate"]),
        ):
            if value is not None:
                values[key].append(float(value))
    output = {"samples": samples, "seed": seed, "unit": "video"}
    output["ci95"] = {
        key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for key, value in values.items()
        if value
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle-key", default="")
    parser.add_argument("--iou-thresholds", default="0.3,0.5")
    parser.add_argument("--duration-cutoffs", default="0.25,0.75")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    bundle = payload["bundles"][args.bundle_key] if args.bundle_key else payload
    rows = _flatten(bundle)
    cutoffs = tuple(float(value) for value in args.duration_cutoffs.split(","))
    if len(cutoffs) != 2 or not 0 < cutoffs[0] < cutoffs[1] < 1:
        raise ValueError("duration cutoffs must be two increasing fractions between zero and one")
    reports = []
    for value in args.iou_thresholds.split(","):
        threshold = float(value)
        report = expanded_metrics(rows, threshold, cutoffs)
        report["bootstrap"] = _bootstrap(rows, threshold, cutoffs, args.bootstrap_samples, args.seed)
        reports.append(report)
    output = {
        "schema_version": "round2-expanded-evaluation-v1",
        "source": str(args.predictions),
        "bundle_key": args.bundle_key or None,
        "matching_protocol": "prediction-order greedy one-to-one matching, consistent with main evaluator",
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "videos": len(rows), "reports": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
