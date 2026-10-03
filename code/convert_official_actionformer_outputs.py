#!/usr/bin/env python3
"""Calibrate official ActionFormer detections and export a Round-2 bundle."""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

from train_segment_locator import _segment_counts, contiguous_segments


def load_official_detections(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # Official eval.py output, produced locally.
    required = ("video-id", "t-start", "t-end", "score")
    if any(key not in payload for key in required):
        raise ValueError(f"{path} is missing official detection fields")
    lengths = {len(payload[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError(f"{path} detection fields have inconsistent lengths")
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for video_id, start, end, score in zip(*(payload[key] for key in required)):
        values = (float(start), float(end), float(score))
        if not all(math.isfinite(value) for value in values):
            continue
        grouped.setdefault(str(video_id), []).append(values)
    for rows in grouped.values():
        rows.sort(key=lambda row: row[2], reverse=True)
    return grouped


def load_fold_labels(annotation_path: Path, subset: str) -> dict[str, np.ndarray]:
    database = json.loads(annotation_path.read_text(encoding="utf-8"))["database"]
    labels = {}
    for video_id, row in database.items():
        if row["subset"].lower() != subset.lower():
            continue
        length = int(round(float(row["duration"])))
        target = np.zeros(length, dtype=np.int64)
        for annotation in row.get("annotations", []):
            start, end = annotation["segment"]
            left = max(0, int(math.floor(float(start))))
            right = min(length, int(math.ceil(float(end))))
            target[left:right] = 1
        labels[video_id] = target
    return labels


def decode_video(
    candidates: list[tuple[float, float, float]],
    length: int,
    score_threshold: float,
    max_predictions: int | None,
) -> list[tuple[int, int]]:
    selected = [row for row in candidates if row[2] >= score_threshold]
    if max_predictions is not None:
        selected = selected[:max_predictions]
    segments = []
    for start, end, _ in selected:
        left = max(0, min(length - 1, int(math.floor(start))))
        right = max(left, min(length - 1, int(math.ceil(end) - 1)))
        segments.append((left, right))
    return segments


def evaluate_params(
    detections: dict[str, list[tuple[float, float, float]]],
    labels: dict[str, np.ndarray],
    score_threshold: float,
    max_predictions: int | None,
    iou_threshold: float = 0.3,
) -> dict:
    tp = fp = fn = 0
    for video_id, target in labels.items():
        predicted = decode_video(detections.get(video_id, []), len(target), score_threshold, max_predictions)
        row_tp, row_fp, row_fn = _segment_counts(predicted, contiguous_segments(target), iou_threshold)
        tp += row_tp
        fp += row_fp
        fn += row_fn
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def select_calibration_params(detections: dict, labels: dict) -> tuple[dict, dict]:
    best = None
    for threshold in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        for max_predictions in (1, 2, 4, 8, None):
            metrics = evaluate_params(detections, labels, threshold, max_predictions)
            key = (metrics["f1"], metrics["precision"], -float(max_predictions or 1000), threshold)
            if best is None or key > best[0]:
                best = (key, {"score_threshold": threshold, "max_predictions": max_predictions}, metrics)
    assert best is not None
    return best[1], best[2]


def build_prediction_fold(
    fold: int,
    annotation_path: Path,
    calibration_pickle: Path,
    outer_pickle: Path,
    id_to_name: dict[str, str],
) -> tuple[dict, dict]:
    calibration_labels = load_fold_labels(annotation_path, "calibration")
    outer_labels = load_fold_labels(annotation_path, "validation")
    calibration_detections = load_official_detections(calibration_pickle)
    outer_detections = load_official_detections(outer_pickle)
    params, calibration_metrics = select_calibration_params(calibration_detections, calibration_labels)
    predictions, labels, val_names = [], [], []
    for video_id, target in outer_labels.items():
        val_names.append(id_to_name[video_id])
        predictions.append(
            [list(segment) for segment in decode_video(outer_detections.get(video_id, []), len(target), **params)]
        )
        labels.append(target.tolist())
    fold_row = {"fold": fold, "val_names": val_names, "predictions": predictions, "labels": labels}
    audit = {"fold": fold, "params": params, "calibration": calibration_metrics, "outer_videos": len(outer_labels)}
    return fold_row, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", default="binary_error_segment_localization")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    id_to_name = {video_id: name for name, video_id in manifest["name_to_id"].items()}
    bundle_folds, audit_folds = [], []
    for fold in manifest["folds"]:
        index = int(fold["fold"])
        bundle_row, audit_row = build_prediction_fold(
            index,
            Path(fold["annotation"]),
            args.raw_dir / f"fold_{index}_calibration.pkl",
            args.raw_dir / f"fold_{index}_validation.pkl",
            id_to_name,
        )
        bundle_folds.append(bundle_row)
        audit_folds.append(audit_row)
    bundle = {
        "schema_version": "round2-prediction-bundle-v1",
        "task": args.task,
        "method": args.method,
        "folds": bundle_folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    args.audit_output.write_text(
        json.dumps({"protocol": "calibration-only score/count selection; untouched outer folds", "folds": audit_folds}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
