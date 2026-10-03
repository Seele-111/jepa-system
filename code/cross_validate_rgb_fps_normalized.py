#!/usr/bin/env python3
"""Development-only R3D VAD ablation with FPS-normalized input smoothing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cross_validate_rgb_baseline import (
    aggregate,
    fit_rgb_classifier,
    frame_table,
    nested_indices,
    probability_records,
)
from train_segment_locator import _segments_from_params, evaluate_records, load_signal_dataset, make_stratified_folds, select_postprocess_params


def load_fps(card_path: Path) -> dict[str, float]:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    return {str(row["video_name"]): float(row["fps"]) for row in card.get("videos", []) if row.get("fps") is not None}


def smooth_seconds(values: np.ndarray, fps: float, target_fps: float) -> np.ndarray:
    window = max(1, int(round(float(fps) / float(target_fps))))
    if window <= 1:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left, pad_right = window // 2, window - 1 - window // 2
    padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, col], kernel, mode="valid") for col in range(values.shape[1])], axis=1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset-card", required=True, type=Path)
    ap.add_argument("--target-fps", type=float, default=8.0)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--predictions-out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    records = load_signal_dataset(args.data_dir)
    fps = load_fps(args.dataset_card)
    for record in records:
        if record.name not in fps:
            raise ValueError(f"missing FPS metadata for {record.name}")
        record.signals = smooth_seconds(record.signals, fps[record.name], args.target_fps)

    outer_folds = make_stratified_folds(records, args.folds, args.seed)
    fold_rows, bundle_folds = [], []
    for fold, outer_val in enumerate(outer_folds):
        fit_idx, cal_idx = nested_indices(records, outer_val, 0.2, args.seed + fold)
        classifier = fit_rgb_classifier(*frame_table(records, fit_idx), args.seed + fold)
        calibration = probability_records(records, cal_idx, classifier)
        params, calibration_metrics = select_postprocess_params(calibration, iou_threshold=0.3)
        validation = probability_records(records, outer_val, classifier)
        metrics = evaluate_records(validation, params, iou_threshold=0.3)
        fold_rows.append({"fold": fold, "params": params, "calibration": calibration_metrics, "metrics": metrics, "val_names": [records[i].name for i in outer_val]})
        bundle_folds.append({"fold": fold, "val_names": [r["name"] for r in validation], "predictions": [[list(s) for s in _segments_from_params(r["probs"], params)] for r in validation], "labels": [r["labels"].astype(int).tolist() for r in validation]})
        print(f"fold={fold} f1={metrics['segment']['f1']:.4f}", flush=True)
    summary = {"method": "Frozen Kinetics R3D-18 + FPS-normalized VAD", "protocol": "nested video-level CV", "target_fps": args.target_fps, "data_dir": args.data_dir, "dataset_card": str(args.dataset_card), "folds": fold_rows, "aggregate": {"frame": aggregate(fold_rows, "frame"), "segment": aggregate(fold_rows, "segment")}}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.write_text(json.dumps({"schema_version": "round2-prediction-bundle-v1", "task": "rgb_temporal_vad_fps_normalized", "method": summary["method"], "folds": bundle_folds}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
