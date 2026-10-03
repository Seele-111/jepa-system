#!/usr/bin/env python3
"""Leakage-controlled OOF evaluation for frozen RGB temporal features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_segment_locator import (
    _segments_from_params,
    evaluate_records,
    load_signal_dataset,
    make_stratified_folds,
    select_postprocess_params,
    train_val_split,
)


def fit_rgb_classifier(x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-4,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    return model.fit(x, y)


def nested_indices(records, outer_val: list[int], calibration_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    outer_train = [i for i in range(len(records)) if i not in set(outer_val)]
    local_records = [records[i] for i in outer_train]
    local_fit, local_cal = train_val_split(local_records, calibration_ratio, seed)
    fit_idx = [outer_train[i] for i in local_fit]
    cal_idx = [outer_train[i] for i in local_cal]
    if set(fit_idx) & set(cal_idx) or set(outer_val) & (set(fit_idx) | set(cal_idx)):
        raise RuntimeError("nested split leakage detected")
    return fit_idx, cal_idx


def frame_table(records, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([records[i].signals for i in indices]), np.concatenate([records[i].labels for i in indices])


def probability_records(records, indices: list[int], classifier) -> list[dict]:
    result = []
    for i in indices:
        probs = classifier.predict_proba(records[i].signals)[:, 1].astype(np.float32)
        result.append({"name": records[i].name, "labels": records[i].labels, "probs": probs})
    return result


def aggregate(folds: list[dict], section: str) -> dict:
    metrics = [f["metrics"][section] for f in folds]
    return {
        **{f"{k}_mean": float(np.mean([m[k] for m in metrics])) for k in ("precision", "recall", "f1")},
        **{f"{k}_std": float(np.std([m[k] for m in metrics])) for k in ("precision", "recall", "f1")},
        **{k: int(sum(m[k] for m in metrics)) for k in ("tp", "fp", "fn")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--predictions-out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibration-ratio", type=float, default=0.2)
    ap.add_argument("--method", default="Frozen Kinetics R3D-18 + frame classifier")
    ap.add_argument("--task", default="rgb_temporal_vad")
    args = ap.parse_args()
    records = load_signal_dataset(args.data_dir)
    outer_folds = make_stratified_folds(records, args.folds, args.seed)
    fold_rows = []
    bundle_folds = []
    for fold, outer_val in enumerate(outer_folds):
        fit_idx, cal_idx = nested_indices(records, outer_val, args.calibration_ratio, args.seed + fold)
        x, y = frame_table(records, fit_idx)
        classifier = fit_rgb_classifier(x, y, args.seed + fold)
        calibration = probability_records(records, cal_idx, classifier)
        params, calibration_metrics = select_postprocess_params(calibration, iou_threshold=0.3)
        validation = probability_records(records, outer_val, classifier)
        metrics = evaluate_records(validation, params, iou_threshold=0.3)
        fold_rows.append({"fold": fold, "fit_names": [records[i].name for i in fit_idx], "calibration_names": [records[i].name for i in cal_idx], "val_names": [records[i].name for i in outer_val], "params": params, "calibration": calibration_metrics, "metrics": metrics})
        bundle_folds.append({"fold": fold, "val_names": [r["name"] for r in validation], "predictions": [[list(s) for s in _segments_from_params(r["probs"], params)] for r in validation], "labels": [r["labels"].astype(int).tolist() for r in validation]})
        print(f"fold={fold} f1={metrics['segment']['f1']:.4f}", flush=True)
    summary = {"method": args.method, "input": "RGB", "protocol": "nested video-level CV", "data_dir": args.data_dir, "folds": fold_rows, "aggregate": {"frame": aggregate(fold_rows, "frame"), "segment": aggregate(fold_rows, "segment")}}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bundle = {"schema_version": "round2-prediction-bundle-v1", "task": args.task, "method": summary["method"], "folds": bundle_folds}
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions_out).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
