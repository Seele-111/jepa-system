#!/usr/bin/env python3
"""Strict nested-CV predictor ablation for V-JEPA/I-JEPA/dual inputs."""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cross_validate_rgb_baseline import aggregate, nested_indices
from train_segment_locator import (
    TemporalSegmentLocator,
    VideoRecord,
    compute_normalizer,
    compute_pos_weight,
    evaluate_records,
    load_signal_dataset,
    locator_loss,
    make_stratified_folds,
    normalized_signals,
    predict_records,
    select_postprocess_params,
    _load_feature_names,
)


def subset_records(records: list[VideoRecord], channels: list[int]) -> list[VideoRecord]:
    return [replace(record, signals=record.signals[:, channels].copy()) for record in records]


def train_ablation_model(records, fit_idx, cal_idx, args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mean, std = compute_normalizer(records, fit_idx)
    model = TemporalSegmentLocator(records[0].signals.shape[1], args.hidden, args.dropout, "tcn").to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = compute_pos_weight(records, fit_idx)
    best_loss = float("inf"); best_state = copy.deepcopy(model.state_dict()); stale = 0
    for epoch in range(args.epochs):
        model.train(); order = fit_idx[:]; random.shuffle(order)
        for idx in order:
            x = torch.from_numpy(normalized_signals(records[idx], mean, std)).unsqueeze(0).to(args.device)
            y = torch.from_numpy(records[idx].labels).unsqueeze(0).to(args.device)
            loss = locator_loss(model(x), y, pos_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval(); losses = []
        with torch.no_grad():
            for idx in cal_idx:
                x = torch.from_numpy(normalized_signals(records[idx], mean, std)).unsqueeze(0).to(args.device)
                y = torch.from_numpy(records[idx].labels).unsqueeze(0).to(args.device)
                losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(model(x), y.float()).cpu()))
        cal_loss = float(np.mean(losses))
        if cal_loss < best_loss - 1e-4:
            best_loss = cal_loss; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state)
    return model, mean, std, {"best_calibration_bce": best_loss, "epochs_ran": epoch + 1}


def mode_channels(mode: str, feature_names: list[str]) -> list[int]:
    if mode == "vjepa":
        names = ["true_vjepa_raw"]
    elif mode == "ijepa":
        names = ["true_ijepa_dense_raw"]
    elif mode == "dual":
        names = ["true_vjepa_raw", "true_ijepa_dense_raw", "dual_jepa_composite"]
    elif mode == "vjepa_rank":
        names = ["true_vjepa_raw_rank"]
    elif mode == "ijepa_rank":
        names = ["true_ijepa_dense_raw_rank"]
    elif mode == "rank_triplet":
        names = ["true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"]
    elif mode == "full":
        return list(range(len(feature_names)))
    else:
        raise ValueError(f"unsupported ablation mode: {mode}")
    missing = [name for name in names if name not in feature_names]
    if missing:
        raise ValueError(f"missing feature names for mode {mode}: {missing}")
    return [feature_names.index(name) for name in names]


def run_mode(records: list[VideoRecord], folds: list[list[int]], mode: str, feature_names: list[str], args):
    channels = mode_channels(mode, feature_names)
    view = subset_records(records, channels)
    fold_rows = []
    bundle_folds = []
    for fold, outer_val in enumerate(folds):
        fit_idx, cal_idx = nested_indices(view, outer_val, args.calibration_ratio, args.seed + fold)
        model, mean, std, training = train_ablation_model(view, fit_idx, cal_idx, args, args.seed + fold)
        calibration = predict_records(model, view, cal_idx, mean, std, args.device)
        params, cal_metrics = select_postprocess_params(calibration, iou_threshold=0.3)
        validation = predict_records(model, view, outer_val, mean, std, args.device)
        metrics = evaluate_records(validation, params, iou_threshold=0.3)
        fold_rows.append({"fold": fold, "channels": channels, "fit_names": [view[i].name for i in fit_idx], "calibration_names": [view[i].name for i in cal_idx], "val_names": [view[i].name for i in outer_val], "training": training, "params": params, "calibration": cal_metrics, "metrics": metrics})
        bundle_folds.append({"fold": fold, "val_names": [row["name"] for row in validation], "predictions": [[list(s) for s in _segments_from_params(row["probs"], params)] for row in validation], "labels": [row["labels"].astype(int).tolist() for row in validation]})
        print(f"mode={mode} fold={fold} f1={metrics['segment']['f1']:.4f}", flush=True)
    return {"mode": mode, "channels": channels, "folds": fold_rows, "aggregate": {"frame": aggregate(fold_rows, "frame"), "segment": aggregate(fold_rows, "segment")}, "bundle": bundle_folds}


def _segments_from_params(probs: np.ndarray, params: dict):
    from train_segment_locator import _segments_from_params as convert
    return convert(probs, params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--calibration-ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=45); ap.add_argument("--patience", type=int, default=8); ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", default="vjepa,ijepa,dual,full")
    args = ap.parse_args()
    records = load_signal_dataset(args.data_dir); folds = make_stratified_folds(records, args.folds, args.seed)
    feature_names = _load_feature_names(args.data_dir)
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    reports = [run_mode(records, folds, mode, feature_names, args) for mode in modes]
    output = {"schema_version": "round2-jepa-ablation-v1", "protocol": "nested video-level CV", "data_dir": args.data_dir, "config": vars(args), "reports": [{k: v for k, v in report.items() if k != "bundle"} for report in reports], "bundles": {report["mode"]: {"schema_version": "round2-prediction-bundle-v1", "task": "jepa_predictor_ablation", "method": report["mode"], "folds": report["bundle"]} for report in reports}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    for report in reports:
        s = report["aggregate"]["segment"]; print(f"{report['mode']}: F1={s['f1_mean']:.4f} +/- {s['f1_std']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
