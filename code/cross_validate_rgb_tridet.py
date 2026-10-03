#!/usr/bin/env python3
"""Independent TriDet-style RGB boundary detector on frozen R3D features."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from cross_validate_rgb_baseline import aggregate, nested_indices
from selector_fusion import evaluate_fused_predictions
from train_jepa_tridet_lite import (
    JEPATriDetLite,
    decode_dense_predictions,
    make_dense_batch,
    select_decode_params,
    tridet_loss,
)
from train_segment_locator import compute_normalizer, load_signal_dataset, make_stratified_folds


def predict(model, records, indices, mean, std, args, params):
    model.eval()
    predictions, labels, names = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            batch = make_dense_batch(records, batch_indices, mean, std, model.strides, model.max_regression_bin, 1.0, args.device, "event_center")
            outputs = model(batch.x, batch.mask)
            predictions.extend(decode_dense_predictions(outputs, batch.lengths, params["score_threshold"], params["nms_iou"], params["max_predictions"]))
            labels.extend(batch.labels)
            names.extend(batch.names)
    return predictions, labels, names


def train_one(records, fit_idx, cal_idx, args, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    mean, std = compute_normalizer(records, fit_idx)
    model = JEPATriDetLite(records[0].signals.shape[1], args.hidden, tuple(args.strides), args.max_regression_bin, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict()); best_loss = float("inf"); stale = 0
    for epoch in range(args.epochs):
        model.train(); order = rng.permutation(fit_idx); losses = []
        for start in range(0, len(order), args.batch_size):
            batch = make_dense_batch(records, [int(i) for i in order[start:start + args.batch_size]], mean, std, tuple(args.strides), args.max_regression_bin, 1.0, args.device, "event_center")
            optimizer.zero_grad(set_to_none=True)
            loss, _ = tridet_loss(model(batch.x, batch.mask), batch.targets, args.center_weight, args.distribution_weight, args.iou_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        # Calibration uses only framewise center confidence, never outer labels.
        cal_batch = make_dense_batch(records, cal_idx, mean, std, tuple(args.strides), args.max_regression_bin, 1.0, args.device, "event_center")
        model.eval()
        with torch.no_grad():
            outputs = model(cal_batch.x, cal_batch.mask)
            center = torch.sigmoid(outputs[0]["center_logits"]).cpu().numpy()
        cal_loss = float(np.mean((center[:, : cal_batch.mask.shape[1]] - cal_batch.mask.cpu().numpy() * np.asarray([np.pad(r.labels, (0, cal_batch.mask.shape[1] - len(r.labels))) for r in [records[i] for i in cal_idx]])) ** 2))
        if cal_loss < best_loss - 1e-5:
            best_loss = cal_loss; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state)
    return model, mean, std, {"best_calibration_center_mse": best_loss, "epochs_ran": epoch + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--summary", required=True); ap.add_argument("--predictions-out", required=True)
    ap.add_argument("--checkpoints-dir", default=""); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4]); ap.add_argument("--max-regression-bin", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--center-weight", type=float, default=1.0); ap.add_argument("--distribution-weight", type=float, default=1.0); ap.add_argument("--iou-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); records = load_signal_dataset(args.data_dir); folds = make_stratified_folds(records, args.folds, args.seed)
    fold_rows = []; bundle_folds = []
    for fold, outer_val in enumerate(folds):
        fit_idx, cal_idx = nested_indices(records, outer_val, 0.2, args.seed + fold)
        model, mean, std, training = train_one(records, fit_idx, cal_idx, args, args.seed + fold)
        cal_predictions, cal_labels, _ = predict(model, records, cal_idx, mean, std, args, {"score_threshold": 0.5, "nms_iou": 0.3, "max_predictions": None})
        params, cal_metrics, _ = select_decode_params(model, records, cal_idx, mean, std, args.device, args.batch_size, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.1, 0.3, 0.5], [None], 0.3)
        predictions, labels, names = predict(model, records, outer_val, mean, std, args, params)
        metrics = evaluate_fused_predictions(predictions, labels, 0.3)
        checkpoint = None; checkpoint_sha256 = None
        if args.checkpoints_dir:
            path = Path(args.checkpoints_dir) / f"fold_{fold}.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "mean": mean, "std": std, "params": params, "config": vars(args)}, path); checkpoint = str(path); checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        fold_rows.append({"fold": fold, "fit_names": [records[i].name for i in fit_idx], "calibration_names": [records[i].name for i in cal_idx], "val_names": names, "training": training, "params": params, "calibration": cal_metrics, "metrics": metrics, "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256})
        bundle_folds.append({"fold": fold, "val_names": names, "predictions": [[list(s) for s in row] for row in predictions], "labels": [np.asarray(y, dtype=int).tolist() for y in labels]})
        print(f"fold={fold} f1={metrics['segment']['f1']:.4f}", flush=True)
    method = "Independent TriDet-style boundary detector on frozen R3D-18 RGB"
    summary = {"method": method, "implementation": "independent adapter, not official TriDet", "input": "RGB", "protocol": "nested video-level CV", "config": vars(args), "folds": fold_rows, "aggregate": {"segment": aggregate(fold_rows, "segment")}}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True); Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True); Path(args.predictions_out).write_text(json.dumps({"schema_version": "round2-prediction-bundle-v1", "task": "rgb_temporal_localization", "method": method, "folds": bundle_folds}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
