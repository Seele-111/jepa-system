#!/usr/bin/env python3
"""Cross-validation for GPU boundary-aware JEPA event locator."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from train_boundary_event_locator import evaluate_boundary_records, predict_records, train_model
from train_segment_locator import load_signal_dataset, make_stratified_folds


def make_fold_args(args, fold_idx: int):
    fold_args = copy.copy(args)
    fold_args.seed = int(args.seed) + int(fold_idx)
    return fold_args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--smooth-windows", default="1,3")
    parser.add_argument("--min-gaps", default="0,1,2")
    parser.add_argument("--min-lengths", default="1,2,4")
    parser.add_argument("--refine-radii", default="0,2,4,8")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    folds = make_stratified_folds(records, args.folds, args.seed)
    all_indices = set(range(len(records)))
    fold_summaries: list[dict] = []
    for fold_idx, val_idx in enumerate(folds):
        train_idx = sorted(all_indices - set(val_idx))
        fold_args = make_fold_args(args, fold_idx)
        print(f"fold={fold_idx} train={len(train_idx)} val={len(val_idx)} device={args.device}", flush=True)
        model, selected_metrics, mean, std = train_model(records, train_idx, val_idx, fold_args)
        val_records = predict_records(model, records, val_idx, mean, std, args.device)
        validation = evaluate_boundary_records(
            val_records,
            selected_metrics["params"],
            iou_threshold=args.iou_threshold,
        )
        validation = {key: value for key, value in validation.items() if key != "predictions"}
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_names": [records[idx].name for idx in val_idx],
                "selected": selected_metrics,
                "validation": {
                    "epoch": selected_metrics.get("epoch"),
                    "loss": selected_metrics.get("loss"),
                    "params": selected_metrics["params"],
                    **validation,
                },
            }
        )
    summary = {
        "data_dir": args.data_dir,
        "n_videos": len(records),
        "frames": int(sum(len(record.labels) for record in records)),
        "positive_frames": int(sum(record.labels.sum() for record in records)),
        "folds": fold_summaries,
        "aggregate": aggregate_fold_metrics(fold_summaries),
        "config": {
            "folds": args.folds,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "boundary_radius": args.boundary_radius,
            "thresholds": args.thresholds,
            "smooth_windows": args.smooth_windows,
            "min_gaps": args.min_gaps,
            "min_lengths": args.min_lengths,
            "refine_radii": args.refine_radii,
            "iou_threshold": args.iou_threshold,
            "device": args.device,
            "task": "binary_error_segment_boundary_locator_cv",
        },
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
