#!/usr/bin/env python3
"""Attach train/validation fold indices to a prediction-only OOF export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-export", required=True)
    parser.add_argument("--fold-index-export", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pred = json.loads(Path(args.prediction_export).read_text(encoding="utf-8"))
    fold_index = json.loads(Path(args.fold_index_export).read_text(encoding="utf-8"))
    pred_folds = pred.get("folds", [])
    index_folds = fold_index.get("folds", [])
    if len(pred_folds) != len(index_folds):
        raise ValueError(f"fold count mismatch: {len(pred_folds)} vs {len(index_folds)}")

    aligned_folds = []
    for pos, (pred_fold, index_fold) in enumerate(zip(pred_folds, index_folds)):
        predictions = pred_fold.get("predictions", [])
        val_idx = [int(idx) for idx in index_fold.get("val_idx", [])]
        train_idx = [int(idx) for idx in index_fold.get("train_idx", [])]
        if len(predictions) != len(val_idx):
            raise ValueError(f"fold {pos} predictions/val_idx mismatch: {len(predictions)} vs {len(val_idx)}")
        merged = dict(pred_fold)
        merged["fold"] = int(index_fold.get("fold", pred_fold.get("fold", pos)))
        merged["train_idx"] = train_idx
        merged["val_idx"] = val_idx
        if "val_names" in index_fold:
            merged["val_names"] = index_fold["val_names"]
        aligned_folds.append(merged)

    out_data = dict(pred)
    out_data["folds"] = aligned_folds
    out_data["aligned_from"] = {
        "prediction_export": str(args.prediction_export),
        "fold_index_export": str(args.fold_index_export),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "folds": len(aligned_folds)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

