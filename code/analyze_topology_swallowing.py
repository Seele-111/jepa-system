#!/usr/bin/env python3
"""Diagnose false negatives swallowed by broad JEPA-fused predictions.

This script replays an existing full-train CV summary and counts unmatched
ground-truth events that lie inside a prediction already matched to another
event. It is train/CV-only and must not be pointed at held-out test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_fn_candidate_attribution import DEFAULT_DATA_DIR, DEFAULT_SOURCE_SUMMARY, replay_fold
from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_event_topology_splitter import diagnose_swallowed_events
from train_segment_locator import load_signal_dataset


DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_topology_swallowing_diag.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    source_path = Path(args.source_summary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [str(name) for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]]
    folds = list(source["folds"])
    if int(args.fold_limit) > 0:
        folds = folds[: int(args.fold_limit)]

    fold_results = [
        replay_fold(
            records,
            feature_names,
            config=source["config"],
            fold_summary=fold,
            device=str(args.device),
            iou_threshold=float(args.iou_threshold),
        )
        for fold in folds
    ]
    diagnostics = [
        diagnose_swallowed_events(
            fold["fused_predictions"],
            fold["labels"],
            iou_threshold=float(args.iou_threshold),
        )
        for fold in fold_results
    ]
    total_fn = sum(int(item["segment"]["fn"]) for item in diagnostics)
    swallowed_fn = sum(int(item["swallowed_fn"]) for item in diagnostics)
    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(source_path),
        "fold_limit": int(args.fold_limit),
        "iou_threshold": float(args.iou_threshold),
        "device": str(args.device),
        "aggregate": aggregate_fold_metrics(fold_results),
        "source_aggregate": source.get("aggregate"),
        "swallowed_fn": int(swallowed_fn),
        "total_fn": int(total_fn),
        "swallowed_fn_fraction_of_fn": float(swallowed_fn / max(1, total_fn)),
        "videos_with_swallowed_fn": int(sum(int(item["videos_with_swallowed_fn"]) for item in diagnostics)),
        "folds": diagnostics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate_segment": result["aggregate"]["segment"],
                "swallowed_fn": result["swallowed_fn"],
                "total_fn": result["total_fn"],
                "swallowed_fn_fraction_of_fn": result["swallowed_fn_fraction_of_fn"],
                "out": str(out_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
