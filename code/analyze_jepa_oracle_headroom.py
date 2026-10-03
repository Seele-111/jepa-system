#!/usr/bin/env python3
"""Replay full-train CV and measure JEPA candidate oracle headroom.

This is a development diagnostic. It estimates how much recall the current
V/I-JEPA candidate pool could recover if a perfect set selector existed.
Do not use it on held-out test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fn_candidate_attribution import replay_fold
from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_oracle_diagnostics import evaluate_budgeted_oracle
from train_segment_locator import load_signal_dataset


DEFAULT_DATA_DIR = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2"
DEFAULT_SOURCE_SUMMARY = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json"
DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_oracle_headroom.json"


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_or_none_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(None if stripped.lower() in {"none", "null"} else float(stripped))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-rescues-per-video-options", default="1,2,3,4")
    parser.add_argument("--max-base-iou-options", default="0,0.05,0.1,0.25,none")
    parser.add_argument("--selector-nms-iou-options", default="none,0.1,0.3,0.5")
    parser.add_argument("--fp-budget-options", default="0,2,5,10,20")
    args = parser.parse_args()

    source_path = Path(args.source_summary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [
        str(name)
        for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]
    ]
    folds = list(source["folds"])
    if args.fold_limit > 0:
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

    base_predictions = [video for fold in fold_results for video in fold["fused_predictions"]]
    candidate_predictions = [video for fold in fold_results for video in fold["raw_candidates"]]
    labels = [np.asarray(video, dtype=np.int64) for fold in fold_results for video in fold["labels"]]
    oracle = evaluate_budgeted_oracle(
        base_predictions,
        candidate_predictions,
        labels,
        iou_threshold=float(args.iou_threshold),
        max_rescues_per_video_options=_parse_int_list(args.max_rescues_per_video_options),
        max_base_iou_options=_parse_float_or_none_list(args.max_base_iou_options),
        selector_nms_iou_options=_parse_float_or_none_list(args.selector_nms_iou_options),
        fp_budget_options=_parse_int_list(args.fp_budget_options),
    )
    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(source_path),
        "fold_limit": int(args.fold_limit),
        "iou_threshold": float(args.iou_threshold),
        "device": str(args.device),
        "source_aggregate": source.get("aggregate"),
        "replay_aggregate": aggregate_fold_metrics(fold_results),
        "oracle": oracle,
        "folds": [
            {
                "fold": int(fold["fold"]),
                "validation": fold["validation"],
                "fn": len(fold["fn_rows"]),
                "fp": len(fold["fp_rows"]),
                "videos": len(fold["val_idx"]),
                "raw_candidate_mean": float(np.mean([len(items) for items in fold["raw_candidates"]]))
                if fold["raw_candidates"]
                else 0.0,
                "raw_candidate_max": int(max((len(items) for items in fold["raw_candidates"]), default=0)),
            }
            for fold in fold_results
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        "replay_segment": result["replay_aggregate"]["segment"],
        "oracle_miss_upper_bound_counts": oracle["miss_upper_bound_counts"],
        "best_by_fp_budget": {
            key: {
                "config": value["config"],
                "segment": value["metrics"]["segment"],
                "rescued_tp": value["rescued_tp"],
                "fp_increase": value["fp_increase"],
                "rescued_segments": value["rescued_segments"],
            }
            for key, value in oracle["best_by_fp_budget"].items()
        },
        "out": str(out),
    }
    print(json.dumps(compact, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
