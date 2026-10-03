#!/usr/bin/env python3
"""Topology oracle headroom from an OOF prediction export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_topology_oracle import evaluate_topology_oracle
from jepa_prediction_set_switcher import _as_segments
from run_strict_dp_event_set_oof import build_fold_local_raw_candidates
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES
from train_segment_locator import (
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    load_signal_dataset,
)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(text).split(",") if item.strip())


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip())


def _load_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--predictions", default="/home/zzy/jepa_data/segment_train_full_event_v2_prediction_set_switcher.predictions.json")
    parser.add_argument("--out", default="/home/zzy/jepa_data/segment_train_full_event_v2_switcher_topology_oracle.json")
    parser.add_argument("--selector-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--selector-target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--selector-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selector-min-gaps", default="0,1,2,4,8")
    parser.add_argument("--selector-min-lengths", default="1,2,4,8")
    parser.add_argument("--selector-epochs", type=int, default=120)
    parser.add_argument("--selector-batch-size", type=int, default=512)
    parser.add_argument("--parent-min-lengths", default="12,24,48")
    parser.add_argument("--min-parent-gt-counts", default="2")
    parser.add_argument("--min-child-parent-coverages", default="0.5,0.7,0.9")
    parser.add_argument("--max-child-parent-ratios", default="0.5,0.7,1.0")
    parser.add_argument("--max-replaced-parents-per-video", default="1,2")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    export = _load_export(args.predictions)
    raw_by_video, selector_diagnostics = build_fold_local_raw_candidates(
        records,
        export,
        feature_names,
        channel_names,
        _parse_float_list(args.selector_thresholds),
        _parse_int_list(args.selector_min_gaps),
        _parse_int_list(args.selector_min_lengths),
        selector_model=args.selector_model,
        selector_target=args.selector_target,
        iou_threshold=float(args.iou_threshold),
        seed=int(args.seed),
        device=args.device,
        selector_epochs=int(args.selector_epochs),
        selector_batch_size=int(args.selector_batch_size),
    )
    base_predictions = []
    candidate_predictions = []
    labels = []
    for fold in export.get("folds", []):
        val_idx = [int(idx) for idx in fold.get("val_idx", [])]
        base_predictions.extend(fold.get("predictions", []))
        labels.extend([np.asarray(item, dtype=np.int64) for item in fold.get("labels", [records[idx].labels for idx in val_idx])])
        for idx in val_idx:
            candidates, _ = raw_by_video[int(idx)]
            candidate_predictions.append([(int(s), int(e)) for s, e in candidates])
    oracle = evaluate_topology_oracle(
        base_predictions,
        candidate_predictions,
        labels,
        iou_threshold=float(args.iou_threshold),
        parent_min_lengths=_parse_int_tuple(args.parent_min_lengths),
        min_parent_gt_counts=_parse_int_tuple(args.min_parent_gt_counts),
        min_child_parent_coverages=_parse_float_tuple(args.min_child_parent_coverages),
        max_child_parent_ratios=_parse_float_tuple(args.max_child_parent_ratios),
        max_replaced_parents_per_videos=_parse_int_tuple(args.max_replaced_parents_per_video),
    )
    result = {
        "data_dir": args.data_dir,
        "predictions": args.predictions,
        "selector_diagnostics": selector_diagnostics,
        "oracle": oracle,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "base_segment": oracle["base_metrics"]["segment"],
                "best": {
                    "config": oracle["best"]["config"],
                    "segment": oracle["best"]["metrics"]["segment"],
                    "tp_delta": oracle["best"]["tp_delta"],
                    "fp_delta": oracle["best"]["fp_delta"],
                    "fn_delta": oracle["best"]["fn_delta"],
                    "replacement_events": oracle["best"]["replacement_events"],
                },
                "best_no_fp_increase": {
                    "config": oracle["best_no_fp_increase"]["config"],
                    "segment": oracle["best_no_fp_increase"]["metrics"]["segment"],
                    "tp_delta": oracle["best_no_fp_increase"]["tp_delta"],
                    "fp_delta": oracle["best_no_fp_increase"]["fp_delta"],
                    "fn_delta": oracle["best_no_fp_increase"]["fn_delta"],
                    "replacement_events": oracle["best_no_fp_increase"]["replacement_events"],
                },
                "out": str(out),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
