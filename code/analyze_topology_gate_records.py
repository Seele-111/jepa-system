#!/usr/bin/env python3
"""Diagnose train/validation sample balance for the JEPA topology split gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fn_candidate_attribution import replay_fold
from cross_validate_selector_fusion import _event_topology_evidence
from jepa_topology_split_gate import build_split_gate_records
from train_segment_locator import load_signal_dataset


DEFAULT_DATA_DIR = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2"
DEFAULT_SOURCE_SUMMARY = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json"
DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_topology_gate_records.json"


def _parse_float_list(value) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _parse_int_list(value) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(arr)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _records_summary(records) -> dict:
    labels = [int(record.label) for record in records]
    return {
        "records": int(len(records)),
        "positive": int(sum(labels)),
        "positive_rate": float(sum(labels) / max(1, len(labels))),
        "children_per_parent": _percentiles([len(record.children) for record in records]),
        "parent_length": _percentiles([record.features[0] for record in records]),
        "child_coverage_ratio": _percentiles([record.features[3] for record in records]),
        "child_score_mean": _percentiles([record.features[12] for record in records]),
        "child_rank_mean": _percentiles([record.features[15] for record in records]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--parent-min-length", type=int, default=12)
    parser.add_argument("--min-child-parent-coverage", type=float, default=0.5)
    parser.add_argument("--max-child-parent-ratio", type=float, default=0.5)
    parser.add_argument("--max-children-per-parent", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    source_path = Path(args.source_summary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [str(name) for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]]
    config = source["config"]
    evidence_names = ["true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"]
    folds = list(source["folds"])
    if int(args.fold_limit) > 0:
        folds = folds[: int(args.fold_limit)]

    out_folds = []
    for fold in folds:
        fold_result = replay_fold(records, feature_names, config, fold, str(args.device), float(args.iou_threshold))
        val_evidence = [
            _event_topology_evidence(records[idx], feature_names, evidence_names, "stack")
            for idx in fold_result["val_idx"]
        ]
        val_records = build_split_gate_records(
            fold_result["fused_predictions"],
            fold_result["raw_candidates"],
            fold_result["raw_scores"],
            val_evidence,
            fold_result["labels"],
            iou_threshold=float(args.iou_threshold),
            parent_min_length=int(args.parent_min_length),
            min_child_parent_coverage=float(args.min_child_parent_coverage),
            max_child_parent_ratio=float(args.max_child_parent_ratio),
            max_children_per_parent=int(args.max_children_per_parent),
        )
        out_folds.append(
            {
                "fold": int(fold["fold"]),
                "validation": fold_result["validation"],
                "val_gate_records": _records_summary(val_records),
            }
        )

    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(source_path),
        "config": {
            "parent_min_length": int(args.parent_min_length),
            "min_child_parent_coverage": float(args.min_child_parent_coverage),
            "max_child_parent_ratio": float(args.max_child_parent_ratio),
            "max_children_per_parent": int(args.max_children_per_parent),
        },
        "folds": out_folds,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"folds": out_folds, "out": str(out_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
