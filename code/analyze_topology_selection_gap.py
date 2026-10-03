#!/usr/bin/env python3
"""Inspect why JEPA topology oracle replacements are not selected.

The script replays full-train CV folds, builds label-oracle topology
replacements, then reports selector/evidence statistics for the child
candidates that would have reduced false negatives. It is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fn_candidate_attribution import replay_fold
from analyze_topology_oracle import DEFAULT_DATA_DIR, DEFAULT_SOURCE_SUMMARY, oracle_topology_replace_predictions
from cross_validate_selector_fusion import _event_topology_evidence
from jepa_event_topology_splitter import _coverage, _evidence_features, _normalise_segment, _segment_length
from train_segment_locator import load_signal_dataset


DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_topology_selection_gap.json"


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(arr)),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _rank_desc(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--parent-min-length", type=int, default=12)
    parser.add_argument("--min-parent-gt-count", type=int, default=2)
    parser.add_argument("--min-child-parent-coverage", type=float, default=0.5)
    parser.add_argument("--max-child-parent-ratio", type=float, default=0.5)
    parser.add_argument("--max-replaced-parents-per-video", type=int, default=1)
    parser.add_argument("--evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--evidence-threshold", type=float, default=0.6)
    args = parser.parse_args()

    source_path = Path(args.source_summary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [str(name) for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]]
    evidence_names = [item.strip() for item in str(args.evidence_names).split(",") if item.strip()]
    folds = list(source["folds"])
    if int(args.fold_limit) > 0:
        folds = folds[: int(args.fold_limit)]

    child_rows: list[dict] = []
    parent_rows: list[dict] = []
    for fold in folds:
        fold_result = replay_fold(
            records,
            feature_names,
            config=source["config"],
            fold_summary=fold,
            device=str(args.device),
            iou_threshold=float(args.iou_threshold),
        )
        _, events = oracle_topology_replace_predictions(
            fold_result["fused_predictions"],
            fold_result["raw_candidates"],
            fold_result["labels"],
            iou_threshold=float(args.iou_threshold),
            parent_min_length=int(args.parent_min_length),
            min_parent_gt_count=int(args.min_parent_gt_count),
            min_child_parent_coverage=float(args.min_child_parent_coverage),
            max_child_parent_ratio=float(args.max_child_parent_ratio),
            max_replaced_parents_per_video=int(args.max_replaced_parents_per_video),
        )
        for event in events:
            local_idx = int(event["video_local_idx"])
            record_idx = int(fold_result["val_idx"][local_idx])
            parent = _normalise_segment(tuple(event["parent"]))
            candidates = [_normalise_segment(item) for item in fold_result["raw_candidates"][local_idx]]
            scores = np.nan_to_num(np.asarray(fold_result["raw_scores"][local_idx], dtype=np.float32).reshape(-1), nan=0.0)
            ranks = _rank_desc(scores)
            evidence = _event_topology_evidence(records[record_idx], feature_names, evidence_names, "stack")
            parent_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "video_idx": record_idx,
                    "video": records[record_idx].name,
                    "parent": list(parent),
                    "parent_length": int(_segment_length(parent)),
                    "children": event["children"],
                    "covered_gts": event["covered_gts"],
                    "candidate_count": int(len(candidates)),
                    "inside_candidate_count": int(sum(_coverage(candidate, parent) >= 0.5 for candidate in candidates)),
                }
            )
            for child_raw in event["children"]:
                child = _normalise_segment(tuple(child_raw))
                matching = [idx for idx, candidate in enumerate(candidates) if candidate == child]
                if not matching:
                    continue
                idx = int(matching[0])
                evidence_mean, active_fraction, parent_contrast = _evidence_features(
                    evidence,
                    parent,
                    child,
                    evidence_threshold=float(args.evidence_threshold),
                )
                child_rows.append(
                    {
                        "fold": int(fold["fold"]),
                        "video_idx": record_idx,
                        "video": records[record_idx].name,
                        "parent": list(parent),
                        "child": list(child),
                        "child_length": int(_segment_length(child)),
                        "child_parent_ratio": float(_segment_length(child) / max(1, _segment_length(parent))),
                        "child_parent_coverage": float(_coverage(child, parent)),
                        "selector_score": float(scores[idx]),
                        "selector_rank": int(ranks[idx]),
                        "evidence_mean": float(evidence_mean),
                        "active_fraction": float(active_fraction),
                        "parent_contrast": float(parent_contrast),
                    }
                )

    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(source_path),
        "fold_limit": int(args.fold_limit),
        "oracle_config": {
            "parent_min_length": int(args.parent_min_length),
            "min_parent_gt_count": int(args.min_parent_gt_count),
            "min_child_parent_coverage": float(args.min_child_parent_coverage),
            "max_child_parent_ratio": float(args.max_child_parent_ratio),
            "max_replaced_parents_per_video": int(args.max_replaced_parents_per_video),
        },
        "summary": {
            "parents": int(len(parent_rows)),
            "children": int(len(child_rows)),
            "selector_score": _percentiles([row["selector_score"] for row in child_rows]),
            "selector_rank": _percentiles([row["selector_rank"] for row in child_rows]),
            "evidence_mean": _percentiles([row["evidence_mean"] for row in child_rows]),
            "active_fraction": _percentiles([row["active_fraction"] for row in child_rows]),
            "parent_contrast": _percentiles([row["parent_contrast"] for row in child_rows]),
            "child_parent_ratio": _percentiles([row["child_parent_ratio"] for row in child_rows]),
        },
        "parents": parent_rows,
        "children": child_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "out": str(out_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
