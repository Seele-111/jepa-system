#!/usr/bin/env python3
"""Diagnose why evidence-island rules miss topology-oracle replacements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_evidence_island_splitter import evidence_islands
from jepa_prediction_set_switcher import _as_segments
from run_strict_dp_event_set_oof import build_fold_local_raw_candidates
from run_strict_topology_set_split_oof import event_topology_evidence, load_prediction_export
from train_proposal_calibrator import interval_iou
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES
from train_segment_locator import (
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    contiguous_segments,
    load_signal_dataset,
)


def _normalise(segment):
    s, e = int(segment[0]), int(segment[1])
    return (s, e) if s <= e else (e, s)


def _segment_length(segment):
    s, e = _normalise(segment)
    return max(1, e - s + 1)


def _intersection(a, b):
    a = _normalise(a)
    b = _normalise(b)
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _coverage(segment, container):
    return _intersection(segment, container) / max(1, _segment_length(segment))


def _assign_island_candidates(
    islands,
    parent,
    candidates,
    min_island_coverage,
    max_child_parent_ratio,
    min_child_parent_coverage,
):
    used = set()
    matched = 0
    for island in islands:
        best = None
        best_key = None
        for candidate in candidates:
            if candidate in used or candidate == parent:
                continue
            if _segment_length(candidate) >= _segment_length(parent):
                continue
            if _segment_length(candidate) / max(1, _segment_length(parent)) > float(max_child_parent_ratio):
                continue
            if _coverage(candidate, parent) < float(min_child_parent_coverage):
                continue
            coverage = _coverage(island, candidate)
            if coverage < float(min_island_coverage):
                continue
            key = (coverage, -abs((candidate[0] + candidate[1]) - (island[0] + island[1])) / 2.0)
            if best_key is None or key > best_key:
                best_key = key
                best = candidate
        if best is not None:
            used.add(best)
            matched += 1
    return matched


def _matched_gt(predictions, labels, iou_threshold):
    gt_segments = [_normalise(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched = set()
    pred_to_gt = {}
    for pred_idx, pred_raw in enumerate(predictions):
        pred = _normalise(pred_raw)
        best_idx = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_segments):
            if gt_idx in matched:
                continue
            value = interval_iou(pred, gt)
            if value > best_iou:
                best_iou = value
                best_idx = gt_idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
            pred_to_gt[pred_idx] = gt_segments[best_idx]
    return gt_segments, matched, pred_to_gt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--predictions", default="/home/zzy/jepa_data/segment_train_full_event_v2_prediction_set_switcher.predictions.json")
    parser.add_argument("--out", default="/home/zzy/jepa_data/segment_train_full_event_v2_evidence_island_oracle_gap.json")
    parser.add_argument("--selector-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--selector-target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--selector-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selector-min-gaps", default="0,1,2,4,8")
    parser.add_argument("--selector-min-lengths", default="1,2,4,8")
    parser.add_argument("--evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--evidence-reducer", choices=["mean", "max", "stack"], default="mean")
    parser.add_argument("--island-thresholds", default="0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    parser.add_argument("--min-island-coverages", default="0.1,0.2,0.3,0.5,0.75")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    evidence_names = _parse_name_list(args.evidence_names)
    export = load_prediction_export(args.predictions)
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
        selector_epochs=120,
        selector_batch_size=512,
    )
    rows = []
    for fold in export.get("folds", []):
        val_idx = [int(idx) for idx in fold.get("val_idx", [])]
        labels = [np.asarray(item, dtype=np.int64) for item in fold.get("labels", [records[idx].labels for idx in val_idx])]
        for local_idx, video_idx in enumerate(val_idx):
            predictions = [_normalise(item) for item in fold.get("predictions", [])[local_idx]]
            gt_segments, matched, pred_to_gt = _matched_gt(predictions, labels[local_idx], args.iou_threshold)
            candidates, scores = raw_by_video[int(video_idx)]
            candidates = [_normalise(item) for item in candidates]
            evidence = event_topology_evidence(records[int(video_idx)], feature_names, evidence_names, args.evidence_reducer)
            curve = evidence.mean(axis=1) if evidence.ndim == 2 else evidence.reshape(-1)
            for pred_idx, parent in enumerate(predictions):
                if pred_idx not in pred_to_gt:
                    continue
                inside_gt = [gt for gt in gt_segments if _coverage(gt, parent) >= 0.8]
                if len(inside_gt) < 2:
                    continue
                candidate_hits = []
                for gt in inside_gt:
                    best = max((interval_iou(candidate, gt) for candidate in candidates), default=0.0)
                    candidate_hits.append(float(best))
                local_curve = curve[parent[0] : parent[1] + 1]
                island_counts = {}
                island_candidate_matches = {}
                for th in _parse_float_list(args.island_thresholds):
                    islands = evidence_islands(local_curve, threshold=float(th), min_length=1, min_gap=0)
                    global_islands = [(s + parent[0], e + parent[0]) for s, e in islands]
                    island_counts[str(th)] = len(global_islands)
                    island_candidate_matches[str(th)] = {
                        str(cov): _assign_island_candidates(
                            global_islands,
                            parent,
                            candidates,
                            min_island_coverage=float(cov),
                            max_child_parent_ratio=0.7,
                            min_child_parent_coverage=0.5,
                        )
                        for cov in _parse_float_list(args.min_island_coverages)
                    }
                rows.append(
                    {
                        "fold": int(fold.get("fold", 0)),
                        "video_idx": int(video_idx),
                        "parent": list(parent),
                        "inside_gt_count": int(len(inside_gt)),
                        "candidate_hits": candidate_hits,
                        "all_gt_candidate_covered": bool(all(hit >= float(args.iou_threshold) for hit in candidate_hits)),
                        "curve_mean": float(local_curve.mean()) if len(local_curve) else 0.0,
                        "curve_max": float(local_curve.max()) if len(local_curve) else 0.0,
                        "curve_std": float(local_curve.std()) if len(local_curve) else 0.0,
                        "island_counts": island_counts,
                        "island_candidate_matches": island_candidate_matches,
                    }
                )
    thresholds = _parse_float_list(args.island_thresholds)
    threshold_hits = {}
    threshold_candidate_hits = {}
    for th in thresholds:
        threshold_hits[str(th)] = int(sum(row["island_counts"][str(th)] >= 2 and row["all_gt_candidate_covered"] for row in rows))
        threshold_candidate_hits[str(th)] = {
            str(cov): int(
                sum(
                    row["island_counts"][str(th)] >= 2
                    and row["island_candidate_matches"][str(th)][str(cov)] >= 2
                    for row in rows
                )
            )
            for cov in _parse_float_list(args.min_island_coverages)
        }
    result = {
        "total_multi_gt_parents": len(rows),
        "candidate_covered_multi_gt_parents": int(sum(row["all_gt_candidate_covered"] for row in rows)),
        "threshold_hits": threshold_hits,
        "threshold_candidate_hits": threshold_candidate_hits,
        "selector_diagnostics": selector_diagnostics,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ["total_multi_gt_parents", "candidate_covered_multi_gt_parents", "threshold_hits", "threshold_candidate_hits"]},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
