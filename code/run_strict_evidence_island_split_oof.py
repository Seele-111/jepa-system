#!/usr/bin/env python3
"""Strict OOF JEPA evidence-island splitting for broad predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_evidence_island_splitter import split_evidence_island_predictions
from run_strict_topology_set_split_oof import (
    build_topology_folds,
    event_topology_evidence,
    load_prediction_export,
    parse_nms_list,
    parse_optional_int_list,
)
from run_strict_dp_event_set_oof import build_fold_local_raw_candidates
from selector_fusion import evaluate_fused_predictions
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES
from train_segment_locator import (
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    load_signal_dataset,
)


def flatten(folds: list[dict], key: str) -> list:
    out = []
    for fold in folds:
        out.extend(fold[key])
    return out


def select_params(
    base,
    candidates,
    scores,
    evidence,
    labels,
    args,
) -> tuple[dict, dict, list[list[tuple[int, int]]]]:
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=float(args.iou_threshold))
    best_config = {"enabled": False, "base_metrics": base_metrics}
    best_metrics = base_metrics
    best_predictions = [[(int(s), int(e)) for s, e in video] for video in base]
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for island_threshold in _parse_float_list(args.island_thresholds):
        for island_min_length in _parse_int_list(args.island_min_lengths):
            for island_min_gap in _parse_int_list(args.island_min_gaps):
                for min_islands in _parse_int_list(args.min_islands):
                    for min_candidate_score in _parse_float_list(args.min_candidate_scores):
                        for max_candidate_rank in parse_optional_int_list(args.max_candidate_ranks):
                            for min_island_coverage in _parse_float_list(args.min_island_coverages):
                                for max_child_parent_ratio in _parse_float_list(args.max_child_parent_ratios):
                                    for min_child_parent_coverage in _parse_float_list(args.min_child_parent_coverages):
                                        for child_nms_iou in parse_nms_list(args.child_nms_ious):
                                            for max_children_per_parent in _parse_int_list(args.max_children_per_parent):
                                                for max_replaced_parents_per_video in _parse_int_list(args.max_replaced_parents_per_video):
                                                    split = split_evidence_island_predictions(
                                                        base,
                                                        candidates,
                                                        scores,
                                                        evidence,
                                                        island_threshold=float(island_threshold),
                                                        island_min_length=int(island_min_length),
                                                        island_min_gap=int(island_min_gap),
                                                        min_islands=int(min_islands),
                                                        min_candidate_score=float(min_candidate_score),
                                                        max_candidate_rank=max_candidate_rank,
                                                        min_island_coverage=float(min_island_coverage),
                                                        max_child_parent_ratio=float(max_child_parent_ratio),
                                                        min_child_parent_coverage=float(min_child_parent_coverage),
                                                        child_nms_iou=child_nms_iou,
                                                        max_children_per_parent=int(max_children_per_parent),
                                                        max_replaced_parents_per_video=int(max_replaced_parents_per_video),
                                                    )
                                                    metrics = evaluate_fused_predictions(split, labels, iou_threshold=float(args.iou_threshold))
                                                    fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                                                    if args.max_fp_increase is not None and fp_delta > int(args.max_fp_increase):
                                                        continue
                                                    changed = sum(int(sorted(a) != sorted(b)) for a, b in zip(base, split))
                                                    if changed <= 0:
                                                        continue
                                                    key = (
                                                        float(metrics["segment"]["f1"]),
                                                        float(metrics["segment"]["precision"]),
                                                        float(metrics["segment"]["recall"]),
                                                        float(metrics["frame"]["f1"]),
                                                        -float(changed),
                                                    )
                                                    if key > best_key:
                                                        best_key = key
                                                        best_metrics = metrics
                                                        best_predictions = split
                                                        best_config = {
                                                            "enabled": True,
                                                            "island_threshold": float(island_threshold),
                                                            "island_min_length": int(island_min_length),
                                                            "island_min_gap": int(island_min_gap),
                                                            "min_islands": int(min_islands),
                                                            "min_candidate_score": float(min_candidate_score),
                                                            "max_candidate_rank": max_candidate_rank,
                                                            "min_island_coverage": float(min_island_coverage),
                                                            "max_child_parent_ratio": float(max_child_parent_ratio),
                                                            "min_child_parent_coverage": float(min_child_parent_coverage),
                                                            "child_nms_iou": child_nms_iou,
                                                            "max_children_per_parent": int(max_children_per_parent),
                                                            "max_replaced_parents_per_video": int(max_replaced_parents_per_video),
                                                            "changed_videos": int(changed),
                                                            "fp_delta": int(fp_delta),
                                                            "base_metrics": base_metrics,
                                                        }
    return best_config, best_metrics, best_predictions


def apply_config(base, candidates, scores, evidence, config):
    if not config.get("enabled"):
        return [[(int(s), int(e)) for s, e in video] for video in base]
    return split_evidence_island_predictions(
        base,
        candidates,
        scores,
        evidence,
        island_threshold=float(config["island_threshold"]),
        island_min_length=int(config["island_min_length"]),
        island_min_gap=int(config["island_min_gap"]),
        min_islands=int(config["min_islands"]),
        min_candidate_score=float(config["min_candidate_score"]),
        max_candidate_rank=config.get("max_candidate_rank"),
        min_island_coverage=float(config["min_island_coverage"]),
        max_child_parent_ratio=float(config["max_child_parent_ratio"]),
        min_child_parent_coverage=float(config["min_child_parent_coverage"]),
        child_nms_iou=config.get("child_nms_iou"),
        max_children_per_parent=int(config["max_children_per_parent"]),
        max_replaced_parents_per_video=int(config["max_replaced_parents_per_video"]),
    )


def run_strict(folds: list[dict], args) -> dict:
    results = []
    for heldout_pos, heldout in enumerate(folds):
        calibration = [fold for pos, fold in enumerate(folds) if pos != heldout_pos]
        config, calibration_metrics, _ = select_params(
            flatten(calibration, "base"),
            flatten(calibration, "candidates"),
            flatten(calibration, "scores"),
            flatten(calibration, "evidence"),
            flatten(calibration, "labels"),
            args,
        )
        predictions = apply_config(heldout["base"], heldout["candidates"], heldout["scores"], heldout["evidence"], config)
        metrics = evaluate_fused_predictions(predictions, heldout["labels"], iou_threshold=float(args.iou_threshold))
        results.append(
            {
                "fold": int(heldout["fold"]),
                "config": config,
                "calibration": calibration_metrics,
                "validation": metrics,
                "predictions": [[[int(s), int(e)] for s, e in video] for video in predictions],
            }
        )
    return {"folds": results, "aggregate": aggregate_fold_metrics(results)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--selector-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--selector-target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--selector-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selector-min-gaps", default="0,1,2,4,8")
    parser.add_argument("--selector-min-lengths", default="1,2,4,8")
    parser.add_argument("--selector-epochs", type=int, default=120)
    parser.add_argument("--selector-batch-size", type=int, default=512)
    parser.add_argument("--evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--evidence-reducer", choices=["stack", "mean", "max"], default="mean")
    parser.add_argument("--island-thresholds", default="0.35,0.45,0.55,0.65")
    parser.add_argument("--island-min-lengths", default="1,2,4")
    parser.add_argument("--island-min-gaps", default="0,1")
    parser.add_argument("--min-islands", default="2")
    parser.add_argument("--min-candidate-scores", default="0,0.05,0.1")
    parser.add_argument("--max-candidate-ranks", default="none,60")
    parser.add_argument("--min-island-coverages", default="0.5,0.75")
    parser.add_argument("--max-child-parent-ratios", default="0.5,0.7")
    parser.add_argument("--min-child-parent-coverages", default="0.5,0.7")
    parser.add_argument("--child-nms-ious", default="0.3")
    parser.add_argument("--max-children-per-parent", default="2,3")
    parser.add_argument("--max-replaced-parents-per-video", default="1")
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    evidence_names = _parse_name_list(args.evidence_names)
    export = load_prediction_export(args.base_predictions)
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
    folds = build_topology_folds(export, records, raw_by_video, feature_names, evidence_names, args.evidence_reducer)
    result = run_strict(folds, args)
    result.update(
        {
            "data_dir": args.data_dir,
            "base_predictions": args.base_predictions,
            "selector_diagnostics": selector_diagnostics,
            "evidence_names": evidence_names,
            "evidence_reducer": args.evidence_reducer,
            "max_fp_increase": args.max_fp_increase,
            "iou_threshold": args.iou_threshold,
        }
    )
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
