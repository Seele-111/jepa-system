#!/usr/bin/env python3
"""Strict OOF topology-set splitting for current-best OOF predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_prediction_set_switcher import _as_segments
from jepa_topology_set_splitter import select_topology_set_split_params, split_topology_set_predictions
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


def parse_optional_int_list(text: str) -> list[int | None]:
    values: list[int | None] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else int(item))
    return values


def parse_nms_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else float(item))
    return values


def load_prediction_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def event_topology_evidence(record, feature_names: list[str], evidence_names: list[str], reducer: str) -> np.ndarray:
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    channels = [name_to_idx[name] for name in evidence_names if name in name_to_idx]
    if not channels:
        raise ValueError(f"none of topology evidence channels found: {evidence_names}")
    values = np.nan_to_num(record.signals[:, channels], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if reducer == "stack":
        return values
    if reducer == "mean":
        return values.mean(axis=1).astype(np.float32)
    if reducer == "max":
        return values.max(axis=1).astype(np.float32)
    raise ValueError(f"unsupported reducer: {reducer}")


def build_topology_folds(export: dict, records, raw_by_video: dict, feature_names: list[str], evidence_names: list[str], reducer: str) -> list[dict]:
    folds = []
    all_indices = set(range(len(records)))
    for fold_pos, fold in enumerate(export.get("folds", [])):
        val_idx = [int(idx) for idx in fold.get("val_idx", [])]
        train_idx = [int(idx) for idx in fold.get("train_idx", [])] or sorted(all_indices - set(val_idx))
        labels = fold.get("labels") or [records[idx].labels for idx in val_idx]
        candidates = []
        scores = []
        evidence = []
        for idx in val_idx:
            video_candidates, video_scores = raw_by_video[int(idx)]
            candidates.append([(int(s), int(e)) for s, e in video_candidates])
            scores.append(np.asarray(video_scores, dtype=np.float32))
            evidence.append(event_topology_evidence(records[int(idx)], feature_names, evidence_names, reducer))
        folds.append(
            {
                "fold": int(fold.get("fold", fold_pos)),
                "train_idx": train_idx,
                "val_idx": val_idx,
                "base": fold.get("predictions", []),
                "candidates": candidates,
                "scores": scores,
                "evidence": evidence,
                "labels": [np.asarray(item, dtype=np.int64) for item in labels],
            }
        )
    return folds


def flatten(folds: list[dict], key: str) -> list:
    values = []
    for fold in folds:
        values.extend(fold[key])
    return values


def apply_config(base, candidates, scores, evidence, config):
    if not config.get("enabled"):
        return [[(int(s), int(e)) for s, e in video] for video in base]
    return split_topology_set_predictions(
        base,
        candidates,
        scores,
        evidence,
        evidence_threshold=float(config["evidence_threshold"]),
        min_selector_score=float(config["min_selector_score"]),
        max_selector_rank=config.get("max_selector_rank"),
        min_evidence_mean=float(config["min_evidence_mean"]),
        min_active_fraction=float(config["min_active_fraction"]),
        min_parent_contrast=float(config["min_parent_contrast"]),
        parent_min_length=int(config["parent_min_length"]),
        max_child_parent_ratio=float(config["max_child_parent_ratio"]),
        min_child_parent_coverage=float(config["min_child_parent_coverage"]),
        min_gap_between_children=int(config["min_gap_between_children"]),
        child_score_threshold=float(config["child_score_threshold"]),
        set_score_threshold=float(config["set_score_threshold"]),
        selector_weight=float(config["selector_weight"]),
        evidence_weight=float(config["evidence_weight"]),
        active_fraction_weight=float(config["active_fraction_weight"]),
        contrast_weight=float(config["contrast_weight"]),
        length_penalty=float(config["length_penalty"]),
        nms_iou=config.get("nms_iou"),
        min_children_per_parent=int(config["min_children_per_parent"]),
        max_children_per_parent=int(config["max_children_per_parent"]),
        max_replaced_parents_per_video=int(config["max_replaced_parents_per_video"]),
    )


def run_strict_topology_set_oof(folds: list[dict], args) -> dict:
    results = []
    for heldout_pos, heldout in enumerate(folds):
        calibration = [fold for pos, fold in enumerate(folds) if pos != heldout_pos]
        config, calibration_metrics, _ = select_topology_set_split_params(
            flatten(calibration, "base"),
            flatten(calibration, "candidates"),
            flatten(calibration, "scores"),
            flatten(calibration, "evidence"),
            flatten(calibration, "labels"),
            evidence_thresholds=_parse_float_list(args.evidence_thresholds),
            min_selector_scores=_parse_float_list(args.min_selector_scores),
            max_selector_ranks=parse_optional_int_list(args.max_selector_ranks),
            min_evidence_means=_parse_float_list(args.min_evidence_means),
            min_active_fractions=_parse_float_list(args.min_active_fractions),
            min_parent_contrasts=_parse_float_list(args.min_parent_contrasts),
            parent_min_lengths=_parse_int_list(args.parent_min_lengths),
            max_child_parent_ratios=_parse_float_list(args.max_child_parent_ratios),
            min_child_parent_coverages=_parse_float_list(args.min_child_parent_coverages),
            min_gap_between_children_values=_parse_int_list(args.min_gaps),
            child_score_thresholds=_parse_float_list(args.child_score_thresholds),
            set_score_thresholds=_parse_float_list(args.set_score_thresholds),
            selector_weights=_parse_float_list(args.selector_weights),
            evidence_weights=_parse_float_list(args.evidence_weights),
            active_fraction_weights=_parse_float_list(args.active_fraction_weights),
            contrast_weights=_parse_float_list(args.contrast_weights),
            length_penalties=_parse_float_list(args.length_penalties),
            nms_ious=parse_nms_list(args.nms_ious),
            min_children_per_parents=_parse_int_list(args.min_children_per_parent),
            max_children_per_parents=_parse_int_list(args.max_children_per_parent),
            max_replaced_parents_per_videos=_parse_int_list(args.max_replaced_parents_per_video),
            max_fp_increase=args.max_fp_increase,
            iou_threshold=float(args.iou_threshold),
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


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


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
    parser.add_argument("--evidence-reducer", choices=["stack", "mean", "max"], default="stack")
    parser.add_argument("--evidence-thresholds", default="0.35,0.4,0.45,0.5,0.55")
    parser.add_argument("--min-selector-scores", default="0,0.05,0.1,0.2")
    parser.add_argument("--max-selector-ranks", default="none,60")
    parser.add_argument("--min-evidence-means", default="0.15,0.2,0.25,0.3")
    parser.add_argument("--min-active-fractions", default="0,0.1,0.25")
    parser.add_argument("--min-parent-contrasts", default="-0.3,-0.1,0")
    parser.add_argument("--parent-min-lengths", default="12,24,48")
    parser.add_argument("--max-child-parent-ratios", default="0.5,0.7")
    parser.add_argument("--min-child-parent-coverages", default="0.5,0.7,0.8")
    parser.add_argument("--min-gaps", default="0,2")
    parser.add_argument("--child-score-thresholds", default="0,0.05,0.1")
    parser.add_argument("--set-score-thresholds", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--selector-weights", default="0.5,1")
    parser.add_argument("--evidence-weights", default="0.5,1,1.5")
    parser.add_argument("--active-fraction-weights", default="0,0.25")
    parser.add_argument("--contrast-weights", default="0,0.5")
    parser.add_argument("--length-penalties", default="0,0.005")
    parser.add_argument("--nms-ious", default="0.3")
    parser.add_argument("--min-children-per-parent", default="2")
    parser.add_argument("--max-children-per-parent", default="2,3")
    parser.add_argument("--max-replaced-parents-per-video", default="1,2")
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
    result = run_strict_topology_set_oof(folds, args)
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
    out.write_text(json.dumps(result, indent=2, default=_jsonable), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
