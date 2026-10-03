#!/usr/bin/env python3
"""Strict OOF JEPA topology matching from exported base predictions.

This runner evaluates structured parent-to-child-set replacement without using
the held-out fold to choose topology parameters. It reuses fold-local JEPA raw
candidates and applies the learned topology matching network only after the
configuration has been selected on the other folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_prediction_set_switcher import _as_segments
from jepa_topology_matching_network import (
    apply_topology_matching_predictions,
    build_topology_inference_records,
    build_topology_matching_records,
    fit_topology_matching_model,
    fit_topology_matching_risk_models,
    score_topology_matching_risk_records,
    select_topology_matching_params,
)
from run_strict_dp_event_set_oof import build_fold_local_raw_candidates
from run_strict_topology_set_split_oof import event_topology_evidence
from selector_fusion import evaluate_fused_predictions
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES
from train_segment_locator import (
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    load_signal_dataset,
)


def load_prediction_export(source: str | Path | dict[str, Any], n_records: int | None = None) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        data = dict(source)
    all_indices = set(range(int(n_records))) if n_records is not None else None
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        fold["val_idx"] = [int(idx) for idx in fold.get("val_idx", [])]
        fold["train_idx"] = [int(idx) for idx in fold.get("train_idx", [])]
        if all_indices is not None and fold["val_idx"] and not fold["train_idx"]:
            fold["train_idx"] = sorted(all_indices - set(fold["val_idx"]))
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def parse_optional_int_list(text: str) -> list[int | None]:
    values: list[int | None] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else int(item))
    return values


def parse_optional_float_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else float(item))
    return values


def build_evidence_by_video(records, feature_names: list[str], evidence_names: list[str], reducer: str) -> dict[int, np.ndarray]:
    return {
        int(idx): event_topology_evidence(record, feature_names, evidence_names, reducer)
        for idx, record in enumerate(records)
    }


def build_matching_folds(
    export: dict[str, Any],
    records,
    raw_by_video: dict[int, tuple[list[tuple[int, int]], np.ndarray]],
    evidence_by_video: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
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
            candidates.append([(int(start), int(end)) for start, end in video_candidates])
            scores.append(np.asarray(video_scores, dtype=np.float32))
            evidence.append(np.asarray(evidence_by_video[int(idx)], dtype=np.float32))
        base_predictions = [_as_segments(video) for video in fold.get("predictions", [])]
        folds.append(
            {
                "fold": int(fold.get("fold", fold_pos)),
                "train_idx": train_idx,
                "val_idx": val_idx,
                "base": base_predictions,
                "candidates": candidates,
                "scores": scores,
                "evidence": evidence,
                "labels": [np.asarray(item, dtype=np.int64) for item in labels],
            }
        )
    return folds


def flatten(folds: list[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for fold in folds:
        values.extend(fold[key])
    return values


def _score_records(model, records):
    if not records:
        return np.zeros(0, dtype=np.float32)
    features = np.stack([record.features for record in records])
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)


def apply_matching_config(base, candidates, scores, evidence, train_labels, config: dict[str, Any], args) -> list[list[tuple[int, int]]]:
    if not config.get("enabled"):
        return [[(int(start), int(end)) for start, end in video] for video in base]

    train_records = build_topology_matching_records(
        flatten(train_labels, "base"),
        flatten(train_labels, "candidates"),
        flatten(train_labels, "scores"),
        flatten(train_labels, "evidence"),
        flatten(train_labels, "labels"),
        iou_threshold=float(args.iou_threshold),
        parent_min_length=int(config["parent_min_length"]),
        min_child_parent_coverage=float(config["min_child_parent_coverage"]),
        max_child_parent_ratio=float(config["max_child_parent_ratio"]),
        max_children_per_parent=int(config["max_children_per_parent"]),
        max_set_proposals_per_parent=int(config["max_set_proposals_per_parent"]),
        max_child_pool_per_parent=int(config["max_child_pool_per_parent"]),
        include_teacher=True,
    )
    inference_records = build_topology_inference_records(
        base,
        candidates,
        scores,
        evidence,
        parent_min_length=int(config["parent_min_length"]),
        min_child_parent_coverage=float(config["min_child_parent_coverage"]),
        max_child_parent_ratio=float(config["max_child_parent_ratio"]),
        max_children_per_parent=int(config["max_children_per_parent"]),
        max_set_proposals_per_parent=int(config["max_set_proposals_per_parent"]),
        max_child_pool_per_parent=int(config["max_child_pool_per_parent"]),
    )
    if not train_records or not inference_records:
        return [[(int(start), int(end)) for start, end in video] for video in base]

    if bool(config.get("risk_aware")):
        gain_model, safety_model = fit_topology_matching_risk_models(
            train_records,
            model_name=str(config["model"]),
            seed=int(args.seed) + 911,
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
        )
        combined, _, safety = score_topology_matching_risk_records(
            gain_model,
            safety_model,
            inference_records,
            risk_penalty=float(config.get("risk_penalty") or 0.5),
            safety_weight=float(config.get("safety_weight") or 0.25),
        )
        proposal_scores = combined
        safety_scores = safety
    else:
        model = fit_topology_matching_model(
            train_records,
            model_name=str(config["model"]),
            seed=int(args.seed) + 911,
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
        )
        proposal_scores = _score_records(model, inference_records)
        safety_scores = None

    return apply_topology_matching_predictions(
        base,
        inference_records,
        proposal_scores,
        threshold=float(config["threshold"]),
        max_replaced_parents_per_video=int(config["max_replaced_parents_per_video"]),
        safety_scores=safety_scores,
        safety_threshold=config.get("safety_threshold"),
        min_evidence_island_coverage=config.get("min_evidence_island_coverage"),
        max_residual_active_fraction=config.get("max_residual_active_fraction"),
        min_topology_count_alignment=config.get("min_topology_count_alignment"),
        min_child_evidence_mass_fraction=config.get("min_child_evidence_mass_fraction"),
    )


def run_strict_topology_matching_oof(folds: list[dict[str, Any]], args) -> dict[str, Any]:
    results = []
    for heldout_pos, heldout in enumerate(folds):
        calibration = [fold for pos, fold in enumerate(folds) if pos != heldout_pos]
        config, calibration_metrics, _ = select_topology_matching_params(
            train_base_predictions=flatten(calibration, "base"),
            train_candidate_predictions=flatten(calibration, "candidates"),
            train_candidate_scores=flatten(calibration, "scores"),
            train_evidence_arrays=flatten(calibration, "evidence"),
            train_labels=flatten(calibration, "labels"),
            val_base_predictions=flatten(calibration, "base"),
            val_candidate_predictions=flatten(calibration, "candidates"),
            val_candidate_scores=flatten(calibration, "scores"),
            val_evidence_arrays=flatten(calibration, "evidence"),
            val_labels=flatten(calibration, "labels"),
            model_name=str(args.matching_model),
            seed=int(args.seed) + int(heldout_pos),
            thresholds=_parse_float_list(args.matching_thresholds),
            parent_min_lengths=_parse_int_list(args.parent_min_lengths),
            min_child_parent_coverages=_parse_float_list(args.min_child_parent_coverages),
            max_child_parent_ratios=_parse_float_list(args.max_child_parent_ratios),
            max_children_per_parents=_parse_int_list(args.max_children_per_parent),
            max_set_proposals_per_parents=_parse_int_list(args.max_set_proposals_per_parent),
            max_child_pool_per_parents=_parse_int_list(args.max_child_pool_per_parent),
            max_replaced_parents_per_videos=_parse_int_list(args.max_replaced_parents_per_video),
            max_fp_increase=args.max_fp_increase,
            risk_aware=bool(args.risk_aware),
            risk_penalties=_parse_float_list(args.risk_penalties),
            safety_thresholds=_parse_float_list(args.safety_thresholds),
            safety_weight=float(args.safety_weight),
            min_evidence_island_coverages=parse_optional_float_list(args.min_evidence_island_coverages),
            max_residual_active_fractions=parse_optional_float_list(args.max_residual_active_fractions),
            min_topology_count_alignments=parse_optional_float_list(args.min_topology_count_alignments),
            min_child_evidence_mass_fractions=parse_optional_float_list(args.min_child_evidence_mass_fractions),
            iou_threshold=float(args.iou_threshold),
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
        )
        predictions = apply_matching_config(
            heldout["base"],
            heldout["candidates"],
            heldout["scores"],
            heldout["evidence"],
            calibration,
            config,
            args,
        )
        metrics = evaluate_fused_predictions(predictions, heldout["labels"], iou_threshold=float(args.iou_threshold))
        results.append(
            {
                "fold": int(heldout["fold"]),
                "config": config,
                "calibration": calibration_metrics,
                "validation": metrics,
                "predictions": [[[int(start), int(end)] for start, end in video] for video in predictions],
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
    parser.add_argument("--matching-model", choices=["prototype", "gbdt", "rf", "extratrees", "logreg", "mlp"], default="prototype")
    parser.add_argument("--matching-thresholds", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--parent-min-lengths", default="12,24,48")
    parser.add_argument("--min-child-parent-coverages", default="0.5,0.7")
    parser.add_argument("--max-child-parent-ratios", default="0.5,0.7")
    parser.add_argument("--max-children-per-parent", default="2,3")
    parser.add_argument("--max-set-proposals-per-parent", default="4,8")
    parser.add_argument("--max-child-pool-per-parent", default="8,16")
    parser.add_argument("--max-replaced-parents-per-video", default="1,2")
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--risk-aware", action="store_true")
    parser.add_argument("--risk-penalties", default="0.25,0.5,1.0")
    parser.add_argument("--safety-thresholds", default="0,0.25,0.5")
    parser.add_argument("--safety-weight", type=float, default=0.25)
    parser.add_argument("--min-evidence-island-coverages", default="none,0.6,0.75,0.9")
    parser.add_argument("--max-residual-active-fractions", default="none,0.15,0.3,0.5")
    parser.add_argument("--min-topology-count-alignments", default="none,0.5,0.8,1.0")
    parser.add_argument("--min-child-evidence-mass-fractions", default="none,0.5,0.7,0.9")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    if not channel_names:
        raise ValueError("no selector channels found in event dataset summary")
    evidence_names = _parse_name_list(args.evidence_names)
    export = load_prediction_export(args.base_predictions, n_records=len(records))
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
    evidence_by_video = build_evidence_by_video(records, feature_names, evidence_names, args.evidence_reducer)
    folds = build_matching_folds(export, records, raw_by_video, evidence_by_video)
    result = run_strict_topology_matching_oof(folds, args)
    result.update(
        {
            "data_dir": args.data_dir,
            "base_predictions": args.base_predictions,
            "selector_diagnostics": selector_diagnostics,
            "selector_channel_names": channel_names,
            "evidence_names": evidence_names,
            "evidence_reducer": args.evidence_reducer,
            "risk_aware": bool(args.risk_aware),
            "min_evidence_island_coverages": parse_optional_float_list(args.min_evidence_island_coverages),
            "max_residual_active_fractions": parse_optional_float_list(args.max_residual_active_fractions),
            "min_topology_count_alignments": parse_optional_float_list(args.min_topology_count_alignments),
            "min_child_evidence_mass_fractions": parse_optional_float_list(args.min_child_evidence_mass_fractions),
            "iou_threshold": float(args.iou_threshold),
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
