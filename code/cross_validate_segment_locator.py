#!/usr/bin/env python3
"""Video-level cross-validation for binary JEPA error-segment localization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_segment_locator import (
    _feature_indices,
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    apply_valley_split_to_records,
    blend_prediction_records,
    load_signal_dataset,
    make_stratified_folds,
    predict_records,
    refine_prediction_records,
    select_blended_postprocess_params,
    select_event_refined_postprocess_params,
    select_recall_repair_params,
    select_valley_split_params,
    train_model,
)


def aggregate_fold_metrics(folds: list[dict]) -> dict:
    aggregate: dict[str, dict] = {}
    for section in ["frame", "segment"]:
        values = {key: [] for key in ["precision", "recall", "f1"]}
        counts = {key: 0 for key in ["tp", "fp", "fn"]}
        for fold in folds:
            metrics = fold["validation"][section]
            for key in values:
                values[key].append(float(metrics[key]))
            for key in counts:
                counts[key] += int(metrics[key])
        aggregate[section] = {
            **{f"{key}_mean": float(np.mean(items)) for key, items in values.items()},
            **{f"{key}_std": float(np.std(items)) for key, items in values.items()},
            **counts,
        }
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lambda-dice", type=float, default=0.0)
    parser.add_argument("--architecture", choices=["tcn", "attn_tcn", "bilstm"], default="tcn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-aux-name", default="")
    parser.add_argument("--blend-alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--event-refine", action="store_true")
    parser.add_argument("--recall-repair", action="store_true")
    parser.add_argument(
        "--recall-repair-aux-names",
        default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank",
    )
    parser.add_argument("--recall-repair-thresholds", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--recall-repair-smooth-windows", default="1,3")
    parser.add_argument("--recall-repair-min-gaps", default="0,1")
    parser.add_argument("--recall-repair-min-lengths", default="1,2,4")
    parser.add_argument("--recall-repair-pads", default="0,1,2")
    parser.add_argument("--recall-repair-min-contrasts", default="0,0.1,0.2")
    parser.add_argument("--recall-repair-coverages", default="0.25,0.5")
    parser.add_argument("--recall-repair-max-per-video", default="1,2,3")
    parser.add_argument("--valley-split", action="store_true")
    parser.add_argument(
        "--valley-split-aux-names",
        default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank",
    )
    parser.add_argument("--valley-split-thresholds", default="0.45,0.55,0.65")
    parser.add_argument("--valley-split-smooth-windows", default="1,3")
    parser.add_argument("--valley-split-island-min-gaps", default="0,1")
    parser.add_argument("--valley-split-island-min-lengths", default="1,2,4")
    parser.add_argument("--valley-split-pads", default="0,1")
    parser.add_argument("--valley-split-parent-min-lengths", default="24,48")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    folds = make_stratified_folds(records, args.folds, args.seed)
    feature_names = _load_feature_names(args.data_dir)
    blend_aux_channel = None
    if args.blend_aux_name:
        if args.blend_aux_name not in feature_names:
            raise ValueError(f"blend feature {args.blend_aux_name!r} not found in {Path(args.data_dir) / 'summary.json'}")
        blend_aux_channel = feature_names.index(args.blend_aux_name)
    repair_names = _parse_name_list(args.recall_repair_aux_names)
    repair_channels = _feature_indices(feature_names, repair_names, "recall-repair") if args.recall_repair else []
    split_names = _parse_name_list(args.valley_split_aux_names)
    split_channels = _feature_indices(feature_names, split_names, "valley-split") if args.valley_split else []

    fold_summaries: list[dict] = []
    all_indices = set(range(len(records)))
    for fold_idx, val_idx in enumerate(folds):
        train_idx = sorted(all_indices - set(val_idx))
        print(f"fold={fold_idx} train={len(train_idx)} val={len(val_idx)}", flush=True)
        model, pure_metrics, mean, std = train_model(
            records,
            train_idx=train_idx,
            val_idx=val_idx,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            dropout=args.dropout,
            patience=args.patience,
            device=args.device,
            seed=args.seed + fold_idx,
            lambda_dice=args.lambda_dice,
            architecture=args.architecture,
        )
        metrics = pure_metrics
        if blend_aux_channel is not None:
            val_outputs = predict_records(model, records, val_idx, mean, std, args.device)
            alpha, params, blended_metrics = select_blended_postprocess_params(
                val_outputs,
                records,
                val_idx,
                aux_channel=blend_aux_channel,
                alphas=_parse_float_list(args.blend_alphas),
            )
            metrics = {
                "epoch": pure_metrics.get("epoch"),
                "loss": pure_metrics.get("loss"),
                "params": params,
                **blended_metrics,
                "hybrid": {
                    "aux_name": args.blend_aux_name,
                    "aux_channel": blend_aux_channel,
                    "alpha_model": alpha,
                    "pure_validation": pure_metrics,
                },
            }
        if args.event_refine:
            val_outputs = predict_records(model, records, val_idx, mean, std, args.device)
            if blend_aux_channel is not None:
                alpha_model = float(metrics.get("hybrid", {}).get("alpha_model", 1.0))
                val_outputs = blend_prediction_records(
                    val_outputs,
                    records,
                    val_idx,
                    aux_channel=blend_aux_channel,
                    alpha_model=alpha_model,
                )
            refine_config, refine_params, refine_metrics = select_event_refined_postprocess_params(val_outputs)
            metrics = {
                "epoch": metrics.get("epoch"),
                "loss": metrics.get("loss"),
                "params": refine_params,
                **refine_metrics,
                "event_refine": {
                    **refine_config,
                    "pre_refine_validation": metrics,
                },
            }
        if args.valley_split:
            val_outputs = predict_records(model, records, val_idx, mean, std, args.device)
            if blend_aux_channel is not None:
                alpha_model = float(metrics.get("hybrid", {}).get("alpha_model", 1.0))
                val_outputs = blend_prediction_records(
                    val_outputs,
                    records,
                    val_idx,
                    aux_channel=blend_aux_channel,
                    alpha_model=alpha_model,
                )
            if args.event_refine:
                refine_config = metrics.get("event_refine", {})
                val_outputs = refine_prediction_records(
                    val_outputs,
                    raw_weight=float(refine_config.get("raw_weight", 1.0)),
                    sigmas=refine_config.get("sigmas", []),
                )
            split_config, _, split_metrics = select_valley_split_params(
                val_outputs,
                records,
                val_idx,
                base_params=metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
                aux_channels=split_channels,
                evidence_thresholds=_parse_float_list(args.valley_split_thresholds),
                evidence_smooth_windows=_parse_int_list(args.valley_split_smooth_windows),
                island_min_gaps=_parse_int_list(args.valley_split_island_min_gaps),
                island_min_lengths=_parse_int_list(args.valley_split_island_min_lengths),
                pads=_parse_int_list(args.valley_split_pads),
                parent_min_lengths=_parse_int_list(args.valley_split_parent_min_lengths),
            )
            metrics = {
                "epoch": metrics.get("epoch"),
                "loss": metrics.get("loss"),
                "params": split_config.get("params", metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1})),
                **split_metrics,
                "valley_split": {
                    **split_config,
                    "aux_names": split_names,
                    "pre_split_validation": metrics,
                },
            }
        if args.recall_repair:
            val_outputs = predict_records(model, records, val_idx, mean, std, args.device)
            if blend_aux_channel is not None:
                alpha_model = float(metrics.get("hybrid", {}).get("alpha_model", 1.0))
                val_outputs = blend_prediction_records(
                    val_outputs,
                    records,
                    val_idx,
                    aux_channel=blend_aux_channel,
                    alpha_model=alpha_model,
                )
            if args.event_refine:
                refine_config = metrics.get("event_refine", {})
                val_outputs = refine_prediction_records(
                    val_outputs,
                    raw_weight=float(refine_config.get("raw_weight", 1.0)),
                    sigmas=refine_config.get("sigmas", []),
                )
            if args.valley_split:
                split_config = metrics.get("valley_split", {})
                if split_config.get("enabled"):
                    val_outputs = apply_valley_split_to_records(
                        val_outputs,
                        records,
                        val_idx,
                        base_params=split_config.get("base_params", metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1})),
                        aux_channels=split_channels,
                        evidence_threshold=float(split_config.get("evidence_threshold", 0.7)),
                        evidence_smooth_window=int(split_config.get("evidence_smooth_window", 1)),
                        island_min_gap=int(split_config.get("island_min_gap", 0)),
                        island_min_length=int(split_config.get("island_min_length", 1)),
                        pad=int(split_config.get("pad", 0)),
                        parent_min_length=int(split_config.get("parent_min_length", 24)),
                    )
            repair_config, _, repair_metrics = select_recall_repair_params(
                val_outputs,
                records,
                val_idx,
                base_params=metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
                aux_channels=repair_channels,
                aux_thresholds=_parse_float_list(args.recall_repair_thresholds),
                aux_smooth_windows=_parse_int_list(args.recall_repair_smooth_windows),
                aux_min_gaps=_parse_int_list(args.recall_repair_min_gaps),
                aux_min_lengths=_parse_int_list(args.recall_repair_min_lengths),
                pads=_parse_int_list(args.recall_repair_pads),
                min_contrasts=_parse_float_list(args.recall_repair_min_contrasts),
                max_existing_coverages=_parse_float_list(args.recall_repair_coverages),
                max_rescues_per_videos=_parse_int_list(args.recall_repair_max_per_video),
            )
            metrics = {
                "epoch": metrics.get("epoch"),
                "loss": metrics.get("loss"),
                "params": metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
                **repair_metrics,
                "recall_repair": {
                    **repair_config,
                    "aux_names": repair_names,
                    "pre_repair_validation": metrics,
                },
            }
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_names": [records[idx].name for idx in val_idx],
                "validation": metrics,
            }
        )

    summary = {
        "data_dir": args.data_dir,
        "n_videos": len(records),
        "frames": int(sum(len(record.labels) for record in records)),
        "positive_frames": int(sum(record.labels.sum() for record in records)),
        "folds": fold_summaries,
        "aggregate": aggregate_fold_metrics(fold_summaries),
        "config": {
            "folds": args.folds,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "lambda_dice": args.lambda_dice,
            "architecture": args.architecture,
            "seed": args.seed,
            "blend_aux_name": args.blend_aux_name,
            "event_refine": args.event_refine,
            "recall_repair": args.recall_repair,
            "recall_repair_aux_names": repair_names,
            "valley_split": args.valley_split,
            "valley_split_aux_names": split_names,
            "task": "binary_error_segment_localization",
        },
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
