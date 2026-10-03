#!/usr/bin/env python3
"""Cross-validate a multi-seed temporal CNN ensemble on JEPA event tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from cross_validate_selector_fusion import predict_selector_fold, train_selector_for_fold
from segment_ensemble import average_prediction_records, records_to_segments
from selector_fusion import evaluate_fused_predictions, fuse_protected_segment_predictions, select_protected_fusion_params
from train_proposal_calibrator import evaluate_segment_predictions
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES
from train_segment_locator import (
    _feature_indices,
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    blend_prediction_records,
    load_signal_dataset,
    make_stratified_folds,
    predict_records,
    select_blended_postprocess_params,
    select_postprocess_params,
    train_model,
)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ensemble-seeds", default="0,17,37")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lambda-dice", type=float, default=0.0)
    parser.add_argument("--lambda-boundary", type=float, default=0.0)
    parser.add_argument("--architecture", choices=["tcn", "attn_tcn"], default="tcn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-aux-name", default="true_vjepa_raw_rank")
    parser.add_argument("--blend-alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--protected-selector", action="store_true")
    parser.add_argument("--selector-model", choices=["gbdt", "rf", "extratrees", "logreg"], default="extratrees")
    parser.add_argument("--selector-target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--selector-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selector-min-gaps", default="0,1,2,4,8")
    parser.add_argument("--selector-min-lengths", default="1,2,4,8")
    parser.add_argument("--selector-nms-candidates", default="none,0.1,0.3,0.5")
    parser.add_argument("--protected-mainline-iou-candidates", default="0,0.1,0.25,0.5")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    folds = make_stratified_folds(records, args.folds, args.seed)
    all_indices = set(range(len(records)))
    feature_names = _load_feature_names(args.data_dir)
    ensemble_offsets = parse_int_list(args.ensemble_seeds)
    if not ensemble_offsets:
        raise ValueError("ensemble-seeds must contain at least one integer")

    blend_aux_channel = None
    if args.blend_aux_name:
        blend_aux_channel = _feature_indices(feature_names, [args.blend_aux_name], "blend")[0]

    selector_channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    selector_thresholds = _parse_float_list(args.selector_thresholds)
    selector_min_gaps = _parse_int_list(args.selector_min_gaps)
    selector_min_lengths = _parse_int_list(args.selector_min_lengths)
    protected_mainline_iou_candidates = _parse_float_list(args.protected_mainline_iou_candidates)
    selector_nms_candidates = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.selector_nms_candidates.split(",")
        if item.strip()
    ]

    fold_summaries: list[dict] = []
    for fold_idx, val_idx in enumerate(folds):
        train_idx = sorted(all_indices - set(val_idx))
        print(
            f"fold={fold_idx} train={len(train_idx)} val={len(val_idx)} "
            f"ensemble={len(ensemble_offsets)} device={args.device}",
            flush=True,
        )
        member_outputs = []
        member_metrics = []
        for member_idx, seed_offset in enumerate(ensemble_offsets):
            member_seed = args.seed + fold_idx * 1000 + int(seed_offset)
            print(f"fold={fold_idx} member={member_idx} seed={member_seed}", flush=True)
            model, metrics, mean, std = train_model(
                records,
                train_idx=train_idx,
                val_idx=val_idx,
                epochs=args.epochs,
                lr=args.lr,
                hidden=args.hidden,
                dropout=args.dropout,
                patience=args.patience,
                device=args.device,
                seed=member_seed,
                lambda_dice=args.lambda_dice,
                lambda_boundary=args.lambda_boundary,
                architecture=args.architecture,
            )
            member_metrics.append(metrics)
            member_outputs.append(predict_records(model, records, val_idx, mean, std, args.device))

        averaged_outputs = average_prediction_records(member_outputs)
        ensemble_outputs = averaged_outputs
        if blend_aux_channel is not None:
            alpha, params, ensemble_metrics = select_blended_postprocess_params(
                averaged_outputs,
                records,
                val_idx,
                aux_channel=blend_aux_channel,
                alphas=_parse_float_list(args.blend_alphas),
                iou_threshold=args.iou_threshold,
            )
            ensemble_outputs = blend_prediction_records(
                averaged_outputs,
                records,
                val_idx,
                aux_channel=blend_aux_channel,
                alpha_model=alpha,
            )
            ensemble_validation = {
                "params": params,
                **ensemble_metrics,
                "hybrid": {
                    "aux_name": args.blend_aux_name,
                    "aux_channel": blend_aux_channel,
                    "alpha_model": alpha,
                },
            }
        else:
            params, ensemble_metrics = select_postprocess_params(averaged_outputs, iou_threshold=args.iou_threshold)
            ensemble_validation = {"params": params, **ensemble_metrics}

        labels = [records[idx].labels for idx in val_idx]
        mainline_predictions = records_to_segments(ensemble_outputs, ensemble_validation["params"])
        validation = evaluate_fused_predictions(mainline_predictions, labels, iou_threshold=args.iou_threshold)
        validation = {
            "params": ensemble_validation["params"],
            **validation,
            **({"hybrid": ensemble_validation["hybrid"]} if "hybrid" in ensemble_validation else {}),
        }
        selector_summary = {"enabled": False}
        if args.protected_selector:
            classifier, selector_params, selector_val_metrics, n_candidates, n_positive = train_selector_for_fold(
                records,
                train_idx,
                val_idx,
                feature_names,
                selector_channel_names,
                selector_thresholds,
                selector_min_gaps,
                selector_min_lengths,
                iou_threshold=args.iou_threshold,
                seed=args.seed + 5000 + fold_idx,
                model_name=args.selector_model,
                target=args.selector_target,
            )
            selector_predictions = predict_selector_fold(
                classifier,
                records,
                val_idx,
                feature_names,
                selector_channel_names,
                selector_thresholds,
                selector_min_gaps,
                selector_min_lengths,
                selector_params,
            )
            protected_config, protected_metrics = select_protected_fusion_params(
                mainline_predictions,
                selector_predictions,
                labels,
                max_mainline_iou_candidates=protected_mainline_iou_candidates,
                selector_nms_iou_candidates=selector_nms_candidates,
                iou_threshold=args.iou_threshold,
            )
            if protected_config.get("enabled"):
                fused_predictions = fuse_protected_segment_predictions(
                    mainline_predictions,
                    selector_predictions,
                    max_mainline_iou=protected_config["max_mainline_iou"],
                    selector_nms_iou=protected_config["selector_nms_iou"],
                )
                validation = {
                    "params": ensemble_validation["params"],
                    **evaluate_fused_predictions(fused_predictions, labels, iou_threshold=args.iou_threshold),
                    **({"hybrid": ensemble_validation["hybrid"]} if "hybrid" in ensemble_validation else {}),
                }
            selector_summary = {
                "enabled": True,
                "params": selector_params,
                "validation": selector_val_metrics,
                "metrics": evaluate_segment_predictions(selector_predictions, labels, iou_threshold=args.iou_threshold),
                "n_train_candidates": n_candidates,
                "n_positive_train_candidates": n_positive,
                "protected": protected_config,
                "protected_metrics": protected_metrics,
            }

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_names": [records[idx].name for idx in val_idx],
                "member_metrics": member_metrics,
                "ensemble": ensemble_validation,
                "selector": selector_summary,
                "validation": validation,
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
            "ensemble_seeds": ensemble_offsets,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "lambda_dice": args.lambda_dice,
            "lambda_boundary": args.lambda_boundary,
            "architecture": args.architecture,
            "seed": args.seed,
            "blend_aux_name": args.blend_aux_name,
            "protected_selector": args.protected_selector,
            "selector_model": args.selector_model,
            "selector_target": args.selector_target,
            "task": "binary_error_segment_localization_temporal_cnn_ensemble",
        },
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
