#!/usr/bin/env python3
"""Cross-validate protected fusion of h32 JEPA hybrid and JEPA-TriDet-lite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from selector_fusion import (
    evaluate_fused_predictions,
    fuse_protected_segment_predictions,
    select_protected_fusion_params,
)
from train_jepa_tridet_lite import (
    _parse_float_list,
    _parse_int_tuple,
    _parse_optional_float_list,
    _parse_optional_int_list,
    predict_indices as predict_tridet_indices,
    train_fold as train_tridet_fold,
)
from train_segment_locator import (
    _load_feature_names,
    _segments_from_params,
    blend_prediction_records,
    load_signal_dataset,
    make_stratified_folds,
    predict_records as predict_mainline_records,
    select_blended_postprocess_params,
    train_model as train_mainline_model,
)


Segment = tuple[int, int]


def probability_records_to_segments(records: list[dict], params: dict) -> list[list[Segment]]:
    predictions: list[list[Segment]] = []
    for record in records:
        probs = np.asarray(record["probs"], dtype=np.float32).reshape(-1)
        predictions.append([tuple(item) for item in _segments_from_params(probs, params)])
    return predictions


def apply_protected_config(
    base_predictions: list[list[tuple[int, int]]],
    tridet_predictions: list[list[tuple[int, int]]],
    config: dict,
) -> list[list[Segment]]:
    base = [[tuple(item) for item in video] for video in base_predictions]
    if not config.get("enabled", False):
        return [sorted(set(video)) for video in base]
    return fuse_protected_segment_predictions(
        base,
        tridet_predictions,
        max_mainline_iou=config.get("max_mainline_iou"),
        selector_nms_iou=config.get("selector_nms_iou"),
    )


def _mainline_hybrid_predictions(
    model,
    records,
    val_idx: list[int],
    mean,
    std,
    device: str,
    aux_channel: int | None,
    blend_alphas: list[float],
    iou_threshold: float,
) -> tuple[list[list[Segment]], list[np.ndarray], list[str], dict]:
    raw_outputs = predict_mainline_records(model, records, val_idx, mean, std, device)
    labels = [np.asarray(output["labels"], dtype=np.int64) for output in raw_outputs]
    names = [records[idx].name for idx in val_idx]
    if aux_channel is None:
        from train_segment_locator import select_postprocess_params

        params, metrics = select_postprocess_params(raw_outputs, iou_threshold=float(iou_threshold))
        return probability_records_to_segments(raw_outputs, params), labels, names, {"params": params, **metrics}

    alpha, params, metrics = select_blended_postprocess_params(
        raw_outputs,
        records,
        val_idx,
        aux_channel=int(aux_channel),
        alphas=blend_alphas,
        iou_threshold=float(iou_threshold),
    )
    blended = blend_prediction_records(raw_outputs, records, val_idx, aux_channel=int(aux_channel), alpha_model=float(alpha))
    predictions = probability_records_to_segments(blended, params)
    return predictions, labels, names, {
        "params": params,
        **metrics,
        "hybrid": {
            "aux_channel": int(aux_channel),
            "alpha_model": float(alpha),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--predictions-out", help="write an evaluator-compatible OOF prediction bundle")
    parser.add_argument("--predictions-kind", choices=["tridet", "fused", "mainline"], default="fused")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.3)

    parser.add_argument("--mainline-epochs", type=int, default=80)
    parser.add_argument("--mainline-lr", type=float, default=1e-3)
    parser.add_argument("--mainline-hidden", type=int, default=32)
    parser.add_argument("--mainline-dropout", type=float, default=0.1)
    parser.add_argument("--mainline-patience", type=int, default=15)
    parser.add_argument("--mainline-lambda-dice", type=float, default=0.0)
    parser.add_argument("--mainline-architecture", choices=["tcn", "attn_tcn"], default="tcn")
    parser.add_argument("--blend-aux-name", default="true_vjepa_raw_rank")
    parser.add_argument("--blend-alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

    parser.add_argument("--tridet-epochs", type=int, default=50)
    parser.add_argument("--tridet-lr", type=float, default=5e-4)
    parser.add_argument("--tridet-hidden", type=int, default=96)
    parser.add_argument("--tridet-dropout", type=float, default=0.1)
    parser.add_argument("--tridet-batch-size", type=int, default=16)
    parser.add_argument("--tridet-patience", type=int, default=12)
    parser.add_argument("--tridet-strides", default="1,2,4")
    parser.add_argument("--tridet-max-regression-bin", type=int, default=32)
    parser.add_argument("--tridet-center-sampling-radius", type=float, default=1.5)
    parser.add_argument("--tridet-center-target-mode", choices=["inside", "event_center"], default="event_center")
    parser.add_argument("--tridet-center-weight", type=float, default=1.0)
    parser.add_argument("--tridet-distribution-weight", type=float, default=1.0)
    parser.add_argument("--tridet-iou-weight", type=float, default=0.5)
    parser.add_argument("--tridet-decode-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--tridet-decode-nms-ious", default="0.1,0.3,0.5,none")
    parser.add_argument("--tridet-decode-max-predictions", default="none,4,8,12")

    parser.add_argument("--fusion-max-mainline-ious", default="0,0.05,0.1,0.25")
    parser.add_argument("--fusion-selector-nms-ious", default="0.1,0.3,0.5,none")
    args = parser.parse_args()

    device = str(args.device)
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    records = load_signal_dataset(args.data_dir)
    folds = make_stratified_folds(records, args.folds, args.seed)
    feature_names = _load_feature_names(args.data_dir)
    aux_channel = None
    if args.blend_aux_name:
        if args.blend_aux_name not in feature_names:
            raise ValueError(f"blend feature {args.blend_aux_name!r} not found")
        aux_channel = int(feature_names.index(args.blend_aux_name))

    stride_values = _parse_int_tuple(args.tridet_strides)
    tridet_decode_thresholds = _parse_float_list(args.tridet_decode_thresholds)
    tridet_decode_nms_ious = _parse_optional_float_list(args.tridet_decode_nms_ious)
    tridet_decode_max_predictions = _parse_optional_int_list(args.tridet_decode_max_predictions)
    blend_alphas = _parse_float_list(args.blend_alphas)
    fusion_max_mainline_ious = _parse_optional_float_list(args.fusion_max_mainline_ious)
    fusion_selector_nms_ious = _parse_optional_float_list(args.fusion_selector_nms_ious)

    fold_summaries: list[dict] = []
    prediction_folds: list[dict] = []
    all_indices = set(range(len(records)))
    for fold_idx, val_idx in enumerate(folds):
        train_idx = sorted(all_indices - set(val_idx))
        print(f"fold={fold_idx} train={len(train_idx)} val={len(val_idx)} device={device}", flush=True)
        mainline_model, _, mainline_mean, mainline_std = train_mainline_model(
            records,
            train_idx=train_idx,
            val_idx=val_idx,
            epochs=args.mainline_epochs,
            lr=args.mainline_lr,
            hidden=args.mainline_hidden,
            dropout=args.mainline_dropout,
            patience=args.mainline_patience,
            device=device,
            seed=args.seed + fold_idx,
            lambda_dice=args.mainline_lambda_dice,
            architecture=args.mainline_architecture,
        )
        base_predictions, labels, names, mainline_metrics = _mainline_hybrid_predictions(
            mainline_model,
            records,
            val_idx,
            mainline_mean,
            mainline_std,
            device=device,
            aux_channel=aux_channel,
            blend_alphas=blend_alphas,
            iou_threshold=args.iou_threshold,
        )

        tridet_model, tridet_metrics, tridet_mean, tridet_std = train_tridet_fold(
            records,
            train_idx=train_idx,
            val_idx=val_idx,
            hidden=args.tridet_hidden,
            strides=stride_values,
            max_regression_bin=args.tridet_max_regression_bin,
            center_sampling_radius=args.tridet_center_sampling_radius,
            center_target_mode=args.tridet_center_target_mode,
            dropout=args.tridet_dropout,
            epochs=args.tridet_epochs,
            lr=args.tridet_lr,
            batch_size=args.tridet_batch_size,
            patience=args.tridet_patience,
            device=device,
            seed=args.seed + 1000 + fold_idx,
            center_weight=args.tridet_center_weight,
            distribution_weight=args.tridet_distribution_weight,
            iou_weight=args.tridet_iou_weight,
            decode_thresholds=tridet_decode_thresholds,
            decode_nms_ious=tridet_decode_nms_ious,
            decode_max_predictions=tridet_decode_max_predictions,
            iou_threshold=args.iou_threshold,
        )
        tridet_predictions, _, _ = predict_tridet_indices(
            tridet_model,
            records,
            val_idx,
            tridet_mean,
            tridet_std,
            device=device,
            batch_size=args.tridet_batch_size,
            score_threshold=float(tridet_metrics["params"]["score_threshold"]),
            nms_iou=tridet_metrics["params"]["nms_iou"],
            max_predictions=tridet_metrics["params"]["max_predictions"],
        )
        tridet_validation = evaluate_fused_predictions(tridet_predictions, labels, iou_threshold=args.iou_threshold)
        fusion_config, fusion_metrics = select_protected_fusion_params(
            base_predictions,
            tridet_predictions,
            labels,
            max_mainline_iou_candidates=fusion_max_mainline_ious,
            selector_nms_iou_candidates=fusion_selector_nms_ious,
            iou_threshold=args.iou_threshold,
        )
        fused_predictions = apply_protected_config(base_predictions, tridet_predictions, fusion_config)
        validation = evaluate_fused_predictions(fused_predictions, labels, iou_threshold=args.iou_threshold)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_names": names,
                "mainline": mainline_metrics,
                "tridet": tridet_validation,
                "tridet_selected": tridet_metrics,
                "fusion": {"config": fusion_config, "selected_metrics": fusion_metrics},
                "validation": validation,
            }
        )
        selected_predictions = {
            "tridet": tridet_predictions,
            "fused": fused_predictions,
            "mainline": base_predictions,
        }[args.predictions_kind]
        prediction_folds.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_names": names,
                "predictions": [
                    [[int(start), int(end)] for start, end in video]
                    for video in selected_predictions
                ],
                "labels": [
                    np.asarray(label, dtype=np.int64).reshape(-1).astype(int).tolist()
                    for label in labels
                ],
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
            **vars(args),
            "device": device,
            "tridet_strides": stride_values,
            "task": "binary_error_segment_localization_jepa_tridet_protected_fusion",
        },
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.predictions_out:
        prediction_path = Path(args.predictions_out)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            json.dumps(
                {
                    "schema_version": "round2-prediction-bundle-v1",
                    "data_dir": args.data_dir,
                    "task": "binary_error_segment_localization_jepa_tridet",
                    "prediction_kind": args.predictions_kind,
                    "n_videos": len(records),
                    "folds": prediction_folds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {prediction_path}", flush=True)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
