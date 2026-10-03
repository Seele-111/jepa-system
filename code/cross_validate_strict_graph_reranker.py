#!/usr/bin/env python3
"""Strict nested-CV JEPA graph/protected pipeline with untouched outer folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cross_validate_rgb_baseline import aggregate
from cross_validate_selector_fusion import predict_selector_fold, train_selector_for_fold
from jepa_graph_reranker import (
    is_graph_reranker_improvement,
    select_graph_reranked_predictions,
    select_graph_reranker_params,
)
from jepa_proposal_replacement import replace_merged_predictions, select_proposal_replacement_params
from selector_fusion import (
    evaluate_fused_predictions,
    fuse_protected_segment_predictions,
    select_protected_fusion_params,
)
from train_proposal_set_selector import score_video_candidates
from train_segment_locator import (
    _load_feature_names,
    _segments_from_params,
    blend_prediction_records,
    load_signal_dataset,
    make_stratified_folds,
    predict_records,
    select_blended_postprocess_params,
    train_model,
    train_val_split,
)


DEFAULT_SELECTOR_CHANNELS = [
    "true_vjepa_raw_rank",
    "true_ijepa_dense_raw_rank",
    "dual_jepa_composite_rank",
    "true_vjepa_raw",
    "true_ijepa_dense_raw",
    "dual_jepa_composite",
    "vi_max",
    "vi_product",
    "composite_times_vi_agreement",
    "true_vjepa_raw_abs_delta",
    "true_ijepa_dense_raw_abs_delta",
    "dual_jepa_composite_abs_delta",
]


def nested_split(records, outer_val: list[int], calibration_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    outer_train = [idx for idx in range(len(records)) if idx not in set(outer_val)]
    local_fit, local_cal = train_val_split([records[idx] for idx in outer_train], calibration_ratio, seed)
    fit_idx = [outer_train[idx] for idx in local_fit]
    cal_idx = [outer_train[idx] for idx in local_cal]
    if set(fit_idx) & set(cal_idx) or set(outer_val) & (set(fit_idx) | set(cal_idx)):
        raise RuntimeError("fit/calibration/outer split leakage detected")
    if set(fit_idx) | set(cal_idx) | set(outer_val) != set(range(len(records))):
        raise RuntimeError("nested split does not cover all records")
    return fit_idx, cal_idx


def score_candidates(classifier, records, indices, feature_names, channel_names, thresholds, min_gaps, min_lengths):
    candidate_sets, score_sets = [], []
    for idx in indices:
        candidates, scores = score_video_candidates(
            classifier,
            records[idx],
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
        )
        candidate_sets.append(candidates)
        score_sets.append(scores)
    return candidate_sets, score_sets


def apply_replacement_config(mainline, candidates, scores, config):
    if not config.get("enabled"):
        return mainline
    return replace_merged_predictions(
        mainline,
        candidates,
        scores,
        prob_threshold=float(config["prob_threshold"]),
        parent_min_length=int(config["parent_min_length"]),
        min_replacements=int(config["min_replacements"]),
        max_replacements_per_parent=int(config["max_replacements_per_parent"]),
        min_candidate_parent_coverage=float(config["min_candidate_parent_coverage"]),
        max_candidate_parent_ratio=float(config["max_candidate_parent_ratio"]),
        length_penalty=float(config["length_penalty"]),
    )


def apply_graph_config(mainline, candidates, scores, fallback, config):
    if not config.get("enabled"):
        return fallback
    return select_graph_reranked_predictions(
        mainline,
        candidates,
        scores,
        prob_threshold=float(config["prob_threshold"]),
        support_iou=float(config["support_iou"]),
        support_weight=float(config["support_weight"]),
        support_count_weight=float(config["support_count_weight"]),
        split_penalty=float(config["split_penalty"]),
        mainline_overlap_penalty=float(config["mainline_overlap_penalty"]),
        length_penalty=float(config["length_penalty"]),
    )


def apply_protected_config(mainline, selector, config):
    if not config.get("enabled"):
        return mainline
    return fuse_protected_segment_predictions(
        mainline,
        selector,
        max_mainline_iou=config["max_mainline_iou"],
        selector_nms_iou=config["selector_nms_iou"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blend-aux-name", default="true_vjepa_raw_rank")
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_SELECTOR_CHANNELS))
    parser.add_argument(
        "--compact-grid",
        action="store_true",
        help="Post-hoc diagnostic grid restricted to configurations selected by the original development CV",
    )
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name.strip() for name in args.selector_channel_names.split(",") if name.strip()]
    missing = [name for name in [args.blend_aux_name, *channel_names] if name not in feature_names]
    if missing:
        raise ValueError(f"missing required features: {missing}")
    blend_channel = feature_names.index(args.blend_aux_name)
    folds = make_stratified_folds(records, args.folds, args.seed)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    min_gaps = [0, 1, 2, 4, 8]
    min_lengths = [1, 2, 4, 8]
    nms_candidates = [None, 0.1, 0.3, 0.5]
    protected_iou_candidates = [0.0, 0.1, 0.25, 0.5]
    replacement_grid = {
        "prob_thresholds": [0.5] if args.compact_grid else [0.5, 0.6, 0.7, 0.8, 0.9],
        "parent_min_lengths": [72] if args.compact_grid else [24, 48, 72],
        "min_replacements_list": [3] if args.compact_grid else [2, 3],
        "max_replacements_per_parents": [3] if args.compact_grid else [2, 3, 4],
        "min_candidate_parent_coverages": [1.0] if args.compact_grid else [0.8, 0.9, 1.0],
        "max_candidate_parent_ratios": [0.6] if args.compact_grid else [0.25, 0.4, 0.6, 0.8],
        "length_penalties": [0.01] if args.compact_grid else [0.0, 0.005, 0.01, 0.02],
    }
    graph_grid = {
        "prob_thresholds": [0.75] if args.compact_grid else [0.55, 0.6, 0.65, 0.7, 0.75],
        "support_ious": [0.2, 0.3],
        "support_weights": [0.0, 0.2] if args.compact_grid else [0.0, 0.1, 0.2],
        "support_count_weights": [0.0] if args.compact_grid else [0.0, 0.05],
        "split_penalties": [0.0] if args.compact_grid else [0.0, 0.1, 0.2],
        "mainline_overlap_penalties": [0.0, 0.2],
        "length_penalties": [0.0] if args.compact_grid else [0.0, 0.005],
    }
    fold_rows, bundle_folds = [], []

    for fold, outer_val in enumerate(folds):
        fit_idx, cal_idx = nested_split(records, outer_val, args.calibration_ratio, args.seed + fold)
        model, training_metrics, mean, std = train_model(
            records,
            train_idx=fit_idx,
            val_idx=cal_idx,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            dropout=args.dropout,
            patience=args.patience,
            device=args.device,
            seed=args.seed + fold,
            lambda_dice=0.0,
            lambda_boundary=0.0,
            architecture="tcn",
        )
        cal_outputs = predict_records(model, records, cal_idx, mean, std, args.device)
        alpha, postprocess, calibration_mainline_metrics = select_blended_postprocess_params(
            cal_outputs,
            records,
            cal_idx,
            aux_channel=blend_channel,
            alphas=np.linspace(0.0, 1.0, 11),
            iou_threshold=0.3,
        )

        def mainline_for(indices):
            outputs = predict_records(model, records, indices, mean, std, args.device)
            outputs = blend_prediction_records(outputs, records, indices, blend_channel, alpha)
            return [_segments_from_params(row["probs"], postprocess) for row in outputs]

        cal_mainline = mainline_for(cal_idx)
        outer_mainline = mainline_for(outer_val)
        classifier, selector_params, selector_cal_metrics, n_candidates, n_positive = train_selector_for_fold(
            records,
            fit_idx,
            cal_idx,
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
            iou_threshold=0.3,
            seed=args.seed + 1000 + fold,
            model_name="extratrees",
            target="binary",
            device=args.device,
        )
        cal_selector = predict_selector_fold(
            classifier, records, cal_idx, feature_names, channel_names, thresholds, min_gaps, min_lengths, selector_params
        )
        outer_selector = predict_selector_fold(
            classifier, records, outer_val, feature_names, channel_names, thresholds, min_gaps, min_lengths, selector_params
        )
        cal_candidates, cal_scores = score_candidates(
            classifier, records, cal_idx, feature_names, channel_names, thresholds, min_gaps, min_lengths
        )
        outer_candidates, outer_scores = score_candidates(
            classifier, records, outer_val, feature_names, channel_names, thresholds, min_gaps, min_lengths
        )
        cal_labels = [records[idx].labels for idx in cal_idx]
        outer_labels = [records[idx].labels for idx in outer_val]

        replacement_config, replacement_cal_metrics, cal_replaced = select_proposal_replacement_params(
            cal_mainline,
            cal_candidates,
            cal_scores,
            cal_labels,
            **replacement_grid,
            max_fp_increase=None,
            iou_threshold=0.3,
        )
        outer_replaced = apply_replacement_config(outer_mainline, outer_candidates, outer_scores, replacement_config)

        baseline_protected_config, baseline_protected_metrics = select_protected_fusion_params(
            cal_replaced,
            cal_selector,
            cal_labels,
            max_mainline_iou_candidates=protected_iou_candidates,
            selector_nms_iou_candidates=nms_candidates,
            iou_threshold=0.3,
        )
        graph_config, graph_cal_metrics, cal_graph_selector = select_graph_reranker_params(
            cal_replaced,
            cal_candidates,
            cal_scores,
            cal_labels,
            **graph_grid,
            protected_mainline_iou_candidates=protected_iou_candidates,
            selector_nms_iou_candidates=nms_candidates,
            iou_threshold=0.3,
        )
        graph_enabled = bool(
            graph_config.get("enabled") and is_graph_reranker_improvement(graph_cal_metrics, baseline_protected_metrics)
        )
        if not graph_enabled:
            graph_config = {**graph_config, "enabled": False, "rejected_against_selector_baseline": True}
            active_cal_selector = cal_selector
            active_outer_selector = outer_selector
        else:
            active_cal_selector = cal_graph_selector
            active_outer_selector = apply_graph_config(
                outer_replaced, outer_candidates, outer_scores, outer_selector, graph_config
            )

        protected_config, protected_cal_metrics = select_protected_fusion_params(
            cal_replaced,
            active_cal_selector,
            cal_labels,
            max_mainline_iou_candidates=protected_iou_candidates,
            selector_nms_iou_candidates=nms_candidates,
            iou_threshold=0.3,
        )
        outer_predictions = apply_protected_config(outer_replaced, active_outer_selector, protected_config)
        outer_metrics = evaluate_fused_predictions(outer_predictions, outer_labels, iou_threshold=0.3)
        outer_metrics_05 = evaluate_fused_predictions(outer_predictions, outer_labels, iou_threshold=0.5)
        fold_rows.append(
            {
                "fold": fold,
                "fit_names": [records[idx].name for idx in fit_idx],
                "calibration_names": [records[idx].name for idx in cal_idx],
                "outer_names": [records[idx].name for idx in outer_val],
                "training": training_metrics,
                "blend": {"alpha": alpha, "params": postprocess, "calibration": calibration_mainline_metrics},
                "selector": {
                    "params": selector_params,
                    "calibration": selector_cal_metrics,
                    "n_fit_candidates": n_candidates,
                    "n_positive_fit_candidates": n_positive,
                },
                "proposal_replacement": {"config": replacement_config, "calibration": replacement_cal_metrics},
                "graph_reranker": {"config": graph_config, "calibration": graph_cal_metrics},
                "protected_fusion": {"config": protected_config, "calibration": protected_cal_metrics},
                "metrics": outer_metrics,
                "metrics_iou_0.5": outer_metrics_05,
            }
        )
        bundle_folds.append(
            {
                "fold": fold,
                "val_names": [records[idx].name for idx in outer_val],
                "predictions": [[list(segment) for segment in video] for video in outer_predictions],
                "labels": [label.astype(int).tolist() for label in outer_labels],
            }
        )
        print(
            f"fold={fold} fit={len(fit_idx)} cal={len(cal_idx)} outer={len(outer_val)} "
            f"f1@0.3={outer_metrics['segment']['f1']:.4f} f1@0.5={outer_metrics_05['segment']['f1']:.4f}",
            flush=True,
        )

    summary = {
        "schema_version": "round2-strict-graph-nested-cv-v1",
        "method": "JEPA-Loc graph/protected strict nested CV",
        "protocol": "outer folds untouched by fitting, early stopping, calibration, and graph selection",
        "data_dir": args.data_dir,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "grid_scope": "post-hoc compact development-selected grid" if args.compact_grid else "full original grid",
        "folds": fold_rows,
        "aggregate": {"frame": aggregate(fold_rows, "frame"), "segment": aggregate(fold_rows, "segment")},
        "aggregate_iou_0.5": {
            section: {
                **{
                    f"{metric}_mean": float(np.mean([fold["metrics_iou_0.5"][section][metric] for fold in fold_rows]))
                    for metric in ("precision", "recall", "f1")
                },
                **{
                    f"{metric}_std": float(np.std([fold["metrics_iou_0.5"][section][metric] for fold in fold_rows]))
                    for metric in ("precision", "recall", "f1")
                },
            }
            for section in ("frame", "segment")
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bundle = {
        "schema_version": "round2-prediction-bundle-v1",
        "task": "binary_error_segment_localization_strict_nested_cv",
        "method": summary["method"],
        "folds": bundle_folds,
    }
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"]["segment"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
