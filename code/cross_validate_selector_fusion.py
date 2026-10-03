#!/usr/bin/env python3
"""Cross-validate mainline temporal CNN + JEPA proposal set selector fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from fusion_aware_selector import select_fusion_aware_selector_params_from_scores
from jepa_clean_anchor_rectification import (
    CleanAnchorRectifyConfig,
    compute_clean_anchor_frame_weights_for_fold,
    rectify_records_for_fold,
)
from jepa_consensus_denoising import ConsensusDenoiseConfig, denoise_records_for_fold
from jepa_dp_event_set_selector import select_dp_event_set_params
from jepa_boundary_distribution_refine import select_boundary_distribution_refine_params
from jepa_candidate_rescue import select_candidate_rescue_params
from jepa_clean_calibrated_selector import (
    fit_clean_candidate_calibrator,
    score_clean_calibrated_candidates,
    select_clean_calibrated_event_set_params,
)
from jepa_boundary_snap import select_boundary_snap_params
from jepa_evidence_set_reasoner import select_evidence_set_reasoner_params
from jepa_event_topology_splitter import select_event_topology_split_params
from jepa_fn_aware_reranker import (
    apply_fn_aware_reranker_config,
    build_fn_acceptance_candidate_records,
    build_fn_aware_candidate_records,
    fit_fn_aware_reranker,
    label_fn_acceptance_oracle,
    label_fn_aware_candidates,
    score_fn_acceptance_candidates,
    score_fn_aware_candidates,
    select_fn_aware_reranker_params,
    select_strict_fn_aware_reranker_params,
    split_indices_for_inner_calibration,
)
from jepa_graph_reranker import (
    is_graph_reranker_improvement,
    select_graph_reranked_predictions,
    select_graph_reranker_params,
)
from jepa_ot_event_set_matcher import select_ot_event_set_params
from jepa_proposal_set_distillation import (
    build_distillation_candidate_records,
    fit_distillation_student,
    fit_soft_quality_distillation_student,
    score_distillation_candidates,
    select_distilled_proposal_set_params,
    select_distilled_topology_replacement_params,
)
from jepa_proposal_replacement import replace_merged_predictions, select_proposal_replacement_params
from jepa_residual_sub_event_expansion import (
    select_residual_sub_event_expansion_params,
    select_residual_sub_event_predictions,
)
from jepa_scale_completeness_rescue import select_scale_completeness_rescue_params
from jepa_topology_matching_network import select_topology_matching_params
from jepa_topology_split_gate import select_topology_split_gate_params
from jepa_watershed_split import select_watershed_split_params
from selector_fusion import (
    evaluate_fused_predictions,
    fuse_protected_segment_predictions,
    fuse_segment_predictions,
    select_fusion_nms_iou,
    select_protected_fusion_params,
)
from train_proposal_calibrator import evaluate_segment_predictions
from train_proposal_set_selector import (
    DEFAULT_CHANNEL_NAMES,
    build_candidate_records,
    candidate_quality_targets,
    fit_classifier,
    fit_quality_model,
    predict_video_segments,
    score_video_candidates,
    select_inference_params,
    select_weighted_proposal_set,
)
from train_proposal_rescuer import (
    build_rescue_candidate_records,
    fit_rescuer,
    select_rescuer_params,
)
from train_segment_locator import (
    _feature_indices,
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    _segments_from_params,
    blend_prediction_records,
    load_signal_dataset,
    make_stratified_folds,
    predict_records,
    select_blended_postprocess_params,
    train_model,
)


def _parse_nms_candidates(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else float(item))
    if not values:
        raise ValueError("at least one NMS candidate is required")
    return values


def _parse_optional_float(text: str) -> float | None:
    value = str(text).strip()
    return None if value.lower() in {"none", "null"} else float(value)


def _parse_optional_int_list(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(None if stripped.lower() in {"none", "null"} else int(stripped))
    return values


def _reduce_evidence_channels(record, feature_names: list[str], evidence_names: list[str], reducer: str) -> np.ndarray:
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    channels = [name_to_idx[name] for name in evidence_names if name in name_to_idx]
    if not channels:
        raise ValueError(f"none of evidence channels were found: {evidence_names}")
    values = np.nan_to_num(record.signals[:, channels], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if reducer == "mean":
        return values.mean(axis=1).astype(np.float32)
    if reducer == "max":
        return values.max(axis=1).astype(np.float32)
    raise ValueError(f"unsupported watershed reducer: {reducer}")


def _event_topology_evidence(record, feature_names: list[str], evidence_names: list[str], reducer: str) -> np.ndarray:
    if reducer == "stack":
        name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
        channels = [name_to_idx[name] for name in evidence_names if name in name_to_idx]
        if not channels:
            raise ValueError(f"none of event-topology evidence channels were found: {evidence_names}")
        return np.nan_to_num(record.signals[:, channels], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return _reduce_evidence_channels(record, feature_names, evidence_names, reducer)


def aggregate_named_metrics(folds: list[dict], key: str) -> dict:
    return aggregate_fold_metrics([{"validation": fold[key]} for fold in folds])


def _json_segments(segments: list[tuple[int, int]]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in segments]


def _build_prediction_export(
    data_dir: str,
    records,
    fold_summaries: list[dict],
    predictions_by_fold: list[list[list[tuple[int, int]]]],
    include_labels: bool = False,
) -> dict:
    """Build a compact JSON-safe OOF prediction export."""
    if len(fold_summaries) != len(predictions_by_fold):
        raise ValueError("fold_summaries and predictions_by_fold must have the same length")
    folds: list[dict] = []
    for fold, fold_predictions in zip(fold_summaries, predictions_by_fold):
        val_idx = [int(idx) for idx in fold["val_idx"]]
        if len(val_idx) != len(fold_predictions):
            raise ValueError(
                f"fold={fold.get('fold')} val/prediction count mismatch: "
                f"{len(val_idx)} vs {len(fold_predictions)}"
            )
        item = {
            "fold": int(fold["fold"]),
            "train_idx": [int(idx) for idx in fold.get("train_idx", [])],
            "val_idx": val_idx,
            "val_names": [str(records[idx].name) for idx in val_idx],
            "predictions": [_json_segments(video_predictions) for video_predictions in fold_predictions],
            "validation": fold.get("validation", {}),
        }
        if include_labels:
            item["labels"] = [
                np.asarray(records[idx].labels, dtype=np.int64).reshape(-1).astype(int).tolist()
                for idx in val_idx
            ]
        folds.append(item)
    return {
        "data_dir": str(data_dir),
        "task": "binary_error_segment_localization_oof_predictions",
        "n_videos": int(len(records)),
        "folds": folds,
    }


def train_selector_for_fold(
    records,
    train_idx: list[int],
    val_idx: list[int],
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    iou_threshold: float,
    seed: int,
    model_name: str,
    target: str,
    device: str = "cpu",
    selector_epochs: int = 120,
    selector_batch_size: int = 512,
) -> tuple[object, dict, dict, int, int]:
    candidates = build_candidate_records(
        records,
        train_idx,
        feature_names=feature_names,
        channel_names=channel_names,
        thresholds=thresholds,
        min_gaps=min_gaps,
        min_lengths=min_lengths,
        iou_threshold=iou_threshold,
    )
    if not candidates:
        raise RuntimeError("no training candidates generated")
    x_train = np.stack([item.features for item in candidates])
    y_train = np.asarray([item.label for item in candidates], dtype=np.int64)
    best_iou_train = np.asarray([item.best_iou for item in candidates], dtype=np.float32)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"candidate labels need both classes, got positives={int(y_train.sum())}/{len(y_train)}")
    if target == "quality":
        if model_name == "mlp":
            raise ValueError("selector-target=quality is not supported with selector-model=mlp")
        classifier = fit_quality_model(x_train, best_iou_train, seed=seed, model_name=model_name)
    else:
        classifier = fit_classifier(
            x_train,
            y_train,
            seed=seed,
            model_name=model_name,
            device=device,
            epochs=selector_epochs,
            batch_size=selector_batch_size,
        )
    params, metrics = select_inference_params(
        classifier,
        records,
        val_idx,
        feature_names,
        channel_names,
        thresholds,
        min_gaps,
        min_lengths,
        iou_threshold,
    )
    return classifier, params, metrics, len(candidates), int(y_train.sum())


def predict_selector_fold(
    classifier,
    records,
    val_idx: list[int],
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    params: dict,
) -> list[list[tuple[int, int]]]:
    return [
        predict_video_segments(
            classifier,
            records[idx],
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
            prob_threshold=float(params["prob_threshold"]),
            length_penalty=float(params["length_penalty"]),
        )
        for idx in val_idx
    ]


def predict_selector_from_scored_candidates(
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    prob_threshold: float,
    length_penalty: float,
) -> list[list[tuple[int, int]]]:
    predictions: list[list[tuple[int, int]]] = []
    for candidates, scores_raw in zip(candidate_predictions, candidate_scores):
        scores = np.asarray(scores_raw, dtype=np.float32).reshape(-1)
        filtered_candidates = []
        filtered_scores = []
        for segment, score in zip(candidates, scores):
            if float(score) < float(prob_threshold):
                continue
            filtered_candidates.append(segment)
            filtered_scores.append(float(score))
        predictions.append(
            select_weighted_proposal_set(
                filtered_candidates,
                filtered_scores,
                length_penalty=float(length_penalty),
            )
        )
    return predictions


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
    parser.add_argument("--lambda-boundary", type=float, default=0.0)
    parser.add_argument("--architecture", choices=["tcn", "attn_tcn"], default="tcn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-aux-name", default="true_vjepa_raw_rank")
    parser.add_argument("--blend-alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--consensus-denoise", action="store_true")
    parser.add_argument(
        "--consensus-evidence-names",
        default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank",
    )
    parser.add_argument("--consensus-agreement-name", default="vi_agreement")
    parser.add_argument("--consensus-add-threshold", type=float, default=0.9)
    parser.add_argument("--consensus-remove-threshold", type=float, default=0.05)
    parser.add_argument("--consensus-min-add-length", type=int, default=2)
    parser.add_argument("--consensus-min-keep-length", type=int, default=2)
    parser.add_argument("--consensus-smooth-window", type=int, default=1)
    parser.add_argument("--consensus-agreement-weight", type=float, default=0.2)
    parser.add_argument("--consensus-max-added-fraction", type=float, default=None)
    parser.add_argument("--clean-anchor-rectify", action="store_true")
    parser.add_argument("--clean-anchor-soft-weights", action="store_true")
    parser.add_argument("--clean-anchor-data-dir", default="/home/zzy/jepa_data/clean79_event_jepa_v2")
    parser.add_argument("--clean-anchor-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--clean-anchor-teacher-model", choices=["prototype", "gbdt", "rf", "extratrees", "logreg"], default="prototype")
    parser.add_argument("--clean-anchor-positive-threshold", type=float, default=0.7)
    parser.add_argument("--clean-anchor-negative-threshold", type=float, default=0.3)
    parser.add_argument("--clean-anchor-add-threshold", type=float, default=0.85)
    parser.add_argument("--clean-anchor-remove-threshold", type=float, default=0.2)
    parser.add_argument("--clean-anchor-min-add-length", type=int, default=2)
    parser.add_argument("--clean-anchor-min-keep-length", type=int, default=2)
    parser.add_argument("--clean-anchor-smooth-window", type=int, default=1)
    parser.add_argument("--clean-anchor-max-added-fraction", type=float, default=None)
    parser.add_argument("--clean-anchor-positive-min-weight", type=float, default=0.35)
    parser.add_argument("--clean-anchor-negative-min-weight", type=float, default=0.35)
    parser.add_argument("--selector-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--selector-target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--selector-epochs", type=int, default=120)
    parser.add_argument("--selector-batch-size", type=int, default=512)
    parser.add_argument("--fusion-aware-selector", action="store_true")
    parser.add_argument("--fusion-aware-thresholds", default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    parser.add_argument("--fusion-aware-length-penalties", default="0,0.005,0.01,0.02,0.04,0.08")
    parser.add_argument("--clean-calibrated-selector", action="store_true")
    parser.add_argument("--clean-calibration-data-dir", default="/home/zzy/jepa_data/clean69_event_jepa_v2")
    parser.add_argument("--clean-calibration-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--clean-calibration-target", choices=["binary", "quality"], default="quality")
    parser.add_argument("--clean-calibration-epochs", type=int, default=120)
    parser.add_argument("--clean-calibration-batch-size", type=int, default=512)
    parser.add_argument("--clean-calibrated-thresholds", default="0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--clean-calibrated-raw-thresholds", default="0,0.1,0.2,0.3")
    parser.add_argument("--clean-calibrated-clean-weights", default="0.5,0.7,0.9,1.1")
    parser.add_argument("--clean-calibrated-raw-weights", default="0,0.1,0.3,0.5")
    parser.add_argument("--clean-calibrated-max-base-ious", default="0,0.1,0.25,0.5,none")
    parser.add_argument("--clean-calibrated-length-penalties", default="0,0.005,0.01,0.02")
    parser.add_argument("--clean-calibrated-max-rescues-per-video", default="1,2,3")
    parser.add_argument("--clean-calibrated-rescue-nms-ious", default="0.1,0.3,none")
    parser.add_argument("--clean-calibrated-max-fp-increase", type=int, default=0)
    parser.add_argument("--graph-reranker", action="store_true")
    parser.add_argument("--graph-reranker-thresholds", default="0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--graph-reranker-support-ious", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--graph-reranker-support-weights", default="0,0.05,0.1,0.2")
    parser.add_argument("--graph-reranker-support-count-weights", default="0,0.02,0.05,0.1")
    parser.add_argument("--graph-reranker-split-penalties", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--graph-reranker-mainline-overlap-penalties", default="0,0.1,0.2,0.4")
    parser.add_argument("--graph-reranker-length-penalties", default="0,0.005,0.01,0.02")
    parser.add_argument("--dp-event-set-selector", action="store_true")
    parser.add_argument("--dp-event-set-base-keep-scores", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--dp-event-set-candidate-score-weights", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--dp-event-set-min-candidate-scores", default="0,0.1,0.2,0.3")
    parser.add_argument("--dp-event-set-length-penalties", default="0,0.005,0.01")
    parser.add_argument("--dp-event-set-base-overlap-penalties", default="0,0.1,0.25")
    parser.add_argument("--dp-event-set-max-fp-increase", type=int, default=0)
    parser.add_argument("--ot-event-set-matcher", action="store_true")
    parser.add_argument("--ot-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--ot-reducer", choices=["max", "mean", "stack"], default="stack")
    parser.add_argument("--ot-score-thresholds", default="0.4,0.6,0.8,1.0")
    parser.add_argument("--ot-base-keep-scores", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--ot-evidence-thresholds", default="0.45,0.55,0.65")
    parser.add_argument("--ot-selector-weights", default="0,0.1,0.25")
    parser.add_argument("--ot-contrast-weights", default="0.25,0.5")
    parser.add_argument("--ot-active-fraction-weights", default="0.25,0.5")
    parser.add_argument("--ot-agreement-weights", default="0.25,0.5")
    parser.add_argument("--ot-island-coverage-weights", default="0.5,1")
    parser.add_argument("--ot-peak-alignment-weights", default="0.25,0.5")
    parser.add_argument("--ot-base-overlap-penalties", default="0,0.25")
    parser.add_argument("--ot-length-penalties", default="0,0.005")
    parser.add_argument("--ot-smooth-windows", default="1,3")
    parser.add_argument("--ot-min-evidence-island-lengths", default="1,2")
    parser.add_argument("--ot-max-fp-increase", type=int, default=0)
    parser.add_argument("--rescuer-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp", "prototype", "none"], default="extratrees")
    parser.add_argument("--rescuer-epochs", type=int, default=120)
    parser.add_argument("--rescuer-batch-size", type=int, default=512)
    parser.add_argument("--boundary-snap", action="store_true")
    parser.add_argument("--boundary-snap-min-scores", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--boundary-snap-coverages", default="0.25,0.5,0.75,0.9")
    parser.add_argument("--boundary-snap-max-ratios", default="1.5,2,3,4")
    parser.add_argument("--watershed-split", action="store_true")
    parser.add_argument("--watershed-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--watershed-reducer", choices=["max", "mean"], default="max")
    parser.add_argument("--watershed-thresholds", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--watershed-smooth-windows", default="1,3")
    parser.add_argument("--watershed-min-gaps", default="0,1")
    parser.add_argument("--watershed-min-lengths", default="1,2,4")
    parser.add_argument("--watershed-pads", default="0,1")
    parser.add_argument("--watershed-parent-min-lengths", default="24,48")
    parser.add_argument("--proposal-replacement", action="store_true")
    parser.add_argument("--proposal-replacement-thresholds", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--proposal-replacement-parent-min-lengths", default="24,48,72")
    parser.add_argument("--proposal-replacement-min-replacements", default="2,3")
    parser.add_argument("--proposal-replacement-max-per-parent", default="2,3,4")
    parser.add_argument("--proposal-replacement-coverages", default="0.8,0.9,1.0")
    parser.add_argument("--proposal-replacement-max-ratios", default="0.25,0.4,0.6,0.8")
    parser.add_argument("--proposal-replacement-length-penalties", default="0,0.005,0.01,0.02")
    parser.add_argument("--proposal-replacement-max-fp-increase", type=int, default=None)
    parser.add_argument("--candidate-rescue", action="store_true")
    parser.add_argument("--candidate-rescue-thresholds", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--candidate-rescue-mainline-ious", default="0,0.05,0.1,0.25,none")
    parser.add_argument("--candidate-rescue-nms-ious", default="none,0.1,0.3,0.5")
    parser.add_argument("--candidate-rescue-length-penalties", default="0,0.005,0.01,0.02,0.04")
    parser.add_argument("--candidate-rescue-max-per-video", default="1,2,3")
    parser.add_argument("--scale-completeness-rescue", action="store_true")
    parser.add_argument("--scale-completeness-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--scale-completeness-reducer", choices=["max", "mean"], default="max")
    parser.add_argument("--scale-completeness-prob-thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--scale-completeness-evidence-thresholds", default="0.55,0.65,0.75,0.85")
    parser.add_argument("--scale-completeness-min-active-fractions", default="0.5,0.67,0.8,0.9")
    parser.add_argument("--scale-completeness-min-means", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--scale-completeness-min-contrasts", default="0.05,0.1,0.2,0.3,0.4")
    parser.add_argument("--scale-completeness-max-base-ious", default="0,0.05,0.1,0.25")
    parser.add_argument("--scale-completeness-length-penalties", default="0,0.005,0.01")
    parser.add_argument("--scale-completeness-nms-ious", default="none,0.1,0.3,0.5")
    parser.add_argument("--scale-completeness-max-per-video", default="1,2")
    parser.add_argument("--evidence-set-reasoner", action="store_true")
    parser.add_argument("--evidence-set-reasoner-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--evidence-set-reasoner-min-selector-scores", default="0,0.05")
    parser.add_argument("--evidence-set-reasoner-evidence-thresholds", default="0.5,0.6,0.7")
    parser.add_argument("--evidence-set-reasoner-min-evidence-means", default="0.4,0.5,0.6")
    parser.add_argument("--evidence-set-reasoner-min-active-fractions", default="0.25,0.5")
    parser.add_argument("--evidence-set-reasoner-min-contrasts", default="0,0.05,0.1")
    parser.add_argument("--evidence-set-reasoner-score-thresholds", default="0.8,1.0,1.2")
    parser.add_argument("--evidence-set-reasoner-selector-weights", default="0,0.1")
    parser.add_argument("--evidence-set-reasoner-contrast-weights", default="0.5")
    parser.add_argument("--evidence-set-reasoner-active-fraction-weights", default="0.5")
    parser.add_argument("--evidence-set-reasoner-consensus-weights", default="0.5,1")
    parser.add_argument("--evidence-set-reasoner-support-ious", default="0.3")
    parser.add_argument("--evidence-set-reasoner-support-count-weights", default="0,0.05,0.1")
    parser.add_argument("--evidence-set-reasoner-max-base-ious", default="none")
    parser.add_argument("--evidence-set-reasoner-base-iou-penalties", default="0,0.2,0.5")
    parser.add_argument("--evidence-set-reasoner-length-penalties", default="0,0.005,0.01")
    parser.add_argument("--evidence-set-reasoner-nms-ious", default="none,0.1,0.3,0.5")
    parser.add_argument("--evidence-set-reasoner-max-per-video", default="1,2,3,4")
    parser.add_argument("--evidence-set-reasoner-prefilter-top-k", type=int, default=80)
    parser.add_argument("--boundary-distribution-refine", action="store_true")
    parser.add_argument("--boundary-distribution-thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--boundary-distribution-base-coverages", default="0.5,0.75,0.9")
    parser.add_argument("--boundary-distribution-max-ratios", default="2,3,4")
    parser.add_argument("--boundary-distribution-start-quantiles", default="0.25,0.5")
    parser.add_argument("--boundary-distribution-end-quantiles", default="0.5,0.75")
    parser.add_argument("--boundary-distribution-base-weights", default="0,0.25,0.5")
    parser.add_argument("--boundary-distribution-max-shift-ratios", default="1,2,4")
    parser.add_argument("--residual-sub-event-expansion", action="store_true")
    parser.add_argument("--residual-sub-event-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--residual-sub-event-reducer", choices=["max", "mean"], default="max")
    parser.add_argument("--residual-sub-event-min-selector-scores", default="0.05,0.1")
    parser.add_argument("--residual-sub-event-evidence-thresholds", default="0.65,0.75")
    parser.add_argument("--residual-sub-event-min-evidence-means", default="0.55,0.65")
    parser.add_argument("--residual-sub-event-min-active-fractions", default="0.5,0.67")
    parser.add_argument("--residual-sub-event-min-parent-contrasts", default="0.05,0.1,0.2")
    parser.add_argument("--residual-sub-event-parent-min-lengths", default="48,72")
    parser.add_argument("--residual-sub-event-max-child-parent-ratios", default="0.25,0.4")
    parser.add_argument("--residual-sub-event-min-child-parent-coverages", default="0.8,0.9")
    parser.add_argument("--residual-sub-event-length-penalties", default="0,0.005")
    parser.add_argument("--residual-sub-event-nms-ious", default="0.3")
    parser.add_argument("--residual-sub-event-max-per-parent", default="1,2")
    parser.add_argument("--residual-sub-event-max-per-video", default="1,2")
    parser.add_argument("--residual-sub-event-modes", default="add")
    parser.add_argument("--residual-sub-event-min-per-parent", default="2")
    parser.add_argument("--event-topology-splitter", action="store_true")
    parser.add_argument("--event-topology-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--event-topology-reducer", choices=["max", "mean", "stack"], default="stack")
    parser.add_argument("--event-topology-min-selector-scores", default="0.45,0.55,0.65")
    parser.add_argument("--event-topology-evidence-thresholds", default="0.6,0.7,0.8")
    parser.add_argument("--event-topology-min-evidence-means", default="0.5,0.6,0.7")
    parser.add_argument("--event-topology-min-active-fractions", default="0.5,0.67,0.8")
    parser.add_argument("--event-topology-min-parent-contrasts", default="0.05,0.1,0.2")
    parser.add_argument("--event-topology-parent-min-lengths", default="24,48,72")
    parser.add_argument("--event-topology-max-child-parent-ratios", default="0.25,0.4,0.6")
    parser.add_argument("--event-topology-min-child-parent-coverages", default="0.8,0.9")
    parser.add_argument("--event-topology-min-gaps", default="0,2,4")
    parser.add_argument("--event-topology-support-ious", default="0.2,0.3")
    parser.add_argument("--event-topology-support-count-weights", default="0,0.05,0.1")
    parser.add_argument("--event-topology-length-penalties", default="0,0.005")
    parser.add_argument("--event-topology-nms-ious", default="0.3")
    parser.add_argument("--event-topology-min-children-per-parent", default="2")
    parser.add_argument("--event-topology-max-children-per-parent", default="2,3")
    parser.add_argument("--event-topology-max-replaced-parents-per-video", default="1,2")
    parser.add_argument("--event-topology-max-fp-increase", type=int, default=0)
    parser.add_argument("--topology-split-gate", action="store_true")
    parser.add_argument("--topology-split-gate-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--topology-split-gate-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--topology-split-gate-reducer", choices=["max", "mean", "stack"], default="stack")
    parser.add_argument("--topology-split-gate-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--topology-split-gate-parent-min-lengths", default="12,24,48")
    parser.add_argument("--topology-split-gate-min-child-parent-coverages", default="0.5,0.7")
    parser.add_argument("--topology-split-gate-max-child-parent-ratios", default="0.5,0.7")
    parser.add_argument("--topology-split-gate-max-children-per-parent", default="2,3")
    parser.add_argument("--topology-split-gate-max-replaced-parents-per-video", default="1")
    parser.add_argument("--topology-split-gate-max-fp-increase", type=int, default=0)
    parser.add_argument("--topology-split-gate-synthetic", action="store_true")
    parser.add_argument("--topology-split-gate-synthetic-max-gt-gap", type=int, default=24)
    parser.add_argument("--topology-split-gate-synthetic-parent-pad", type=int, default=0)
    parser.add_argument("--topology-matching-network", action="store_true")
    parser.add_argument("--topology-matching-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp", "prototype"], default="prototype")
    parser.add_argument("--topology-matching-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--topology-matching-reducer", choices=["max", "mean", "stack"], default="stack")
    parser.add_argument("--topology-matching-thresholds", default="0,0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--topology-matching-parent-min-lengths", default="12,24,48")
    parser.add_argument("--topology-matching-min-child-parent-coverages", default="0.5,0.7")
    parser.add_argument("--topology-matching-max-child-parent-ratios", default="0.5,0.7")
    parser.add_argument("--topology-matching-max-children-per-parent", default="2,3")
    parser.add_argument("--topology-matching-max-set-proposals-per-parent", default="6,12")
    parser.add_argument("--topology-matching-max-child-pool-per-parent", default="12,16")
    parser.add_argument("--topology-matching-max-replaced-parents-per-video", default="1,2")
    parser.add_argument("--topology-matching-max-fp-increase", type=int, default=0)
    parser.add_argument("--topology-matching-risk-aware", action="store_true")
    parser.add_argument("--topology-matching-risk-penalties", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--topology-matching-safety-thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--topology-matching-safety-weight", type=float, default=0.25)
    parser.add_argument("--topology-matching-epochs", type=int, default=120)
    parser.add_argument("--topology-matching-batch-size", type=int, default=512)
    parser.add_argument("--fn-aware-reranker", action="store_true")
    parser.add_argument("--fn-aware-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp", "prototype"], default="extratrees")
    parser.add_argument("--fn-aware-epochs", type=int, default=120)
    parser.add_argument("--fn-aware-batch-size", type=int, default=512)
    parser.add_argument("--fn-aware-train-max-per-video", type=int, default=120)
    parser.add_argument("--fn-aware-train-min-score", type=float, default=None)
    parser.add_argument("--fn-aware-thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--fn-aware-max-base-ious", default="0,0.05,0.1,0.25,none")
    parser.add_argument("--fn-aware-nms-ious", default="none,0.1,0.3,0.5")
    parser.add_argument("--fn-aware-length-penalties", default="0,0.005,0.01,0.02")
    parser.add_argument("--fn-aware-max-per-video", default="1,2,3")
    parser.add_argument("--fn-aware-max-fp-increase", type=int, default=None)
    parser.add_argument("--fn-aware-max-candidates-per-video", type=int, default=None)
    parser.add_argument("--fn-aware-acceptance-distill", action="store_true")
    parser.add_argument("--fn-aware-acceptance-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp", "prototype"], default="extratrees")
    parser.add_argument("--fn-aware-acceptance-train-max-per-video", type=int, default=120)
    parser.add_argument("--fn-aware-acceptance-threshold", type=float, default=0.4)
    parser.add_argument("--fn-aware-acceptance-max-rescues-per-video", type=int, default=2)
    parser.add_argument("--fn-aware-acceptance-max-candidates-per-video", type=int, default=60)
    parser.add_argument("--fn-aware-acceptance-max-fp-increase", type=int, default=0)
    parser.add_argument("--fn-aware-acceptance-use-aux-positives", action="store_true")
    parser.add_argument("--fn-aware-acceptance-inner-calibration", action="store_true")
    parser.add_argument("--fn-aware-acceptance-calibration-fraction", type=float, default=0.25)
    parser.add_argument("--fn-aware-acceptance-calibration-seed-offset", type=int, default=3700)
    parser.add_argument("--proposal-set-distillation", action="store_true")
    parser.add_argument("--proposal-set-distillation-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp", "prototype"], default="extratrees")
    parser.add_argument("--proposal-set-distillation-soft-quality", action="store_true")
    parser.add_argument("--proposal-set-distillation-epochs", type=int, default=120)
    parser.add_argument("--proposal-set-distillation-batch-size", type=int, default=512)
    parser.add_argument("--proposal-set-distillation-train-max-per-video", type=int, default=120)
    parser.add_argument("--proposal-set-distillation-hard-negative-top-k", type=int, default=40)
    parser.add_argument("--proposal-set-distillation-train-min-score", type=float, default=None)
    parser.add_argument("--proposal-set-distillation-teacher-max-rescues", type=int, default=2)
    parser.add_argument("--proposal-set-distillation-teacher-max-base-iou", default="0")
    parser.add_argument("--proposal-set-distillation-teacher-nms-iou", default="0.3")
    parser.add_argument("--proposal-set-distillation-teacher-mode", choices=["oracle", "oracle_or_overlap", "oracle_or_topology"], default="oracle")
    parser.add_argument("--proposal-set-distillation-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--proposal-set-distillation-max-base-ious", default="0,0.05,0.1,0.25,none")
    parser.add_argument("--proposal-set-distillation-nms-ious", default="none,0.1,0.3,0.5")
    parser.add_argument("--proposal-set-distillation-length-penalties", default="0,0.005,0.01,0.02")
    parser.add_argument("--proposal-set-distillation-max-per-video", default="1,2,3")
    parser.add_argument("--proposal-set-distillation-topology-replacement", action="store_true")
    parser.add_argument("--proposal-set-distillation-topology-evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--proposal-set-distillation-topology-reducer", choices=["max", "mean", "stack"], default="stack")
    parser.add_argument("--proposal-set-distillation-topology-evidence-thresholds", default="0.4,0.5,0.6")
    parser.add_argument("--proposal-set-distillation-topology-min-student-scores", default="0.1,0.2,0.3")
    parser.add_argument("--proposal-set-distillation-topology-max-student-ranks", default="none,80")
    parser.add_argument("--proposal-set-distillation-topology-min-evidence-means", default="0.2,0.3,0.4")
    parser.add_argument("--proposal-set-distillation-topology-min-active-fractions", default="0,0.1,0.25")
    parser.add_argument("--proposal-set-distillation-topology-min-parent-contrasts", default="-0.3,-0.1,0")
    parser.add_argument("--proposal-set-distillation-topology-parent-min-lengths", default="12,24")
    parser.add_argument("--proposal-set-distillation-topology-max-child-parent-ratios", default="0.5,0.7")
    parser.add_argument("--proposal-set-distillation-topology-min-child-parent-coverages", default="0.5,0.7")
    parser.add_argument("--proposal-set-distillation-topology-min-gaps", default="0")
    parser.add_argument("--proposal-set-distillation-topology-child-score-thresholds", default="0.1,0.2")
    parser.add_argument("--proposal-set-distillation-topology-set-score-thresholds", default="0.1,0.2,0.3")
    parser.add_argument("--proposal-set-distillation-topology-student-weights", default="1")
    parser.add_argument("--proposal-set-distillation-topology-evidence-weights", default="0,0.5")
    parser.add_argument("--proposal-set-distillation-topology-active-fraction-weights", default="0,0.25")
    parser.add_argument("--proposal-set-distillation-topology-contrast-weights", default="0,0.5")
    parser.add_argument("--proposal-set-distillation-topology-length-penalties", default="0,0.005")
    parser.add_argument("--proposal-set-distillation-topology-nms-ious", default="0.3")
    parser.add_argument("--proposal-set-distillation-topology-min-children-per-parent", default="2")
    parser.add_argument("--proposal-set-distillation-topology-max-children-per-parent", default="2,3")
    parser.add_argument("--proposal-set-distillation-topology-max-replaced-parents-per-video", default="1")
    parser.add_argument("--proposal-set-distillation-topology-max-fp-increase", type=int, default=0)
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--selector-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selector-min-gaps", default="0,1,2,4,8")
    parser.add_argument("--selector-min-lengths", default="1,2,4,8")
    parser.add_argument("--nms-candidates", default="none,0.1,0.3,0.5")
    parser.add_argument("--protected-mainline-iou-candidates", default="0,0.1,0.25,0.5")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--include-prediction-labels", action="store_true")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    consensus_evidence_names = _parse_name_list(args.consensus_evidence_names)
    if args.consensus_denoise and not any(name in feature_names for name in consensus_evidence_names):
        raise ValueError(f"none of consensus evidence channels were found: {consensus_evidence_names}")
    clean_anchor_evidence_names = _parse_name_list(args.clean_anchor_evidence_names)
    if args.clean_anchor_rectify and not any(name in feature_names for name in clean_anchor_evidence_names):
        raise ValueError(f"none of clean-anchor evidence channels were found: {clean_anchor_evidence_names}")
    consensus_config = ConsensusDenoiseConfig(
        add_threshold=args.consensus_add_threshold,
        remove_threshold=args.consensus_remove_threshold,
        min_add_length=args.consensus_min_add_length,
        min_keep_length=args.consensus_min_keep_length,
        smooth_window=args.consensus_smooth_window,
        agreement_weight=args.consensus_agreement_weight,
        max_added_fraction=args.consensus_max_added_fraction,
    )
    clean_anchor_rectify_base_config = CleanAnchorRectifyConfig(
        evidence_names=clean_anchor_evidence_names,
        teacher_model=args.clean_anchor_teacher_model,
        positive_threshold=args.clean_anchor_positive_threshold,
        negative_threshold=args.clean_anchor_negative_threshold,
        add_threshold=args.clean_anchor_add_threshold,
        remove_threshold=args.clean_anchor_remove_threshold,
        min_add_length=args.clean_anchor_min_add_length,
        min_keep_length=args.clean_anchor_min_keep_length,
        smooth_window=args.clean_anchor_smooth_window,
        max_added_fraction=args.clean_anchor_max_added_fraction,
        positive_min_weight=args.clean_anchor_positive_min_weight,
        negative_min_weight=args.clean_anchor_negative_min_weight,
    )
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    if not channel_names:
        raise ValueError("no selected selector channel names were found in summary.json")
    thresholds = _parse_float_list(args.selector_thresholds)
    min_gaps = _parse_int_list(args.selector_min_gaps)
    min_lengths = _parse_int_list(args.selector_min_lengths)
    nms_candidates = _parse_nms_candidates(args.nms_candidates)
    protected_mainline_iou_candidates = [
        float(item.strip()) for item in args.protected_mainline_iou_candidates.split(",") if item.strip()
    ]
    fusion_aware_thresholds = _parse_float_list(args.fusion_aware_thresholds)
    fusion_aware_length_penalties = _parse_float_list(args.fusion_aware_length_penalties)
    clean_calibrated_thresholds = _parse_float_list(args.clean_calibrated_thresholds)
    clean_calibrated_raw_thresholds = _parse_float_list(args.clean_calibrated_raw_thresholds)
    clean_calibrated_clean_weights = _parse_float_list(args.clean_calibrated_clean_weights)
    clean_calibrated_raw_weights = _parse_float_list(args.clean_calibrated_raw_weights)
    clean_calibrated_max_base_ious = _parse_nms_candidates(args.clean_calibrated_max_base_ious)
    clean_calibrated_length_penalties = _parse_float_list(args.clean_calibrated_length_penalties)
    clean_calibrated_max_rescues_per_video = _parse_int_list(args.clean_calibrated_max_rescues_per_video)
    clean_calibrated_rescue_nms_ious = _parse_nms_candidates(args.clean_calibrated_rescue_nms_ious)
    graph_reranker_thresholds = _parse_float_list(args.graph_reranker_thresholds)
    graph_reranker_support_ious = _parse_float_list(args.graph_reranker_support_ious)
    graph_reranker_support_weights = _parse_float_list(args.graph_reranker_support_weights)
    graph_reranker_support_count_weights = _parse_float_list(args.graph_reranker_support_count_weights)
    graph_reranker_split_penalties = _parse_float_list(args.graph_reranker_split_penalties)
    graph_reranker_mainline_overlap_penalties = _parse_float_list(args.graph_reranker_mainline_overlap_penalties)
    graph_reranker_length_penalties = _parse_float_list(args.graph_reranker_length_penalties)
    dp_event_set_base_keep_scores = _parse_float_list(args.dp_event_set_base_keep_scores)
    dp_event_set_candidate_score_weights = _parse_float_list(args.dp_event_set_candidate_score_weights)
    dp_event_set_min_candidate_scores = _parse_float_list(args.dp_event_set_min_candidate_scores)
    dp_event_set_length_penalties = _parse_float_list(args.dp_event_set_length_penalties)
    dp_event_set_base_overlap_penalties = _parse_float_list(args.dp_event_set_base_overlap_penalties)
    ot_evidence_names = _parse_name_list(args.ot_evidence_names)
    if args.ot_event_set_matcher and not any(name in feature_names for name in ot_evidence_names):
        raise ValueError(f"none of OT evidence channels were found: {ot_evidence_names}")
    ot_score_thresholds = _parse_float_list(args.ot_score_thresholds)
    ot_base_keep_scores = _parse_float_list(args.ot_base_keep_scores)
    ot_evidence_thresholds = _parse_float_list(args.ot_evidence_thresholds)
    ot_selector_weights = _parse_float_list(args.ot_selector_weights)
    ot_contrast_weights = _parse_float_list(args.ot_contrast_weights)
    ot_active_fraction_weights = _parse_float_list(args.ot_active_fraction_weights)
    ot_agreement_weights = _parse_float_list(args.ot_agreement_weights)
    ot_island_coverage_weights = _parse_float_list(args.ot_island_coverage_weights)
    ot_peak_alignment_weights = _parse_float_list(args.ot_peak_alignment_weights)
    ot_base_overlap_penalties = _parse_float_list(args.ot_base_overlap_penalties)
    ot_length_penalties = _parse_float_list(args.ot_length_penalties)
    ot_smooth_windows = _parse_int_list(args.ot_smooth_windows)
    ot_min_evidence_island_lengths = _parse_int_list(args.ot_min_evidence_island_lengths)
    boundary_snap_min_scores = _parse_float_list(args.boundary_snap_min_scores)
    boundary_snap_coverages = _parse_float_list(args.boundary_snap_coverages)
    boundary_snap_max_ratios = _parse_float_list(args.boundary_snap_max_ratios)
    watershed_evidence_names = _parse_name_list(args.watershed_evidence_names)
    watershed_thresholds = _parse_float_list(args.watershed_thresholds)
    watershed_smooth_windows = _parse_int_list(args.watershed_smooth_windows)
    watershed_min_gaps = _parse_int_list(args.watershed_min_gaps)
    watershed_min_lengths = _parse_int_list(args.watershed_min_lengths)
    watershed_pads = _parse_int_list(args.watershed_pads)
    watershed_parent_min_lengths = _parse_int_list(args.watershed_parent_min_lengths)
    proposal_replacement_thresholds = _parse_float_list(args.proposal_replacement_thresholds)
    proposal_replacement_parent_min_lengths = _parse_int_list(args.proposal_replacement_parent_min_lengths)
    proposal_replacement_min_replacements = _parse_int_list(args.proposal_replacement_min_replacements)
    proposal_replacement_max_per_parent = _parse_int_list(args.proposal_replacement_max_per_parent)
    proposal_replacement_coverages = _parse_float_list(args.proposal_replacement_coverages)
    proposal_replacement_max_ratios = _parse_float_list(args.proposal_replacement_max_ratios)
    proposal_replacement_length_penalties = _parse_float_list(args.proposal_replacement_length_penalties)
    proposal_replacement_max_fp_increase = args.proposal_replacement_max_fp_increase
    candidate_rescue_thresholds = _parse_float_list(args.candidate_rescue_thresholds)
    candidate_rescue_mainline_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.candidate_rescue_mainline_ious.split(",")
        if item.strip()
    ]
    candidate_rescue_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.candidate_rescue_nms_ious.split(",")
        if item.strip()
    ]
    candidate_rescue_length_penalties = _parse_float_list(args.candidate_rescue_length_penalties)
    candidate_rescue_max_per_video = _parse_int_list(args.candidate_rescue_max_per_video)
    scale_completeness_evidence_names = _parse_name_list(args.scale_completeness_evidence_names)
    scale_completeness_prob_thresholds = _parse_float_list(args.scale_completeness_prob_thresholds)
    scale_completeness_evidence_thresholds = _parse_float_list(args.scale_completeness_evidence_thresholds)
    scale_completeness_min_active_fractions = _parse_float_list(args.scale_completeness_min_active_fractions)
    scale_completeness_min_means = _parse_float_list(args.scale_completeness_min_means)
    scale_completeness_min_contrasts = _parse_float_list(args.scale_completeness_min_contrasts)
    scale_completeness_max_base_ious = _parse_float_list(args.scale_completeness_max_base_ious)
    scale_completeness_length_penalties = _parse_float_list(args.scale_completeness_length_penalties)
    scale_completeness_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.scale_completeness_nms_ious.split(",")
        if item.strip()
    ]
    scale_completeness_max_per_video = _parse_int_list(args.scale_completeness_max_per_video)
    evidence_set_reasoner_evidence_names = _parse_name_list(args.evidence_set_reasoner_evidence_names)
    if args.evidence_set_reasoner and not any(name in feature_names for name in evidence_set_reasoner_evidence_names):
        raise ValueError(f"none of evidence-set-reasoner channels were found: {evidence_set_reasoner_evidence_names}")
    evidence_set_reasoner_min_selector_scores = _parse_float_list(args.evidence_set_reasoner_min_selector_scores)
    evidence_set_reasoner_evidence_thresholds = _parse_float_list(args.evidence_set_reasoner_evidence_thresholds)
    evidence_set_reasoner_min_evidence_means = _parse_float_list(args.evidence_set_reasoner_min_evidence_means)
    evidence_set_reasoner_min_active_fractions = _parse_float_list(args.evidence_set_reasoner_min_active_fractions)
    evidence_set_reasoner_min_contrasts = _parse_float_list(args.evidence_set_reasoner_min_contrasts)
    evidence_set_reasoner_score_thresholds = _parse_float_list(args.evidence_set_reasoner_score_thresholds)
    evidence_set_reasoner_selector_weights = _parse_float_list(args.evidence_set_reasoner_selector_weights)
    evidence_set_reasoner_contrast_weights = _parse_float_list(args.evidence_set_reasoner_contrast_weights)
    evidence_set_reasoner_active_fraction_weights = _parse_float_list(args.evidence_set_reasoner_active_fraction_weights)
    evidence_set_reasoner_consensus_weights = _parse_float_list(args.evidence_set_reasoner_consensus_weights)
    evidence_set_reasoner_support_ious = _parse_float_list(args.evidence_set_reasoner_support_ious)
    evidence_set_reasoner_support_count_weights = _parse_float_list(args.evidence_set_reasoner_support_count_weights)
    evidence_set_reasoner_max_base_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.evidence_set_reasoner_max_base_ious.split(",")
        if item.strip()
    ]
    evidence_set_reasoner_base_iou_penalties = _parse_float_list(args.evidence_set_reasoner_base_iou_penalties)
    evidence_set_reasoner_length_penalties = _parse_float_list(args.evidence_set_reasoner_length_penalties)
    evidence_set_reasoner_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.evidence_set_reasoner_nms_ious.split(",")
        if item.strip()
    ]
    evidence_set_reasoner_max_per_video = _parse_int_list(args.evidence_set_reasoner_max_per_video)
    boundary_distribution_thresholds = _parse_float_list(args.boundary_distribution_thresholds)
    boundary_distribution_base_coverages = _parse_float_list(args.boundary_distribution_base_coverages)
    boundary_distribution_max_ratios = _parse_float_list(args.boundary_distribution_max_ratios)
    boundary_distribution_start_quantiles = _parse_float_list(args.boundary_distribution_start_quantiles)
    boundary_distribution_end_quantiles = _parse_float_list(args.boundary_distribution_end_quantiles)
    boundary_distribution_base_weights = _parse_float_list(args.boundary_distribution_base_weights)
    boundary_distribution_max_shift_ratios = _parse_float_list(args.boundary_distribution_max_shift_ratios)
    residual_sub_event_evidence_names = _parse_name_list(args.residual_sub_event_evidence_names)
    residual_sub_event_min_selector_scores = _parse_float_list(args.residual_sub_event_min_selector_scores)
    residual_sub_event_evidence_thresholds = _parse_float_list(args.residual_sub_event_evidence_thresholds)
    residual_sub_event_min_evidence_means = _parse_float_list(args.residual_sub_event_min_evidence_means)
    residual_sub_event_min_active_fractions = _parse_float_list(args.residual_sub_event_min_active_fractions)
    residual_sub_event_min_parent_contrasts = _parse_float_list(args.residual_sub_event_min_parent_contrasts)
    residual_sub_event_parent_min_lengths = _parse_int_list(args.residual_sub_event_parent_min_lengths)
    residual_sub_event_max_child_parent_ratios = _parse_float_list(args.residual_sub_event_max_child_parent_ratios)
    residual_sub_event_min_child_parent_coverages = _parse_float_list(args.residual_sub_event_min_child_parent_coverages)
    residual_sub_event_length_penalties = _parse_float_list(args.residual_sub_event_length_penalties)
    residual_sub_event_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.residual_sub_event_nms_ious.split(",")
        if item.strip()
    ]
    residual_sub_event_max_per_parent = _parse_int_list(args.residual_sub_event_max_per_parent)
    residual_sub_event_max_per_video = _parse_int_list(args.residual_sub_event_max_per_video)
    residual_sub_event_modes = _parse_name_list(args.residual_sub_event_modes)
    residual_sub_event_min_per_parent = _parse_int_list(args.residual_sub_event_min_per_parent)
    event_topology_evidence_names = _parse_name_list(args.event_topology_evidence_names)
    event_topology_min_selector_scores = _parse_float_list(args.event_topology_min_selector_scores)
    event_topology_evidence_thresholds = _parse_float_list(args.event_topology_evidence_thresholds)
    event_topology_min_evidence_means = _parse_float_list(args.event_topology_min_evidence_means)
    event_topology_min_active_fractions = _parse_float_list(args.event_topology_min_active_fractions)
    event_topology_min_parent_contrasts = _parse_float_list(args.event_topology_min_parent_contrasts)
    event_topology_parent_min_lengths = _parse_int_list(args.event_topology_parent_min_lengths)
    event_topology_max_child_parent_ratios = _parse_float_list(args.event_topology_max_child_parent_ratios)
    event_topology_min_child_parent_coverages = _parse_float_list(args.event_topology_min_child_parent_coverages)
    event_topology_min_gaps = _parse_int_list(args.event_topology_min_gaps)
    event_topology_support_ious = _parse_float_list(args.event_topology_support_ious)
    event_topology_support_count_weights = _parse_float_list(args.event_topology_support_count_weights)
    event_topology_length_penalties = _parse_float_list(args.event_topology_length_penalties)
    event_topology_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.event_topology_nms_ious.split(",")
        if item.strip()
    ]
    event_topology_min_children_per_parent = _parse_int_list(args.event_topology_min_children_per_parent)
    event_topology_max_children_per_parent = _parse_int_list(args.event_topology_max_children_per_parent)
    event_topology_max_replaced_parents_per_video = _parse_int_list(args.event_topology_max_replaced_parents_per_video)
    topology_split_gate_evidence_names = _parse_name_list(args.topology_split_gate_evidence_names)
    topology_split_gate_thresholds = _parse_float_list(args.topology_split_gate_thresholds)
    topology_split_gate_parent_min_lengths = _parse_int_list(args.topology_split_gate_parent_min_lengths)
    topology_split_gate_min_child_parent_coverages = _parse_float_list(args.topology_split_gate_min_child_parent_coverages)
    topology_split_gate_max_child_parent_ratios = _parse_float_list(args.topology_split_gate_max_child_parent_ratios)
    topology_split_gate_max_children_per_parent = _parse_int_list(args.topology_split_gate_max_children_per_parent)
    topology_split_gate_max_replaced_parents_per_video = _parse_int_list(args.topology_split_gate_max_replaced_parents_per_video)
    topology_matching_evidence_names = _parse_name_list(args.topology_matching_evidence_names)
    topology_matching_thresholds = _parse_float_list(args.topology_matching_thresholds)
    topology_matching_parent_min_lengths = _parse_int_list(args.topology_matching_parent_min_lengths)
    topology_matching_min_child_parent_coverages = _parse_float_list(args.topology_matching_min_child_parent_coverages)
    topology_matching_max_child_parent_ratios = _parse_float_list(args.topology_matching_max_child_parent_ratios)
    topology_matching_max_children_per_parent = _parse_int_list(args.topology_matching_max_children_per_parent)
    topology_matching_max_set_proposals_per_parent = _parse_int_list(args.topology_matching_max_set_proposals_per_parent)
    topology_matching_max_child_pool_per_parent = _parse_int_list(args.topology_matching_max_child_pool_per_parent)
    topology_matching_max_replaced_parents_per_video = _parse_int_list(args.topology_matching_max_replaced_parents_per_video)
    topology_matching_risk_penalties = _parse_float_list(args.topology_matching_risk_penalties)
    topology_matching_safety_thresholds = _parse_float_list(args.topology_matching_safety_thresholds)
    fn_aware_thresholds = _parse_float_list(args.fn_aware_thresholds)
    fn_aware_max_base_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.fn_aware_max_base_ious.split(",")
        if item.strip()
    ]
    fn_aware_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.fn_aware_nms_ious.split(",")
        if item.strip()
    ]
    fn_aware_length_penalties = _parse_float_list(args.fn_aware_length_penalties)
    fn_aware_max_per_video = _parse_int_list(args.fn_aware_max_per_video)
    proposal_set_distillation_teacher_max_base_iou = _parse_optional_float(args.proposal_set_distillation_teacher_max_base_iou)
    proposal_set_distillation_teacher_nms_iou = _parse_optional_float(args.proposal_set_distillation_teacher_nms_iou)
    proposal_set_distillation_thresholds = _parse_float_list(args.proposal_set_distillation_thresholds)
    proposal_set_distillation_max_base_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.proposal_set_distillation_max_base_ious.split(",")
        if item.strip()
    ]
    proposal_set_distillation_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.proposal_set_distillation_nms_ious.split(",")
        if item.strip()
    ]
    proposal_set_distillation_length_penalties = _parse_float_list(args.proposal_set_distillation_length_penalties)
    proposal_set_distillation_max_per_video = _parse_int_list(args.proposal_set_distillation_max_per_video)
    proposal_set_distillation_topology_evidence_names = _parse_name_list(args.proposal_set_distillation_topology_evidence_names)
    proposal_set_distillation_topology_evidence_thresholds = _parse_float_list(args.proposal_set_distillation_topology_evidence_thresholds)
    proposal_set_distillation_topology_min_student_scores = _parse_float_list(args.proposal_set_distillation_topology_min_student_scores)
    proposal_set_distillation_topology_max_student_ranks = _parse_optional_int_list(args.proposal_set_distillation_topology_max_student_ranks)
    proposal_set_distillation_topology_min_evidence_means = _parse_float_list(args.proposal_set_distillation_topology_min_evidence_means)
    proposal_set_distillation_topology_min_active_fractions = _parse_float_list(args.proposal_set_distillation_topology_min_active_fractions)
    proposal_set_distillation_topology_min_parent_contrasts = _parse_float_list(args.proposal_set_distillation_topology_min_parent_contrasts)
    proposal_set_distillation_topology_parent_min_lengths = _parse_int_list(args.proposal_set_distillation_topology_parent_min_lengths)
    proposal_set_distillation_topology_max_child_parent_ratios = _parse_float_list(args.proposal_set_distillation_topology_max_child_parent_ratios)
    proposal_set_distillation_topology_min_child_parent_coverages = _parse_float_list(args.proposal_set_distillation_topology_min_child_parent_coverages)
    proposal_set_distillation_topology_min_gaps = _parse_int_list(args.proposal_set_distillation_topology_min_gaps)
    proposal_set_distillation_topology_child_score_thresholds = _parse_float_list(args.proposal_set_distillation_topology_child_score_thresholds)
    proposal_set_distillation_topology_set_score_thresholds = _parse_float_list(args.proposal_set_distillation_topology_set_score_thresholds)
    proposal_set_distillation_topology_student_weights = _parse_float_list(args.proposal_set_distillation_topology_student_weights)
    proposal_set_distillation_topology_evidence_weights = _parse_float_list(args.proposal_set_distillation_topology_evidence_weights)
    proposal_set_distillation_topology_active_fraction_weights = _parse_float_list(args.proposal_set_distillation_topology_active_fraction_weights)
    proposal_set_distillation_topology_contrast_weights = _parse_float_list(args.proposal_set_distillation_topology_contrast_weights)
    proposal_set_distillation_topology_length_penalties = _parse_float_list(args.proposal_set_distillation_topology_length_penalties)
    proposal_set_distillation_topology_nms_ious = [
        None if item.strip().lower() in {"none", "null"} else float(item.strip())
        for item in args.proposal_set_distillation_topology_nms_ious.split(",")
        if item.strip()
    ]
    proposal_set_distillation_topology_min_children_per_parent = _parse_int_list(args.proposal_set_distillation_topology_min_children_per_parent)
    proposal_set_distillation_topology_max_children_per_parent = _parse_int_list(args.proposal_set_distillation_topology_max_children_per_parent)
    proposal_set_distillation_topology_max_replaced_parents_per_video = _parse_int_list(args.proposal_set_distillation_topology_max_replaced_parents_per_video)
    if args.clean_calibrated_selector:
        print(
            f"clean_calibration data={args.clean_calibration_data_dir} "
            f"model={args.clean_calibration_model} target={args.clean_calibration_target} fold_local=true",
            flush=True,
        )
    folds = make_stratified_folds(records, args.folds, args.seed)
    all_indices = set(range(len(records)))
    source_records = records

    blend_aux_channel = None
    if args.blend_aux_name:
        blend_aux_channel = _feature_indices(feature_names, [args.blend_aux_name], "blend")[0]

    fold_summaries: list[dict] = []
    predictions_by_fold: list[list[list[tuple[int, int]]]] = []
    for fold_idx, val_idx in enumerate(folds):
        records = source_records
        train_idx = sorted(all_indices - set(val_idx))
        print(f"fold={fold_idx} train={len(train_idx)} val={len(val_idx)}", flush=True)
        clean_calibrator = None
        if args.clean_calibrated_selector:
            val_names_for_clean_exclusion = {source_records[idx].name for idx in val_idx}
            clean_calibrator = fit_clean_candidate_calibrator(
                args.clean_calibration_data_dir,
                channel_names=channel_names,
                thresholds=thresholds,
                min_gaps=min_gaps,
                min_lengths=min_lengths,
                iou_threshold=args.iou_threshold,
                model_name=args.clean_calibration_model,
                target=args.clean_calibration_target,
                seed=args.seed + 9000 + fold_idx,
                device=args.device,
                epochs=args.clean_calibration_epochs,
                batch_size=args.clean_calibration_batch_size,
                exclude_video_names=val_names_for_clean_exclusion,
            )
            print(
                f"fold={fold_idx} clean_calibration videos={clean_calibrator.n_train_videos} "
                f"excluded={len(clean_calibrator.excluded_video_names)} "
                f"candidates={clean_calibrator.n_train_candidates} "
                f"positives={clean_calibrator.n_positive_train_candidates}",
                flush=True,
            )
        consensus_summary = {"enabled": False}
        if args.consensus_denoise:
            records, consensus_stats = denoise_records_for_fold(
                source_records,
                train_idx=train_idx,
                feature_names=feature_names,
                evidence_names=consensus_evidence_names,
                agreement_name=args.consensus_agreement_name,
                config=consensus_config,
            )
            consensus_summary = {
                "enabled": True,
                "evidence_names": consensus_evidence_names,
                "agreement_name": args.consensus_agreement_name,
                **consensus_stats,
            }
            print(
                "consensus_denoise "
                f"changed_videos={consensus_summary['changed_videos']} "
                f"added_frames={consensus_summary['added_frames']} "
                f"removed_frames={consensus_summary['removed_frames']}",
                flush=True,
            )
        clean_anchor_rectify_summary = {"enabled": False}
        if args.clean_anchor_rectify:
            rect_config = CleanAnchorRectifyConfig(
                **{
                    **clean_anchor_rectify_base_config.__dict__,
                    "exclude_video_names": [source_records[idx].name for idx in val_idx],
                }
            )
            records, clean_anchor_rectify_summary = rectify_records_for_fold(
                records,
                train_idx=train_idx,
                feature_names=feature_names,
                clean_data_dir=args.clean_anchor_data_dir,
                config=rect_config,
            )
            print(
                "clean_anchor_rectify "
                f"changed_videos={clean_anchor_rectify_summary['changed_videos']} "
                f"added_frames={clean_anchor_rectify_summary['added_frames']} "
                f"removed_frames={clean_anchor_rectify_summary['removed_frames']} "
                f"teacher_videos={clean_anchor_rectify_summary['teacher_videos']}",
                flush=True,
            )
        clean_anchor_soft_weights_summary = {"enabled": False}
        frame_weights_by_idx = None
        if args.clean_anchor_soft_weights:
            weight_config = CleanAnchorRectifyConfig(
                **{
                    **clean_anchor_rectify_base_config.__dict__,
                    "exclude_video_names": [source_records[idx].name for idx in val_idx],
                }
            )
            frame_weights_by_idx, clean_anchor_soft_weights_summary = compute_clean_anchor_frame_weights_for_fold(
                records,
                train_idx=train_idx,
                feature_names=feature_names,
                clean_data_dir=args.clean_anchor_data_dir,
                config=weight_config,
            )
            print(
                "clean_anchor_soft_weights "
                f"videos={clean_anchor_soft_weights_summary['weighted_train_videos']} "
                f"frames={clean_anchor_soft_weights_summary['weighted_frames']} "
                f"mean_weight={clean_anchor_soft_weights_summary['mean_weight']:.3f} "
                f"min_weight={clean_anchor_soft_weights_summary['min_weight']:.3f}",
                flush=True,
            )
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
            lambda_boundary=args.lambda_boundary,
            architecture=args.architecture,
            frame_weights_by_idx=frame_weights_by_idx,
        )
        val_outputs = predict_records(model, records, val_idx, mean, std, args.device)
        mainline_metrics = pure_metrics
        if blend_aux_channel is not None:
            alpha, params, blended_metrics = select_blended_postprocess_params(
                val_outputs,
                records,
                val_idx,
                aux_channel=blend_aux_channel,
                alphas=_parse_float_list(args.blend_alphas),
                iou_threshold=args.iou_threshold,
            )
            val_outputs = blend_prediction_records(
                val_outputs,
                records,
                val_idx,
                aux_channel=blend_aux_channel,
                alpha_model=alpha,
            )
            mainline_metrics = {
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
        mainline_predictions = [_segments_from_params(output["probs"], mainline_metrics["params"]) for output in val_outputs]
        train_outputs = predict_records(model, records, train_idx, mean, std, args.device)
        if blend_aux_channel is not None:
            alpha_model = float(mainline_metrics.get("hybrid", {}).get("alpha_model", 1.0))
            train_outputs = blend_prediction_records(
                train_outputs,
                records,
                train_idx,
                aux_channel=blend_aux_channel,
                alpha_model=alpha_model,
            )
        train_mainline_predictions = [_segments_from_params(output["probs"], mainline_metrics["params"]) for output in train_outputs]
        mainline_probs_by_idx = {
            int(idx): np.asarray(output["probs"], dtype=np.float32)
            for idx, output in zip(val_idx, val_outputs)
        }
        mainline_probs_by_idx.update(
            {
                int(idx): np.asarray(output["probs"], dtype=np.float32)
                for idx, output in zip(train_idx, train_outputs)
            }
        )
        mainline_segments_by_idx = {int(idx): segments for idx, segments in zip(val_idx, mainline_predictions)}
        mainline_segments_by_idx.update({int(idx): segments for idx, segments in zip(train_idx, train_mainline_predictions)})

        classifier, selector_params, selector_val_metrics, n_candidates, n_positive = train_selector_for_fold(
            records,
            train_idx,
            val_idx,
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
            iou_threshold=args.iou_threshold,
            seed=args.seed + 1000 + fold_idx,
            model_name=args.selector_model,
            target=args.selector_target,
            device=args.device,
            selector_epochs=args.selector_epochs,
            selector_batch_size=args.selector_batch_size,
        )
        selector_predictions = predict_selector_fold(
            classifier,
            records,
            val_idx,
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
            selector_params,
        )
        labels = [records[idx].labels for idx in val_idx]
        raw_candidates = None
        raw_scores = None
        if (
            args.boundary_snap
            or args.candidate_rescue
            or args.proposal_replacement
            or args.fusion_aware_selector
            or args.clean_calibrated_selector
            or args.graph_reranker
            or args.dp_event_set_selector
            or args.ot_event_set_matcher
            or args.scale_completeness_rescue
            or args.evidence_set_reasoner
            or args.boundary_distribution_refine
            or args.residual_sub_event_expansion
            or args.event_topology_splitter
            or args.topology_split_gate
            or args.topology_matching_network
            or args.fn_aware_reranker
            or args.proposal_set_distillation
        ):
            raw_candidates = []
            raw_scores = []
            for idx in val_idx:
                candidates, scores = score_video_candidates(
                    classifier,
                    records[idx],
                    feature_names,
                    channel_names,
                    thresholds,
                    min_gaps,
                    min_lengths,
                )
                raw_candidates.append(candidates)
                raw_scores.append(scores)
        train_raw_candidates = None
        train_raw_scores = None
        if args.fn_aware_reranker or args.topology_split_gate or args.topology_matching_network or args.proposal_set_distillation:
            train_raw_candidates = []
            train_raw_scores = []
            for idx in train_idx:
                candidates, scores = score_video_candidates(
                    classifier,
                    records[idx],
                    feature_names,
                    channel_names,
                    thresholds,
                    min_gaps,
                    min_lengths,
                )
                train_raw_candidates.append(candidates)
                train_raw_scores.append(scores)
            train_active_mainline_predictions = list(train_mainline_predictions)
            train_active_selector_predictions = predict_selector_from_scored_candidates(
                train_raw_candidates,
                train_raw_scores,
                prob_threshold=float(selector_params["prob_threshold"]),
                length_penalty=float(selector_params["length_penalty"]),
            )
        boundary_snap_summary = {"enabled": False}
        if args.boundary_snap:
            snap_config, snap_metrics, snapped_predictions = select_boundary_snap_params(
                mainline_predictions,
                raw_candidates or [],
                raw_scores or [],
                labels,
                min_scores=boundary_snap_min_scores,
                min_mainline_coverages=boundary_snap_coverages,
                max_length_ratios=boundary_snap_max_ratios,
                iou_threshold=args.iou_threshold,
            )
            boundary_snap_summary = {
                "config": snap_config,
                "metrics": snap_metrics,
            }
            if snap_config.get("enabled"):
                mainline_predictions = snapped_predictions
                boundary_snap_summary["enabled"] = True
                mainline_metrics = {
                    "epoch": mainline_metrics.get("epoch"),
                    "loss": mainline_metrics.get("loss"),
                    "params": mainline_metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
                    **snap_metrics,
                    "boundary_snap": snap_config,
                    **({"hybrid": mainline_metrics["hybrid"]} if "hybrid" in mainline_metrics else {}),
                }
        best_nms, best_segment_metrics = select_fusion_nms_iou(
            mainline_predictions,
            selector_predictions,
            labels,
            candidates=nms_candidates,
            iou_threshold=args.iou_threshold,
        )
        watershed_split_summary = {"enabled": False}
        if args.watershed_split:
            watershed_evidence = [
                _reduce_evidence_channels(records[idx], feature_names, watershed_evidence_names, args.watershed_reducer)
                for idx in val_idx
            ]
            split_config, split_metrics, split_predictions = select_watershed_split_params(
                mainline_predictions,
                watershed_evidence,
                labels,
                evidence_thresholds=watershed_thresholds,
                smooth_windows=watershed_smooth_windows,
                min_gaps=watershed_min_gaps,
                min_lengths=watershed_min_lengths,
                pads=watershed_pads,
                parent_min_lengths=watershed_parent_min_lengths,
                iou_threshold=args.iou_threshold,
            )
            watershed_split_summary = {
                "config": split_config,
                "metrics": split_metrics,
            }
            if split_config.get("enabled"):
                mainline_predictions = split_predictions
                watershed_split_summary["enabled"] = True
                mainline_metrics = {
                    "epoch": mainline_metrics.get("epoch"),
                    "loss": mainline_metrics.get("loss"),
                    "params": mainline_metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
                    **split_metrics,
                    "watershed_split": {
                        **split_config,
                        "evidence_names": watershed_evidence_names,
                        "reducer": args.watershed_reducer,
                    },
                    **({"hybrid": mainline_metrics["hybrid"]} if "hybrid" in mainline_metrics else {}),
                }
            best_nms, best_segment_metrics = select_fusion_nms_iou(
                mainline_predictions,
                selector_predictions,
                labels,
                candidates=nms_candidates,
                iou_threshold=args.iou_threshold,
            )
        proposal_replacement_summary = {"enabled": False}
        if args.proposal_replacement:
            replacement_config, replacement_metrics, replacement_predictions = select_proposal_replacement_params(
                mainline_predictions,
                raw_candidates or [],
                raw_scores or [],
                labels,
                prob_thresholds=proposal_replacement_thresholds,
                parent_min_lengths=proposal_replacement_parent_min_lengths,
                min_replacements_list=proposal_replacement_min_replacements,
                max_replacements_per_parents=proposal_replacement_max_per_parent,
                min_candidate_parent_coverages=proposal_replacement_coverages,
                max_candidate_parent_ratios=proposal_replacement_max_ratios,
                length_penalties=proposal_replacement_length_penalties,
                max_fp_increase=proposal_replacement_max_fp_increase,
                iou_threshold=args.iou_threshold,
            )
            proposal_replacement_summary = {
                "config": replacement_config,
                "metrics": replacement_metrics,
            }
            if replacement_config.get("enabled"):
                mainline_predictions = replacement_predictions
                proposal_replacement_summary["enabled"] = True
                if args.fn_aware_reranker or args.topology_split_gate or args.topology_matching_network or args.proposal_set_distillation:
                    train_active_mainline_predictions = replace_merged_predictions(
                        train_active_mainline_predictions,
                        train_raw_candidates or [],
                        train_raw_scores or [],
                        prob_threshold=float(replacement_config["prob_threshold"]),
                        parent_min_length=int(replacement_config["parent_min_length"]),
                        min_replacements=int(replacement_config["min_replacements"]),
                        max_replacements_per_parent=int(replacement_config["max_replacements_per_parent"]),
                        min_candidate_parent_coverage=float(replacement_config["min_candidate_parent_coverage"]),
                        max_candidate_parent_ratio=float(replacement_config["max_candidate_parent_ratio"]),
                        length_penalty=float(replacement_config["length_penalty"]),
                    )
                mainline_metrics = {
                    "epoch": mainline_metrics.get("epoch"),
                    "loss": mainline_metrics.get("loss"),
                    "params": mainline_metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
                    **replacement_metrics,
                    "proposal_replacement": replacement_config,
                    **({"hybrid": mainline_metrics["hybrid"]} if "hybrid" in mainline_metrics else {}),
                }
            best_nms, best_segment_metrics = select_fusion_nms_iou(
                mainline_predictions,
                selector_predictions,
                labels,
                candidates=nms_candidates,
                iou_threshold=args.iou_threshold,
            )
        fusion_aware_selector_summary = {"enabled": False}
        if args.fusion_aware_selector:
            fusion_params, fusion_aware_metrics, fusion_aware_predictions = select_fusion_aware_selector_params_from_scores(
                mainline_predictions,
                raw_candidates or [],
                raw_scores or [],
                labels,
                prob_thresholds=fusion_aware_thresholds,
                length_penalties=fusion_aware_length_penalties,
                protected_mainline_iou_candidates=protected_mainline_iou_candidates,
                selector_nms_iou_candidates=nms_candidates,
                iou_threshold=args.iou_threshold,
            )
            fusion_aware_selector_summary = {
                "config": fusion_params,
                "metrics": fusion_aware_metrics,
                "baseline_selector_params": selector_params,
                "baseline_selector_validation": selector_val_metrics,
            }
            if fusion_params.get("enabled"):
                selector_predictions = fusion_aware_predictions
                selector_params = {
                    "prob_threshold": float(fusion_params["prob_threshold"]),
                    "length_penalty": float(fusion_params["length_penalty"]),
                    "fusion_aware": True,
                }
                if args.fn_aware_reranker or args.topology_split_gate or args.topology_matching_network or args.proposal_set_distillation:
                    train_active_selector_predictions = predict_selector_from_scored_candidates(
                        train_raw_candidates or [],
                        train_raw_scores or [],
                        prob_threshold=float(fusion_params["prob_threshold"]),
                        length_penalty=float(fusion_params["length_penalty"]),
                    )
                selector_val_metrics = evaluate_segment_predictions(
                    selector_predictions,
                    labels,
                    iou_threshold=args.iou_threshold,
                )
                fusion_aware_selector_summary["enabled"] = True
            best_nms, best_segment_metrics = select_fusion_nms_iou(
                mainline_predictions,
                selector_predictions,
                labels,
                candidates=nms_candidates,
                iou_threshold=args.iou_threshold,
            )
        graph_reranker_summary = {"enabled": False}
        if args.graph_reranker:
            graph_baseline_protected_config, graph_baseline_protected_metrics = select_protected_fusion_params(
                mainline_predictions,
                selector_predictions,
                labels,
                max_mainline_iou_candidates=protected_mainline_iou_candidates,
                selector_nms_iou_candidates=nms_candidates,
                iou_threshold=args.iou_threshold,
            )
            graph_current_metrics = graph_baseline_protected_metrics
            graph_config, graph_metrics, graph_selector_predictions = select_graph_reranker_params(
                mainline_predictions,
                raw_candidates or [],
                raw_scores or [],
                labels,
                prob_thresholds=graph_reranker_thresholds,
                support_ious=graph_reranker_support_ious,
                support_weights=graph_reranker_support_weights,
                support_count_weights=graph_reranker_support_count_weights,
                split_penalties=graph_reranker_split_penalties,
                mainline_overlap_penalties=graph_reranker_mainline_overlap_penalties,
                length_penalties=graph_reranker_length_penalties,
                protected_mainline_iou_candidates=protected_mainline_iou_candidates,
                selector_nms_iou_candidates=nms_candidates,
                iou_threshold=args.iou_threshold,
            )
            graph_reranker_summary = {
                "config": graph_config,
                "metrics": graph_metrics,
                "baseline_selector_params": selector_params,
                "baseline_selector_validation": selector_val_metrics,
                "current_fusion_metrics": graph_current_metrics,
                "current_protected": graph_baseline_protected_config,
            }
            if graph_config.get("enabled") and is_graph_reranker_improvement(graph_metrics, graph_current_metrics):
                selector_predictions = graph_selector_predictions
                selector_params = {
                    "prob_threshold": float(graph_config["prob_threshold"]),
                    "length_penalty": float(graph_config["length_penalty"]),
                    "graph_reranker": True,
                    "support_iou": float(graph_config["support_iou"]),
                    "support_weight": float(graph_config["support_weight"]),
                    "support_count_weight": float(graph_config["support_count_weight"]),
                    "split_penalty": float(graph_config["split_penalty"]),
                    "mainline_overlap_penalty": float(graph_config["mainline_overlap_penalty"]),
                }
                if args.fn_aware_reranker or args.topology_split_gate or args.topology_matching_network or args.proposal_set_distillation:
                    train_active_selector_predictions = select_graph_reranked_predictions(
                        train_active_mainline_predictions,
                        train_raw_candidates or [],
                        train_raw_scores or [],
                        prob_threshold=float(graph_config["prob_threshold"]),
                        support_iou=float(graph_config["support_iou"]),
                        support_weight=float(graph_config["support_weight"]),
                        support_count_weight=float(graph_config["support_count_weight"]),
                        split_penalty=float(graph_config["split_penalty"]),
                        mainline_overlap_penalty=float(graph_config["mainline_overlap_penalty"]),
                        length_penalty=float(graph_config["length_penalty"]),
                    )
                selector_val_metrics = evaluate_segment_predictions(
                    selector_predictions,
                    labels,
                    iou_threshold=args.iou_threshold,
                )
                graph_reranker_summary["enabled"] = True
            best_nms, best_segment_metrics = select_fusion_nms_iou(
                mainline_predictions,
                selector_predictions,
                labels,
                candidates=nms_candidates,
                iou_threshold=args.iou_threshold,
            )
        unprotected_predictions = fuse_segment_predictions(mainline_predictions, selector_predictions, nms_iou=best_nms)
        unprotected_metrics = evaluate_fused_predictions(unprotected_predictions, labels, iou_threshold=args.iou_threshold)
        protected_config, protected_metrics = select_protected_fusion_params(
            mainline_predictions,
            selector_predictions,
            labels,
            max_mainline_iou_candidates=protected_mainline_iou_candidates,
            selector_nms_iou_candidates=nms_candidates,
            iou_threshold=args.iou_threshold,
        )
        if protected_config.get("enabled"):
            fused_predictions = fuse_protected_segment_predictions(
                mainline_predictions,
                selector_predictions,
                max_mainline_iou=protected_config["max_mainline_iou"],
                selector_nms_iou=protected_config["selector_nms_iou"],
            )
        else:
            fused_predictions = mainline_predictions
        fusion_metrics = evaluate_fused_predictions(fused_predictions, labels, iou_threshold=args.iou_threshold)
        dp_event_set_summary = {"enabled": False, "reason": "disabled"}
        if args.dp_event_set_selector:
            dp_config, dp_metrics, dp_predictions = select_dp_event_set_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                labels,
                base_keep_scores=dp_event_set_base_keep_scores,
                candidate_score_weights=dp_event_set_candidate_score_weights,
                min_candidate_scores=dp_event_set_min_candidate_scores,
                length_penalties=dp_event_set_length_penalties,
                base_overlap_penalties=dp_event_set_base_overlap_penalties,
                max_fp_increase=args.dp_event_set_max_fp_increase,
                iou_threshold=args.iou_threshold,
            )
            dp_event_set_summary = {
                "enabled": False,
                "config": dp_config,
                "metrics": dp_metrics,
                "base": "protected_fusion",
            }
            dp_key = (
                float(dp_metrics["segment"]["f1"]),
                float(dp_metrics["segment"]["recall"]),
                float(dp_metrics["segment"]["precision"]),
                float(dp_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if dp_config.get("enabled") and dp_key > fusion_key:
                fused_predictions = dp_predictions
                fusion_metrics = dp_metrics
                dp_event_set_summary["enabled"] = True
        ot_event_set_summary = {"enabled": False, "reason": "disabled"}
        if args.ot_event_set_matcher:
            ot_evidence = [
                _event_topology_evidence(records[idx], feature_names, ot_evidence_names, args.ot_reducer)
                for idx in val_idx
            ]
            ot_config, ot_metrics, ot_predictions = select_ot_event_set_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                evidence=ot_evidence,
                labels=labels,
                score_thresholds=ot_score_thresholds,
                base_keep_scores=ot_base_keep_scores,
                evidence_thresholds=ot_evidence_thresholds,
                selector_weights=ot_selector_weights,
                contrast_weights=ot_contrast_weights,
                active_fraction_weights=ot_active_fraction_weights,
                agreement_weights=ot_agreement_weights,
                island_coverage_weights=ot_island_coverage_weights,
                peak_alignment_weights=ot_peak_alignment_weights,
                base_overlap_penalties=ot_base_overlap_penalties,
                length_penalties=ot_length_penalties,
                smooth_windows=ot_smooth_windows,
                min_evidence_island_lengths=ot_min_evidence_island_lengths,
                max_fp_increase=args.ot_max_fp_increase,
                iou_threshold=args.iou_threshold,
            )
            ot_event_set_summary = {
                "enabled": False,
                "config": ot_config,
                "metrics": ot_metrics,
                "base": "current_fused_predictions",
                "evidence_names": ot_evidence_names,
                "reducer": args.ot_reducer,
            }
            ot_key = (
                float(ot_metrics["segment"]["f1"]),
                float(ot_metrics["segment"]["recall"]),
                float(ot_metrics["segment"]["precision"]),
                float(ot_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if ot_config.get("enabled") and ot_key > fusion_key:
                fused_predictions = ot_predictions
                fusion_metrics = ot_metrics
                ot_event_set_summary["enabled"] = True
        clean_calibrated_selector_summary = {"enabled": False, "reason": "disabled"}
        if args.clean_calibrated_selector:
            if clean_calibrator is None:
                raise RuntimeError("clean_calibrated_selector enabled without a fitted clean calibrator")
            clean_scores = [
                score_clean_calibrated_candidates(
                    clean_calibrator.model,
                    records[idx],
                    candidates,
                    feature_names,
                    clean_calibrator.channel_names,
                )
                for idx, candidates in zip(val_idx, raw_candidates or [])
            ]
            clean_config, clean_metrics, clean_predictions = select_clean_calibrated_event_set_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                clean_scores,
                labels,
                clean_thresholds=clean_calibrated_thresholds,
                raw_thresholds=clean_calibrated_raw_thresholds,
                clean_weights=clean_calibrated_clean_weights,
                raw_weights=clean_calibrated_raw_weights,
                max_base_ious=clean_calibrated_max_base_ious,
                length_penalties=clean_calibrated_length_penalties,
                max_rescues_per_videos=clean_calibrated_max_rescues_per_video,
                rescue_nms_ious=clean_calibrated_rescue_nms_ious,
                max_fp_increase=args.clean_calibrated_max_fp_increase,
                iou_threshold=args.iou_threshold,
            )
            clean_calibrated_selector_summary = {
                "enabled": False,
                "config": clean_config,
                "metrics": clean_metrics,
                "base": "protected_fusion",
                "clean_calibration_data_dir": args.clean_calibration_data_dir,
                "clean_calibration_model": args.clean_calibration_model,
                "clean_calibration_target": args.clean_calibration_target,
                "n_clean_train_candidates": clean_calibrator.n_train_candidates,
                "n_clean_positive_train_candidates": clean_calibrator.n_positive_train_candidates,
                "n_clean_train_videos": clean_calibrator.n_train_videos,
                "excluded_clean_video_names": clean_calibrator.excluded_video_names,
                "channel_names": clean_calibrator.channel_names,
            }
            clean_key = (
                float(clean_metrics["segment"]["f1"]),
                float(clean_metrics["segment"]["precision"]),
                float(clean_metrics["segment"]["recall"]),
                float(clean_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if clean_config.get("enabled") and clean_key > fusion_key:
                fused_predictions = clean_predictions
                fusion_metrics = clean_metrics
                clean_calibrated_selector_summary["enabled"] = True
        topology_split_gate_summary = {"enabled": False, "reason": "disabled"}
        if args.topology_split_gate:
            if protected_config.get("enabled"):
                train_gate_base_predictions = fuse_protected_segment_predictions(
                    train_active_mainline_predictions,
                    train_active_selector_predictions,
                    max_mainline_iou=protected_config["max_mainline_iou"],
                    selector_nms_iou=protected_config["selector_nms_iou"],
                )
            else:
                train_gate_base_predictions = train_active_mainline_predictions
            train_gate_evidence = [
                _event_topology_evidence(
                    records[idx],
                    feature_names,
                    topology_split_gate_evidence_names,
                    args.topology_split_gate_reducer,
                )
                for idx in train_idx
            ]
            val_gate_evidence = [
                _event_topology_evidence(
                    records[idx],
                    feature_names,
                    topology_split_gate_evidence_names,
                    args.topology_split_gate_reducer,
                )
                for idx in val_idx
            ]
            gate_config, gate_metrics, gate_predictions = select_topology_split_gate_params(
                train_gate_base_predictions,
                train_raw_candidates or [],
                train_raw_scores or [],
                train_gate_evidence,
                [records[idx].labels for idx in train_idx],
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                val_gate_evidence,
                labels,
                model_name=args.topology_split_gate_model,
                seed=args.seed + 4000 + fold_idx,
                thresholds=topology_split_gate_thresholds,
                parent_min_lengths=topology_split_gate_parent_min_lengths,
                min_child_parent_coverages=topology_split_gate_min_child_parent_coverages,
                max_child_parent_ratios=topology_split_gate_max_child_parent_ratios,
                max_children_per_parents=topology_split_gate_max_children_per_parent,
                max_replaced_parents_per_videos=topology_split_gate_max_replaced_parents_per_video,
                max_fp_increase=args.topology_split_gate_max_fp_increase,
                use_synthetic_positives=args.topology_split_gate_synthetic,
                synthetic_max_gt_gap=args.topology_split_gate_synthetic_max_gt_gap,
                synthetic_parent_pad=args.topology_split_gate_synthetic_parent_pad,
                iou_threshold=args.iou_threshold,
            )
            topology_split_gate_summary = {
                "enabled": False,
                "config": gate_config,
                "metrics": gate_metrics,
                "base": "current_fused_predictions",
                "evidence_names": topology_split_gate_evidence_names,
                "reducer": args.topology_split_gate_reducer,
                "model": args.topology_split_gate_model,
            }
            gate_key = (
                float(gate_metrics["segment"]["f1"]),
                float(gate_metrics["segment"]["precision"]),
                float(gate_metrics["segment"]["recall"]),
                float(gate_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if gate_config.get("enabled") and gate_key > fusion_key:
                fused_predictions = gate_predictions
                fusion_metrics = gate_metrics
                topology_split_gate_summary["enabled"] = True
        topology_matching_summary = {"enabled": False, "reason": "disabled"}
        if args.topology_matching_network:
            if protected_config.get("enabled"):
                train_matching_base_predictions = fuse_protected_segment_predictions(
                    train_active_mainline_predictions,
                    train_active_selector_predictions,
                    max_mainline_iou=protected_config["max_mainline_iou"],
                    selector_nms_iou=protected_config["selector_nms_iou"],
                )
            else:
                train_matching_base_predictions = train_active_mainline_predictions
            train_matching_evidence = [
                _event_topology_evidence(
                    records[idx],
                    feature_names,
                    topology_matching_evidence_names,
                    args.topology_matching_reducer,
                )
                for idx in train_idx
            ]
            val_matching_evidence = [
                _event_topology_evidence(
                    records[idx],
                    feature_names,
                    topology_matching_evidence_names,
                    args.topology_matching_reducer,
                )
                for idx in val_idx
            ]
            matching_config, matching_metrics, matching_predictions = select_topology_matching_params(
                train_matching_base_predictions,
                train_raw_candidates or [],
                train_raw_scores or [],
                train_matching_evidence,
                [records[idx].labels for idx in train_idx],
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                val_matching_evidence,
                labels,
                model_name=args.topology_matching_model,
                seed=args.seed + 6000 + fold_idx,
                thresholds=topology_matching_thresholds,
                parent_min_lengths=topology_matching_parent_min_lengths,
                min_child_parent_coverages=topology_matching_min_child_parent_coverages,
                max_child_parent_ratios=topology_matching_max_child_parent_ratios,
                max_children_per_parents=topology_matching_max_children_per_parent,
                max_set_proposals_per_parents=topology_matching_max_set_proposals_per_parent,
                max_child_pool_per_parents=topology_matching_max_child_pool_per_parent,
                max_replaced_parents_per_videos=topology_matching_max_replaced_parents_per_video,
                max_fp_increase=args.topology_matching_max_fp_increase,
                risk_aware=args.topology_matching_risk_aware,
                risk_penalties=topology_matching_risk_penalties,
                safety_thresholds=topology_matching_safety_thresholds,
                safety_weight=args.topology_matching_safety_weight,
                iou_threshold=args.iou_threshold,
                device=args.device,
                epochs=args.topology_matching_epochs,
                batch_size=args.topology_matching_batch_size,
            )
            topology_matching_summary = {
                "enabled": False,
                "config": matching_config,
                "metrics": matching_metrics,
                "base": "current_fused_predictions",
                "evidence_names": topology_matching_evidence_names,
                "reducer": args.topology_matching_reducer,
                "model": args.topology_matching_model,
                "risk_aware": args.topology_matching_risk_aware,
            }
            matching_key = (
                float(matching_metrics["segment"]["f1"]),
                float(matching_metrics["segment"]["precision"]),
                float(matching_metrics["segment"]["recall"]),
                float(matching_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if matching_config.get("enabled") and matching_key > fusion_key:
                fused_predictions = matching_predictions
                fusion_metrics = matching_metrics
                topology_matching_summary["enabled"] = True
        event_topology_splitter_summary = {"enabled": False, "reason": "disabled"}
        if args.event_topology_splitter:
            topology_evidence = [
                _event_topology_evidence(
                    records[idx],
                    feature_names,
                    event_topology_evidence_names,
                    args.event_topology_reducer,
                )
                for idx in val_idx
            ]
            topology_config, topology_metrics, topology_predictions = select_event_topology_split_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                topology_evidence,
                labels,
                min_selector_scores=event_topology_min_selector_scores,
                evidence_thresholds=event_topology_evidence_thresholds,
                min_evidence_means=event_topology_min_evidence_means,
                min_active_fractions=event_topology_min_active_fractions,
                min_parent_contrasts=event_topology_min_parent_contrasts,
                parent_min_lengths=event_topology_parent_min_lengths,
                max_child_parent_ratios=event_topology_max_child_parent_ratios,
                min_child_parent_coverages=event_topology_min_child_parent_coverages,
                min_gap_between_children_values=event_topology_min_gaps,
                support_ious=event_topology_support_ious,
                support_count_weights=event_topology_support_count_weights,
                length_penalties=event_topology_length_penalties,
                nms_ious=event_topology_nms_ious,
                min_children_per_parents=event_topology_min_children_per_parent,
                max_children_per_parents=event_topology_max_children_per_parent,
                max_replaced_parents_per_videos=event_topology_max_replaced_parents_per_video,
                max_fp_increase=args.event_topology_max_fp_increase,
                iou_threshold=args.iou_threshold,
            )
            event_topology_splitter_summary = {
                "enabled": False,
                "config": topology_config,
                "metrics": topology_metrics,
                "base": "current_fused_predictions",
                "evidence_names": event_topology_evidence_names,
                "reducer": args.event_topology_reducer,
            }
            topology_key = (
                float(topology_metrics["segment"]["f1"]),
                float(topology_metrics["segment"]["precision"]),
                float(topology_metrics["segment"]["recall"]),
                float(topology_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if topology_config.get("enabled") and topology_key > fusion_key:
                fused_predictions = topology_predictions
                fusion_metrics = topology_metrics
                event_topology_splitter_summary["enabled"] = True
        residual_sub_event_summary = {"enabled": False, "reason": "disabled"}
        if args.residual_sub_event_expansion:
            residual_evidence = [
                _reduce_evidence_channels(
                    records[idx],
                    feature_names,
                    residual_sub_event_evidence_names,
                    args.residual_sub_event_reducer,
                )
                for idx in val_idx
            ]
            residual_config, residual_metrics, residual_predictions = select_residual_sub_event_expansion_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                residual_evidence,
                labels,
                min_selector_scores=residual_sub_event_min_selector_scores,
                evidence_thresholds=residual_sub_event_evidence_thresholds,
                min_evidence_means=residual_sub_event_min_evidence_means,
                min_active_fractions=residual_sub_event_min_active_fractions,
                min_parent_contrasts=residual_sub_event_min_parent_contrasts,
                parent_min_lengths=residual_sub_event_parent_min_lengths,
                max_child_parent_ratios=residual_sub_event_max_child_parent_ratios,
                min_child_parent_coverages=residual_sub_event_min_child_parent_coverages,
                length_penalties=residual_sub_event_length_penalties,
                nms_ious=residual_sub_event_nms_ious,
                max_sub_events_per_parents=residual_sub_event_max_per_parent,
                max_sub_events_per_videos=residual_sub_event_max_per_video,
                modes=residual_sub_event_modes,
                min_sub_events_per_parents=residual_sub_event_min_per_parent,
                iou_threshold=args.iou_threshold,
            )
            residual_sub_event_summary = {
                "enabled": False,
                "config": residual_config,
                "metrics": residual_metrics,
                "evidence_names": residual_sub_event_evidence_names,
                "reducer": args.residual_sub_event_reducer,
            }
            residual_key = (
                float(residual_metrics["segment"]["f1"]),
                float(residual_metrics["segment"]["precision"]),
                float(residual_metrics["segment"]["recall"]),
                float(residual_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if residual_config.get("enabled") and residual_key > fusion_key:
                fused_predictions = residual_predictions
                fusion_metrics = residual_metrics
                residual_sub_event_summary["enabled"] = True
        if args.fn_aware_reranker:
            if protected_config.get("enabled"):
                train_active_base_predictions = fuse_protected_segment_predictions(
                    train_active_mainline_predictions,
                    train_active_selector_predictions,
                    max_mainline_iou=protected_config["max_mainline_iou"],
                    selector_nms_iou=protected_config["selector_nms_iou"],
                )
            else:
                train_active_base_predictions = train_active_mainline_predictions
            if residual_sub_event_summary.get("enabled"):
                train_residual_evidence = [
                    _reduce_evidence_channels(
                        records[idx],
                        feature_names,
                        residual_sub_event_evidence_names,
                        args.residual_sub_event_reducer,
                    )
                    for idx in train_idx
                ]
                residual_config = residual_sub_event_summary["config"]
                train_active_base_predictions = select_residual_sub_event_predictions(
                    train_active_base_predictions,
                    train_raw_candidates or [],
                    train_raw_scores or [],
                    train_residual_evidence,
                    min_selector_score=float(residual_config["min_selector_score"]),
                    evidence_threshold=float(residual_config["evidence_threshold"]),
                    min_evidence_mean=float(residual_config["min_evidence_mean"]),
                    min_active_fraction=float(residual_config["min_active_fraction"]),
                    min_parent_contrast=float(residual_config["min_parent_contrast"]),
                    parent_min_length=int(residual_config["parent_min_length"]),
                    max_child_parent_ratio=float(residual_config["max_child_parent_ratio"]),
                    min_child_parent_coverage=float(residual_config["min_child_parent_coverage"]),
                    length_penalty=float(residual_config["length_penalty"]),
                    nms_iou=residual_config["nms_iou"],
                    max_sub_events_per_parent=int(residual_config["max_sub_events_per_parent"]),
                    max_sub_events_per_video=int(residual_config["max_sub_events_per_video"]),
                    mode=str(residual_config.get("mode", "add")),
                    min_sub_events_per_parent=int(residual_config.get("min_sub_events_per_parent", 2)),
                )
        fn_aware_summary = {"enabled": False, "reason": "disabled"}
        if args.fn_aware_reranker:
            train_candidate_by_idx = {
                int(idx): candidates for idx, candidates in zip(train_idx, train_raw_candidates or [])
            }
            train_scores_by_idx = {
                int(idx): scores for idx, scores in zip(train_idx, train_raw_scores or [])
            }
            train_base_by_idx = {
                int(idx): segments for idx, segments in zip(train_idx, train_active_base_predictions)
            }
            fn_candidates = build_fn_aware_candidate_records(
                records,
                train_idx,
                candidate_predictions_by_idx=train_candidate_by_idx,
                candidate_scores_by_idx=train_scores_by_idx,
                base_segments_by_idx=train_base_by_idx,
                feature_names=feature_names,
                channel_names=channel_names,
                iou_threshold=args.iou_threshold,
                max_records_per_video=args.fn_aware_train_max_per_video,
                min_selector_score=args.fn_aware_train_min_score,
            )
            if fn_candidates:
                x_fn = np.stack([item.features for item in fn_candidates])
                y_fn = np.asarray([item.label for item in fn_candidates], dtype=np.int64)
                if len(np.unique(y_fn)) >= 2:
                    print(
                        f"fold={fold_idx} fn_aware_train candidates={len(fn_candidates)} "
                        f"positives={int(y_fn.sum())} model={args.fn_aware_model} device={args.device}",
                        flush=True,
                    )
                    fn_model = fit_fn_aware_reranker(
                        x_fn,
                        y_fn,
                        seed=args.seed + 3000 + fold_idx,
                        model_name=args.fn_aware_model,
                        device=args.device,
                        epochs=args.fn_aware_epochs,
                        batch_size=args.fn_aware_batch_size,
                    )
                    train_fn_scores = [
                        score_fn_aware_candidates(
                            fn_model,
                            records[idx],
                            candidates,
                            scores,
                            feature_names,
                            channel_names,
                            base_segments,
                        )
                        for idx, candidates, scores, base_segments in zip(
                            train_idx,
                            train_raw_candidates or [],
                            train_raw_scores or [],
                            train_active_base_predictions,
                        )
                    ]
                    fn_scores = [
                        score_fn_aware_candidates(
                            fn_model,
                            records[idx],
                            candidates,
                            scores,
                            feature_names,
                            channel_names,
                            base_segments,
                        )
                        for idx, candidates, scores, base_segments in zip(
                            val_idx,
                            raw_candidates or [],
                            raw_scores or [],
                            fused_predictions,
                        )
                    ]
                    acceptance_summary = {"enabled": False, "reason": "disabled"}
                    active_fn_scores = fn_scores
                    calibrated_fn_config = None
                    calibrated_fn_metrics = None
                    calibrated_fn_predictions = None
                    if args.fn_aware_acceptance_distill:
                        train_fn_scores_by_idx = {
                            int(idx): scores for idx, scores in zip(train_idx, train_fn_scores)
                        }
                        acceptance_train_idx = list(train_idx)
                        acceptance_calibration_idx: list[int] = []
                        if args.fn_aware_acceptance_inner_calibration:
                            acceptance_train_idx, acceptance_calibration_idx = split_indices_for_inner_calibration(
                                train_idx,
                                {int(idx): records[idx].labels for idx in train_idx},
                                calibration_fraction=args.fn_aware_acceptance_calibration_fraction,
                                seed=args.seed + args.fn_aware_acceptance_calibration_seed_offset + fold_idx,
                            )
                        acceptance_labels = label_fn_acceptance_oracle(
                            [train_base_by_idx[int(idx)] for idx in acceptance_train_idx],
                            [train_candidate_by_idx[int(idx)] for idx in acceptance_train_idx],
                            [train_fn_scores_by_idx[int(idx)] for idx in acceptance_train_idx],
                            [records[idx].labels for idx in acceptance_train_idx],
                            threshold=float(args.fn_aware_acceptance_threshold),
                            max_base_iou=None,
                            nms_iou=None,
                            length_penalty=0.0,
                            max_rescues_per_video=int(args.fn_aware_acceptance_max_rescues_per_video),
                            max_fp_increase=int(args.fn_aware_acceptance_max_fp_increase),
                            max_candidates_per_video=args.fn_aware_acceptance_max_candidates_per_video,
                            iou_threshold=args.iou_threshold,
                        )
                        train_acceptance_labels_by_idx = {
                            int(idx): labels_for_video for idx, labels_for_video in zip(acceptance_train_idx, acceptance_labels)
                        }
                        train_aux_positive_labels_by_idx = None
                        if args.fn_aware_acceptance_use_aux_positives:
                            train_aux_positive_labels_by_idx = {}
                            for idx, candidates, base_segments in zip(
                                acceptance_train_idx,
                                [train_candidate_by_idx[int(idx)] for idx in acceptance_train_idx],
                                [train_base_by_idx[int(idx)] for idx in acceptance_train_idx],
                            ):
                                aux_labels, _ = label_fn_aware_candidates(
                                    candidates,
                                    records[idx].labels,
                                    base_segments,
                                    iou_threshold=args.iou_threshold,
                                )
                                train_aux_positive_labels_by_idx[int(idx)] = aux_labels
                        acceptance_records = build_fn_acceptance_candidate_records(
                            records,
                            acceptance_train_idx,
                            candidate_predictions_by_idx=train_candidate_by_idx,
                            raw_scores_by_idx=train_scores_by_idx,
                            first_stage_scores_by_idx=train_fn_scores_by_idx,
                            base_segments_by_idx=train_base_by_idx,
                            acceptance_labels_by_idx=train_acceptance_labels_by_idx,
                            feature_names=feature_names,
                            channel_names=channel_names,
                            max_records_per_video=args.fn_aware_acceptance_train_max_per_video,
                            aux_positive_labels_by_idx=train_aux_positive_labels_by_idx,
                        )
                        if acceptance_records:
                            x_accept = np.stack([item.features for item in acceptance_records])
                            y_accept = np.asarray([item.label for item in acceptance_records], dtype=np.int64)
                            if len(np.unique(y_accept)) >= 2:
                                print(
                                    f"fold={fold_idx} fn_acceptance_train candidates={len(acceptance_records)} "
                                    f"positives={int(y_accept.sum())} model={args.fn_aware_acceptance_model}",
                                    flush=True,
                                )
                                acceptance_model = fit_fn_aware_reranker(
                                    x_accept,
                                    y_accept,
                                    seed=args.seed + 3500 + fold_idx,
                                    model_name=args.fn_aware_acceptance_model,
                                    device=args.device,
                                    epochs=args.fn_aware_epochs,
                                    batch_size=args.fn_aware_batch_size,
                                )
                                val_candidate_by_idx = {
                                    int(idx): candidates for idx, candidates in zip(val_idx, raw_candidates or [])
                                }
                                val_scores_by_idx = {
                                    int(idx): scores for idx, scores in zip(val_idx, raw_scores or [])
                                }
                                val_first_stage_scores_by_idx = {
                                    int(idx): scores for idx, scores in zip(val_idx, fn_scores)
                                }
                                val_base_by_idx = {
                                    int(idx): segments for idx, segments in zip(val_idx, fused_predictions)
                                }
                                active_fn_scores = score_fn_acceptance_candidates(
                                    acceptance_model,
                                    records,
                                    val_idx,
                                    candidate_predictions_by_idx=val_candidate_by_idx,
                                    raw_scores_by_idx=val_scores_by_idx,
                                    first_stage_scores_by_idx=val_first_stage_scores_by_idx,
                                    base_segments_by_idx=val_base_by_idx,
                                    feature_names=feature_names,
                                    channel_names=channel_names,
                                )
                                calibration_summary = {"enabled": False, "reason": "disabled"}
                                if args.fn_aware_acceptance_inner_calibration:
                                    if acceptance_calibration_idx:
                                        calibration_scores = score_fn_acceptance_candidates(
                                            acceptance_model,
                                            records,
                                            acceptance_calibration_idx,
                                            candidate_predictions_by_idx=train_candidate_by_idx,
                                            raw_scores_by_idx=train_scores_by_idx,
                                            first_stage_scores_by_idx=train_fn_scores_by_idx,
                                            base_segments_by_idx=train_base_by_idx,
                                            feature_names=feature_names,
                                            channel_names=channel_names,
                                        )
                                        calibration_config, calibration_metrics, _ = select_strict_fn_aware_reranker_params(
                                            [train_base_by_idx[int(idx)] for idx in acceptance_calibration_idx],
                                            [train_candidate_by_idx[int(idx)] for idx in acceptance_calibration_idx],
                                            calibration_scores,
                                            [records[idx].labels for idx in acceptance_calibration_idx],
                                            thresholds=fn_aware_thresholds,
                                            max_base_ious=fn_aware_max_base_ious,
                                            nms_ious=fn_aware_nms_ious,
                                            length_penalties=fn_aware_length_penalties,
                                            max_rescues_per_videos=fn_aware_max_per_video,
                                            max_fp_increase=args.fn_aware_acceptance_max_fp_increase,
                                            max_candidates_per_video=args.fn_aware_max_candidates_per_video,
                                            iou_threshold=args.iou_threshold,
                                        )
                                        calibration_summary = {
                                            "enabled": True,
                                            "train_inner_videos": int(len(acceptance_train_idx)),
                                            "calibration_videos": int(len(acceptance_calibration_idx)),
                                            "config": calibration_config,
                                            "metrics": calibration_metrics,
                                            "max_fp_increase": int(args.fn_aware_acceptance_max_fp_increase),
                                        }
                                        calibrated_fn_config = {
                                            **calibration_config,
                                            "inner_calibrated": True,
                                            "calibration_metrics": calibration_metrics,
                                            "calibration_videos": int(len(acceptance_calibration_idx)),
                                        }
                                        calibrated_fn_predictions = apply_fn_aware_reranker_config(
                                            fused_predictions,
                                            raw_candidates or [],
                                            active_fn_scores,
                                            calibrated_fn_config,
                                        )
                                        calibrated_fn_metrics = evaluate_fused_predictions(
                                            calibrated_fn_predictions,
                                            labels,
                                            iou_threshold=args.iou_threshold,
                                        )
                                    else:
                                        calibration_summary = {"enabled": False, "reason": "empty_calibration_split"}
                                acceptance_summary = {
                                    "enabled": True,
                                    "model": args.fn_aware_acceptance_model,
                                    "n_train_candidates": int(len(acceptance_records)),
                                    "n_positive_train_candidates": int(y_accept.sum()),
                                    "inner_calibration": bool(args.fn_aware_acceptance_inner_calibration),
                                    "inner_train_videos": int(len(acceptance_train_idx)),
                                    "calibration_videos": int(len(acceptance_calibration_idx)),
                                    "teacher_threshold": float(args.fn_aware_acceptance_threshold),
                                    "teacher_max_rescues_per_video": int(args.fn_aware_acceptance_max_rescues_per_video),
                                    "teacher_max_candidates_per_video": args.fn_aware_acceptance_max_candidates_per_video,
                                    "teacher_max_fp_increase": int(args.fn_aware_acceptance_max_fp_increase),
                                    "use_aux_positives": bool(args.fn_aware_acceptance_use_aux_positives),
                                    "calibration": calibration_summary,
                                }
                            else:
                                acceptance_summary = {
                                    "enabled": False,
                                    "reason": "single_class_acceptance_labels",
                                    "n_train_candidates": int(len(acceptance_records)),
                                    "n_positive_train_candidates": int(y_accept.sum()),
                                    "inner_calibration": bool(args.fn_aware_acceptance_inner_calibration),
                                    "inner_train_videos": int(len(acceptance_train_idx)),
                                    "calibration_videos": int(len(acceptance_calibration_idx)),
                                }
                    if args.fn_aware_acceptance_inner_calibration:
                        if calibrated_fn_config is None:
                            calibrated_fn_config = {
                                "enabled": False,
                                "reason": acceptance_summary.get("reason", "inner_calibration_unavailable"),
                                "inner_calibrated": True,
                            }
                            calibrated_fn_metrics = evaluate_fused_predictions(
                                fused_predictions,
                                labels,
                                iou_threshold=args.iou_threshold,
                            )
                            calibrated_fn_predictions = fused_predictions
                        fn_config = calibrated_fn_config
                        fn_metrics = calibrated_fn_metrics
                        fn_predictions = calibrated_fn_predictions
                    else:
                        fn_config, fn_metrics, fn_predictions = select_fn_aware_reranker_params(
                            fused_predictions,
                            raw_candidates or [],
                            active_fn_scores,
                            labels,
                            thresholds=fn_aware_thresholds,
                            max_base_ious=fn_aware_max_base_ious,
                            nms_ious=fn_aware_nms_ious,
                            length_penalties=fn_aware_length_penalties,
                            max_rescues_per_videos=fn_aware_max_per_video,
                            max_fp_increase=args.fn_aware_max_fp_increase,
                            max_candidates_per_video=args.fn_aware_max_candidates_per_video,
                            iou_threshold=args.iou_threshold,
                        )
                    fn_aware_summary = {
                        "enabled": False,
                        "config": fn_config,
                        "metrics": fn_metrics,
                        "base": "current_fused_predictions",
                        "model": args.fn_aware_model,
                        "n_train_candidates": len(fn_candidates),
                        "n_positive_train_candidates": int(y_fn.sum()),
                        "train_max_per_video": args.fn_aware_train_max_per_video,
                        "train_min_score": args.fn_aware_train_min_score,
                        "max_fp_increase": args.fn_aware_max_fp_increase,
                        "max_candidates_per_video": args.fn_aware_max_candidates_per_video,
                        "acceptance_distill": acceptance_summary,
                    }
                    fn_key = (
                        float(fn_metrics["segment"]["f1"]),
                        float(fn_metrics["segment"]["precision"]),
                        float(fn_metrics["segment"]["recall"]),
                        float(fn_metrics["frame"]["f1"]),
                    )
                    fusion_key = (
                        float(fusion_metrics["segment"]["f1"]),
                        float(fusion_metrics["segment"]["precision"]),
                        float(fusion_metrics["segment"]["recall"]),
                        float(fusion_metrics["frame"]["f1"]),
                    )
                    if args.fn_aware_acceptance_inner_calibration and fn_config.get("enabled"):
                        fused_predictions = fn_predictions
                        fusion_metrics = fn_metrics
                        fn_aware_summary["enabled"] = True
                        fn_aware_summary["applied_by"] = "inner_calibration"
                    elif fn_config.get("enabled") and fn_key > fusion_key:
                        fused_predictions = fn_predictions
                        fusion_metrics = fn_metrics
                        fn_aware_summary["enabled"] = True
                else:
                    fn_aware_summary = {
                        "enabled": False,
                        "reason": "single_class_fn_aware_labels",
                        "n_train_candidates": len(fn_candidates),
                        "n_positive_train_candidates": int(y_fn.sum()),
                        "train_max_per_video": args.fn_aware_train_max_per_video,
                        "train_min_score": args.fn_aware_train_min_score,
                    }
            else:
                fn_aware_summary = {"enabled": False, "reason": "no_fn_aware_candidates"}
        proposal_set_distillation_summary = {"enabled": False, "reason": "disabled"}
        if args.proposal_set_distillation:
            if protected_config.get("enabled"):
                train_distill_base_predictions = fuse_protected_segment_predictions(
                    train_active_mainline_predictions,
                    train_active_selector_predictions,
                    max_mainline_iou=protected_config["max_mainline_iou"],
                    selector_nms_iou=protected_config["selector_nms_iou"],
                )
            else:
                train_distill_base_predictions = train_active_mainline_predictions
            if residual_sub_event_summary.get("enabled"):
                train_residual_evidence = [
                    _reduce_evidence_channels(
                        records[idx],
                        feature_names,
                        residual_sub_event_evidence_names,
                        args.residual_sub_event_reducer,
                    )
                    for idx in train_idx
                ]
                residual_config = residual_sub_event_summary["config"]
                train_distill_base_predictions = select_residual_sub_event_predictions(
                    train_distill_base_predictions,
                    train_raw_candidates or [],
                    train_raw_scores or [],
                    train_residual_evidence,
                    min_selector_score=float(residual_config["min_selector_score"]),
                    evidence_threshold=float(residual_config["evidence_threshold"]),
                    min_evidence_mean=float(residual_config["min_evidence_mean"]),
                    min_active_fraction=float(residual_config["min_active_fraction"]),
                    min_parent_contrast=float(residual_config["min_parent_contrast"]),
                    parent_min_length=int(residual_config["parent_min_length"]),
                    max_child_parent_ratio=float(residual_config["max_child_parent_ratio"]),
                    min_child_parent_coverage=float(residual_config["min_child_parent_coverage"]),
                    length_penalty=float(residual_config["length_penalty"]),
                    nms_iou=residual_config["nms_iou"],
                    max_sub_events_per_parent=int(residual_config["max_sub_events_per_parent"]),
                    max_sub_events_per_video=int(residual_config["max_sub_events_per_video"]),
                    mode=str(residual_config.get("mode", "add")),
                    min_sub_events_per_parent=int(residual_config.get("min_sub_events_per_parent", 2)),
                )
            train_candidate_by_idx = {
                int(idx): candidates for idx, candidates in zip(train_idx, train_raw_candidates or [])
            }
            train_scores_by_idx = {
                int(idx): scores for idx, scores in zip(train_idx, train_raw_scores or [])
            }
            train_base_by_idx = {
                int(idx): segments for idx, segments in zip(train_idx, train_distill_base_predictions)
            }
            distill_candidates = build_distillation_candidate_records(
                records,
                train_idx,
                candidate_predictions_by_idx=train_candidate_by_idx,
                candidate_scores_by_idx=train_scores_by_idx,
                base_segments_by_idx=train_base_by_idx,
                feature_names=feature_names,
                channel_names=channel_names,
                iou_threshold=args.iou_threshold,
                max_teacher_rescues=args.proposal_set_distillation_teacher_max_rescues,
                max_base_iou=proposal_set_distillation_teacher_max_base_iou,
                teacher_nms_iou=proposal_set_distillation_teacher_nms_iou,
                teacher_mode=args.proposal_set_distillation_teacher_mode,
                max_records_per_video=args.proposal_set_distillation_train_max_per_video,
                hard_negative_top_k=args.proposal_set_distillation_hard_negative_top_k,
                min_selector_score=args.proposal_set_distillation_train_min_score,
            )
            if distill_candidates:
                x_distill = np.stack([item.features for item in distill_candidates])
                y_distill = np.asarray([item.label for item in distill_candidates], dtype=np.int64)
                best_iou_distill = np.asarray([item.best_iou for item in distill_candidates], dtype=np.float32)
                quality_targets = candidate_quality_targets(best_iou_distill)
                distill_student = None
                distill_target = "hard_oracle"
                n_positive_distill = int(y_distill.sum())
                skip_reason = "single_class_distillation_labels"
                if args.proposal_set_distillation_soft_quality:
                    distill_target = "soft_quality"
                    n_positive_distill = int((quality_targets > 0.0).sum())
                    skip_reason = "no_soft_quality_targets"
                    if n_positive_distill > 0:
                        print(
                            f"fold={fold_idx} proposal_set_soft_quality_train candidates={len(distill_candidates)} "
                            f"soft_positives={n_positive_distill} model={args.proposal_set_distillation_model} "
                            f"device={args.device}",
                            flush=True,
                        )
                        distill_student = fit_soft_quality_distillation_student(
                            x_distill,
                            best_iou_distill,
                            seed=args.seed + 5000 + fold_idx,
                            model_name=args.proposal_set_distillation_model,
                            device=args.device,
                            epochs=args.proposal_set_distillation_epochs,
                            batch_size=args.proposal_set_distillation_batch_size,
                        )
                elif len(np.unique(y_distill)) >= 2:
                    print(
                        f"fold={fold_idx} proposal_set_distill_train candidates={len(distill_candidates)} "
                        f"positives={int(y_distill.sum())} model={args.proposal_set_distillation_model} device={args.device}",
                        flush=True,
                    )
                    distill_student = fit_distillation_student(
                        x_distill,
                        y_distill,
                        seed=args.seed + 5000 + fold_idx,
                        model_name=args.proposal_set_distillation_model,
                        device=args.device,
                        epochs=args.proposal_set_distillation_epochs,
                        batch_size=args.proposal_set_distillation_batch_size,
                    )
                if distill_student is not None:
                    distill_scores = [
                        score_distillation_candidates(
                            distill_student,
                            records[idx],
                            candidates,
                            scores,
                            feature_names,
                            channel_names,
                            base_segments,
                        )
                        for idx, candidates, scores, base_segments in zip(
                            val_idx,
                            raw_candidates or [],
                            raw_scores or [],
                            fused_predictions,
                        )
                    ]
                    topology_distill_summary = {"enabled": False, "reason": "disabled"}
                    if args.proposal_set_distillation_topology_replacement:
                        distill_topology_evidence = [
                            _event_topology_evidence(
                                records[idx],
                                feature_names,
                                proposal_set_distillation_topology_evidence_names,
                                args.proposal_set_distillation_topology_reducer,
                            )
                            for idx in val_idx
                        ]
                        topology_config, topology_metrics, topology_predictions = select_distilled_topology_replacement_params(
                            fused_predictions,
                            raw_candidates or [],
                            distill_scores,
                            distill_topology_evidence,
                            labels,
                            evidence_thresholds=proposal_set_distillation_topology_evidence_thresholds,
                            min_student_scores=proposal_set_distillation_topology_min_student_scores,
                            max_student_ranks=proposal_set_distillation_topology_max_student_ranks,
                            min_evidence_means=proposal_set_distillation_topology_min_evidence_means,
                            min_active_fractions=proposal_set_distillation_topology_min_active_fractions,
                            min_parent_contrasts=proposal_set_distillation_topology_min_parent_contrasts,
                            parent_min_lengths=proposal_set_distillation_topology_parent_min_lengths,
                            max_child_parent_ratios=proposal_set_distillation_topology_max_child_parent_ratios,
                            min_child_parent_coverages=proposal_set_distillation_topology_min_child_parent_coverages,
                            min_gap_between_children_values=proposal_set_distillation_topology_min_gaps,
                            child_score_thresholds=proposal_set_distillation_topology_child_score_thresholds,
                            set_score_thresholds=proposal_set_distillation_topology_set_score_thresholds,
                            student_weights=proposal_set_distillation_topology_student_weights,
                            evidence_weights=proposal_set_distillation_topology_evidence_weights,
                            active_fraction_weights=proposal_set_distillation_topology_active_fraction_weights,
                            contrast_weights=proposal_set_distillation_topology_contrast_weights,
                            length_penalties=proposal_set_distillation_topology_length_penalties,
                            nms_ious=proposal_set_distillation_topology_nms_ious,
                            min_children_per_parents=proposal_set_distillation_topology_min_children_per_parent,
                            max_children_per_parents=proposal_set_distillation_topology_max_children_per_parent,
                            max_replaced_parents_per_videos=proposal_set_distillation_topology_max_replaced_parents_per_video,
                            max_fp_increase=args.proposal_set_distillation_topology_max_fp_increase,
                            iou_threshold=args.iou_threshold,
                        )
                        topology_distill_summary = {
                            "enabled": False,
                            "config": topology_config,
                            "metrics": topology_metrics,
                            "base": "current_fused_predictions",
                            "evidence_names": proposal_set_distillation_topology_evidence_names,
                            "reducer": args.proposal_set_distillation_topology_reducer,
                        }
                        topology_key = (
                            float(topology_metrics["segment"]["f1"]),
                            float(topology_metrics["segment"]["precision"]),
                            float(topology_metrics["segment"]["recall"]),
                            float(topology_metrics["frame"]["f1"]),
                        )
                        fusion_key = (
                            float(fusion_metrics["segment"]["f1"]),
                            float(fusion_metrics["segment"]["precision"]),
                            float(fusion_metrics["segment"]["recall"]),
                            float(fusion_metrics["frame"]["f1"]),
                        )
                        if topology_config.get("enabled") and topology_key > fusion_key:
                            fused_predictions = topology_predictions
                            fusion_metrics = topology_metrics
                            topology_distill_summary["enabled"] = True
                    distill_config, distill_metrics, distill_predictions = select_distilled_proposal_set_params(
                        fused_predictions,
                        raw_candidates or [],
                        distill_scores,
                        labels,
                        thresholds=proposal_set_distillation_thresholds,
                        max_base_ious=proposal_set_distillation_max_base_ious,
                        nms_ious=proposal_set_distillation_nms_ious,
                        length_penalties=proposal_set_distillation_length_penalties,
                        max_rescues_per_videos=proposal_set_distillation_max_per_video,
                        iou_threshold=args.iou_threshold,
                    )
                    proposal_set_distillation_summary = {
                        "enabled": bool(topology_distill_summary.get("enabled")),
                        "config": distill_config,
                        "metrics": distill_metrics,
                        "base": "current_fused_predictions",
                        "model": args.proposal_set_distillation_model,
                        "distillation_target": distill_target,
                        "soft_quality": bool(args.proposal_set_distillation_soft_quality),
                        "n_train_candidates": len(distill_candidates),
                        "n_positive_train_candidates": int(n_positive_distill),
                        "train_max_per_video": args.proposal_set_distillation_train_max_per_video,
                        "hard_negative_top_k": args.proposal_set_distillation_hard_negative_top_k,
                        "train_min_score": args.proposal_set_distillation_train_min_score,
                        "teacher_max_rescues": args.proposal_set_distillation_teacher_max_rescues,
                        "teacher_max_base_iou": proposal_set_distillation_teacher_max_base_iou,
                        "teacher_nms_iou": proposal_set_distillation_teacher_nms_iou,
                        "teacher_mode": args.proposal_set_distillation_teacher_mode,
                        "topology_replacement": topology_distill_summary,
                    }
                    distill_key = (
                        float(distill_metrics["segment"]["f1"]),
                        float(distill_metrics["segment"]["precision"]),
                        float(distill_metrics["segment"]["recall"]),
                        float(distill_metrics["frame"]["f1"]),
                    )
                    fusion_key = (
                        float(fusion_metrics["segment"]["f1"]),
                        float(fusion_metrics["segment"]["precision"]),
                        float(fusion_metrics["segment"]["recall"]),
                        float(fusion_metrics["frame"]["f1"]),
                    )
                    if distill_config.get("enabled") and distill_key > fusion_key:
                        fused_predictions = distill_predictions
                        fusion_metrics = distill_metrics
                        proposal_set_distillation_summary["enabled"] = True
                else:
                    proposal_set_distillation_summary = {
                        "enabled": False,
                        "reason": skip_reason,
                        "n_train_candidates": len(distill_candidates),
                        "n_positive_train_candidates": int(n_positive_distill),
                        "distillation_target": distill_target,
                        "soft_quality": bool(args.proposal_set_distillation_soft_quality),
                        "train_max_per_video": args.proposal_set_distillation_train_max_per_video,
                        "hard_negative_top_k": args.proposal_set_distillation_hard_negative_top_k,
                    }
            else:
                proposal_set_distillation_summary = {"enabled": False, "reason": "no_distillation_candidates"}
        evidence_set_reasoner_summary = {"enabled": False}
        if args.evidence_set_reasoner:
            evidence_set_reasoner_evidence = [
                np.stack(
                    [
                        _reduce_evidence_channels(
                            records[idx],
                            feature_names,
                            [name],
                            "mean",
                        )
                        for name in evidence_set_reasoner_evidence_names
                        if name in feature_names
                    ],
                    axis=1,
                )
                for idx in val_idx
            ]
            reasoner_config, reasoner_metrics, reasoner_predictions = select_evidence_set_reasoner_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                evidence_set_reasoner_evidence,
                labels,
                min_selector_scores=evidence_set_reasoner_min_selector_scores,
                evidence_thresholds=evidence_set_reasoner_evidence_thresholds,
                min_evidence_means=evidence_set_reasoner_min_evidence_means,
                min_active_fractions=evidence_set_reasoner_min_active_fractions,
                min_contrasts=evidence_set_reasoner_min_contrasts,
                score_thresholds=evidence_set_reasoner_score_thresholds,
                selector_weights=evidence_set_reasoner_selector_weights,
                contrast_weights=evidence_set_reasoner_contrast_weights,
                active_fraction_weights=evidence_set_reasoner_active_fraction_weights,
                consensus_weights=evidence_set_reasoner_consensus_weights,
                support_ious=evidence_set_reasoner_support_ious,
                support_count_weights=evidence_set_reasoner_support_count_weights,
                max_base_ious=evidence_set_reasoner_max_base_ious,
                base_iou_penalties=evidence_set_reasoner_base_iou_penalties,
                length_penalties=evidence_set_reasoner_length_penalties,
                nms_ious=evidence_set_reasoner_nms_ious,
                max_rescues_per_videos=evidence_set_reasoner_max_per_video,
                prefilter_top_k=args.evidence_set_reasoner_prefilter_top_k,
                iou_threshold=args.iou_threshold,
            )
            evidence_set_reasoner_summary = {
                "config": reasoner_config,
                "metrics": reasoner_metrics,
                "base": "current_fused_predictions",
                "evidence_names": evidence_set_reasoner_evidence_names,
                "prefilter_top_k": args.evidence_set_reasoner_prefilter_top_k,
            }
            reasoner_key = (
                float(reasoner_metrics["segment"]["f1"]),
                float(reasoner_metrics["segment"]["precision"]),
                float(reasoner_metrics["segment"]["recall"]),
                float(reasoner_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if reasoner_config.get("enabled") and reasoner_key > fusion_key:
                fused_predictions = reasoner_predictions
                fusion_metrics = reasoner_metrics
                evidence_set_reasoner_summary["enabled"] = True
        candidate_rescue_summary = {"enabled": False}
        if args.candidate_rescue:
            rescue_base_predictions = fused_predictions
            rescue_config, rescue_metrics, rescue_predictions = select_candidate_rescue_params(
                rescue_base_predictions,
                raw_candidates or [],
                raw_scores or [],
                labels,
                prob_thresholds=candidate_rescue_thresholds,
                max_mainline_ious=candidate_rescue_mainline_ious,
                candidate_nms_ious=candidate_rescue_nms_ious,
                length_penalties=candidate_rescue_length_penalties,
                max_rescues_per_videos=candidate_rescue_max_per_video,
                iou_threshold=args.iou_threshold,
            )
            candidate_rescue_summary = {
                "config": rescue_config,
                "metrics": rescue_metrics,
            }
            rescue_key = (
                float(rescue_metrics["segment"]["f1"]),
                float(rescue_metrics["segment"]["precision"]),
                float(rescue_metrics["segment"]["recall"]),
                float(rescue_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if rescue_config.get("enabled") and rescue_key > fusion_key:
                fused_predictions = rescue_predictions
                fusion_metrics = rescue_metrics
                candidate_rescue_summary["enabled"] = True
            candidate_rescue_summary["base"] = "current_fused_predictions"
        scale_completeness_rescue_summary = {"enabled": False}
        if args.scale_completeness_rescue:
            scale_completeness_evidence = [
                _reduce_evidence_channels(
                    records[idx],
                    feature_names,
                    scale_completeness_evidence_names,
                    args.scale_completeness_reducer,
                )
                for idx in val_idx
            ]
            scale_config, scale_metrics, scale_predictions = select_scale_completeness_rescue_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                scale_completeness_evidence,
                labels,
                prob_thresholds=scale_completeness_prob_thresholds,
                evidence_thresholds=scale_completeness_evidence_thresholds,
                min_active_fractions=scale_completeness_min_active_fractions,
                min_means=scale_completeness_min_means,
                min_contrasts=scale_completeness_min_contrasts,
                max_base_ious=scale_completeness_max_base_ious,
                length_penalties=scale_completeness_length_penalties,
                nms_ious=scale_completeness_nms_ious,
                max_rescues_per_videos=scale_completeness_max_per_video,
                iou_threshold=args.iou_threshold,
            )
            scale_completeness_rescue_summary = {
                "config": scale_config,
                "metrics": scale_metrics,
                "base": "current_fused_predictions",
                "evidence_names": scale_completeness_evidence_names,
                "reducer": args.scale_completeness_reducer,
            }
            scale_key = (
                float(scale_metrics["segment"]["f1"]),
                float(scale_metrics["segment"]["precision"]),
                float(scale_metrics["segment"]["recall"]),
                float(scale_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if scale_config.get("enabled") and scale_key > fusion_key:
                fused_predictions = scale_predictions
                fusion_metrics = scale_metrics
                scale_completeness_rescue_summary["enabled"] = True
        boundary_distribution_refine_summary = {"enabled": False}
        if args.boundary_distribution_refine:
            boundary_config, boundary_metrics, boundary_predictions = select_boundary_distribution_refine_params(
                fused_predictions,
                raw_candidates or [],
                raw_scores or [],
                labels,
                prob_thresholds=boundary_distribution_thresholds,
                min_base_coverages=boundary_distribution_base_coverages,
                max_length_ratios=boundary_distribution_max_ratios,
                start_quantiles=boundary_distribution_start_quantiles,
                end_quantiles=boundary_distribution_end_quantiles,
                base_boundary_weights=boundary_distribution_base_weights,
                max_shift_ratios=boundary_distribution_max_shift_ratios,
                iou_threshold=args.iou_threshold,
            )
            boundary_distribution_refine_summary = {
                "config": boundary_config,
                "metrics": boundary_metrics,
                "base": "current_fused_predictions",
            }
            boundary_key = (
                float(boundary_metrics["segment"]["f1"]),
                float(boundary_metrics["segment"]["precision"]),
                float(boundary_metrics["segment"]["recall"]),
                float(boundary_metrics["frame"]["f1"]),
            )
            fusion_key = (
                float(fusion_metrics["segment"]["f1"]),
                float(fusion_metrics["segment"]["precision"]),
                float(fusion_metrics["segment"]["recall"]),
                float(fusion_metrics["frame"]["f1"]),
            )
            if boundary_config.get("enabled") and boundary_key > fusion_key:
                fused_predictions = boundary_predictions
                fusion_metrics = boundary_metrics
                boundary_distribution_refine_summary["enabled"] = True
        rescuer_summary = {"enabled": False, "reason": "disabled"}
        if args.rescuer_model != "none":
            rescue_candidates = build_rescue_candidate_records(
                records,
                train_idx,
                classifier,
                feature_names,
                channel_names,
                thresholds,
                min_gaps,
                min_lengths,
                mainline_probs_by_idx,
                mainline_segments_by_idx,
                iou_threshold=args.iou_threshold,
            )
            if rescue_candidates:
                x_rescue = np.stack([item.features for item in rescue_candidates])
                y_rescue = np.asarray([item.label for item in rescue_candidates], dtype=np.int64)
                if len(np.unique(y_rescue)) >= 2:
                    print(
                        f"fold={fold_idx} rescuer_train candidates={len(rescue_candidates)} "
                        f"positives={int(y_rescue.sum())} model={args.rescuer_model} device={args.device}",
                        flush=True,
                    )
                    rescuer = fit_rescuer(
                        x_rescue,
                        y_rescue,
                        seed=args.seed + 2000 + fold_idx,
                        model_name=args.rescuer_model,
                        device=args.device,
                        epochs=args.rescuer_epochs,
                        batch_size=args.rescuer_batch_size,
                    )
                    rescuer_config, rescuer_metrics, rescuer_predictions = select_rescuer_params(
                        classifier,
                        rescuer,
                        records,
                        val_idx,
                        feature_names,
                        channel_names,
                        thresholds,
                        min_gaps,
                        min_lengths,
                        mainline_probs_by_idx,
                        mainline_segments_by_idx,
                        iou_threshold=args.iou_threshold,
                        active_base_segments_by_idx={int(idx): segments for idx, segments in zip(val_idx, fused_predictions)},
                    )
                    rescuer_summary = {
                        "enabled": True,
                        "config": rescuer_config,
                        "metrics": rescuer_metrics,
                        "n_train_candidates": len(rescue_candidates),
                        "n_positive_train_candidates": int(y_rescue.sum()),
                    }
                    rescuer_key = (
                        float(rescuer_metrics["segment"]["f1"]),
                        float(rescuer_metrics["segment"]["precision"]),
                        float(rescuer_metrics["segment"]["recall"]),
                        float(rescuer_metrics["frame"]["f1"]),
                    )
                    fusion_key = (
                        float(fusion_metrics["segment"]["f1"]),
                        float(fusion_metrics["segment"]["precision"]),
                        float(fusion_metrics["segment"]["recall"]),
                        float(fusion_metrics["frame"]["f1"]),
                    )
                    if rescuer_key > fusion_key:
                        fused_predictions = rescuer_predictions
                        fusion_metrics = rescuer_metrics
                else:
                    rescuer_summary = {
                        "enabled": False,
                        "reason": "single_class_rescue_labels",
                        "n_train_candidates": len(rescue_candidates),
                        "n_positive_train_candidates": int(y_rescue.sum()),
                    }
            else:
                rescuer_summary = {"enabled": False, "reason": "no_rescue_candidates"}
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_names": [records[idx].name for idx in val_idx],
                "consensus_denoise": consensus_summary,
                "clean_anchor_rectify": clean_anchor_rectify_summary,
                "clean_anchor_soft_weights": clean_anchor_soft_weights_summary,
                "mainline": mainline_metrics,
                "selector": {
                    "params": selector_params,
                    "validation": selector_val_metrics,
                    "metrics": evaluate_segment_predictions(selector_predictions, labels, iou_threshold=args.iou_threshold),
                    "n_train_candidates": n_candidates,
                    "n_positive_train_candidates": n_positive,
                },
                "fusion": {
                    "best_nms_iou": best_nms,
                    "best_segment_metrics": best_segment_metrics,
                    "boundary_snap": boundary_snap_summary,
                    "watershed_split": watershed_split_summary,
                    "proposal_replacement": proposal_replacement_summary,
                    "fusion_aware_selector": fusion_aware_selector_summary,
                    "clean_calibrated_selector": clean_calibrated_selector_summary,
                    "graph_reranker": graph_reranker_summary,
                    "dp_event_set_selector": dp_event_set_summary,
                    "ot_event_set_matcher": ot_event_set_summary,
                    "residual_sub_event_expansion": residual_sub_event_summary,
                    "topology_split_gate": topology_split_gate_summary,
                    "topology_matching_network": topology_matching_summary,
                    "event_topology_splitter": event_topology_splitter_summary,
                    "fn_aware_reranker": fn_aware_summary,
                    "proposal_set_distillation": proposal_set_distillation_summary,
                    "candidate_rescue": candidate_rescue_summary,
                    "scale_completeness_rescue": scale_completeness_rescue_summary,
                    "evidence_set_reasoner": evidence_set_reasoner_summary,
                    "boundary_distribution_refine": boundary_distribution_refine_summary,
                    "unprotected_validation": unprotected_metrics,
                    "protected": protected_config,
                    "protected_metrics": protected_metrics,
                    "rescuer": rescuer_summary,
                },
                "validation": fusion_metrics,
            }
        )
        predictions_by_fold.append(fused_predictions)

    records = source_records
    summary = {
        "data_dir": args.data_dir,
        "n_videos": len(records),
        "frames": int(sum(len(record.labels) for record in records)),
        "positive_frames": int(sum(record.labels.sum() for record in records)),
        "folds": fold_summaries,
        "aggregate": aggregate_fold_metrics(fold_summaries),
        "mainline_aggregate": aggregate_named_metrics(fold_summaries, "mainline"),
        "selector_aggregate": {
            "segment": {
                "precision_mean": float(np.mean([fold["selector"]["metrics"]["precision"] for fold in fold_summaries])),
                "precision_std": float(np.std([fold["selector"]["metrics"]["precision"] for fold in fold_summaries])),
                "recall_mean": float(np.mean([fold["selector"]["metrics"]["recall"] for fold in fold_summaries])),
                "recall_std": float(np.std([fold["selector"]["metrics"]["recall"] for fold in fold_summaries])),
                "f1_mean": float(np.mean([fold["selector"]["metrics"]["f1"] for fold in fold_summaries])),
                "f1_std": float(np.std([fold["selector"]["metrics"]["f1"] for fold in fold_summaries])),
                "tp": int(sum(fold["selector"]["metrics"]["tp"] for fold in fold_summaries)),
                "fp": int(sum(fold["selector"]["metrics"]["fp"] for fold in fold_summaries)),
                "fn": int(sum(fold["selector"]["metrics"]["fn"] for fold in fold_summaries)),
            }
        },
        "config": {
            "folds": args.folds,
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
            "consensus_denoise": args.consensus_denoise,
            "consensus_evidence_names": consensus_evidence_names,
            "consensus_agreement_name": args.consensus_agreement_name,
            "consensus_config": consensus_config.__dict__,
            "clean_anchor_rectify": args.clean_anchor_rectify,
            "clean_anchor_soft_weights": args.clean_anchor_soft_weights,
            "clean_anchor_data_dir": args.clean_anchor_data_dir,
            "clean_anchor_rectify_config": clean_anchor_rectify_base_config.__dict__,
            "selector_model": args.selector_model,
            "selector_target": args.selector_target,
            "selector_epochs": args.selector_epochs,
            "selector_batch_size": args.selector_batch_size,
            "fusion_aware_selector": args.fusion_aware_selector,
            "fusion_aware_thresholds": fusion_aware_thresholds,
            "fusion_aware_length_penalties": fusion_aware_length_penalties,
            "clean_calibrated_selector": args.clean_calibrated_selector,
            "clean_calibration_data_dir": args.clean_calibration_data_dir,
            "clean_calibration_model": args.clean_calibration_model,
            "clean_calibration_target": args.clean_calibration_target,
            "clean_calibration_epochs": args.clean_calibration_epochs,
            "clean_calibration_batch_size": args.clean_calibration_batch_size,
            "clean_calibrated_thresholds": clean_calibrated_thresholds,
            "clean_calibrated_raw_thresholds": clean_calibrated_raw_thresholds,
            "clean_calibrated_clean_weights": clean_calibrated_clean_weights,
            "clean_calibrated_raw_weights": clean_calibrated_raw_weights,
            "clean_calibrated_max_base_ious": clean_calibrated_max_base_ious,
            "clean_calibrated_length_penalties": clean_calibrated_length_penalties,
            "clean_calibrated_max_rescues_per_video": clean_calibrated_max_rescues_per_video,
            "clean_calibrated_rescue_nms_ious": clean_calibrated_rescue_nms_ious,
            "clean_calibrated_max_fp_increase": args.clean_calibrated_max_fp_increase,
            "graph_reranker": args.graph_reranker,
            "graph_reranker_thresholds": graph_reranker_thresholds,
            "graph_reranker_support_ious": graph_reranker_support_ious,
            "graph_reranker_support_weights": graph_reranker_support_weights,
            "graph_reranker_support_count_weights": graph_reranker_support_count_weights,
            "graph_reranker_split_penalties": graph_reranker_split_penalties,
            "graph_reranker_mainline_overlap_penalties": graph_reranker_mainline_overlap_penalties,
            "graph_reranker_length_penalties": graph_reranker_length_penalties,
            "dp_event_set_selector": args.dp_event_set_selector,
            "dp_event_set_base_keep_scores": dp_event_set_base_keep_scores,
            "dp_event_set_candidate_score_weights": dp_event_set_candidate_score_weights,
            "dp_event_set_min_candidate_scores": dp_event_set_min_candidate_scores,
            "dp_event_set_length_penalties": dp_event_set_length_penalties,
            "dp_event_set_base_overlap_penalties": dp_event_set_base_overlap_penalties,
            "dp_event_set_max_fp_increase": args.dp_event_set_max_fp_increase,
            "ot_event_set_matcher": args.ot_event_set_matcher,
            "ot_evidence_names": ot_evidence_names,
            "ot_reducer": args.ot_reducer,
            "ot_score_thresholds": ot_score_thresholds,
            "ot_base_keep_scores": ot_base_keep_scores,
            "ot_evidence_thresholds": ot_evidence_thresholds,
            "ot_selector_weights": ot_selector_weights,
            "ot_contrast_weights": ot_contrast_weights,
            "ot_active_fraction_weights": ot_active_fraction_weights,
            "ot_agreement_weights": ot_agreement_weights,
            "ot_island_coverage_weights": ot_island_coverage_weights,
            "ot_peak_alignment_weights": ot_peak_alignment_weights,
            "ot_base_overlap_penalties": ot_base_overlap_penalties,
            "ot_length_penalties": ot_length_penalties,
            "ot_smooth_windows": ot_smooth_windows,
            "ot_min_evidence_island_lengths": ot_min_evidence_island_lengths,
            "ot_max_fp_increase": args.ot_max_fp_increase,
            "rescuer_model": args.rescuer_model,
            "rescuer_epochs": args.rescuer_epochs,
            "rescuer_batch_size": args.rescuer_batch_size,
            "boundary_snap": args.boundary_snap,
            "boundary_snap_min_scores": boundary_snap_min_scores,
            "boundary_snap_coverages": boundary_snap_coverages,
            "boundary_snap_max_ratios": boundary_snap_max_ratios,
            "watershed_split": args.watershed_split,
            "watershed_evidence_names": watershed_evidence_names,
            "watershed_reducer": args.watershed_reducer,
            "watershed_thresholds": watershed_thresholds,
            "watershed_smooth_windows": watershed_smooth_windows,
            "watershed_min_gaps": watershed_min_gaps,
            "watershed_min_lengths": watershed_min_lengths,
            "watershed_pads": watershed_pads,
            "watershed_parent_min_lengths": watershed_parent_min_lengths,
            "proposal_replacement": args.proposal_replacement,
            "proposal_replacement_thresholds": proposal_replacement_thresholds,
            "proposal_replacement_parent_min_lengths": proposal_replacement_parent_min_lengths,
            "proposal_replacement_min_replacements": proposal_replacement_min_replacements,
            "proposal_replacement_max_per_parent": proposal_replacement_max_per_parent,
            "proposal_replacement_coverages": proposal_replacement_coverages,
            "proposal_replacement_max_ratios": proposal_replacement_max_ratios,
            "proposal_replacement_length_penalties": proposal_replacement_length_penalties,
            "proposal_replacement_max_fp_increase": proposal_replacement_max_fp_increase,
            "candidate_rescue": args.candidate_rescue,
            "candidate_rescue_thresholds": candidate_rescue_thresholds,
            "candidate_rescue_mainline_ious": candidate_rescue_mainline_ious,
            "candidate_rescue_nms_ious": candidate_rescue_nms_ious,
            "candidate_rescue_length_penalties": candidate_rescue_length_penalties,
            "candidate_rescue_max_per_video": candidate_rescue_max_per_video,
            "scale_completeness_rescue": args.scale_completeness_rescue,
            "scale_completeness_evidence_names": scale_completeness_evidence_names,
            "scale_completeness_reducer": args.scale_completeness_reducer,
            "scale_completeness_prob_thresholds": scale_completeness_prob_thresholds,
            "scale_completeness_evidence_thresholds": scale_completeness_evidence_thresholds,
            "scale_completeness_min_active_fractions": scale_completeness_min_active_fractions,
            "scale_completeness_min_means": scale_completeness_min_means,
            "scale_completeness_min_contrasts": scale_completeness_min_contrasts,
            "scale_completeness_max_base_ious": scale_completeness_max_base_ious,
            "scale_completeness_length_penalties": scale_completeness_length_penalties,
            "scale_completeness_nms_ious": scale_completeness_nms_ious,
            "scale_completeness_max_per_video": scale_completeness_max_per_video,
            "evidence_set_reasoner": args.evidence_set_reasoner,
            "evidence_set_reasoner_evidence_names": evidence_set_reasoner_evidence_names,
            "evidence_set_reasoner_min_selector_scores": evidence_set_reasoner_min_selector_scores,
            "evidence_set_reasoner_evidence_thresholds": evidence_set_reasoner_evidence_thresholds,
            "evidence_set_reasoner_min_evidence_means": evidence_set_reasoner_min_evidence_means,
            "evidence_set_reasoner_min_active_fractions": evidence_set_reasoner_min_active_fractions,
            "evidence_set_reasoner_min_contrasts": evidence_set_reasoner_min_contrasts,
            "evidence_set_reasoner_score_thresholds": evidence_set_reasoner_score_thresholds,
            "evidence_set_reasoner_selector_weights": evidence_set_reasoner_selector_weights,
            "evidence_set_reasoner_contrast_weights": evidence_set_reasoner_contrast_weights,
            "evidence_set_reasoner_active_fraction_weights": evidence_set_reasoner_active_fraction_weights,
            "evidence_set_reasoner_consensus_weights": evidence_set_reasoner_consensus_weights,
            "evidence_set_reasoner_support_ious": evidence_set_reasoner_support_ious,
            "evidence_set_reasoner_support_count_weights": evidence_set_reasoner_support_count_weights,
            "evidence_set_reasoner_max_base_ious": evidence_set_reasoner_max_base_ious,
            "evidence_set_reasoner_base_iou_penalties": evidence_set_reasoner_base_iou_penalties,
            "evidence_set_reasoner_length_penalties": evidence_set_reasoner_length_penalties,
            "evidence_set_reasoner_nms_ious": evidence_set_reasoner_nms_ious,
            "evidence_set_reasoner_max_per_video": evidence_set_reasoner_max_per_video,
            "evidence_set_reasoner_prefilter_top_k": args.evidence_set_reasoner_prefilter_top_k,
            "boundary_distribution_refine": args.boundary_distribution_refine,
            "boundary_distribution_thresholds": boundary_distribution_thresholds,
            "boundary_distribution_base_coverages": boundary_distribution_base_coverages,
            "boundary_distribution_max_ratios": boundary_distribution_max_ratios,
            "boundary_distribution_start_quantiles": boundary_distribution_start_quantiles,
            "boundary_distribution_end_quantiles": boundary_distribution_end_quantiles,
            "boundary_distribution_base_weights": boundary_distribution_base_weights,
            "boundary_distribution_max_shift_ratios": boundary_distribution_max_shift_ratios,
            "residual_sub_event_expansion": args.residual_sub_event_expansion,
            "residual_sub_event_evidence_names": residual_sub_event_evidence_names,
            "residual_sub_event_reducer": args.residual_sub_event_reducer,
            "residual_sub_event_min_selector_scores": residual_sub_event_min_selector_scores,
            "residual_sub_event_evidence_thresholds": residual_sub_event_evidence_thresholds,
            "residual_sub_event_min_evidence_means": residual_sub_event_min_evidence_means,
            "residual_sub_event_min_active_fractions": residual_sub_event_min_active_fractions,
            "residual_sub_event_min_parent_contrasts": residual_sub_event_min_parent_contrasts,
            "residual_sub_event_parent_min_lengths": residual_sub_event_parent_min_lengths,
            "residual_sub_event_max_child_parent_ratios": residual_sub_event_max_child_parent_ratios,
            "residual_sub_event_min_child_parent_coverages": residual_sub_event_min_child_parent_coverages,
            "residual_sub_event_length_penalties": residual_sub_event_length_penalties,
            "residual_sub_event_nms_ious": residual_sub_event_nms_ious,
            "residual_sub_event_max_per_parent": residual_sub_event_max_per_parent,
            "residual_sub_event_max_per_video": residual_sub_event_max_per_video,
            "residual_sub_event_modes": residual_sub_event_modes,
            "residual_sub_event_min_per_parent": residual_sub_event_min_per_parent,
            "event_topology_splitter": args.event_topology_splitter,
            "event_topology_evidence_names": event_topology_evidence_names,
            "event_topology_reducer": args.event_topology_reducer,
            "event_topology_min_selector_scores": event_topology_min_selector_scores,
            "event_topology_evidence_thresholds": event_topology_evidence_thresholds,
            "event_topology_min_evidence_means": event_topology_min_evidence_means,
            "event_topology_min_active_fractions": event_topology_min_active_fractions,
            "event_topology_min_parent_contrasts": event_topology_min_parent_contrasts,
            "event_topology_parent_min_lengths": event_topology_parent_min_lengths,
            "event_topology_max_child_parent_ratios": event_topology_max_child_parent_ratios,
            "event_topology_min_child_parent_coverages": event_topology_min_child_parent_coverages,
            "event_topology_min_gaps": event_topology_min_gaps,
            "event_topology_support_ious": event_topology_support_ious,
            "event_topology_support_count_weights": event_topology_support_count_weights,
            "event_topology_length_penalties": event_topology_length_penalties,
            "event_topology_nms_ious": event_topology_nms_ious,
            "event_topology_min_children_per_parent": event_topology_min_children_per_parent,
            "event_topology_max_children_per_parent": event_topology_max_children_per_parent,
            "event_topology_max_replaced_parents_per_video": event_topology_max_replaced_parents_per_video,
            "event_topology_max_fp_increase": args.event_topology_max_fp_increase,
            "topology_split_gate": args.topology_split_gate,
            "topology_split_gate_model": args.topology_split_gate_model,
            "topology_split_gate_evidence_names": topology_split_gate_evidence_names,
            "topology_split_gate_reducer": args.topology_split_gate_reducer,
            "topology_split_gate_thresholds": topology_split_gate_thresholds,
            "topology_split_gate_parent_min_lengths": topology_split_gate_parent_min_lengths,
            "topology_split_gate_min_child_parent_coverages": topology_split_gate_min_child_parent_coverages,
            "topology_split_gate_max_child_parent_ratios": topology_split_gate_max_child_parent_ratios,
            "topology_split_gate_max_children_per_parent": topology_split_gate_max_children_per_parent,
            "topology_split_gate_max_replaced_parents_per_video": topology_split_gate_max_replaced_parents_per_video,
            "topology_split_gate_max_fp_increase": args.topology_split_gate_max_fp_increase,
            "topology_split_gate_synthetic": args.topology_split_gate_synthetic,
            "topology_split_gate_synthetic_max_gt_gap": args.topology_split_gate_synthetic_max_gt_gap,
            "topology_split_gate_synthetic_parent_pad": args.topology_split_gate_synthetic_parent_pad,
            "topology_matching_network": args.topology_matching_network,
            "topology_matching_model": args.topology_matching_model,
            "topology_matching_evidence_names": topology_matching_evidence_names,
            "topology_matching_reducer": args.topology_matching_reducer,
            "topology_matching_thresholds": topology_matching_thresholds,
            "topology_matching_parent_min_lengths": topology_matching_parent_min_lengths,
            "topology_matching_min_child_parent_coverages": topology_matching_min_child_parent_coverages,
            "topology_matching_max_child_parent_ratios": topology_matching_max_child_parent_ratios,
            "topology_matching_max_children_per_parent": topology_matching_max_children_per_parent,
            "topology_matching_max_set_proposals_per_parent": topology_matching_max_set_proposals_per_parent,
            "topology_matching_max_child_pool_per_parent": topology_matching_max_child_pool_per_parent,
            "topology_matching_max_replaced_parents_per_video": topology_matching_max_replaced_parents_per_video,
            "topology_matching_max_fp_increase": args.topology_matching_max_fp_increase,
            "topology_matching_risk_aware": args.topology_matching_risk_aware,
            "topology_matching_risk_penalties": topology_matching_risk_penalties,
            "topology_matching_safety_thresholds": topology_matching_safety_thresholds,
            "topology_matching_safety_weight": args.topology_matching_safety_weight,
            "topology_matching_epochs": args.topology_matching_epochs,
            "topology_matching_batch_size": args.topology_matching_batch_size,
            "fn_aware_reranker": args.fn_aware_reranker,
            "fn_aware_model": args.fn_aware_model,
            "fn_aware_epochs": args.fn_aware_epochs,
            "fn_aware_batch_size": args.fn_aware_batch_size,
            "fn_aware_train_max_per_video": args.fn_aware_train_max_per_video,
            "fn_aware_train_min_score": args.fn_aware_train_min_score,
            "fn_aware_thresholds": fn_aware_thresholds,
            "fn_aware_max_base_ious": fn_aware_max_base_ious,
            "fn_aware_nms_ious": fn_aware_nms_ious,
            "fn_aware_length_penalties": fn_aware_length_penalties,
            "fn_aware_max_per_video": fn_aware_max_per_video,
            "fn_aware_max_fp_increase": args.fn_aware_max_fp_increase,
            "fn_aware_max_candidates_per_video": args.fn_aware_max_candidates_per_video,
            "fn_aware_acceptance_distill": args.fn_aware_acceptance_distill,
            "fn_aware_acceptance_model": args.fn_aware_acceptance_model,
            "fn_aware_acceptance_train_max_per_video": args.fn_aware_acceptance_train_max_per_video,
            "fn_aware_acceptance_threshold": args.fn_aware_acceptance_threshold,
            "fn_aware_acceptance_max_rescues_per_video": args.fn_aware_acceptance_max_rescues_per_video,
            "fn_aware_acceptance_max_candidates_per_video": args.fn_aware_acceptance_max_candidates_per_video,
            "fn_aware_acceptance_max_fp_increase": args.fn_aware_acceptance_max_fp_increase,
            "fn_aware_acceptance_use_aux_positives": args.fn_aware_acceptance_use_aux_positives,
            "fn_aware_acceptance_inner_calibration": args.fn_aware_acceptance_inner_calibration,
            "fn_aware_acceptance_calibration_fraction": args.fn_aware_acceptance_calibration_fraction,
            "fn_aware_acceptance_calibration_seed_offset": args.fn_aware_acceptance_calibration_seed_offset,
            "proposal_set_distillation": args.proposal_set_distillation,
            "proposal_set_distillation_model": args.proposal_set_distillation_model,
            "proposal_set_distillation_epochs": args.proposal_set_distillation_epochs,
            "proposal_set_distillation_batch_size": args.proposal_set_distillation_batch_size,
            "proposal_set_distillation_train_max_per_video": args.proposal_set_distillation_train_max_per_video,
            "proposal_set_distillation_hard_negative_top_k": args.proposal_set_distillation_hard_negative_top_k,
            "proposal_set_distillation_train_min_score": args.proposal_set_distillation_train_min_score,
            "proposal_set_distillation_teacher_max_rescues": args.proposal_set_distillation_teacher_max_rescues,
            "proposal_set_distillation_teacher_max_base_iou": proposal_set_distillation_teacher_max_base_iou,
            "proposal_set_distillation_teacher_nms_iou": proposal_set_distillation_teacher_nms_iou,
            "proposal_set_distillation_teacher_mode": args.proposal_set_distillation_teacher_mode,
            "proposal_set_distillation_thresholds": proposal_set_distillation_thresholds,
            "proposal_set_distillation_max_base_ious": proposal_set_distillation_max_base_ious,
            "proposal_set_distillation_nms_ious": proposal_set_distillation_nms_ious,
            "proposal_set_distillation_length_penalties": proposal_set_distillation_length_penalties,
            "proposal_set_distillation_max_per_video": proposal_set_distillation_max_per_video,
            "proposal_set_distillation_topology_replacement": args.proposal_set_distillation_topology_replacement,
            "proposal_set_distillation_topology_evidence_names": proposal_set_distillation_topology_evidence_names,
            "proposal_set_distillation_topology_reducer": args.proposal_set_distillation_topology_reducer,
            "proposal_set_distillation_topology_evidence_thresholds": proposal_set_distillation_topology_evidence_thresholds,
            "proposal_set_distillation_topology_min_student_scores": proposal_set_distillation_topology_min_student_scores,
            "proposal_set_distillation_topology_max_student_ranks": proposal_set_distillation_topology_max_student_ranks,
            "proposal_set_distillation_topology_min_evidence_means": proposal_set_distillation_topology_min_evidence_means,
            "proposal_set_distillation_topology_min_active_fractions": proposal_set_distillation_topology_min_active_fractions,
            "proposal_set_distillation_topology_min_parent_contrasts": proposal_set_distillation_topology_min_parent_contrasts,
            "proposal_set_distillation_topology_parent_min_lengths": proposal_set_distillation_topology_parent_min_lengths,
            "proposal_set_distillation_topology_max_child_parent_ratios": proposal_set_distillation_topology_max_child_parent_ratios,
            "proposal_set_distillation_topology_min_child_parent_coverages": proposal_set_distillation_topology_min_child_parent_coverages,
            "proposal_set_distillation_topology_min_gaps": proposal_set_distillation_topology_min_gaps,
            "proposal_set_distillation_topology_child_score_thresholds": proposal_set_distillation_topology_child_score_thresholds,
            "proposal_set_distillation_topology_set_score_thresholds": proposal_set_distillation_topology_set_score_thresholds,
            "proposal_set_distillation_topology_student_weights": proposal_set_distillation_topology_student_weights,
            "proposal_set_distillation_topology_evidence_weights": proposal_set_distillation_topology_evidence_weights,
            "proposal_set_distillation_topology_active_fraction_weights": proposal_set_distillation_topology_active_fraction_weights,
            "proposal_set_distillation_topology_contrast_weights": proposal_set_distillation_topology_contrast_weights,
            "proposal_set_distillation_topology_length_penalties": proposal_set_distillation_topology_length_penalties,
            "proposal_set_distillation_topology_nms_ious": proposal_set_distillation_topology_nms_ious,
            "proposal_set_distillation_topology_min_children_per_parent": proposal_set_distillation_topology_min_children_per_parent,
            "proposal_set_distillation_topology_max_children_per_parent": proposal_set_distillation_topology_max_children_per_parent,
            "proposal_set_distillation_topology_max_replaced_parents_per_video": proposal_set_distillation_topology_max_replaced_parents_per_video,
            "proposal_set_distillation_topology_max_fp_increase": args.proposal_set_distillation_topology_max_fp_increase,
            "selector_channel_names": channel_names,
            "selector_thresholds": thresholds,
            "selector_min_gaps": min_gaps,
            "selector_min_lengths": min_lengths,
            "nms_candidates": args.nms_candidates,
            "protected_mainline_iou_candidates": protected_mainline_iou_candidates,
            "task": "binary_error_segment_localization_mainline_selector_fusion_cv",
        },
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.predictions_out:
        predictions_export = _build_prediction_export(
            data_dir=args.data_dir,
            records=records,
            fold_summaries=fold_summaries,
            predictions_by_fold=predictions_by_fold,
            include_labels=bool(args.include_prediction_labels),
        )
        predictions_out = Path(args.predictions_out)
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        predictions_out.write_text(json.dumps(predictions_export, indent=2), encoding="utf-8")
        print(f"wrote predictions {predictions_out}", flush=True)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
