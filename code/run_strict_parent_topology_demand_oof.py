#!/usr/bin/env python3
"""Strict OOF parent-level JEPA topology demand/safety selection.

The runner starts from an OOF base-prediction export. For each held-out fold it
trains parent-to-child-set gain/safety heads on the other folds, selects the
replacement threshold through inner OOF calibration on those other folds, and
then applies the frozen config to the held-out fold without labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_prediction_set_switcher import _as_segments
from jepa_topology_matching_network import (
    build_topology_inference_records,
    build_topology_matching_records,
    fit_topology_matching_risk_models,
    apply_topology_matching_predictions,
    score_topology_matching_risk_records,
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


def _load_prediction_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def _flatten(folds: list[dict], key: str) -> list:
    out = []
    for fold in folds:
        out.extend(fold[key])
    return out


def _as_segment_lists(items) -> list[list[tuple[int, int]]]:
    return [[(int(s), int(e)) for s, e in video] for video in items]


def _build_train_records(folds: list[dict], config: dict, iou_threshold: float):
    return build_topology_matching_records(
        _flatten(folds, "base"),
        _flatten(folds, "candidates"),
        _flatten(folds, "scores"),
        _flatten(folds, "evidence"),
        _flatten(folds, "labels"),
        iou_threshold=float(iou_threshold),
        parent_min_length=int(config["parent_min_length"]),
        min_child_parent_coverage=float(config["min_child_parent_coverage"]),
        max_child_parent_ratio=float(config["max_child_parent_ratio"]),
        max_children_per_parent=int(config["max_children_per_parent"]),
        max_set_proposals_per_parent=int(config["max_set_proposals_per_parent"]),
        max_child_pool_per_parent=int(config["max_child_pool_per_parent"]),
        include_teacher=True,
    )


def _build_inference_records(fold: dict, config: dict):
    return build_topology_inference_records(
        fold["base"],
        fold["candidates"],
        fold["scores"],
        fold["evidence"],
        parent_min_length=int(config["parent_min_length"]),
        min_child_parent_coverage=float(config["min_child_parent_coverage"]),
        max_child_parent_ratio=float(config["max_child_parent_ratio"]),
        max_children_per_parent=int(config["max_children_per_parent"]),
        max_set_proposals_per_parent=int(config["max_set_proposals_per_parent"]),
        max_child_pool_per_parent=int(config["max_child_pool_per_parent"]),
    )


def _score_and_apply(
    fold: dict,
    gain_model,
    safety_model,
    config: dict,
    args,
) -> tuple[list[list[tuple[int, int]]], dict]:
    proposals = _build_inference_records(fold, config)
    if not proposals:
        predictions = _as_segment_lists(fold["base"])
        return predictions, {"n_proposals": 0}
    scores, gain_scores, safety_scores = score_topology_matching_risk_records(
        gain_model,
        safety_model,
        proposals,
        risk_penalty=float(config["risk_penalty"]),
        safety_weight=float(args.safety_weight),
    )
    predictions = apply_topology_matching_predictions(
        fold["base"],
        proposals,
        scores,
        threshold=float(config["threshold"]),
        max_replaced_parents_per_video=int(config["max_replaced_parents_per_video"]),
        safety_scores=safety_scores,
        safety_threshold=float(config["safety_threshold"]),
    )
    diagnostics = {
        "n_proposals": int(len(proposals)),
        "score_max": float(np.max(scores)) if len(scores) else 0.0,
        "gain_score_max": float(np.max(gain_scores)) if len(gain_scores) else 0.0,
        "safety_score_max": float(np.max(safety_scores)) if len(safety_scores) else 0.0,
    }
    return predictions, diagnostics


def apply_voted_topology_predictions(
    base_predictions: list[list[tuple[int, int]]],
    proposals: list,
    score_matrix: np.ndarray,
    safety_matrix: np.ndarray,
    threshold: float,
    safety_threshold: float,
    min_votes: int,
    max_replaced_parents_per_video: int,
) -> tuple[list[list[tuple[int, int]]], dict]:
    """Apply topology replacements only when enough models agree."""
    out = _as_segment_lists(base_predictions)
    scores = np.nan_to_num(np.asarray(score_matrix, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    safety = np.nan_to_num(np.asarray(safety_matrix, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if scores.ndim != 2 or safety.ndim != 2:
        raise ValueError("score_matrix and safety_matrix must be 2D")
    if scores.shape != safety.shape:
        raise ValueError("score_matrix and safety_matrix shapes must match")
    if scores.shape[1] != len(proposals):
        raise ValueError(f"proposal/score count mismatch: {len(proposals)} vs {scores.shape[1]}")
    votes = ((scores >= float(threshold)) & (safety >= float(safety_threshold))).sum(axis=0)
    by_video: dict[int, list[tuple[object, int, float]]] = {}
    for proposal, vote_count, mean_score in zip(proposals, votes, scores.mean(axis=0) if len(proposals) else []):
        if int(vote_count) < int(min_votes):
            continue
        by_video.setdefault(int(proposal.video_idx), []).append((proposal, int(vote_count), float(mean_score)))
    accepted = 0
    for video_idx, items in by_video.items():
        items.sort(
            key=lambda item: (
                item[1],
                item[2],
                len(item[0].children),
                -int(item[0].parent[0]),
            ),
            reverse=True,
        )
        selected = []
        used_parents = set()
        for proposal, vote_count, mean_score in items:
            if proposal.parent in used_parents:
                continue
            used_parents.add(proposal.parent)
            selected.append((proposal, vote_count, mean_score))
            if int(max_replaced_parents_per_video) > 0 and len(selected) >= int(max_replaced_parents_per_video):
                break
        replace_parents = {proposal.parent for proposal, _, _ in selected}
        children = [child for proposal, _, _ in selected for child in proposal.children]
        out[video_idx] = sorted(set([segment for segment in out[video_idx] if segment not in replace_parents] + children))
        accepted += len(selected)
    diagnostics = {
        "accepted_proposals": int(accepted),
        "max_votes": int(votes.max()) if len(votes) else 0,
        "n_proposals": int(len(proposals)),
        "n_models": int(scores.shape[0]),
        "min_votes": int(min_votes),
    }
    return out, diagnostics


def _candidate_configs(args) -> list[dict]:
    configs = []
    for parent_min_length in args.parent_min_lengths:
        for min_child_parent_coverage in args.min_child_parent_coverages:
            for max_child_parent_ratio in args.max_child_parent_ratios:
                for max_children_per_parent in args.max_children_per_parent:
                    for max_set_proposals_per_parent in args.max_set_proposals_per_parent:
                        for max_child_pool_per_parent in args.max_child_pool_per_parent:
                            for risk_penalty in args.risk_penalties:
                                for safety_threshold in args.safety_thresholds:
                                    for threshold in args.thresholds:
                                        for max_replaced in args.max_replaced_parents_per_video:
                                            configs.append(
                                                {
                                                    "enabled": True,
                                                    "inner_oof": True,
                                                    "model": str(args.model),
                                                    "parent_min_length": int(parent_min_length),
                                                    "min_child_parent_coverage": float(min_child_parent_coverage),
                                                    "max_child_parent_ratio": float(max_child_parent_ratio),
                                                    "max_children_per_parent": int(max_children_per_parent),
                                                    "max_set_proposals_per_parent": int(max_set_proposals_per_parent),
                                                    "max_child_pool_per_parent": int(max_child_pool_per_parent),
                                                    "risk_penalty": float(risk_penalty),
                                                    "safety_threshold": float(safety_threshold),
                                                    "safety_weight": float(args.safety_weight),
                                                    "threshold": float(threshold),
                                                    "max_replaced_parents_per_video": int(max_replaced),
                                                    "max_fp_increase": args.max_fp_increase,
                                                }
                                            )
    return configs


def _structural_configs(args) -> list[dict]:
    configs = []
    for parent_min_length in args.parent_min_lengths:
        for min_child_parent_coverage in args.min_child_parent_coverages:
            for max_child_parent_ratio in args.max_child_parent_ratios:
                for max_children_per_parent in args.max_children_per_parent:
                    for max_set_proposals_per_parent in args.max_set_proposals_per_parent:
                        for max_child_pool_per_parent in args.max_child_pool_per_parent:
                            configs.append(
                                {
                                    "parent_min_length": int(parent_min_length),
                                    "min_child_parent_coverage": float(min_child_parent_coverage),
                                    "max_child_parent_ratio": float(max_child_parent_ratio),
                                    "max_children_per_parent": int(max_children_per_parent),
                                    "max_set_proposals_per_parent": int(max_set_proposals_per_parent),
                                    "max_child_pool_per_parent": int(max_child_pool_per_parent),
                                }
                            )
    return configs


def _decision_configs(args) -> list[dict]:
    configs = []
    for risk_penalty in args.risk_penalties:
        for safety_threshold in args.safety_thresholds:
            for threshold in args.thresholds:
                for max_replaced in args.max_replaced_parents_per_video:
                    configs.append(
                        {
                            "risk_penalty": float(risk_penalty),
                            "safety_threshold": float(safety_threshold),
                            "safety_weight": float(args.safety_weight),
                            "threshold": float(threshold),
                            "max_replaced_parents_per_video": int(max_replaced),
                        }
                    )
    return configs


def select_inner_oof_config(calibration_folds: list[dict], args, seed_offset: int = 0) -> tuple[dict, dict]:
    base_predictions = _as_segment_lists(_flatten(calibration_folds, "base"))
    labels = _flatten(calibration_folds, "labels")
    base_metrics = evaluate_fused_predictions(base_predictions, labels, iou_threshold=float(args.iou_threshold))
    diagnostics = {
        "configs_considered": 0,
        "no_train_records": 0,
        "single_class_gain": 0,
        "fp_rejected_configs": 0,
        "best_rejected_metrics": None,
        "best_rejected_config": None,
    }
    best_config: dict = {"enabled": False, "inner_oof": True, "base_metrics": base_metrics, "diagnostics": diagnostics}
    best_metrics = base_metrics
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for structure in _structural_configs(args):
        prepared_folds = []
        train_record_count = 0
        positive_gain_count = 0
        structure_viable = True
        for inner_pos, inner_val in enumerate(calibration_folds):
            inner_train = [fold for pos, fold in enumerate(calibration_folds) if pos != inner_pos]
            train_records = _build_train_records(inner_train, structure, iou_threshold=float(args.iou_threshold))
            train_record_count += len(train_records)
            positive_gain_count += int(sum(record.gain_label for record in train_records))
            if not train_records:
                diagnostics["no_train_records"] += 1
                structure_viable = False
                break
            if len({int(record.gain_label) for record in train_records}) < 2:
                diagnostics["single_class_gain"] += 1
            gain_model, safety_model = fit_topology_matching_risk_models(
                train_records,
                model_name=str(args.model),
                seed=int(args.seed) + int(seed_offset) + 17 * inner_pos,
                device=args.device,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
            )
            proposals = _build_inference_records(inner_val, structure)
            score_variants = {}
            safety_scores = np.zeros(len(proposals), dtype=np.float32)
            if proposals:
                for risk_penalty in args.risk_penalties:
                    scores, _, safety = score_topology_matching_risk_records(
                        gain_model,
                        safety_model,
                        proposals,
                        risk_penalty=float(risk_penalty),
                        safety_weight=float(args.safety_weight),
                    )
                    score_variants[float(risk_penalty)] = scores
                    safety_scores = safety
            prepared_folds.append(
                {
                    "fold": inner_val,
                    "proposals": proposals,
                    "score_variants": score_variants,
                    "safety_scores": safety_scores,
                }
            )
        if not structure_viable:
            continue

        for decision in _decision_configs(args):
            diagnostics["configs_considered"] += 1
            config = {
                "enabled": True,
                "inner_oof": True,
                "model": str(args.model),
                **structure,
                **decision,
                "max_fp_increase": args.max_fp_increase,
            }
            inner_predictions = []
            inner_labels = []
            for prepared in prepared_folds:
                inner_val = prepared["fold"]
                proposals = prepared["proposals"]
                if proposals:
                    scores = prepared["score_variants"].get(float(decision["risk_penalty"]), np.zeros(len(proposals), dtype=np.float32))
                    preds = apply_topology_matching_predictions(
                        inner_val["base"],
                        proposals,
                        scores,
                        threshold=float(decision["threshold"]),
                        max_replaced_parents_per_video=int(decision["max_replaced_parents_per_video"]),
                        safety_scores=prepared["safety_scores"],
                        safety_threshold=float(decision["safety_threshold"]),
                    )
                else:
                    preds = _as_segment_lists(inner_val["base"])
                inner_predictions.extend(preds)
                inner_labels.extend(inner_val["labels"])
            metrics = evaluate_fused_predictions(inner_predictions, inner_labels, iou_threshold=float(args.iou_threshold))
            fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
            changed = sum(int(sorted(a) != sorted(b)) for a, b in zip(base_predictions, inner_predictions))
            observed = {
                **config,
                "inner_changed_videos": int(changed),
                "inner_train_records": int(train_record_count),
                "inner_positive_gain_records": int(positive_gain_count),
                "inner_fp_delta": int(fp_delta),
            }
            key4 = (
                float(metrics["segment"]["f1"]),
                float(metrics["segment"]["precision"]),
                float(metrics["segment"]["recall"]),
                float(metrics["frame"]["f1"]),
            )
            if args.max_fp_increase is not None and fp_delta > int(args.max_fp_increase):
                diagnostics["fp_rejected_configs"] += 1
                rejected = diagnostics.get("best_rejected_metrics")
                rejected_key = (-1.0, -1.0, -1.0, -1.0) if rejected is None else (
                    float(rejected["segment"]["f1"]),
                    float(rejected["segment"]["precision"]),
                    float(rejected["segment"]["recall"]),
                    float(rejected["frame"]["f1"]),
                )
                if key4 > rejected_key:
                    diagnostics["best_rejected_metrics"] = metrics
                    diagnostics["best_rejected_config"] = observed
                continue
            key = (*key4, -float(changed))
            if key > best_key:
                best_key = key
                best_metrics = metrics
                best_config = {
                    **observed,
                    "base_metrics": base_metrics,
                    "diagnostics": diagnostics,
                }
        continue

    return best_config, best_metrics

def run_strict_parent_topology_demand_oof(folds: list[dict], args) -> dict:
    results = []
    for heldout_pos, heldout in enumerate(folds):
        calibration = [fold for pos, fold in enumerate(folds) if pos != heldout_pos]
        config, calibration_metrics = select_inner_oof_config(
            calibration,
            args,
            seed_offset=1000 + int(heldout.get("fold", heldout_pos)) * 101,
        )
        if config.get("enabled"):
            if int(getattr(args, "inference_vote_min_count", 0)) > 0:
                proposals = _build_inference_records(heldout, config)
                score_rows = []
                safety_rows = []
                for inner_pos, inner_excluded in enumerate(calibration):
                    voter_train = [fold for pos, fold in enumerate(calibration) if pos != inner_pos]
                    train_records = _build_train_records(voter_train, config, iou_threshold=float(args.iou_threshold))
                    if not train_records or not proposals:
                        continue
                    gain_model, safety_model = fit_topology_matching_risk_models(
                        train_records,
                        model_name=str(args.model),
                        seed=int(args.seed) + 7000 + 31 * inner_pos + int(heldout.get("fold", heldout_pos)),
                        device=args.device,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                    )
                    scores, _, safety = score_topology_matching_risk_records(
                        gain_model,
                        safety_model,
                        proposals,
                        risk_penalty=float(config["risk_penalty"]),
                        safety_weight=float(args.safety_weight),
                    )
                    score_rows.append(scores)
                    safety_rows.append(safety)
                if score_rows:
                    predictions, inference_diag = apply_voted_topology_predictions(
                        heldout["base"],
                        proposals,
                        np.stack(score_rows),
                        np.stack(safety_rows),
                        threshold=float(config["threshold"]),
                        safety_threshold=float(config["safety_threshold"]),
                        min_votes=int(args.inference_vote_min_count),
                        max_replaced_parents_per_video=int(config["max_replaced_parents_per_video"]),
                    )
                    inference_diag["voting_enabled"] = True
                else:
                    predictions = _as_segment_lists(heldout["base"])
                    inference_diag = {"n_proposals": int(len(proposals)), "voting_enabled": True, "reason": "no_voter_scores"}
            else:
                train_records = _build_train_records(calibration, config, iou_threshold=float(args.iou_threshold))
                gain_model, safety_model = fit_topology_matching_risk_models(
                    train_records,
                    model_name=str(args.model),
                    seed=int(args.seed) + 5000 + int(heldout.get("fold", heldout_pos)),
                    device=args.device,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                )
                predictions, inference_diag = _score_and_apply(heldout, gain_model, safety_model, config, args)
        else:
            predictions = _as_segment_lists(heldout["base"])
            inference_diag = {"n_proposals": 0}
        metrics = evaluate_fused_predictions(predictions, heldout["labels"], iou_threshold=float(args.iou_threshold))
        results.append(
            {
                "fold": int(heldout.get("fold", heldout_pos)),
                "config": config,
                "calibration": calibration_metrics,
                "validation": metrics,
                "inference_diagnostics": inference_diag,
                "predictions": [[[int(s), int(e)] for s, e in video] for video in predictions],
            }
        )
    return {"folds": results, "aggregate": aggregate_fold_metrics(results)}


def build_topology_demand_folds(
    export: dict,
    records,
    raw_by_video: dict[int, tuple[list[tuple[int, int]], np.ndarray]],
    feature_names: list[str],
    evidence_names: list[str],
    reducer: str,
) -> list[dict]:
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
                "base": _as_segment_lists(fold.get("predictions", [])),
                "candidates": candidates,
                "scores": scores,
                "evidence": evidence,
                "labels": [np.asarray(item, dtype=np.int64) for item in labels],
            }
        )
    return folds


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
    parser.add_argument("--model", choices=["prototype", "logreg", "extratrees", "rf", "gbdt", "mlp"], default="prototype")
    parser.add_argument("--parent-min-lengths", default="12,24,48")
    parser.add_argument("--min-child-parent-coverages", default="0.5,0.7")
    parser.add_argument("--max-child-parent-ratios", default="0.5,0.7")
    parser.add_argument("--max-children-per-parent", default="2,3")
    parser.add_argument("--max-set-proposals-per-parent", default="8,16")
    parser.add_argument("--max-child-pool-per-parent", default="12")
    parser.add_argument("--thresholds", default="0,0.1,0.2,0.3,0.4")
    parser.add_argument("--risk-penalties", default="0.25,0.5,0.75")
    parser.add_argument("--safety-thresholds", default="0,0.25,0.5")
    parser.add_argument("--safety-weight", type=float, default=0.25)
    parser.add_argument("--max-replaced-parents-per-video", default="1")
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--inference-vote-min-count", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    args.parent_min_lengths = _parse_int_list(args.parent_min_lengths)
    args.min_child_parent_coverages = _parse_float_list(args.min_child_parent_coverages)
    args.max_child_parent_ratios = _parse_float_list(args.max_child_parent_ratios)
    args.max_children_per_parent = _parse_int_list(args.max_children_per_parent)
    args.max_set_proposals_per_parent = _parse_int_list(args.max_set_proposals_per_parent)
    args.max_child_pool_per_parent = _parse_int_list(args.max_child_pool_per_parent)
    args.thresholds = _parse_float_list(args.thresholds)
    args.risk_penalties = _parse_float_list(args.risk_penalties)
    args.safety_thresholds = _parse_float_list(args.safety_thresholds)
    args.max_replaced_parents_per_video = _parse_int_list(args.max_replaced_parents_per_video)

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    evidence_names = _parse_name_list(args.evidence_names)
    export = _load_prediction_export(args.base_predictions)
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
    folds = build_topology_demand_folds(export, records, raw_by_video, feature_names, evidence_names, args.evidence_reducer)
    result = run_strict_parent_topology_demand_oof(folds, args)
    result.update(
        {
            "data_dir": args.data_dir,
            "base_predictions": args.base_predictions,
            "selector_diagnostics": selector_diagnostics,
            "evidence_names": evidence_names,
            "evidence_reducer": args.evidence_reducer,
            "model": args.model,
            "strict_protocol": "outer held-out labels are used only for final evaluation; configs are selected by inner OOF on other folds",
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
