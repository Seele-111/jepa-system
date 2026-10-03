#!/usr/bin/env python3
"""Attribute remaining false negatives in the frozen full-train JEPA pipeline.

The script replays the current cross-validation pipeline on full-train only and
writes a per-GT explanation for every missed error segment. It intentionally
does not touch held-out test data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_graph_reranker import graph_rerank_candidate_scores, select_graph_reranked_predictions
from jepa_proposal_replacement import replace_merged_predictions
from selector_fusion import evaluate_fused_predictions, fuse_protected_segment_predictions
from train_proposal_calibrator import evaluate_segment_predictions, interval_iou
from train_proposal_set_selector import (
    build_candidate_records,
    candidate_quality_targets,
    fit_classifier,
    fit_quality_model,
    score_video_candidates,
    select_weighted_proposal_set,
)
from train_segment_locator import (
    _feature_indices,
    _segments_from_params,
    blend_prediction_records,
    contiguous_segments,
    load_signal_dataset,
    predict_records,
    train_model,
)


DEFAULT_DATA_DIR = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2"
DEFAULT_SOURCE_SUMMARY = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json"
DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_fn_candidate_attribution.json"


Segment = tuple[int, int]


def _as_float_list(value) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _as_int_list(value) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _parse_nms_candidates(value) -> list[float | None]:
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_items = list(value)
    parsed: list[float | None] = []
    for item in raw_items:
        if item is None:
            parsed.append(None)
            continue
        text = str(item).strip().lower()
        parsed.append(None if text in {"none", "null"} else float(text))
    return parsed


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _segment_overlap_fraction(candidate: tuple[int, int], target: tuple[int, int]) -> float:
    """Return how much of candidate is covered by target."""
    candidate = _normalise_segment(candidate)
    target = _normalise_segment(target)
    intersection = max(0, min(candidate[1], target[1]) - max(candidate[0], target[0]) + 1)
    return intersection / max(1, _segment_length(candidate))


def _match_prediction_segments(
    pred_segments: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float,
) -> dict:
    """Greedy segment matching compatible with evaluate_segment_predictions."""
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched: set[int] = set()
    matched_pairs: list[dict] = []
    false_positive_predictions: list[Segment] = []
    for pred_raw in pred_segments:
        pred = _normalise_segment(pred_raw)
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_segments):
            if idx in matched:
                continue
            iou = interval_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
            matched_pairs.append({"prediction": pred, "gt": gt_segments[best_idx], "iou": float(best_iou)})
        else:
            false_positive_predictions.append(pred)
    return {
        "gt_segments": gt_segments,
        "matched_gt_indices": sorted(matched),
        "matched_pairs": matched_pairs,
        "unmatched_gt": [gt for idx, gt in enumerate(gt_segments) if idx not in matched],
        "false_positive_predictions": false_positive_predictions,
    }


def _best_iou_with(segment: tuple[int, int], others: Iterable[tuple[int, int]]) -> tuple[float, Segment | None]:
    best_iou = 0.0
    best_segment: Segment | None = None
    target = _normalise_segment(segment)
    for other_raw in others:
        other = _normalise_segment(other_raw)
        iou = interval_iou(target, other)
        if iou > best_iou:
            best_iou = float(iou)
            best_segment = other
    return best_iou, best_segment


def _rank_desc(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def _candidate_diagnostics(
    gt: tuple[int, int],
    candidates: list[tuple[int, int]],
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray | None,
) -> dict:
    if not candidates:
        return {
            "best_candidate_iou": 0.0,
            "best_candidate": None,
            "best_candidate_score": 0.0,
            "best_candidate_rank": None,
            "best_adjusted_score": None,
            "best_adjusted_rank": None,
            "covering_candidates": 0,
        }
    raw_scores = np.nan_to_num(np.asarray(raw_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    adjusted = None if adjusted_scores is None else np.nan_to_num(np.asarray(adjusted_scores, dtype=np.float32).reshape(-1), nan=0.0)
    raw_ranks = _rank_desc(raw_scores)
    adjusted_ranks = _rank_desc(adjusted) if adjusted is not None else None
    best_idx = -1
    best_iou = 0.0
    best_key: tuple[float, float] | None = None
    covering = 0
    for idx, candidate in enumerate(candidates):
        iou = float(interval_iou(_normalise_segment(candidate), _normalise_segment(gt)))
        if iou >= 0.3:
            covering += 1
        key = (iou, float(raw_scores[idx]))
        if best_key is None or key > best_key:
            best_key = key
            best_iou = iou
            best_idx = idx
    if best_idx < 0:
        return {
            "best_candidate_iou": 0.0,
            "best_candidate": None,
            "best_candidate_score": 0.0,
            "best_candidate_rank": None,
            "best_adjusted_score": None,
            "best_adjusted_rank": None,
            "covering_candidates": 0,
        }
    return {
        "best_candidate_iou": float(best_iou),
        "best_candidate": list(_normalise_segment(candidates[best_idx])),
        "best_candidate_score": float(raw_scores[best_idx]),
        "best_candidate_rank": int(raw_ranks[best_idx]),
        "best_adjusted_score": None if adjusted is None else float(adjusted[best_idx]),
        "best_adjusted_rank": None if adjusted_ranks is None else int(adjusted_ranks[best_idx]),
        "covering_candidates": int(covering),
    }


def _classify_fn_reason(
    best_candidate_iou: float,
    best_candidate_score: float,
    best_candidate_rank: int | None,
    selected_iou: float,
    fused_iou: float,
    protected_suppressed: bool,
    active_threshold: float = 0.55,
    adjusted_score: float | None = None,
) -> str:
    if fused_iou >= 0.3:
        return "segment_matching_conflict"
    if selected_iou >= 0.3:
        if protected_suppressed:
            return "fusion_suppressed_selected_candidate"
        return "selector_positive_not_used_by_final_fusion"
    if best_candidate_iou < 0.3 or best_candidate_rank is None:
        return "candidate_generation_miss"
    active_score = float(best_candidate_score if adjusted_score is None else adjusted_score)
    if active_score < float(active_threshold) or int(best_candidate_rank) > 50:
        return "candidate_scored_too_low"
    return "proposal_set_selection_conflict"


def _train_selector_classifier(
    records,
    train_idx: list[int],
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    iou_threshold: float,
    seed: int,
    model_name: str,
    target: str,
    device: str,
    selector_epochs: int,
    selector_batch_size: int,
):
    candidate_records = build_candidate_records(
        records,
        train_idx,
        feature_names=feature_names,
        channel_names=channel_names,
        thresholds=thresholds,
        min_gaps=min_gaps,
        min_lengths=min_lengths,
        iou_threshold=iou_threshold,
    )
    if not candidate_records:
        raise RuntimeError("no training candidates generated")
    x_train = np.stack([item.features for item in candidate_records])
    y_train = np.asarray([item.label for item in candidate_records], dtype=np.int64)
    best_iou_train = np.asarray([item.best_iou for item in candidate_records], dtype=np.float32)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"candidate labels need both classes, got positives={int(y_train.sum())}/{len(y_train)}")
    if target == "quality":
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
    return classifier, {
        "n_train_candidates": len(candidate_records),
        "n_positive_train_candidates": int(y_train.sum()),
        "train_quality_target_mean": float(candidate_quality_targets(best_iou_train).mean()),
    }


def _select_from_scores(candidates, scores, prob_threshold: float, length_penalty: float) -> list[Segment]:
    filtered_candidates: list[Segment] = []
    filtered_scores: list[float] = []
    for candidate, score in zip(candidates, scores):
        if float(score) >= float(prob_threshold):
            filtered_candidates.append(_normalise_segment(candidate))
            filtered_scores.append(float(score))
    return select_weighted_proposal_set(filtered_candidates, filtered_scores, length_penalty=float(length_penalty))


def replay_fold(
    records,
    feature_names: list[str],
    config: dict,
    fold_summary: dict,
    device: str,
    iou_threshold: float,
) -> dict:
    fold_idx = int(fold_summary["fold"])
    train_idx = [int(idx) for idx in fold_summary["train_idx"]]
    val_idx = [int(idx) for idx in fold_summary["val_idx"]]
    print(f"replay fold={fold_idx} train={len(train_idx)} val={len(val_idx)} device={device}", flush=True)

    model, pure_metrics, mean, std = train_model(
        records,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=int(config.get("epochs", 80)),
        lr=float(config.get("lr", 1e-3)),
        hidden=int(config.get("hidden", 32)),
        dropout=float(config.get("dropout", 0.1)),
        patience=int(config.get("patience", 15)),
        device=device,
        seed=int(config.get("seed", 42)) + fold_idx,
        lambda_dice=float(config.get("lambda_dice", 0.0)),
        lambda_boundary=float(config.get("lambda_boundary", 0.0)),
        architecture=str(config.get("architecture", "tcn")),
    )
    del pure_metrics

    val_outputs = predict_records(model, records, val_idx, mean, std, device)
    blend_aux_name = str(config.get("blend_aux_name") or "")
    if blend_aux_name:
        blend_aux_channel = _feature_indices(feature_names, [blend_aux_name], "blend")[0]
        alpha_model = float(fold_summary["mainline"].get("hybrid", {}).get("alpha_model", 1.0))
        val_outputs = blend_prediction_records(
            val_outputs,
            records,
            val_idx,
            aux_channel=blend_aux_channel,
            alpha_model=alpha_model,
        )

    mainline_params = fold_summary["mainline"]["params"]
    mainline_predictions = [_segments_from_params(output["probs"], mainline_params) for output in val_outputs]

    channel_names = [name for name in config.get("selector_channel_names", []) if name in feature_names]
    thresholds = _as_float_list(config.get("selector_thresholds", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    min_gaps = _as_int_list(config.get("selector_min_gaps", [0, 1, 2, 4, 8]))
    min_lengths = _as_int_list(config.get("selector_min_lengths", [1, 2, 4, 8]))
    classifier, selector_train_info = _train_selector_classifier(
        records,
        train_idx,
        feature_names,
        channel_names,
        thresholds,
        min_gaps,
        min_lengths,
        iou_threshold=iou_threshold,
        seed=int(config.get("seed", 42)) + 1000 + fold_idx,
        model_name=str(config.get("selector_model", "extratrees")),
        target=str(config.get("selector_target", "binary")),
        device=device,
        selector_epochs=int(config.get("selector_epochs", 120)),
        selector_batch_size=int(config.get("selector_batch_size", 512)),
    )

    raw_candidates: list[list[Segment]] = []
    raw_scores: list[np.ndarray] = []
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
        raw_candidates.append([_normalise_segment(item) for item in candidates])
        raw_scores.append(np.asarray(scores, dtype=np.float32))

    selector_params = fold_summary["selector"]["params"]
    selector_predictions = [
        _select_from_scores(
            candidates,
            scores,
            prob_threshold=float(selector_params["prob_threshold"]),
            length_penalty=float(selector_params["length_penalty"]),
        )
        for candidates, scores in zip(raw_candidates, raw_scores)
    ]

    proposal_config = fold_summary["fusion"].get("proposal_replacement", {}).get("config", {})
    proposal_replacement_enabled = bool(proposal_config.get("enabled"))
    if proposal_replacement_enabled:
        mainline_predictions = replace_merged_predictions(
            mainline_predictions,
            raw_candidates,
            raw_scores,
            prob_threshold=float(proposal_config["prob_threshold"]),
            parent_min_length=int(proposal_config["parent_min_length"]),
            min_replacements=int(proposal_config["min_replacements"]),
            max_replacements_per_parent=int(proposal_config["max_replacements_per_parent"]),
            min_candidate_parent_coverage=float(proposal_config["min_candidate_parent_coverage"]),
            max_candidate_parent_ratio=float(proposal_config["max_candidate_parent_ratio"]),
            length_penalty=float(proposal_config["length_penalty"]),
        )

    graph_summary = fold_summary["fusion"].get("graph_reranker", {})
    graph_config = graph_summary.get("config", {})
    graph_enabled = bool(graph_summary.get("enabled") or graph_config.get("enabled"))
    adjusted_scores: list[np.ndarray | None] = [None for _ in raw_scores]
    if graph_enabled:
        adjusted_scores = [
            graph_rerank_candidate_scores(
                mainline,
                candidates,
                scores,
                support_iou=float(graph_config["support_iou"]),
                support_weight=float(graph_config["support_weight"]),
                support_count_weight=float(graph_config["support_count_weight"]),
                split_penalty=float(graph_config["split_penalty"]),
                mainline_overlap_penalty=float(graph_config["mainline_overlap_penalty"]),
            )
            for mainline, candidates, scores in zip(mainline_predictions, raw_candidates, raw_scores)
        ]
        selector_predictions = select_graph_reranked_predictions(
            mainline_predictions,
            raw_candidates,
            raw_scores,
            prob_threshold=float(graph_config["prob_threshold"]),
            support_iou=float(graph_config["support_iou"]),
            support_weight=float(graph_config["support_weight"]),
            support_count_weight=float(graph_config["support_count_weight"]),
            split_penalty=float(graph_config["split_penalty"]),
            mainline_overlap_penalty=float(graph_config["mainline_overlap_penalty"]),
            length_penalty=float(graph_config["length_penalty"]),
        )

    protected_config = fold_summary["fusion"].get("protected", {})
    protected_enabled = bool(protected_config.get("enabled"))
    if protected_enabled:
        fused_predictions = fuse_protected_segment_predictions(
            mainline_predictions,
            selector_predictions,
            max_mainline_iou=protected_config["max_mainline_iou"],
            selector_nms_iou=protected_config["selector_nms_iou"],
        )
    else:
        fused_predictions = mainline_predictions

    labels = [records[idx].labels for idx in val_idx]
    metrics = evaluate_fused_predictions(fused_predictions, labels, iou_threshold=iou_threshold)
    selector_metrics = evaluate_segment_predictions(selector_predictions, labels, iou_threshold=iou_threshold)
    mainline_metrics = evaluate_fused_predictions(mainline_predictions, labels, iou_threshold=iou_threshold)

    fn_rows: list[dict] = []
    fp_rows: list[dict] = []
    for local_idx, record_idx in enumerate(val_idx):
        record = records[record_idx]
        match = _match_prediction_segments(fused_predictions[local_idx], record.labels, iou_threshold=iou_threshold)
        for pred in match["false_positive_predictions"]:
            fp_rows.append({"fold": fold_idx, "video_idx": int(record_idx), "video": record.name, "prediction": list(pred)})
        for gt in match["unmatched_gt"]:
            candidate_info = _candidate_diagnostics(gt, raw_candidates[local_idx], raw_scores[local_idx], adjusted_scores[local_idx])
            mainline_iou, mainline_segment = _best_iou_with(gt, mainline_predictions[local_idx])
            selector_iou, selector_segment = _best_iou_with(gt, selector_predictions[local_idx])
            fused_iou, fused_segment = _best_iou_with(gt, fused_predictions[local_idx])
            protected_suppressed = bool(protected_enabled and selector_iou >= iou_threshold and fused_iou < iou_threshold)
            active_threshold = float(
                graph_config.get("prob_threshold", selector_params.get("prob_threshold", 0.55))
                if graph_enabled
                else selector_params.get("prob_threshold", 0.55)
            )
            reason = _classify_fn_reason(
                best_candidate_iou=float(candidate_info["best_candidate_iou"]),
                best_candidate_score=float(candidate_info["best_candidate_score"]),
                best_candidate_rank=candidate_info["best_candidate_rank"],
                selected_iou=float(selector_iou),
                fused_iou=float(fused_iou),
                protected_suppressed=protected_suppressed,
                active_threshold=active_threshold,
                adjusted_score=candidate_info["best_adjusted_score"],
            )
            fn_rows.append(
                {
                    "fold": fold_idx,
                    "video_idx": int(record_idx),
                    "video": record.name,
                    "gt": list(gt),
                    "gt_length": int(_segment_length(gt)),
                    "reason": reason,
                    "mainline_iou": float(mainline_iou),
                    "mainline_segment": None if mainline_segment is None else list(mainline_segment),
                    "selector_iou": float(selector_iou),
                    "selector_segment": None if selector_segment is None else list(selector_segment),
                    "fused_iou": float(fused_iou),
                    "fused_segment": None if fused_segment is None else list(fused_segment),
                    "protected_enabled": protected_enabled,
                    "protected_suppressed": protected_suppressed,
                    "graph_enabled": graph_enabled,
                    "proposal_replacement_enabled": proposal_replacement_enabled,
                    "active_threshold": active_threshold,
                    **candidate_info,
                }
            )

    return {
        "fold": fold_idx,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "val_names": [records[idx].name for idx in val_idx],
        "validation": metrics,
        "mainline_validation": mainline_metrics,
        "selector_validation": selector_metrics,
        "selector_train": selector_train_info,
        "fn_rows": fn_rows,
        "fp_rows": fp_rows,
        "fused_predictions": fused_predictions,
        "mainline_predictions": mainline_predictions,
        "selector_predictions": selector_predictions,
        "raw_candidates": raw_candidates,
        "raw_scores": raw_scores,
        "labels": labels,
        "graph_enabled": graph_enabled,
        "proposal_replacement_enabled": proposal_replacement_enabled,
        "protected_enabled": protected_enabled,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    source_path = Path(args.source_summary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [str(name) for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]]
    folds = list(source["folds"])
    if args.fold_limit > 0:
        folds = folds[: int(args.fold_limit)]
    fold_results = [
        replay_fold(
            records,
            feature_names,
            config=source["config"],
            fold_summary=fold,
            device=args.device,
            iou_threshold=float(args.iou_threshold),
        )
        for fold in folds
    ]

    fn_rows = [row for fold in fold_results for row in fold["fn_rows"]]
    fp_rows = [row for fold in fold_results for row in fold["fp_rows"]]
    reason_counts = Counter(row["reason"] for row in fn_rows)
    covered = sum(1 for row in fn_rows if float(row["best_candidate_iou"]) >= float(args.iou_threshold))
    selected = sum(1 for row in fn_rows if float(row["selector_iou"]) >= float(args.iou_threshold))
    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(source_path),
        "fold_limit": int(args.fold_limit),
        "iou_threshold": float(args.iou_threshold),
        "device": str(args.device),
        "aggregate": aggregate_fold_metrics(fold_results),
        "source_aggregate": source.get("aggregate"),
        "folds": [
            {
                "fold": fold["fold"],
                "validation": fold["validation"],
                "mainline_validation": fold["mainline_validation"],
                "selector_validation": fold["selector_validation"],
                "graph_enabled": fold["graph_enabled"],
                "proposal_replacement_enabled": fold["proposal_replacement_enabled"],
                "protected_enabled": fold["protected_enabled"],
                "fn": len(fold["fn_rows"]),
                "fp": len(fold["fp_rows"]),
            }
            for fold in fold_results
        ],
        "fn_summary": {
            "total_fn": len(fn_rows),
            "candidate_covered_fn": int(covered),
            "candidate_covered_rate": float(covered / max(1, len(fn_rows))),
            "selector_selected_fn": int(selected),
            "selector_selected_rate": float(selected / max(1, len(fn_rows))),
            "reason_counts": dict(reason_counts),
        },
        "fn_rows": fn_rows,
        "fp_rows": fp_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate_segment": result["aggregate"]["segment"],
                "fn_summary": result["fn_summary"],
                "out": str(out_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
