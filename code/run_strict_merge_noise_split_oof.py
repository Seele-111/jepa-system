#!/usr/bin/env python3
"""Strict OOF selective splitter for JEPA merge-noise parent segments."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_merge_noise_selective_splitter import (
    MergeNoiseParentRecord,
    extract_merge_noise_parent_records,
)
from jepa_prediction_set_switcher import _as_segments
from run_strict_dp_event_set_oof import build_fold_local_raw_candidates
from run_strict_parent_topology_demand_oof import build_topology_demand_folds
from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou
from train_proposal_rescuer import fit_rescuer
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES
from train_segment_locator import (
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    contiguous_segments,
    load_signal_dataset,
)


class _ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x)
        n = int(values.shape[0]) if values.ndim > 1 else 1
        positive = np.full(n, self.probability, dtype=np.float32)
        return np.stack([1.0 - positive, positive], axis=1)


@dataclass(frozen=True)
class LabeledMergeNoiseRecord:
    video_idx: int
    parent: tuple[int, int]
    children: list[tuple[int, int]]
    features: np.ndarray
    label: int
    tp_delta: int
    fp_delta: int


def _normalise_segment(segment: tuple[int, int]) -> tuple[int, int]:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_metrics(predictions: list[tuple[int, int]], labels: np.ndarray, iou_threshold: float) -> tuple[int, int, int]:
    gt = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched: set[int] = set()
    tp = fp = 0
    for pred_raw in predictions:
        pred = _normalise_segment(pred_raw)
        best_iou = 0.0
        best_idx = -1
        for idx, target in enumerate(gt):
            if idx in matched:
                continue
            value = float(interval_iou(pred, target))
            if value > best_iou:
                best_iou = value
                best_idx = idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1
    return int(tp), int(fp), int(len(gt) - len(matched))


def label_merge_noise_records(
    base_predictions,
    candidate_predictions,
    candidate_scores,
    evidence_arrays,
    labels,
    config: dict,
    iou_threshold: float,
) -> list[LabeledMergeNoiseRecord]:
    records = extract_merge_noise_parent_records(
        base_predictions,
        candidate_predictions,
        candidate_scores,
        evidence_arrays,
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
    )
    out: list[LabeledMergeNoiseRecord] = []
    for record in records:
        video_idx = int(record.video_idx)
        label_array = np.asarray(labels[video_idx], dtype=np.int64)
        base = [_normalise_segment(item) for item in base_predictions[video_idx]]
        split = sorted(set([segment for segment in base if segment != record.parent] + record.children))
        base_tp, base_fp, _ = _segment_metrics(base, label_array, iou_threshold=float(iou_threshold))
        split_tp, split_fp, _ = _segment_metrics(split, label_array, iou_threshold=float(iou_threshold))
        tp_delta = int(split_tp - base_tp)
        fp_delta = int(split_fp - base_fp)
        out.append(
            LabeledMergeNoiseRecord(
                video_idx=video_idx,
                parent=record.parent,
                children=record.children,
                features=record.features,
                label=int(tp_delta > 0 and fp_delta <= 0),
                tp_delta=tp_delta,
                fp_delta=fp_delta,
            )
        )
    return out


def _flatten(folds: list[dict], key: str) -> list:
    out = []
    for fold in folds:
        out.extend(fold[key])
    return out


def _as_segment_lists(items) -> list[list[tuple[int, int]]]:
    return [[(int(s), int(e)) for s, e in video] for video in items]


def _structure_configs(args) -> list[dict]:
    configs = []
    for island_threshold in args.island_thresholds:
        for island_min_length in args.island_min_lengths:
            for island_min_gap in args.island_min_gaps:
                for min_islands in args.min_islands:
                    for min_candidate_score in args.min_candidate_scores:
                        for max_candidate_rank in args.max_candidate_ranks:
                            for min_island_coverage in args.min_island_coverages:
                                for max_child_parent_ratio in args.max_child_parent_ratios:
                                    for min_child_parent_coverage in args.min_child_parent_coverages:
                                        for child_nms_iou in args.child_nms_ious:
                                            for max_children_per_parent in args.max_children_per_parent:
                                                configs.append(
                                                    {
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
                                                    }
                                                )
    return configs


def _score_records(model, records: list[MergeNoiseParentRecord]) -> np.ndarray:
    if not records:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(model.predict_proba(np.stack([record.features for record in records]))[:, 1], dtype=np.float32)


def _fit_parent_model(records: list[LabeledMergeNoiseRecord], args, seed: int):
    x = np.stack([record.features for record in records])
    y = np.asarray([record.label for record in records], dtype=np.int64)
    if len(np.unique(y)) < 2:
        return _ConstantProbabilityModel(float(y.mean()))
    return fit_rescuer(
        x,
        y,
        seed=int(seed),
        model_name=str(args.model),
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )


def _apply_records(base_predictions, records: list[MergeNoiseParentRecord], scores: np.ndarray, threshold: float, max_replaced_per_video: int):
    out = _as_segment_lists(base_predictions)
    by_video: dict[int, list[tuple[MergeNoiseParentRecord, float]]] = {}
    for record, score in zip(records, np.asarray(scores, dtype=np.float32).reshape(-1)):
        if float(score) >= float(threshold):
            by_video.setdefault(int(record.video_idx), []).append((record, float(score)))
    for video_idx, items in by_video.items():
        items.sort(key=lambda item: (item[1], item[0].features[-1]), reverse=True)
        selected = []
        used = set()
        for record, score in items:
            if record.parent in used:
                continue
            used.add(record.parent)
            selected.append((record, score))
            if int(max_replaced_per_video) > 0 and len(selected) >= int(max_replaced_per_video):
                break
        replace = {record.parent for record, _ in selected}
        children = [child for record, _ in selected for child in record.children]
        out[video_idx] = sorted(set([segment for segment in out[video_idx] if segment not in replace] + children))
    return out


def select_inner_config(calibration_folds: list[dict], args) -> tuple[dict, dict]:
    base_predictions = _as_segment_lists(_flatten(calibration_folds, "base"))
    labels = _flatten(calibration_folds, "labels")
    base_metrics = evaluate_fused_predictions(base_predictions, labels, iou_threshold=float(args.iou_threshold))
    best_config = {"enabled": False, "base_metrics": base_metrics, "diagnostics": {"configs_considered": 0}}
    best_metrics = base_metrics
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    diagnostics = best_config["diagnostics"]
    for structure in _structure_configs(args):
        prepared = []
        train_count = positive_count = 0
        viable = True
        for inner_pos, inner_val in enumerate(calibration_folds):
            inner_train = [fold for pos, fold in enumerate(calibration_folds) if pos != inner_pos]
            train_records = label_merge_noise_records(
                _flatten(inner_train, "base"),
                _flatten(inner_train, "candidates"),
                _flatten(inner_train, "scores"),
                _flatten(inner_train, "evidence"),
                _flatten(inner_train, "labels"),
                structure,
                iou_threshold=float(args.iou_threshold),
            )
            train_count += len(train_records)
            positive_count += int(sum(record.label for record in train_records))
            if not train_records:
                viable = False
                break
            model = _fit_parent_model(train_records, args, seed=int(args.seed) + 101 * inner_pos)
            val_records = extract_merge_noise_parent_records(
                inner_val["base"],
                inner_val["candidates"],
                inner_val["scores"],
                inner_val["evidence"],
                **structure,
            )
            scores = _score_records(model, val_records)
            prepared.append({"fold": inner_val, "records": val_records, "scores": scores})
        if not viable:
            continue
        for threshold in args.thresholds:
            for max_replaced in args.max_replaced_parents_per_video:
                diagnostics["configs_considered"] += 1
                preds = []
                pred_labels = []
                for item in prepared:
                    preds.extend(_apply_records(item["fold"]["base"], item["records"], item["scores"], threshold, max_replaced))
                    pred_labels.extend(item["fold"]["labels"])
                metrics = evaluate_fused_predictions(preds, pred_labels, iou_threshold=float(args.iou_threshold))
                fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                changed = sum(int(sorted(a) != sorted(b)) for a, b in zip(base_predictions, preds))
                config = {
                    "enabled": True,
                    "model": str(args.model),
                    **structure,
                    "threshold": float(threshold),
                    "max_replaced_parents_per_video": int(max_replaced),
                    "max_fp_increase": args.max_fp_increase,
                    "inner_changed_videos": int(changed),
                    "inner_fp_delta": int(fp_delta),
                    "n_train_records": int(train_count),
                    "n_positive_train_records": int(positive_count),
                }
                if args.max_fp_increase is not None and fp_delta > int(args.max_fp_increase):
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
                    best_config = {**config, "base_metrics": base_metrics, "diagnostics": diagnostics}
                    best_metrics = metrics
    return best_config, best_metrics


def run_strict_merge_noise_split_oof(folds: list[dict], args) -> dict:
    results = []
    for heldout_pos, heldout in enumerate(folds):
        calibration = [fold for pos, fold in enumerate(folds) if pos != heldout_pos]
        config, calibration_metrics = select_inner_config(calibration, args)
        if config.get("enabled"):
            train_records = label_merge_noise_records(
                _flatten(calibration, "base"),
                _flatten(calibration, "candidates"),
                _flatten(calibration, "scores"),
                _flatten(calibration, "evidence"),
                _flatten(calibration, "labels"),
                config,
                iou_threshold=float(args.iou_threshold),
            )
            model = _fit_parent_model(
                train_records,
                args,
                seed=int(args.seed) + 5000 + int(heldout.get("fold", heldout_pos)),
            )
            heldout_records = extract_merge_noise_parent_records(
                heldout["base"],
                heldout["candidates"],
                heldout["scores"],
                heldout["evidence"],
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
            )
            scores = _score_records(model, heldout_records)
            predictions = _apply_records(
                heldout["base"],
                heldout_records,
                scores,
                threshold=float(config["threshold"]),
                max_replaced_per_video=int(config["max_replaced_parents_per_video"]),
            )
            inference = {"n_records": int(len(heldout_records)), "score_max": float(scores.max()) if len(scores) else 0.0}
        else:
            predictions = _as_segment_lists(heldout["base"])
            inference = {"n_records": 0}
        metrics = evaluate_fused_predictions(predictions, heldout["labels"], iou_threshold=float(args.iou_threshold))
        results.append(
            {
                "fold": int(heldout.get("fold", heldout_pos)),
                "config": config,
                "calibration": calibration_metrics,
                "validation": metrics,
                "inference": inference,
                "predictions": [[[int(s), int(e)] for s, e in video] for video in predictions],
            }
        )
    return {"folds": results, "aggregate": aggregate_fold_metrics(results)}


def _load_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def _parse_optional_ints(text: str) -> list[int | None]:
    out = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(None if item.lower() in {"none", "null"} else int(item))
    return out


def _parse_optional_floats(text: str) -> list[float | None]:
    out = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(None if item.lower() in {"none", "null"} else float(item))
    return out


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
    parser.add_argument("--island-thresholds", default="0.45,0.55,0.65")
    parser.add_argument("--island-min-lengths", default="1,2")
    parser.add_argument("--island-min-gaps", default="0,1")
    parser.add_argument("--min-islands", default="2")
    parser.add_argument("--min-candidate-scores", default="0,0.05")
    parser.add_argument("--max-candidate-ranks", default="none,60")
    parser.add_argument("--min-island-coverages", default="0.5,0.75")
    parser.add_argument("--max-child-parent-ratios", default="0.7")
    parser.add_argument("--min-child-parent-coverages", default="0.5")
    parser.add_argument("--child-nms-ious", default="0.3")
    parser.add_argument("--max-children-per-parent", default="3")
    parser.add_argument("--thresholds", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--max-replaced-parents-per-video", default="1")
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    args.island_thresholds = _parse_float_list(args.island_thresholds)
    args.island_min_lengths = _parse_int_list(args.island_min_lengths)
    args.island_min_gaps = _parse_int_list(args.island_min_gaps)
    args.min_islands = _parse_int_list(args.min_islands)
    args.min_candidate_scores = _parse_float_list(args.min_candidate_scores)
    args.max_candidate_ranks = _parse_optional_ints(args.max_candidate_ranks)
    args.min_island_coverages = _parse_float_list(args.min_island_coverages)
    args.max_child_parent_ratios = _parse_float_list(args.max_child_parent_ratios)
    args.min_child_parent_coverages = _parse_float_list(args.min_child_parent_coverages)
    args.child_nms_ious = _parse_optional_floats(args.child_nms_ious)
    args.max_children_per_parent = _parse_int_list(args.max_children_per_parent)
    args.thresholds = _parse_float_list(args.thresholds)
    args.max_replaced_parents_per_video = _parse_int_list(args.max_replaced_parents_per_video)

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    export = _load_export(args.base_predictions)
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
    folds = build_topology_demand_folds(
        export,
        records,
        raw_by_video,
        feature_names,
        _parse_name_list(args.evidence_names),
        args.evidence_reducer,
    )
    result = run_strict_merge_noise_split_oof(folds, args)
    result.update(
        {
            "data_dir": args.data_dir,
            "base_predictions": args.base_predictions,
            "selector_diagnostics": selector_diagnostics,
            "model": args.model,
            "frontier_inspiration": "NoCo-style merged-adjacent-instance noise plus selective risk-controlled acceptance",
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
