#!/usr/bin/env python3
"""Strict OOF prediction-set switcher for JEPA locator variants.

This module decides whether to keep a high-precision base prediction set or
switch an entire video to a recall-oriented variant. Rule selection is performed
on other folds only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from cross_validate_segment_locator import aggregate_fold_metrics
from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _as_segments(raw: Iterable[Iterable[int]]) -> list[Segment]:
    return [_normalise_segment((int(item[0]), int(item[1]))) for item in raw]


def _segment_length(segment: Segment) -> int:
    return max(1, int(segment[1]) - int(segment[0]) + 1)


def _total_length(segments: list[Segment]) -> int:
    return int(sum(_segment_length(segment) for segment in segments))


def _mean_best_iou(source: list[Segment], target: list[Segment]) -> float:
    if not source:
        return 1.0 if not target else 0.0
    return float(np.mean([max((interval_iou(item, other) for other in target), default=0.0) for item in source]))


def prediction_set_features(base: list[tuple[int, int]], recall: list[tuple[int, int]]) -> dict:
    base_segments = sorted({_normalise_segment(item) for item in base})
    recall_segments = sorted({_normalise_segment(item) for item in recall})
    base_count = len(base_segments)
    recall_count = len(recall_segments)
    base_length = _total_length(base_segments)
    recall_length = _total_length(recall_segments)
    recall_to_base = _mean_best_iou(recall_segments, base_segments)
    base_to_recall = _mean_best_iou(base_segments, recall_segments)
    return {
        "base_count": int(base_count),
        "recall_count": int(recall_count),
        "count_delta": int(recall_count - base_count),
        "base_length": int(base_length),
        "recall_length": int(recall_length),
        "length_ratio": float(recall_length / max(1, base_length)),
        "mean_recall_to_base_iou": float(recall_to_base),
        "mean_base_to_recall_iou": float(base_to_recall),
        "mutual_iou": float(min(recall_to_base, base_to_recall)),
        "extra_segments": int(sum(1 for item in recall_segments if max((interval_iou(item, b) for b in base_segments), default=0.0) < 0.3)),
        "dropped_segments": int(sum(1 for item in base_segments if max((interval_iou(item, r) for r in recall_segments), default=0.0) < 0.3)),
    }


def _should_switch(features: dict, config: dict) -> bool:
    if not config.get("enabled", False):
        return False
    return (
        int(config["min_count_delta"]) <= int(features["count_delta"]) <= int(config["max_count_delta"])
        and float(config["min_length_ratio"]) <= float(features["length_ratio"]) <= float(config["max_length_ratio"])
        and float(features["mutual_iou"]) >= float(config["min_mutual_iou"])
        and int(features["extra_segments"]) <= int(config["max_extra_segments"])
        and int(features["dropped_segments"]) <= int(config["max_dropped_segments"])
    )


def apply_set_switch_config(
    base_predictions: list[list[tuple[int, int]]],
    recall_predictions: list[list[tuple[int, int]]],
    config: dict,
) -> list[list[Segment]]:
    if len(base_predictions) != len(recall_predictions):
        raise ValueError("base and recall prediction video counts must match")
    selected: list[list[Segment]] = []
    for base_raw, recall_raw in zip(base_predictions, recall_predictions):
        base = sorted({_normalise_segment(item) for item in base_raw})
        recall = sorted({_normalise_segment(item) for item in recall_raw})
        features = prediction_set_features(base, recall)
        selected.append(recall if _should_switch(features, config) else base)
    return selected


def _flatten(folds: list[dict], key: str) -> list:
    values = []
    for fold in folds:
        values.extend(fold[key])
    return values


def _json_segments(segments: list[Segment]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in segments]


def build_prediction_export_from_switch_result(
    result: dict,
    source_folds: list[dict],
    include_labels: bool = False,
) -> dict:
    """Build an OOF prediction export from a set-switching result."""
    out_folds: list[dict] = []
    source_by_fold = {int(fold.get("fold", idx)): fold for idx, fold in enumerate(source_folds)}
    for fold_pos, fold_result in enumerate(result.get("folds", [])):
        fold_id = int(fold_result.get("fold", fold_pos))
        source = source_by_fold.get(fold_id, source_folds[fold_pos] if fold_pos < len(source_folds) else {})
        item = {
            "fold": fold_id,
            "train_idx": [int(idx) for idx in source.get("train_idx", [])],
            "val_idx": [int(idx) for idx in source.get("val_idx", [])],
            "val_names": [str(name) for name in source.get("val_names", [])],
            "predictions": fold_result.get("predictions", []),
            "validation": fold_result.get("validation", {}),
            "config": fold_result.get("config", {}),
        }
        if include_labels:
            item["labels"] = [
                np.asarray(label, dtype=np.int64).reshape(-1).astype(int).tolist()
                for label in source.get("labels", [])
            ]
        out_folds.append(item)
    return {
        "task": "binary_error_segment_localization_oof_predictions",
        "source": "strict_prediction_set_switcher",
        "n_videos": int(sum(len(fold.get("val_idx", [])) for fold in source_folds)),
        "folds": out_folds,
    }


def select_set_switch_config(
    folds: list[dict],
    max_fp_increase: int | None,
    max_count_deltas: Iterable[int],
    min_length_ratios: Iterable[float],
    max_length_ratios: Iterable[float],
    min_mutual_ious: Iterable[float],
    max_extra_segments_list: Iterable[int],
    max_dropped_segments_list: Iterable[int],
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in _flatten(folds, "base")]
    recall = [[_normalise_segment(item) for item in video] for video in _flatten(folds, "recall")]
    labels = _flatten(folds, "labels")
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=float(iou_threshold))
    best_config: dict = {"enabled": False, "base_metrics": base_metrics}
    best_metrics = base_metrics
    best_predictions = base
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["segment"]["precision"]),
        0,
    )
    base_fp = int(base_metrics["segment"]["fp"])
    for max_count_delta in max_count_deltas:
        for min_length_ratio in min_length_ratios:
            for max_length_ratio in max_length_ratios:
                if float(max_length_ratio) < float(min_length_ratio):
                    continue
                for min_mutual_iou in min_mutual_ious:
                    for max_extra_segments in max_extra_segments_list:
                        for max_dropped_segments in max_dropped_segments_list:
                            config = {
                                "enabled": True,
                                "min_count_delta": -int(max_dropped_segments),
                                "max_count_delta": int(max_count_delta),
                                "min_length_ratio": float(min_length_ratio),
                                "max_length_ratio": float(max_length_ratio),
                                "min_mutual_iou": float(min_mutual_iou),
                                "max_extra_segments": int(max_extra_segments),
                                "max_dropped_segments": int(max_dropped_segments),
                            }
                            predictions = apply_set_switch_config(base, recall, config)
                            changed = sum(int(sorted(before) != sorted(after)) for before, after in zip(base, predictions))
                            if changed <= 0:
                                continue
                            metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(iou_threshold))
                            if max_fp_increase is not None and int(metrics["segment"]["fp"]) > base_fp + int(max_fp_increase):
                                continue
                            key = (
                                float(metrics["segment"]["f1"]),
                                float(metrics["segment"]["recall"]),
                                float(metrics["segment"]["precision"]),
                                -int(changed),
                            )
                            if key > best_key:
                                best_key = key
                                best_config = {
                                    **config,
                                    "changed_videos": int(changed),
                                    "base_metrics": base_metrics,
                                }
                                best_metrics = metrics
                                best_predictions = predictions
    return best_config, best_metrics, best_predictions


def run_fold_heldout_set_switching(
    folds: list[dict],
    max_fp_increase: int | None,
    max_count_deltas: Iterable[int],
    min_length_ratios: Iterable[float],
    max_length_ratios: Iterable[float],
    min_mutual_ious: Iterable[float],
    max_extra_segments_list: Iterable[int],
    max_dropped_segments_list: Iterable[int],
    iou_threshold: float = 0.3,
) -> dict:
    results = []
    for heldout_idx, heldout in enumerate(folds):
        calibration = [fold for idx, fold in enumerate(folds) if idx != heldout_idx]
        config, calibration_metrics, _ = select_set_switch_config(
            calibration,
            max_fp_increase=max_fp_increase,
            max_count_deltas=max_count_deltas,
            min_length_ratios=min_length_ratios,
            max_length_ratios=max_length_ratios,
            min_mutual_ious=min_mutual_ious,
            max_extra_segments_list=max_extra_segments_list,
            max_dropped_segments_list=max_dropped_segments_list,
            iou_threshold=float(iou_threshold),
        )
        predictions = apply_set_switch_config(heldout["base"], heldout["recall"], config)
        metrics = evaluate_fused_predictions(predictions, heldout["labels"], iou_threshold=float(iou_threshold))
        results.append(
            {
                "fold": int(heldout.get("fold", heldout_idx)),
                "config": config,
                "calibration": calibration_metrics,
                "validation": metrics,
                "predictions": [_json_segments(video) for video in predictions],
            }
        )
    return {"folds": results, "aggregate": aggregate_fold_metrics(results)}


def _parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def load_prediction_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def build_folds_from_exports(base_export: dict, recall_export: dict) -> list[dict]:
    folds = []
    for base_fold, recall_fold in zip(base_export.get("folds", []), recall_export.get("folds", [])):
        if list(base_fold.get("val_idx", [])) != list(recall_fold.get("val_idx", [])):
            raise ValueError(f"fold={base_fold.get('fold')} val_idx mismatch")
        labels = base_fold.get("labels") or recall_fold.get("labels")
        if labels is None:
            raise ValueError("prediction exports need labels for offline CV set switching")
        folds.append(
            {
                "fold": int(base_fold.get("fold", len(folds))),
                "train_idx": [int(idx) for idx in base_fold.get("train_idx", [])],
                "val_idx": [int(idx) for idx in base_fold.get("val_idx", [])],
                "val_names": [str(name) for name in base_fold.get("val_names", [])],
                "base": base_fold.get("predictions", []),
                "recall": recall_fold.get("predictions", []),
                "labels": [np.asarray(item, dtype=np.int64) for item in labels],
            }
        )
    return folds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--recall-predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--max-count-deltas", default="0,1,2,3")
    parser.add_argument("--min-length-ratios", default="0.25,0.5,0.75")
    parser.add_argument("--max-length-ratios", default="1.0,1.25,1.5")
    parser.add_argument("--min-mutual-ious", default="0,0.25,0.5,0.75")
    parser.add_argument("--max-extra-segments", default="0,1,2")
    parser.add_argument("--max-dropped-segments", default="0,1,2")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--include-prediction-labels", action="store_true")
    args = parser.parse_args()

    folds = build_folds_from_exports(load_prediction_export(args.base_predictions), load_prediction_export(args.recall_predictions))
    result = run_fold_heldout_set_switching(
        folds,
        max_fp_increase=args.max_fp_increase,
        max_count_deltas=_parse_ints(args.max_count_deltas),
        min_length_ratios=_parse_floats(args.min_length_ratios),
        max_length_ratios=_parse_floats(args.max_length_ratios),
        min_mutual_ious=_parse_floats(args.min_mutual_ious),
        max_extra_segments_list=_parse_ints(args.max_extra_segments),
        max_dropped_segments_list=_parse_ints(args.max_dropped_segments),
        iou_threshold=args.iou_threshold,
    )
    result.update(
        {
            "base_predictions": str(args.base_predictions),
            "recall_predictions": str(args.recall_predictions),
            "max_fp_increase": args.max_fp_increase,
            "max_count_deltas": args.max_count_deltas,
            "min_length_ratios": args.min_length_ratios,
            "max_length_ratios": args.max_length_ratios,
            "min_mutual_ious": args.min_mutual_ious,
            "max_extra_segments": args.max_extra_segments,
            "max_dropped_segments": args.max_dropped_segments,
            "iou_threshold": args.iou_threshold,
        }
    )
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.predictions_out:
        prediction_export = build_prediction_export_from_switch_result(
            result,
            folds,
            include_labels=bool(args.include_prediction_labels),
        )
        pred_out = Path(args.predictions_out)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        pred_out.write_text(json.dumps(prediction_export, indent=2), encoding="utf-8")
        print(f"wrote {pred_out}", flush=True)
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
