#!/usr/bin/env python3
"""Strict OOF prediction-level fusion for JEPA segment locators.

The module combines a high-precision base export with a recall-enhanced export.
For each held-out fold, it selects the fusion rule on all other folds and then
applies that frozen rule to the held-out fold.
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


def _parse_optional_floats(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in str(text).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(None if stripped.lower() in {"none", "null"} else float(stripped))
    if not values:
        raise ValueError("at least one value is required")
    return values


def _parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _json_segments(segments: list[Segment]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in segments]


def _as_segments(raw: Iterable[Iterable[int]]) -> list[Segment]:
    return [_normalise_segment((int(item[0]), int(item[1]))) for item in raw]


def candidate_additions(
    base: list[tuple[int, int]],
    recall: list[tuple[int, int]],
    max_base_iou: float | None,
) -> list[Segment]:
    """Return recall predictions that are not already covered by base."""
    base_segments = [_normalise_segment(item) for item in base]
    additions: list[Segment] = []
    seen = set(base_segments)
    for item in recall:
        candidate = _normalise_segment(item)
        if candidate in seen:
            continue
        if max_base_iou is not None:
            overlap = max((float(interval_iou(candidate, base_item)) for base_item in base_segments), default=0.0)
            if overlap > float(max_base_iou):
                continue
        additions.append(candidate)
        seen.add(candidate)
    return sorted(additions)


def _nms_additions(additions: list[Segment], nms_iou: float | None, max_items: int) -> list[Segment]:
    if not additions or int(max_items) <= 0:
        return []
    kept: list[Segment] = []
    for candidate in sorted(additions, key=lambda item: (item[0], item[1])):
        if nms_iou is not None and any(interval_iou(candidate, existing) > float(nms_iou) for existing in kept):
            continue
        kept.append(candidate)
        if len(kept) >= int(max_items):
            break
    return kept


def apply_protected_addition_config(
    base_predictions: list[list[tuple[int, int]]],
    recall_predictions: list[list[tuple[int, int]]],
    max_base_iou: float | None,
    max_additions_per_video: int,
    nms_iou: float | None,
) -> list[list[Segment]]:
    if len(base_predictions) != len(recall_predictions):
        raise ValueError("base and recall prediction video counts must match")
    fused: list[list[Segment]] = []
    for base_raw, recall_raw in zip(base_predictions, recall_predictions):
        base = sorted({_normalise_segment(item) for item in base_raw})
        additions = candidate_additions(base, recall_raw, max_base_iou=max_base_iou)
        additions = _nms_additions(additions, nms_iou=nms_iou, max_items=int(max_additions_per_video))
        fused.append(sorted(set(base + additions)))
    return fused


def _flatten_fold_field(folds: list[dict], key: str) -> list:
    values = []
    for fold in folds:
        values.extend(fold[key])
    return values


def _evaluate_fold_bundle(folds: list[dict], predictions: list[list[Segment]], iou_threshold: float) -> dict:
    labels = _flatten_fold_field(folds, "labels")
    segment_metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(iou_threshold))
    return segment_metrics


def select_config_on_folds(
    folds: list[dict],
    max_fp_increase: int | None,
    max_base_ious: Iterable[float | None],
    max_additions_per_videos: Iterable[int],
    nms_ious: Iterable[float | None],
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in _flatten_fold_field(folds, "base")]
    recall = [[_normalise_segment(item) for item in video] for video in _flatten_fold_field(folds, "recall")]
    labels = _flatten_fold_field(folds, "labels")
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
    for max_base_iou in max_base_ious:
        for max_additions_per_video in max_additions_per_videos:
            for nms_iou in nms_ious:
                predictions = apply_protected_addition_config(
                    base,
                    recall,
                    max_base_iou=max_base_iou,
                    max_additions_per_video=int(max_additions_per_video),
                    nms_iou=nms_iou,
                )
                metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(iou_threshold))
                if max_fp_increase is not None and int(metrics["segment"]["fp"]) > base_fp + int(max_fp_increase):
                    continue
                changed = sum(int(sorted(before) != sorted(after)) for before, after in zip(base, predictions))
                key = (
                    float(metrics["segment"]["f1"]),
                    float(metrics["segment"]["recall"]),
                    float(metrics["segment"]["precision"]),
                    -int(changed),
                )
                if key > best_key:
                    best_key = key
                    best_metrics = metrics
                    best_predictions = predictions
                    best_config = {
                        "enabled": True,
                        "max_base_iou": max_base_iou,
                        "max_additions_per_video": int(max_additions_per_video),
                        "nms_iou": nms_iou,
                        "changed_videos": int(changed),
                        "base_metrics": base_metrics,
                    }
    return best_config, best_metrics, best_predictions


def run_fold_heldout_fusion(
    folds: list[dict],
    max_fp_increase: int | None,
    max_base_ious: Iterable[float | None],
    max_additions_per_videos: Iterable[int],
    nms_ious: Iterable[float | None],
    iou_threshold: float = 0.3,
) -> dict:
    results = []
    for heldout_idx, heldout in enumerate(folds):
        calibration = [fold for idx, fold in enumerate(folds) if idx != heldout_idx]
        config, calibration_metrics, _ = select_config_on_folds(
            calibration,
            max_fp_increase=max_fp_increase,
            max_base_ious=max_base_ious,
            max_additions_per_videos=max_additions_per_videos,
            nms_ious=nms_ious,
            iou_threshold=float(iou_threshold),
        )
        if config.get("enabled"):
            predictions = apply_protected_addition_config(
                heldout["base"],
                heldout["recall"],
                max_base_iou=config["max_base_iou"],
                max_additions_per_video=int(config["max_additions_per_video"]),
                nms_iou=config["nms_iou"],
            )
        else:
            predictions = [[_normalise_segment(item) for item in video] for video in heldout["base"]]
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
    aggregate = aggregate_fold_metrics(results)
    return {"folds": results, "aggregate": aggregate}


def load_prediction_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def build_folds_from_exports(base_export: dict, recall_export: dict) -> list[dict]:
    base_folds = base_export.get("folds", [])
    recall_folds = recall_export.get("folds", [])
    if len(base_folds) != len(recall_folds):
        raise ValueError("base and recall exports must have the same number of folds")
    folds: list[dict] = []
    for base_fold, recall_fold in zip(base_folds, recall_folds):
        if list(base_fold.get("val_idx", [])) != list(recall_fold.get("val_idx", [])):
            raise ValueError(f"fold={base_fold.get('fold')} val_idx mismatch")
        labels = base_fold.get("labels") or recall_fold.get("labels")
        if labels is None:
            raise ValueError("prediction exports need labels for offline CV fusion")
        folds.append(
            {
                "fold": int(base_fold.get("fold", len(folds))),
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
    parser.add_argument("--max-base-ious", default="0,0.05,0.1,0.25,none")
    parser.add_argument("--max-additions-per-video", default="1,2,3")
    parser.add_argument("--nms-ious", default="none,0.1,0.3")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    folds = build_folds_from_exports(
        load_prediction_export(args.base_predictions),
        load_prediction_export(args.recall_predictions),
    )
    result = run_fold_heldout_fusion(
        folds,
        max_fp_increase=args.max_fp_increase,
        max_base_ious=_parse_optional_floats(args.max_base_ious),
        max_additions_per_videos=_parse_ints(args.max_additions_per_video),
        nms_ious=_parse_optional_floats(args.nms_ious),
        iou_threshold=args.iou_threshold,
    )
    result.update(
        {
            "base_predictions": str(args.base_predictions),
            "recall_predictions": str(args.recall_predictions),
            "max_fp_increase": args.max_fp_increase,
            "max_base_ious": args.max_base_ious,
            "max_additions_per_video": args.max_additions_per_video,
            "nms_ious": args.nms_ious,
            "iou_threshold": args.iou_threshold,
        }
    )
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
