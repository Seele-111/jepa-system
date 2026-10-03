#!/usr/bin/env python3
"""Evaluate OOF prediction bundles with the Round-2 matching protocol.

The bundle format is the JSON emitted by ``cross_validate_selector_fusion``
with ``--include-prediction-labels``. External baselines can export the same
shape without importing any JEPA code.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from selector_fusion import evaluate_fused_predictions


def _load_groups(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            str(row.get("video_name") or row.get("name") or row.get("filename")): str(
                row.get("generator") or row.get("group") or "unknown"
            )
            for row in csv.DictReader(handle)
        }


def _flatten(bundle: dict[str, Any]) -> list[tuple[str, list[tuple[int, int]], np.ndarray]]:
    rows: list[tuple[str, list[tuple[int, int]], np.ndarray]] = []
    for fold in bundle.get("folds", []):
        names = fold.get("val_names", [])
        predictions = fold.get("predictions", [])
        labels = fold.get("labels")
        if labels is None:
            raise ValueError("prediction bundle must include labels for protocol evaluation")
        if not (len(names) == len(predictions) == len(labels)):
            raise ValueError(f"fold={fold.get('fold')} names/predictions/labels mismatch")
        for name, prediction, label in zip(names, predictions, labels):
            segments = [(int(item[0]), int(item[1])) for item in prediction]
            rows.append((str(name), segments, np.asarray(label, dtype=np.int64)))
    if not rows:
        raise ValueError("prediction bundle contains no folds")
    return rows


def exclude_names(bundle: dict[str, Any], names: set[str]) -> dict[str, Any]:
    if not names:
        return bundle
    filtered = dict(bundle)
    filtered_folds = []
    for fold in bundle.get("folds", []):
        keep = [idx for idx, name in enumerate(fold.get("val_names", [])) if str(name) not in names]
        row = dict(fold)
        for key in ("val_names", "predictions", "labels"):
            if row.get(key) is not None:
                row[key] = [row[key][idx] for idx in keep]
        if keep:
            filtered_folds.append(row)
    filtered["folds"] = filtered_folds
    return filtered


def evaluate_bundle(bundle: dict[str, Any], iou_threshold: float, groups: dict[str, str]) -> dict[str, Any]:
    rows = _flatten(bundle)
    predictions = [row[1] for row in rows]
    labels = [row[2] for row in rows]
    fold_reports = []
    for fold in bundle.get("folds", []):
        fold_reports.append(
            evaluate_fused_predictions(
                [[(int(item[0]), int(item[1])) for item in prediction] for prediction in fold.get("predictions", [])],
                [np.asarray(label, dtype=np.int64) for label in fold.get("labels", [])],
                iou_threshold=iou_threshold,
            )
        )

    def macro_summary(key: str) -> dict[str, float]:
        values = {
            metric: np.asarray([float(report[key][metric]) for report in fold_reports], dtype=np.float64)
            for metric in ("precision", "recall", "f1")
        }
        return {
            f"{metric}_mean": float(value.mean()) for metric, value in values.items()
        } | {
            f"{metric}_std": float(value.std(ddof=0)) for metric, value in values.items()
        }

    result: dict[str, Any] = {
        "task": bundle.get("task", "unknown"),
        "videos": len(rows),
        "iou_threshold": float(iou_threshold),
        "overall": evaluate_fused_predictions(predictions, labels, iou_threshold=iou_threshold),
        "fold_mean": {"frame": macro_summary("frame"), "segment": macro_summary("segment")},
        "fold_reports": fold_reports,
    }
    grouped: dict[str, list[tuple[list[tuple[int, int]], np.ndarray]]] = defaultdict(list)
    for name, prediction, label in rows:
        grouped[groups.get(name, "unknown")].append((prediction, label))
    result["groups"] = {
        group: evaluate_fused_predictions([item[0] for item in values], [item[1] for item in values], iou_threshold=iou_threshold)
        for group, values in sorted(grouped.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iou-thresholds", default="0.3,0.5")
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--bundle-key", default="", help="Optional key under a top-level 'bundles' mapping")
    parser.add_argument("--exclude-names", default="", help="Comma-separated video names to exclude")
    args = parser.parse_args()
    bundle = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.bundle_key:
        try:
            bundle = bundle["bundles"][args.bundle_key]
        except KeyError as exc:
            raise ValueError(f"bundle key {args.bundle_key!r} not found") from exc
    excluded = {value.strip() for value in args.exclude_names.split(",") if value.strip()}
    bundle = exclude_names(bundle, excluded)
    groups = _load_groups(args.groups_csv)
    output = {
        "schema_version": "round2-prediction-report-v1",
        "source": str(args.predictions),
        "bundle_key": args.bundle_key or None,
        "excluded_names": sorted(excluded),
        "reports": [evaluate_bundle(bundle, float(value), groups) for value in args.iou_thresholds.split(",") if value.strip()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reports": len(output["reports"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
