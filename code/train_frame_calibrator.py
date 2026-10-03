#!/usr/bin/env python3
"""
Frame-level calibrator for JEPA event-token features.

This is a binary-only baseline: train frame error probabilities, then convert
probabilities to error segments with validation-selected temporal postprocess.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from train_proposal_calibrator import NumpyLogisticClassifier
from train_segment_locator import VideoRecord, evaluate_records, load_signal_dataset, select_postprocess_params, train_val_split


@dataclass
class FrameTable:
    x: np.ndarray
    y: np.ndarray
    slices: list[tuple[int, int, int]]
    names: list[str]
    labels: list[np.ndarray]


def make_frame_table(records: list[VideoRecord], indices: list[int]) -> FrameTable:
    x_parts = []
    y_parts = []
    slices = []
    names = []
    labels = []
    offset = 0
    for video_idx in indices:
        record = records[video_idx]
        n = len(record.labels)
        x_parts.append(record.signals.astype(np.float32))
        y_parts.append(record.labels.astype(np.int64))
        slices.append((video_idx, offset, offset + n))
        names.append(record.name)
        labels.append(record.labels.copy())
        offset += n
    return FrameTable(
        x=np.concatenate(x_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        slices=slices,
        names=names,
        labels=labels,
    )


def records_from_probabilities(table: FrameTable, probabilities: np.ndarray) -> list[dict]:
    records = []
    for local_idx, (_, start, end) in enumerate(table.slices):
        records.append(
            {
                "name": table.names[local_idx],
                "labels": table.labels[local_idx],
                "probs": probabilities[start:end].astype(np.float32),
            }
        )
    return records


def fit_classifier(x: np.ndarray, y: np.ndarray, seed: int, model_name: str):
    if model_name == "logreg":
        return NumpyLogisticClassifier(lr=0.03, steps=1200, l2=1e-3).fit(x, y)
    try:
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    except Exception:
        return NumpyLogisticClassifier(lr=0.03, steps=1200, l2=1e-3).fit(x, y)

    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    sample_weight = np.where(y > 0, neg / pos, 1.0)
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(x, y)
        return clf
    if model_name == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(x, y)
        return clf
    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.75,
        random_state=seed,
    )
    clf.fit(x, y, sample_weight=sample_weight)
    return clf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="/home/zzy/jepa_data/frame_calibrator.pkl")
    parser.add_argument("--summary", default="")
    parser.add_argument("--model", choices=["gbdt", "rf", "extratrees", "logreg"], default="gbdt")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    train_idx, val_idx = train_val_split(records, args.val_ratio, args.seed)
    train_table = make_frame_table(records, train_idx)
    val_table = make_frame_table(records, val_idx)
    clf = fit_classifier(train_table.x, train_table.y, args.seed, args.model)
    val_probs = clf.predict_proba(val_table.x)[:, 1].astype(np.float32)
    val_records = records_from_probabilities(val_table, val_probs)
    params, val_metrics = select_postprocess_params(val_records, iou_threshold=args.iou_threshold)

    train_probs = clf.predict_proba(train_table.x)[:, 1].astype(np.float32)
    train_metrics = evaluate_records(records_from_probabilities(train_table, train_probs), params, iou_threshold=args.iou_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "params": params,
                "model": args.model,
                "task": "binary_error_segment_localization",
            },
            f,
        )
    summary = {
        "data_dir": args.data_dir,
        "output": str(output),
        "model": args.model,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_names": [records[idx].name for idx in train_idx],
        "val_names": [records[idx].name for idx in val_idx],
        "n_train_frames": int(len(train_table.y)),
        "n_positive_train_frames": int(train_table.y.sum()),
        "params": params,
        "train": train_metrics,
        "validation": val_metrics,
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
