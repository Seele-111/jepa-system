#!/usr/bin/env python3
"""
Train a proposal-level calibrator for true dual-JEPA segment localization.

This module keeps the task binary: candidate segment overlaps an annotated error
segment or it does not. Category and severity labels are ignored.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from train_segment_locator import VideoRecord, contiguous_segments, load_signal_dataset, train_val_split


PROPOSAL_FEATURE_NAMES = [
    "duration",
    "log_duration",
    "relative_start",
    "relative_end",
    "vjepa_mean",
    "vjepa_max",
    "vjepa_topk",
    "vjepa_std",
    "vjepa_range",
    "vjepa_slope",
    "vjepa_contrast",
    "ijepa_mean",
    "ijepa_max",
    "ijepa_topk",
    "ijepa_std",
    "ijepa_range",
    "ijepa_slope",
    "ijepa_contrast",
    "composite_mean",
    "composite_max",
    "composite_topk",
    "composite_std",
    "composite_range",
    "composite_slope",
    "composite_contrast",
    "vi_agreement",
    "composite_minus_vjepa",
    "composite_minus_ijepa",
]


@dataclass
class Proposal:
    video_idx: int
    segment: tuple[int, int]
    features: np.ndarray
    label: int
    best_iou: float


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0, end - start + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / max(1, union)


def _segments_from_scores(scores: np.ndarray, threshold: float, min_gap: int, min_length: int) -> list[tuple[int, int]]:
    binary = (np.asarray(scores) >= float(threshold)).astype(np.int64)
    segments = contiguous_segments(binary)
    if min_gap > 0 and len(segments) > 1:
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end - 1 <= min_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        segments = merged
    return [(start, end) for start, end in segments if end - start + 1 >= min_length]


def generate_candidates(
    composite_scores: np.ndarray,
    thresholds: Iterable[float] = (0.2, 0.3, 0.4, 0.5),
    min_gap: int = 2,
    min_length: int = 1,
) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for threshold in thresholds:
        candidates.update(_segments_from_scores(composite_scores, float(threshold), min_gap, min_length))
    return sorted(candidates)


def _topk_mean(values: np.ndarray, fraction: float = 0.25) -> float:
    if len(values) == 0:
        return 0.0
    k = max(1, int(round(len(values) * fraction)))
    return float(np.sort(values)[-k:].mean())


def _outside_values(values: np.ndarray, start: int, end: int) -> np.ndarray:
    length = end - start + 1
    left_start = max(0, start - length)
    left = values[left_start:start]
    right = values[end + 1 : min(len(values), end + 1 + length)]
    if len(left) == 0 and len(right) == 0:
        return np.asarray([], dtype=np.float32)
    return np.concatenate([left, right])


def extract_proposal_features(signals: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float32)
    start, end = segment
    start = max(0, min(start, len(signals) - 1))
    end = max(start, min(end, len(signals) - 1))
    duration = end - start + 1
    values = signals[start : end + 1]

    features: list[float] = [
        float(duration),
        float(np.log1p(duration)),
        float(start / max(1, len(signals) - 1)),
        float(end / max(1, len(signals) - 1)),
    ]

    for channel in range(3):
        channel_values = values[:, channel]
        outside = _outside_values(signals[:, channel], start, end)
        contrast = float(channel_values.mean() - outside.mean()) if len(outside) else 0.0
        slope = float(channel_values[-1] - channel_values[0]) / max(1, duration - 1)
        features.extend(
            [
                float(channel_values.mean()),
                float(channel_values.max()),
                _topk_mean(channel_values),
                float(channel_values.std()),
                float(channel_values.max() - channel_values.min()),
                slope,
                contrast,
            ]
        )

    v = values[:, 0]
    i = values[:, 1]
    c = values[:, 2]
    agreement = np.minimum(v, i) / (np.maximum(v, i) + 1e-6)
    features.extend(
        [
            float(agreement.mean()),
            float(c.mean() - v.mean()),
            float(c.mean() - i.mean()),
        ]
    )
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def label_candidates(
    candidates: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    gt_segments = contiguous_segments(np.asarray(labels, dtype=np.int64))
    y = np.zeros(len(candidates), dtype=np.int64)
    best = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        if gt_segments:
            best[idx] = max(interval_iou(candidate, gt) for gt in gt_segments)
        y[idx] = int(best[idx] >= iou_threshold)
    return y, best


def build_proposals(
    records: list[VideoRecord],
    indices: Iterable[int],
    candidate_thresholds: Iterable[float],
    iou_threshold: float,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    for video_idx in indices:
        record = records[video_idx]
        candidates = generate_candidates(record.signals[:, 2], thresholds=candidate_thresholds)
        labels, best_iou = label_candidates(candidates, record.labels, iou_threshold=iou_threshold)
        for segment, label, iou in zip(candidates, labels, best_iou):
            proposals.append(
                Proposal(
                    video_idx=video_idx,
                    segment=segment,
                    features=extract_proposal_features(record.signals, segment),
                    label=int(label),
                    best_iou=float(iou),
                )
            )
    return proposals


def nms_segments(segments: list[tuple[int, int, float]], iou_threshold: float = 0.5) -> list[tuple[int, int]]:
    ordered = sorted(segments, key=lambda item: item[2], reverse=True)
    kept: list[tuple[int, int, float]] = []
    for candidate in ordered:
        if all(interval_iou((candidate[0], candidate[1]), (item[0], item[1])) <= iou_threshold for item in kept):
            kept.append(candidate)
    return [(start, end) for start, end, _ in sorted(kept, key=lambda item: item[0])]


def evaluate_segment_predictions(
    predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    iou_threshold: float = 0.3,
) -> dict:
    tp = fp = fn = 0
    for pred_segments, label_array in zip(predictions, labels):
        gt_segments = contiguous_segments(np.asarray(label_array, dtype=np.int64))
        matched: set[int] = set()
        local_tp = 0
        for pred in pred_segments:
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gt_segments):
                if idx in matched:
                    continue
                iou = interval_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched.add(best_idx)
                local_tp += 1
        tp += local_tp
        fp += len(pred_segments) - local_tp
        fn += len(gt_segments) - local_tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


class NumpyLogisticClassifier:
    def __init__(self, lr: float = 0.05, steps: int = 800, l2: float = 1e-3):
        self.lr = lr
        self.steps = steps
        self.l2 = l2
        self.mean = None
        self.std = None
        self.weights = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-6)
        z = (x - self.mean) / self.std
        z = np.c_[np.ones(len(z)), z]
        self.weights = np.zeros(z.shape[1], dtype=np.float64)
        pos_weight = max(1.0, float((len(y) - y.sum()) / max(1, y.sum())))
        weights = np.where(y > 0, pos_weight, 1.0)
        for _ in range(self.steps):
            logits = z @ self.weights
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = (z.T @ ((probs - y) * weights)) / max(1, len(y))
            grad[1:] += self.l2 * self.weights[1:]
            self.weights -= self.lr * grad
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        z = np.c_[np.ones(len(z)), z]
        probs = 1.0 / (1.0 + np.exp(-(z @ self.weights)))
        return np.stack([1.0 - probs, probs], axis=1)


def fit_classifier(x: np.ndarray, y: np.ndarray, seed: int):
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except Exception:
        return NumpyLogisticClassifier().fit(x, y)

    clf = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=2,
        subsample=0.8,
        random_state=seed,
    )
    clf.fit(x, y)
    return clf


def predict_video_segments(
    classifier,
    record: VideoRecord,
    candidate_thresholds: Iterable[float],
    prob_threshold: float,
    nms_iou: float,
) -> list[tuple[int, int]]:
    candidates = generate_candidates(record.signals[:, 2], thresholds=candidate_thresholds)
    if not candidates:
        return []
    x = np.stack([extract_proposal_features(record.signals, segment) for segment in candidates])
    probs = classifier.predict_proba(x)[:, 1]
    scored = [
        (segment[0], segment[1], float(prob))
        for segment, prob in zip(candidates, probs)
        if prob >= prob_threshold
    ]
    return nms_segments(scored, iou_threshold=nms_iou)


def select_inference_params(
    classifier,
    records: list[VideoRecord],
    val_idx: list[int],
    candidate_thresholds: Iterable[float],
    iou_threshold: float,
) -> tuple[dict, dict]:
    best_params = None
    best_metrics = None
    best_key = None
    for prob_threshold in np.linspace(0.15, 0.85, 15):
        for nms_iou in [0.3, 0.5, 0.7]:
            predictions = [
                predict_video_segments(classifier, records[idx], candidate_thresholds, float(prob_threshold), nms_iou)
                for idx in val_idx
            ]
            metrics = evaluate_segment_predictions(predictions, [records[idx].labels for idx in val_idx], iou_threshold)
            key = (metrics["f1"], metrics["recall"], -metrics["fp"])
            if best_key is None or key > best_key:
                best_key = key
                best_params = {"prob_threshold": float(prob_threshold), "nms_iou": float(nms_iou)}
                best_metrics = metrics
    return best_params, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="/home/zzy/jepa_data/proposal_calibrator.pkl")
    parser.add_argument("--summary", default="")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--candidate-thresholds", default="0.2,0.3,0.4,0.5")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    train_idx, val_idx = train_val_split(records, args.val_ratio, args.seed)
    candidate_thresholds = [float(x) for x in args.candidate_thresholds.split(",") if x.strip()]
    train_props = build_proposals(records, train_idx, candidate_thresholds, args.iou_threshold)
    if not train_props:
        raise RuntimeError("no training proposals generated")
    x_train = np.stack([p.features for p in train_props])
    y_train = np.asarray([p.label for p in train_props], dtype=np.int64)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"proposal labels need both classes, got positives={int(y_train.sum())}/{len(y_train)}")

    classifier = fit_classifier(x_train, y_train, args.seed)
    params, val_metrics = select_inference_params(classifier, records, val_idx, candidate_thresholds, args.iou_threshold)
    train_predictions = [
        predict_video_segments(classifier, records[idx], candidate_thresholds, params["prob_threshold"], params["nms_iou"])
        for idx in train_idx
    ]
    train_metrics = evaluate_segment_predictions(train_predictions, [records[idx].labels for idx in train_idx], args.iou_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(
            {
                "classifier": classifier,
                "feature_names": PROPOSAL_FEATURE_NAMES,
                "candidate_thresholds": candidate_thresholds,
                "params": params,
                "iou_threshold": args.iou_threshold,
            },
            f,
        )

    summary = {
        "data_dir": args.data_dir,
        "output": str(output),
        "n_videos": len(records),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_names": [records[idx].name for idx in train_idx],
        "val_names": [records[idx].name for idx in val_idx],
        "n_train_proposals": len(train_props),
        "n_positive_train_proposals": int(y_train.sum()),
        "candidate_thresholds": candidate_thresholds,
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
