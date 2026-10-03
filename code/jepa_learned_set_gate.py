#!/usr/bin/env python3
"""Learned JEPA reliability-aware prediction-set gate.

For each video, the gate chooses between a high-precision base prediction set
and a recall-oriented variant. Outer held-out folds are never used to train or
choose thresholds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_prediction_set_switcher import (
    build_folds_from_exports,
    load_prediction_export,
    prediction_set_features,
)
from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou
from train_proposal_set_selector import fit_classifier
from train_segment_locator import VideoRecord, _load_feature_names, load_signal_dataset


Segment = tuple[int, int]


DEFAULT_EVIDENCE_NAMES = [
    "true_vjepa_raw_rank",
    "true_ijepa_dense_raw_rank",
    "dual_jepa_composite_rank",
    "vi_agreement",
    "vi_product",
    "composite_times_vi_agreement",
]


@dataclass
class GateExample:
    fold_idx: int
    video_idx: int
    features: np.ndarray
    label: int
    base: list[Segment]
    recall: list[Segment]
    labels: np.ndarray


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: Segment) -> int:
    return max(1, int(segment[1]) - int(segment[0]) + 1)


def _segment_mask(n: int, segments: list[Segment]) -> np.ndarray:
    mask = np.zeros(max(0, int(n)), dtype=bool)
    for segment_raw in segments:
        start, end = _normalise_segment(segment_raw)
        if len(mask) == 0:
            continue
        left = max(0, min(start, len(mask) - 1))
        right = max(left, min(end, len(mask) - 1))
        mask[left : right + 1] = True
    return mask


def _evidence_matrix(record: VideoRecord, feature_names: list[str], evidence_names: list[str]) -> np.ndarray:
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    channels = [name_to_idx[name] for name in evidence_names if name in name_to_idx]
    if not channels:
        channels = list(range(min(3, record.signals.shape[1])))
    return np.nan_to_num(record.signals[:, channels], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _masked_stats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if len(matrix) == 0 or not mask.any():
        return 0.0, 0.0, 0.0, 0.0
    selected = matrix[np.asarray(mask, dtype=bool)]
    per_frame = selected.mean(axis=1)
    return (
        float(per_frame.mean()),
        float(per_frame.max()),
        float(per_frame.std()),
        float((per_frame >= max(0.5, float(per_frame.mean()))).mean()),
    )


def set_gate_feature_vector(
    record: VideoRecord,
    feature_names: list[str],
    base: list[tuple[int, int]],
    recall: list[tuple[int, int]],
    evidence_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    base_segments = sorted({_normalise_segment(item) for item in base})
    recall_segments = sorted({_normalise_segment(item) for item in recall})
    evidence = _evidence_matrix(record, feature_names, evidence_names or DEFAULT_EVIDENCE_NAMES)
    base_mask = _segment_mask(len(record.signals), base_segments)
    recall_mask = _segment_mask(len(record.signals), recall_segments)
    added_mask = recall_mask & ~base_mask
    dropped_mask = base_mask & ~recall_mask
    base_mean, base_max, base_std, base_active = _masked_stats(evidence, base_mask)
    recall_mean, recall_max, recall_std, recall_active = _masked_stats(evidence, recall_mask)
    added_mean, added_max, added_std, added_active = _masked_stats(evidence, added_mask)
    dropped_mean, dropped_max, dropped_std, dropped_active = _masked_stats(evidence, dropped_mask)
    shape = prediction_set_features(base_segments, recall_segments)
    values = [
        float(shape["base_count"]),
        float(shape["recall_count"]),
        float(shape["count_delta"]),
        float(shape["base_length"]),
        float(shape["recall_length"]),
        float(shape["length_ratio"]),
        float(shape["mean_recall_to_base_iou"]),
        float(shape["mean_base_to_recall_iou"]),
        float(shape["mutual_iou"]),
        float(shape["extra_segments"]),
        float(shape["dropped_segments"]),
        base_mean,
        base_max,
        base_std,
        base_active,
        recall_mean,
        recall_max,
        recall_std,
        recall_active,
        added_mean,
        added_max,
        added_std,
        added_active,
        dropped_mean,
        dropped_max,
        dropped_std,
        dropped_active,
        recall_mean - base_mean,
        recall_max - base_max,
        added_mean - dropped_mean,
    ]
    names = [
        "base_count",
        "recall_count",
        "count_delta",
        "base_length",
        "recall_length",
        "length_ratio",
        "mean_recall_to_base_iou",
        "mean_base_to_recall_iou",
        "mutual_iou",
        "extra_segments",
        "dropped_segments",
        "base_evidence_mean",
        "base_evidence_max",
        "base_evidence_std",
        "base_evidence_active",
        "recall_evidence_mean",
        "recall_evidence_max",
        "recall_evidence_std",
        "recall_evidence_active",
        "added_evidence_mean",
        "added_evidence_max",
        "added_evidence_std",
        "added_evidence_active",
        "dropped_evidence_mean",
        "dropped_evidence_max",
        "dropped_evidence_std",
        "dropped_evidence_active",
        "delta_evidence_mean",
        "delta_evidence_max",
        "added_minus_dropped_evidence_mean",
    ]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), names


def label_switch_preference(
    base: list[tuple[int, int]],
    recall: list[tuple[int, int]],
    labels: np.ndarray,
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> tuple[int, dict]:
    label_array = np.asarray(labels, dtype=np.int64)
    base_segments = [[_normalise_segment(item) for item in base]]
    recall_segments = [[_normalise_segment(item) for item in recall]]
    label_list = [label_array]
    base_metrics = evaluate_fused_predictions(base_segments, label_list, iou_threshold=float(iou_threshold))["segment"]
    recall_metrics = evaluate_fused_predictions(recall_segments, label_list, iou_threshold=float(iou_threshold))["segment"]
    fp_allowed = max_fp_increase is None or int(recall_metrics["fp"]) <= int(base_metrics["fp"]) + int(max_fp_increase)
    recall_key = (float(recall_metrics["f1"]), float(recall_metrics["recall"]), float(recall_metrics["precision"]))
    base_key = (float(base_metrics["f1"]), float(base_metrics["recall"]), float(base_metrics["precision"]))
    label = int(fp_allowed and recall_key > base_key)
    return label, {
        "base_tp": int(base_metrics["tp"]),
        "base_fp": int(base_metrics["fp"]),
        "base_fn": int(base_metrics["fn"]),
        "recall_tp": int(recall_metrics["tp"]),
        "recall_fp": int(recall_metrics["fp"]),
        "recall_fn": int(recall_metrics["fn"]),
    }


def build_gate_examples_from_folds(
    folds: list[dict],
    records: list[VideoRecord],
    feature_names: list[str],
    evidence_names: list[str],
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> tuple[list[GateExample], list[str]]:
    examples: list[GateExample] = []
    feature_names_out: list[str] = []
    for fold_pos, fold in enumerate(folds):
        for local_idx, video_idx in enumerate(fold.get("val_idx", [])):
            base = [_normalise_segment(item) for item in fold["base"][local_idx]]
            recall = [_normalise_segment(item) for item in fold["recall"][local_idx]]
            labels = np.asarray(fold["labels"][local_idx], dtype=np.int64)
            features, names = set_gate_feature_vector(
                records[int(video_idx)],
                feature_names=feature_names,
                base=base,
                recall=recall,
                evidence_names=evidence_names,
            )
            if not feature_names_out:
                feature_names_out = names
            label, _ = label_switch_preference(
                base,
                recall,
                labels,
                max_fp_increase=max_fp_increase,
                iou_threshold=float(iou_threshold),
            )
            examples.append(
                GateExample(
                    fold_idx=int(fold.get("fold", fold_pos)),
                    video_idx=int(video_idx),
                    features=features,
                    label=int(label),
                    base=base,
                    recall=recall,
                    labels=labels,
                )
            )
    return examples, feature_names_out


def _predict_scores(model, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)


def _fit_gate_model(
    examples: list[GateExample],
    model_name: str,
    seed: int,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    x = np.stack([item.features for item in examples])
    y = np.asarray([item.label for item in examples], dtype=np.int64)
    if len(np.unique(y)) < 2:
        return None
    return fit_classifier(
        x,
        y,
        seed=int(seed),
        model_name=model_name,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
    )


def _apply_gate_scores(
    base_predictions: list[list[tuple[int, int]]],
    recall_predictions: list[list[tuple[int, int]]],
    scores: np.ndarray,
    threshold: float,
) -> list[list[Segment]]:
    out: list[list[Segment]] = []
    for base, recall, score in zip(base_predictions, recall_predictions, np.asarray(scores, dtype=np.float32)):
        selected = recall if float(score) >= float(threshold) else base
        out.append(sorted({_normalise_segment(item) for item in selected}))
    return out


def select_threshold_on_gate_scores(
    base_predictions: list[list[tuple[int, int]]],
    recall_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    scores: np.ndarray,
    thresholds: Iterable[float],
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    recall = [[_normalise_segment(item) for item in video] for video in recall_predictions]
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=float(iou_threshold))
    best_config: dict = {"enabled": False, "base_metrics": base_metrics}
    best_metrics = base_metrics
    best_predictions = base
    base_fp = int(base_metrics["segment"]["fp"])
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["segment"]["precision"]),
        0,
    )
    for threshold in thresholds:
        predictions = _apply_gate_scores(base, recall, scores, threshold=float(threshold))
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
                "enabled": True,
                "threshold": float(threshold),
                "changed_videos": int(changed),
                "base_metrics": base_metrics,
            }
            best_metrics = metrics
            best_predictions = predictions
    return best_config, best_metrics, best_predictions


def _inner_oof_scores(
    calibration_folds: list[dict],
    records: list[VideoRecord],
    feature_names: list[str],
    evidence_names: list[str],
    model_name: str,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    iou_threshold: float,
) -> tuple[list[list[Segment]], list[list[Segment]], list[np.ndarray], np.ndarray]:
    all_base: list[list[Segment]] = []
    all_recall: list[list[Segment]] = []
    all_labels: list[np.ndarray] = []
    all_scores: list[float] = []
    for fold_idx, fold in enumerate(calibration_folds):
        train_folds = [item for pos, item in enumerate(calibration_folds) if pos != fold_idx]
        train_examples, _ = build_gate_examples_from_folds(
            train_folds,
            records,
            feature_names=feature_names,
            evidence_names=evidence_names,
            max_fp_increase=0,
            iou_threshold=float(iou_threshold),
        )
        val_examples, _ = build_gate_examples_from_folds(
            [fold],
            records,
            feature_names=feature_names,
            evidence_names=evidence_names,
            max_fp_increase=0,
            iou_threshold=float(iou_threshold),
        )
        if not val_examples:
            continue
        model = _fit_gate_model(
            train_examples,
            model_name=model_name,
            seed=seed + 100 + fold_idx,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )
        scores = np.zeros(len(val_examples), dtype=np.float32)
        if model is not None:
            scores = _predict_scores(model, np.stack([item.features for item in val_examples]))
        all_scores.extend([float(item) for item in scores])
        all_base.extend([item.base for item in val_examples])
        all_recall.extend([item.recall for item in val_examples])
        all_labels.extend([item.labels for item in val_examples])
    return all_base, all_recall, all_labels, np.asarray(all_scores, dtype=np.float32)


def run_fold_heldout_learned_set_gate(
    folds: list[dict],
    records: list[VideoRecord],
    feature_names: list[str],
    evidence_names: list[str],
    model_name: str = "extratrees",
    thresholds: Iterable[float] = tuple(np.linspace(0.2, 0.9, 15)),
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
    seed: int = 42,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
) -> dict:
    results = []
    for heldout_pos, heldout in enumerate(folds):
        calibration_folds = [fold for pos, fold in enumerate(folds) if pos != heldout_pos]
        cal_base, cal_recall, cal_labels, cal_scores = _inner_oof_scores(
            calibration_folds,
            records,
            feature_names=feature_names,
            evidence_names=evidence_names,
            model_name=model_name,
            seed=seed + heldout_pos * 1000,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            iou_threshold=float(iou_threshold),
        )
        config, cal_metrics, _ = select_threshold_on_gate_scores(
            cal_base,
            cal_recall,
            cal_labels,
            cal_scores,
            thresholds=thresholds,
            max_fp_increase=max_fp_increase,
            iou_threshold=float(iou_threshold),
        )
        train_examples, feature_names_out = build_gate_examples_from_folds(
            calibration_folds,
            records,
            feature_names=feature_names,
            evidence_names=evidence_names,
            max_fp_increase=max_fp_increase,
            iou_threshold=float(iou_threshold),
        )
        heldout_examples, _ = build_gate_examples_from_folds(
            [heldout],
            records,
            feature_names=feature_names,
            evidence_names=evidence_names,
            max_fp_increase=max_fp_increase,
            iou_threshold=float(iou_threshold),
        )
        scores = np.zeros(len(heldout_examples), dtype=np.float32)
        model = _fit_gate_model(
            train_examples,
            model_name=model_name,
            seed=seed + heldout_pos,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )
        if model is not None and len(heldout_examples) > 0:
            scores = _predict_scores(model, np.stack([item.features for item in heldout_examples]))
        if config.get("enabled"):
            predictions = _apply_gate_scores(
                [item.base for item in heldout_examples],
                [item.recall for item in heldout_examples],
                scores,
                threshold=float(config["threshold"]),
            )
        else:
            predictions = [item.base for item in heldout_examples]
        labels = [item.labels for item in heldout_examples]
        metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(iou_threshold))
        results.append(
            {
                "fold": int(heldout.get("fold", heldout_pos)),
                "config": config,
                "calibration": cal_metrics,
                "validation": metrics,
                "feature_names": feature_names_out,
                "score_mean": float(scores.mean()) if len(scores) else 0.0,
                "score_max": float(scores.max()) if len(scores) else 0.0,
                "predictions": [[[int(s), int(e)] for s, e in video] for video in predictions],
            }
        )
    return {"folds": results, "aggregate": aggregate_fold_metrics(results)}


def _parse_names(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--recall-predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--evidence-names", default=",".join(DEFAULT_EVIDENCE_NAMES))
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    folds = build_folds_from_exports(
        load_prediction_export(args.base_predictions),
        load_prediction_export(args.recall_predictions),
    )
    result = run_fold_heldout_learned_set_gate(
        folds,
        records,
        feature_names=feature_names,
        evidence_names=_parse_names(args.evidence_names),
        model_name=args.model,
        thresholds=_parse_floats(args.thresholds),
        max_fp_increase=args.max_fp_increase,
        iou_threshold=args.iou_threshold,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    result.update(
        {
            "data_dir": args.data_dir,
            "base_predictions": args.base_predictions,
            "recall_predictions": args.recall_predictions,
            "model": args.model,
            "evidence_names": _parse_names(args.evidence_names),
            "thresholds": _parse_floats(args.thresholds),
            "max_fp_increase": args.max_fp_increase,
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
