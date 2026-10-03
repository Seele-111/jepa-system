#!/usr/bin/env python3
"""Clean-set calibrated JEPA event-set selection.

The module uses cleaner annotations only to calibrate whether JEPA event
candidates are trustworthy. It keeps the task binary: find erroneous temporal
segments, not category or severity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions, short_first_nms_segments
from train_proposal_calibrator import interval_iou
from train_proposal_set_selector import (
    build_candidate_records,
    extract_set_proposal_features,
    fit_classifier,
    fit_quality_model,
    select_weighted_proposal_set,
)
from train_segment_locator import VideoRecord, load_signal_dataset


Segment = tuple[int, int]


@dataclass
class CleanCalibratorBundle:
    model: object
    feature_names: list[str]
    channel_names: list[str]
    thresholds: list[float]
    min_gaps: list[int]
    min_lengths: list[int]
    target: str
    n_train_candidates: int
    n_positive_train_candidates: int
    n_train_videos: int
    excluded_video_names: list[str]
    clean_video_names: list[str]


class _ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x)
        n = int(values.shape[0]) if values.ndim > 1 else 1
        positive = np.full(n, self.probability, dtype=np.float32)
        return np.stack([1.0 - positive, positive], axis=1).astype(np.float32)


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _load_feature_names(data_dir: str | Path) -> list[str]:
    import json

    summary_path = Path(data_dir) / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return [str(name) for name in data.get("feature_names", [])]


def _max_iou_with_base(candidate: Segment, base_segments: list[Segment]) -> float:
    if not base_segments:
        return 0.0
    return float(max(interval_iou(candidate, base) for base in base_segments))


def combine_clean_and_raw_scores(
    raw_scores: np.ndarray,
    clean_scores: np.ndarray,
    clean_weight: float,
    raw_weight: float,
) -> np.ndarray:
    """Blend fold-local selector confidence with Clean69-calibrated confidence."""
    raw = np.nan_to_num(np.asarray(raw_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    clean = np.nan_to_num(np.asarray(clean_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(raw) != len(clean):
        raise ValueError(f"raw/clean score count mismatch: {len(raw)} vs {len(clean)}")
    return (
        float(raw_weight) * np.clip(raw, 0.0, 1.0)
        + float(clean_weight) * np.clip(clean, 0.0, 1.0)
    ).astype(np.float32)


def fit_clean_candidate_calibrator(
    clean_data_dir: str | Path,
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    iou_threshold: float,
    model_name: str,
    target: str,
    seed: int,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
    exclude_video_names: Iterable[str] | None = None,
) -> CleanCalibratorBundle:
    """Fit a candidate quality calibrator on the cleaner Clean69 dataset."""
    clean_records = load_signal_dataset(clean_data_dir)
    clean_feature_names = _load_feature_names(clean_data_dir)
    active_channels = [name for name in channel_names if name in clean_feature_names]
    if not active_channels:
        raise ValueError("no requested clean calibration channels were found in Clean69 summary.json")
    excluded = {str(name) for name in (exclude_video_names or [])}
    train_indices = [idx for idx, record in enumerate(clean_records) if record.name not in excluded]
    if not train_indices:
        raise RuntimeError("no clean calibration videos remain after fold exclusion")
    candidate_records = build_candidate_records(
        clean_records,
        train_indices,
        feature_names=clean_feature_names,
        channel_names=active_channels,
        thresholds=thresholds,
        min_gaps=min_gaps,
        min_lengths=min_lengths,
        iou_threshold=float(iou_threshold),
    )
    if not candidate_records:
        raise RuntimeError("no clean calibration candidates generated")
    x_train = np.stack([item.features for item in candidate_records])
    y_train = np.asarray([item.label for item in candidate_records], dtype=np.int64)
    best_iou = np.asarray([item.best_iou for item in candidate_records], dtype=np.float32)
    if len(np.unique(y_train)) < 2:
        model = _ConstantProbabilityModel(float(y_train.mean()) if len(y_train) else 0.0)
    elif target == "quality":
        model = fit_quality_model(x_train, best_iou, seed=int(seed), model_name=model_name)
    else:
        model = fit_classifier(
            x_train,
            y_train,
            seed=int(seed),
            model_name=model_name,
            device=device,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
    return CleanCalibratorBundle(
        model=model,
        feature_names=clean_feature_names,
        channel_names=active_channels,
        thresholds=list(thresholds),
        min_gaps=list(min_gaps),
        min_lengths=list(min_lengths),
        target=str(target),
        n_train_candidates=int(len(candidate_records)),
        n_positive_train_candidates=int(y_train.sum()),
        n_train_videos=int(len(train_indices)),
        excluded_video_names=sorted(excluded),
        clean_video_names=[clean_records[idx].name for idx in train_indices],
    )


def score_clean_calibrated_candidates(
    model,
    record: VideoRecord,
    candidates: list[tuple[int, int]],
    feature_names: list[str],
    channel_names: list[str],
) -> np.ndarray:
    """Score an existing candidate list with a clean-trained calibrator."""
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    if not channel_indices:
        raise ValueError("no clean-calibrated scoring channels found in target feature names")
    x = np.stack(
        [
            extract_set_proposal_features(record.signals, _normalise_segment(segment), channel_indices)
            for segment in candidates
        ]
    )
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)


def _select_video_rescues(
    base_segments: list[Segment],
    candidates: list[Segment],
    raw_scores: np.ndarray,
    clean_scores: np.ndarray,
    clean_threshold: float,
    raw_threshold: float,
    clean_weight: float,
    raw_weight: float,
    max_base_iou: float | None,
    length_penalty: float,
    max_rescues: int,
) -> list[Segment]:
    combined = combine_clean_and_raw_scores(
        raw_scores,
        clean_scores,
        clean_weight=float(clean_weight),
        raw_weight=float(raw_weight),
    )
    filtered_candidates: list[Segment] = []
    filtered_scores: list[float] = []
    seen_base = set(base_segments)
    for segment_raw, raw_score, clean_score, score in zip(candidates, raw_scores, clean_scores, combined):
        segment = _normalise_segment(segment_raw)
        if segment in seen_base:
            continue
        if float(raw_score) < float(raw_threshold) or float(clean_score) < float(clean_threshold):
            continue
        if max_base_iou is not None and _max_iou_with_base(segment, base_segments) > float(max_base_iou):
            continue
        filtered_candidates.append(segment)
        filtered_scores.append(float(score))
    if not filtered_candidates:
        return []
    if int(max_rescues) > 0 and len(filtered_candidates) > int(max_rescues) * 8:
        order = np.argsort(-np.asarray(filtered_scores, dtype=np.float32), kind="mergesort")[: int(max_rescues) * 8]
        filtered_candidates = [filtered_candidates[int(idx)] for idx in order]
        filtered_scores = [filtered_scores[int(idx)] for idx in order]
    selected = select_weighted_proposal_set(
        filtered_candidates,
        filtered_scores,
        length_penalty=float(length_penalty),
    )
    if int(max_rescues) > 0 and len(selected) > int(max_rescues):
        score_by_segment = {segment: score for segment, score in zip(filtered_candidates, filtered_scores)}
        selected = sorted(
            selected,
            key=lambda segment: (score_by_segment.get(segment, 0.0), -(segment[1] - segment[0] + 1)),
            reverse=True,
        )[: int(max_rescues)]
    return sorted({_normalise_segment(item) for item in selected})


def apply_clean_calibrated_event_set(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    raw_scores: list[np.ndarray],
    clean_scores: list[np.ndarray],
    clean_threshold: float,
    raw_threshold: float,
    clean_weight: float,
    raw_weight: float,
    max_base_iou: float | None,
    length_penalty: float,
    max_rescues_per_video: int,
    rescue_nms_iou: float | None = 0.3,
) -> list[list[Segment]]:
    if not (len(base_predictions) == len(candidate_predictions) == len(raw_scores) == len(clean_scores)):
        raise ValueError("base/candidate/raw/clean video counts must match")
    predictions: list[list[Segment]] = []
    for base_raw, candidates_raw, raw, clean in zip(base_predictions, candidate_predictions, raw_scores, clean_scores):
        base = sorted({_normalise_segment(segment) for segment in base_raw})
        candidates = [_normalise_segment(segment) for segment in candidates_raw]
        rescues = _select_video_rescues(
            base,
            candidates,
            np.asarray(raw, dtype=np.float32),
            np.asarray(clean, dtype=np.float32),
            clean_threshold=float(clean_threshold),
            raw_threshold=float(raw_threshold),
            clean_weight=float(clean_weight),
            raw_weight=float(raw_weight),
            max_base_iou=max_base_iou,
            length_penalty=float(length_penalty),
            max_rescues=int(max_rescues_per_video),
        )
        if rescue_nms_iou is not None:
            rescues = short_first_nms_segments(rescues, iou_threshold=float(rescue_nms_iou))
        predictions.append(sorted(set(base + rescues)))
    return predictions


def select_clean_calibrated_event_set_params(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    raw_scores: list[np.ndarray],
    clean_scores: list[np.ndarray],
    labels: list[np.ndarray],
    clean_thresholds: Iterable[float] = (0.4, 0.5, 0.6, 0.7),
    raw_thresholds: Iterable[float] = (0.0, 0.1, 0.2),
    clean_weights: Iterable[float] = (0.5, 0.7, 0.9),
    raw_weights: Iterable[float] = (0.1, 0.3, 0.5),
    max_base_ious: Iterable[float | None] = (0.0, 0.1, 0.25, None),
    length_penalties: Iterable[float] = (0.0, 0.005, 0.01),
    max_rescues_per_videos: Iterable[int] = (1, 2),
    rescue_nms_ious: Iterable[float | None] = (0.3,),
    max_fp_increase: int | None = 0,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in base_predictions]
    base_metrics = evaluate_fused_predictions(base, labels, iou_threshold=float(iou_threshold))
    diagnostics = {
        "configs_considered": 0,
        "evaluated_configs": 0,
        "fp_rejected_configs": 0,
        "best_observed_metrics": None,
        "best_observed_config": None,
        "best_rejected_metrics": None,
        "best_rejected_config": None,
    }
    best_config: dict = {"enabled": False, "base_metrics": base_metrics, "diagnostics": diagnostics}
    best_metrics = base_metrics
    best_predictions = base
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for clean_threshold in clean_thresholds:
        for raw_threshold in raw_thresholds:
            for clean_weight in clean_weights:
                for raw_weight in raw_weights:
                    for max_base_iou in max_base_ious:
                        for length_penalty in length_penalties:
                            for max_rescues_per_video in max_rescues_per_videos:
                                for rescue_nms_iou in rescue_nms_ious:
                                    diagnostics["configs_considered"] += 1
                                    predictions = apply_clean_calibrated_event_set(
                                        base,
                                        candidate_predictions,
                                        raw_scores,
                                        clean_scores,
                                        clean_threshold=float(clean_threshold),
                                        raw_threshold=float(raw_threshold),
                                        clean_weight=float(clean_weight),
                                        raw_weight=float(raw_weight),
                                        max_base_iou=max_base_iou,
                                        length_penalty=float(length_penalty),
                                        max_rescues_per_video=int(max_rescues_per_video),
                                        rescue_nms_iou=rescue_nms_iou,
                                    )
                                    metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(iou_threshold))
                                    diagnostics["evaluated_configs"] += 1
                                    rescue_count = sum(max(0, len(pred) - len(base_video)) for pred, base_video in zip(predictions, base))
                                    observed_config = {
                                        "clean_threshold": float(clean_threshold),
                                        "raw_threshold": float(raw_threshold),
                                        "clean_weight": float(clean_weight),
                                        "raw_weight": float(raw_weight),
                                        "max_base_iou": None if max_base_iou is None else float(max_base_iou),
                                        "length_penalty": float(length_penalty),
                                        "max_rescues_per_video": int(max_rescues_per_video),
                                        "rescue_nms_iou": None if rescue_nms_iou is None else float(rescue_nms_iou),
                                        "rescue_segments": int(rescue_count),
                                    }
                                    observed_key = (
                                        float(metrics["segment"]["f1"]),
                                        float(metrics["segment"]["precision"]),
                                        float(metrics["segment"]["recall"]),
                                        float(metrics["frame"]["f1"]),
                                    )
                                    current_observed = diagnostics.get("best_observed_metrics")
                                    if current_observed is None or observed_key > (
                                        float(current_observed["segment"]["f1"]),
                                        float(current_observed["segment"]["precision"]),
                                        float(current_observed["segment"]["recall"]),
                                        float(current_observed["frame"]["f1"]),
                                    ):
                                        diagnostics["best_observed_metrics"] = metrics
                                        diagnostics["best_observed_config"] = observed_config
                                    if max_fp_increase is not None:
                                        fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                                        if fp_delta > int(max_fp_increase):
                                            diagnostics["fp_rejected_configs"] += 1
                                            rejected = diagnostics.get("best_rejected_metrics")
                                            if rejected is None or observed_key > (
                                                float(rejected["segment"]["f1"]),
                                                float(rejected["segment"]["precision"]),
                                                float(rejected["segment"]["recall"]),
                                                float(rejected["frame"]["f1"]),
                                            ):
                                                diagnostics["best_rejected_metrics"] = metrics
                                                diagnostics["best_rejected_config"] = {
                                                    **observed_config,
                                                    "fp_delta": int(fp_delta),
                                                }
                                            continue
                                    key = (*observed_key, -float(rescue_count))
                                    if key > best_key:
                                        best_key = key
                                        best_metrics = metrics
                                        best_predictions = predictions
                                        best_config = {
                                            "enabled": True,
                                            **observed_config,
                                            "max_fp_increase": max_fp_increase,
                                            "base_metrics": base_metrics,
                                            "diagnostics": diagnostics,
                                        }
    if not best_config.get("enabled"):
        best_config = {"enabled": False, "base_metrics": base_metrics, "diagnostics": diagnostics}
    return best_config, best_metrics, best_predictions
