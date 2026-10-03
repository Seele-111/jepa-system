#!/usr/bin/env python3
"""Fold-local clean-anchor label rectification for JEPA temporal localization.

The module uses a cleaner anchor set to estimate which JEPA evidence patterns
look like reliable error frames. It rectifies training labels only; validation
labels remain untouched.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from train_segment_locator import VideoRecord, contiguous_segments, load_signal_dataset, smooth_probabilities


@dataclass
class CleanAnchorRectifyConfig:
    evidence_names: list[str]
    teacher_model: str = "prototype"
    positive_threshold: float = 0.7
    negative_threshold: float = 0.3
    add_threshold: float = 0.85
    remove_threshold: float = 0.2
    min_add_length: int = 2
    min_keep_length: int = 2
    smooth_window: int = 1
    max_added_fraction: float | None = None
    exclude_video_names: list[str] | None = None
    positive_min_weight: float = 0.35
    negative_min_weight: float = 0.35


def _load_feature_names(data_dir: str | Path) -> list[str]:
    path = Path(data_dir) / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(name) for name in data.get("feature_names", [])]


def _channel_indices(feature_names: Sequence[str], evidence_names: Sequence[str]) -> list[int]:
    mapping = {str(name): idx for idx, name in enumerate(feature_names)}
    indices = [mapping[name] for name in evidence_names if name in mapping]
    if not indices:
        raise ValueError(f"none of clean-anchor evidence channels were found: {list(evidence_names)}")
    return indices


def _evidence_score(signals: np.ndarray, channel_indices: Sequence[int]) -> np.ndarray:
    values = np.nan_to_num(np.asarray(signals, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    score = values[:, list(channel_indices)].mean(axis=1)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _fit_frame_teacher(x: np.ndarray, y: np.ndarray, model_name: str, seed: int):
    model_name = str(model_name)
    if model_name == "prototype":
        return None
    if len(np.unique(y)) < 2:
        return None
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    try:
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    except Exception:
        from train_frame_calibrator import fit_classifier

        return fit_classifier(x, y, seed=int(seed), model_name="logreg")
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
    elif model_name == "gbdt":
        model = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            subsample=0.85,
            random_state=int(seed),
        )
    elif model_name == "logreg":
        from train_frame_calibrator import fit_classifier

        return fit_classifier(x, y, seed=int(seed), model_name="logreg")
    else:
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        )
    model.fit(x, y)
    return model


def _fit_anchor_teacher(
    clean_data_dir: str | Path,
    evidence_names: Sequence[str],
    exclude_video_names: Iterable[str] | None,
    teacher_model: str,
) -> dict:
    clean_records = load_signal_dataset(clean_data_dir)
    feature_names = _load_feature_names(clean_data_dir)
    channel_indices = _channel_indices(feature_names, evidence_names)
    excluded = {str(name) for name in (exclude_video_names or [])}
    positive_scores: list[np.ndarray] = []
    negative_scores: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    used_names: list[str] = []
    for record in clean_records:
        if record.name in excluded:
            continue
        score = _evidence_score(record.signals, channel_indices)
        labels = (np.asarray(record.labels).reshape(-1) > 0).astype(np.int64)
        n = min(len(score), len(labels))
        if n <= 0:
            continue
        score = score[:n]
        labels = labels[:n]
        x_parts.append(np.asarray(record.signals[:n, channel_indices], dtype=np.float32))
        y_parts.append(labels.astype(np.int64))
        positive_scores.append(score[labels > 0])
        negative_scores.append(score[labels <= 0])
        used_names.append(record.name)
    pos = np.concatenate([item for item in positive_scores if len(item)], axis=0) if positive_scores else np.zeros(0, dtype=np.float32)
    neg = np.concatenate([item for item in negative_scores if len(item)], axis=0) if negative_scores else np.zeros(0, dtype=np.float32)
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("clean anchor needs both positive and negative frames for rectification")
    x_train = np.concatenate(x_parts, axis=0) if x_parts else np.zeros((0, len(channel_indices)), dtype=np.float32)
    y_train = np.concatenate(y_parts, axis=0) if y_parts else np.zeros(0, dtype=np.int64)
    model = _fit_frame_teacher(x_train, y_train, model_name=teacher_model, seed=9301)
    active_teacher_model = str(teacher_model) if model is not None else "prototype"
    return {
        "feature_names": feature_names,
        "channel_indices": channel_indices,
        "model": model,
        "teacher_model": active_teacher_model,
        "positive_low": float(np.quantile(pos, 0.25)),
        "positive_high": float(np.quantile(pos, 0.75)),
        "negative_low": float(np.quantile(neg, 0.25)),
        "negative_high": float(np.quantile(neg, 0.75)),
        "teacher_positive_frames": int(len(pos)),
        "teacher_negative_frames": int(len(neg)),
        "teacher_videos": int(len(used_names)),
        "excluded_video_names": sorted(excluded),
    }


def _teacher_probability(score: np.ndarray, anchor: dict) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    neg_center = 0.5 * (float(anchor["negative_low"]) + float(anchor["negative_high"]))
    pos_center = 0.5 * (float(anchor["positive_low"]) + float(anchor["positive_high"]))
    if pos_center <= neg_center + 1e-6:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    prob = (values - neg_center) / (pos_center - neg_center)
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def _predict_teacher_probability(signals: np.ndarray, channel_indices: Sequence[int], anchor: dict) -> np.ndarray:
    model = anchor.get("model")
    values = np.nan_to_num(np.asarray(signals, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    x = values[:, list(channel_indices)]
    if model is not None:
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    return _teacher_probability(_evidence_score(values, channel_indices), anchor)


def _cap_added_frames(original: np.ndarray, proposed: np.ndarray, teacher: np.ndarray, max_added_fraction: float | None) -> np.ndarray:
    if max_added_fraction is None:
        return proposed
    added = np.flatnonzero((original == 0) & (proposed == 1))
    max_added = int(round(float(max_added_fraction) * max(1, len(original))))
    if len(added) <= max_added:
        return proposed
    capped = proposed.copy()
    order = added[np.argsort(teacher[added])[::-1]]
    keep = set(int(idx) for idx in order[:max_added])
    for idx in added:
        if int(idx) not in keep:
            capped[int(idx)] = 0
    return capped


def rectify_labels_with_teacher(labels: np.ndarray, teacher: np.ndarray, config: CleanAnchorRectifyConfig) -> tuple[np.ndarray, dict]:
    original = (np.asarray(labels).reshape(-1) > 0).astype(np.int64)
    score = np.nan_to_num(np.asarray(teacher, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    n = min(len(original), len(score))
    original = original[:n]
    score = np.clip(score[:n], 0.0, 1.0)
    if int(config.smooth_window) > 1:
        score = smooth_probabilities(score, int(config.smooth_window))
    rectified = original.copy()

    low_positive = ((original == 1) & (score <= float(config.remove_threshold))).astype(np.int64)
    for start, end in contiguous_segments(low_positive):
        if end - start + 1 >= int(config.min_keep_length):
            rectified[start : end + 1] = 0

    high_negative = ((original == 0) & (score >= float(config.add_threshold))).astype(np.int64)
    for start, end in contiguous_segments(high_negative):
        if end - start + 1 >= int(config.min_add_length):
            rectified[start : end + 1] = 1

    rectified = _cap_added_frames(original, rectified, score, config.max_added_fraction)
    added = int(np.sum((original == 0) & (rectified == 1)))
    removed = int(np.sum((original == 1) & (rectified == 0)))
    return rectified.astype(np.int64), {
        "added_frames": added,
        "removed_frames": removed,
        "changed_frames": added + removed,
        "original_positive_frames": int(original.sum()),
        "rectified_positive_frames": int(rectified.sum()),
    }


def rectify_records_for_fold(
    records: Sequence[VideoRecord],
    train_idx: Iterable[int],
    feature_names: Sequence[str],
    clean_data_dir: str | Path,
    config: CleanAnchorRectifyConfig,
) -> tuple[list[VideoRecord], dict]:
    train_set = {int(idx) for idx in train_idx}
    exclude_names = set(config.exclude_video_names or [])
    anchor = _fit_anchor_teacher(clean_data_dir, config.evidence_names, exclude_names, config.teacher_model)
    target_channels = _channel_indices(feature_names, config.evidence_names)
    copied: list[VideoRecord] = []
    summary = {
        "enabled": True,
        "changed_videos": 0,
        "added_frames": 0,
        "removed_frames": 0,
        "changed_frames": 0,
        "train_videos": len(train_set),
        "config": asdict(config),
        **{key: value for key, value in anchor.items() if key not in {"feature_names", "channel_indices", "model"}},
    }
    for idx, record in enumerate(records):
        signals = np.asarray(record.signals, dtype=np.float32).copy()
        labels = np.asarray(record.labels, dtype=np.int64).copy()
        if idx in train_set:
            teacher = _predict_teacher_probability(signals, target_channels, anchor)
            labels, stats = rectify_labels_with_teacher(labels, teacher, config)
            if int(stats["changed_frames"]) > 0:
                summary["changed_videos"] += 1
            summary["added_frames"] += int(stats["added_frames"])
            summary["removed_frames"] += int(stats["removed_frames"])
            summary["changed_frames"] += int(stats["changed_frames"])
        copied.append(VideoRecord(name=record.name, signals=signals, labels=labels))
    return copied, summary


def compute_clean_anchor_frame_weights_for_fold(
    records: Sequence[VideoRecord],
    train_idx: Iterable[int],
    feature_names: Sequence[str],
    clean_data_dir: str | Path,
    config: CleanAnchorRectifyConfig,
) -> tuple[dict[int, np.ndarray], dict]:
    train_set = {int(idx) for idx in train_idx}
    exclude_names = set(config.exclude_video_names or [])
    anchor = _fit_anchor_teacher(clean_data_dir, config.evidence_names, exclude_names, config.teacher_model)
    target_channels = _channel_indices(feature_names, config.evidence_names)
    positive_floor = float(np.clip(config.positive_min_weight, 0.0, 1.0))
    negative_floor = float(np.clip(config.negative_min_weight, 0.0, 1.0))
    weights_by_idx: dict[int, np.ndarray] = {}
    summary = {
        "enabled": True,
        "teacher_model": anchor.get("teacher_model"),
        "weighted_train_videos": 0,
        "weighted_frames": 0,
        "mean_weight": 0.0,
        "min_weight": 1.0,
        "positive_min_weight": positive_floor,
        "negative_min_weight": negative_floor,
        **{key: value for key, value in anchor.items() if key not in {"feature_names", "channel_indices", "model"}},
    }
    all_weights: list[np.ndarray] = []
    for idx, record in enumerate(records):
        if idx not in train_set:
            continue
        signals = np.asarray(record.signals, dtype=np.float32)
        labels = (np.asarray(record.labels).reshape(-1) > 0).astype(np.int64)
        teacher = _predict_teacher_probability(signals, target_channels, anchor)
        n = min(len(labels), len(teacher))
        labels = labels[:n]
        teacher = np.clip(teacher[:n], 0.0, 1.0)
        if int(config.smooth_window) > 1:
            teacher = smooth_probabilities(teacher, int(config.smooth_window))
        weights = np.ones(n, dtype=np.float32)
        positive_mask = labels > 0
        negative_mask = ~positive_mask
        weights[positive_mask] = positive_floor + (1.0 - positive_floor) * teacher[positive_mask]
        weights[negative_mask] = negative_floor + (1.0 - negative_floor) * (1.0 - teacher[negative_mask])
        weights_by_idx[int(idx)] = np.clip(weights, 0.0, 1.0).astype(np.float32)
        all_weights.append(weights_by_idx[int(idx)])
        summary["weighted_train_videos"] += 1
        summary["weighted_frames"] += int(n)
    if all_weights:
        stacked = np.concatenate(all_weights, axis=0)
        summary["mean_weight"] = float(stacked.mean())
        summary["min_weight"] = float(stacked.min())
    return weights_by_idx, summary
