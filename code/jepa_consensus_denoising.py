#!/usr/bin/env python3
"""Consensus label denoising for JEPA temporal error localization.

The denoiser is intentionally fold-local: it can alter noisy training labels
using V/I-JEPA agreement evidence, while validation labels stay untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

from train_segment_locator import VideoRecord, contiguous_segments, smooth_probabilities


@dataclass
class ConsensusDenoiseConfig:
    add_threshold: float
    remove_threshold: float
    min_add_length: int
    min_keep_length: int
    smooth_window: int = 1
    agreement_weight: float = 0.0
    max_added_fraction: float | None = None


def _name_to_index(feature_names: Sequence[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(feature_names)}


def compute_consensus_score(
    signals: np.ndarray,
    feature_names: Sequence[str],
    evidence_names: Sequence[str],
    agreement_name: str = "",
    agreement_weight: float = 0.0,
) -> np.ndarray:
    """Return a frame-wise JEPA consensus score in [0, 1].

    Evidence channels are averaged, then optionally gated by an agreement
    channel. This rewards evidence that is shared by V-JEPA/I-JEPA style signals
    instead of trusting a single noisy rank channel.
    """
    values = np.nan_to_num(np.asarray(signals, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim != 2:
        raise ValueError(f"signals must have shape [T, C], got {values.shape}")
    mapping = _name_to_index(feature_names)
    channels = [mapping[name] for name in evidence_names if name in mapping]
    if not channels:
        raise ValueError(f"none of the evidence channels were found: {list(evidence_names)}")
    evidence = values[:, channels].mean(axis=1)
    weight = float(np.clip(agreement_weight, 0.0, 1.0))
    if agreement_name and agreement_name in mapping and weight > 0.0:
        agreement = np.clip(values[:, mapping[agreement_name]], 0.0, 1.0)
        evidence = evidence * ((1.0 - weight) + weight * agreement)
    return np.clip(evidence, 0.0, 1.0).astype(np.float32)


def _cap_added_frames(original: np.ndarray, proposed: np.ndarray, consensus: np.ndarray, max_added_fraction: float | None) -> np.ndarray:
    if max_added_fraction is None:
        return proposed
    added = np.flatnonzero((original == 0) & (proposed == 1))
    max_added = int(round(float(max_added_fraction) * max(1, len(original))))
    if len(added) <= max_added:
        return proposed
    capped = proposed.copy()
    order = added[np.argsort(consensus[added])[::-1]]
    keep = set(int(idx) for idx in order[:max_added])
    for idx in added:
        if int(idx) not in keep:
            capped[int(idx)] = 0
    return capped


def denoise_labels_with_consensus(
    labels: np.ndarray,
    consensus: np.ndarray,
    config: ConsensusDenoiseConfig,
) -> tuple[np.ndarray, dict]:
    """Add high-consensus unlabeled islands and remove weak positive islands."""
    original = (np.asarray(labels).reshape(-1) > 0).astype(np.int64)
    score = np.nan_to_num(np.asarray(consensus, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    n = min(len(original), len(score))
    original = original[:n]
    score = np.clip(score[:n], 0.0, 1.0)
    if int(config.smooth_window) > 1:
        score = smooth_probabilities(score, int(config.smooth_window))

    denoised = original.copy()

    low_positive = ((original == 1) & (score < float(config.remove_threshold))).astype(np.int64)
    for start, end in contiguous_segments(low_positive):
        if end - start + 1 >= int(config.min_keep_length):
            denoised[start : end + 1] = 0

    high_unlabeled = ((original == 0) & (score >= float(config.add_threshold))).astype(np.int64)
    for start, end in contiguous_segments(high_unlabeled):
        if end - start + 1 >= int(config.min_add_length):
            denoised[start : end + 1] = 1

    denoised = _cap_added_frames(original, denoised, score, config.max_added_fraction)
    added = int(np.sum((original == 0) & (denoised == 1)))
    removed = int(np.sum((original == 1) & (denoised == 0)))
    return denoised.astype(np.int64), {
        "added_frames": added,
        "removed_frames": removed,
        "changed_frames": added + removed,
        "original_positive_frames": int(original.sum()),
        "denoised_positive_frames": int(denoised.sum()),
        "config": asdict(config),
    }


def denoise_records_for_fold(
    records: Sequence[VideoRecord],
    train_idx: Iterable[int],
    feature_names: Sequence[str],
    evidence_names: Sequence[str],
    agreement_name: str,
    config: ConsensusDenoiseConfig,
) -> tuple[list[VideoRecord], dict]:
    """Return copied records with denoising applied only to train_idx."""
    train_set = {int(idx) for idx in train_idx}
    copied: list[VideoRecord] = []
    summary = {
        "changed_videos": 0,
        "added_frames": 0,
        "removed_frames": 0,
        "changed_frames": 0,
        "train_videos": len(train_set),
        "config": asdict(config),
    }
    for idx, record in enumerate(records):
        signals = np.asarray(record.signals, dtype=np.float32).copy()
        labels = np.asarray(record.labels, dtype=np.int64).copy()
        if idx in train_set:
            consensus = compute_consensus_score(
                signals,
                feature_names=feature_names,
                evidence_names=evidence_names,
                agreement_name=agreement_name,
                agreement_weight=float(config.agreement_weight),
            )
            labels, stats = denoise_labels_with_consensus(labels, consensus, config)
            if int(stats["changed_frames"]) > 0:
                summary["changed_videos"] += 1
            summary["added_frames"] += int(stats["added_frames"])
            summary["removed_frames"] += int(stats["removed_frames"])
            summary["changed_frames"] += int(stats["changed_frames"])
        copied.append(VideoRecord(name=record.name, signals=signals, labels=labels))
    return copied, summary
