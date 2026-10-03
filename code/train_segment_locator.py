#!/usr/bin/env python3
"""
Train a binary temporal locator on true V/I-JEPA segment signals.

Input dataset:
  signals.npz: one [T, 3] array per video
  labels.npz:  one [T] binary array per video

The task is only error-segment localization. No category or severity labels are
loaded, trained, or evaluated here.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VideoRecord:
    name: str
    signals: np.ndarray
    labels: np.ndarray


def _npz_arrays(path: Path) -> list[np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return [np.asarray(archive[key]) for key in archive.files]


def load_signal_dataset(data_dir: str | Path) -> list[VideoRecord]:
    root = Path(data_dir)
    signal_path = root / "signals.npz"
    label_path = root / "labels.npz"
    if not signal_path.exists():
        raise FileNotFoundError(f"missing {signal_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"missing {label_path}")

    signals = _npz_arrays(signal_path)
    labels = _npz_arrays(label_path)
    if len(signals) != len(labels):
        raise ValueError(f"signals/labels count mismatch: {len(signals)} vs {len(labels)}")

    names: list[str]
    names_path = root / "video_names.json"
    if names_path.exists():
        names = json.loads(names_path.read_text(encoding="utf-8"))
    else:
        names = [f"video_{idx:04d}" for idx in range(len(signals))]

    records: list[VideoRecord] = []
    for idx, (sig, lab) in enumerate(zip(signals, labels)):
        sig = np.asarray(sig, dtype=np.float32)
        if sig.ndim != 2 or sig.shape[1] < 3:
            raise ValueError(f"record {idx} signals must have shape [T, C>=3], got {sig.shape}")
        lab = (np.asarray(lab).reshape(-1) > 0).astype(np.int64)
        n = min(len(sig), len(lab))
        if n <= 0:
            continue
        sig = np.nan_to_num(sig[:n], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        records.append(VideoRecord(str(names[idx]) if idx < len(names) else f"video_{idx:04d}", sig, lab[:n]))

    if not records:
        raise ValueError(f"no usable records in {root}")
    return records


def train_val_split(records: list[VideoRecord], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {"zero": [], "very_low": [], "low_mid": [], "mid": [], "high": []}
    for idx, record in enumerate(records):
        ratio = float(record.labels.sum()) / max(1, len(record.labels))
        if ratio <= 0.0:
            key = "zero"
        elif ratio <= 0.10:
            key = "very_low"
        elif ratio <= 0.30:
            key = "low_mid"
        elif ratio <= 0.60:
            key = "mid"
        else:
            key = "high"
        groups[key].append(idx)

    val: list[int] = []
    for items in groups.values():
        rng.shuffle(items)
        if len(items) <= 1:
            continue
        val.extend(items[: max(1, int(round(len(items) * val_ratio)))])
    val = sorted(set(val))
    if not val and len(records) > 1:
        val = [int(rng.integers(0, len(records)))]
    if len(val) == len(records) and len(records) > 1:
        val = val[:1]
    train = [idx for idx in range(len(records)) if idx not in set(val)]
    if not train:
        train = val[:]
    return train, val


def make_stratified_folds(records: list[VideoRecord], n_folds: int, seed: int) -> list[list[int]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not records:
        raise ValueError("records must not be empty")
    fold_count = min(int(n_folds), len(records))
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {"zero": [], "very_low": [], "low_mid": [], "mid": [], "high": []}
    for idx, record in enumerate(records):
        ratio = float(record.labels.sum()) / max(1, len(record.labels))
        if ratio <= 0.0:
            key = "zero"
        elif ratio <= 0.10:
            key = "very_low"
        elif ratio <= 0.30:
            key = "low_mid"
        elif ratio <= 0.60:
            key = "mid"
        else:
            key = "high"
        groups[key].append(idx)

    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for items in groups.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        start = int(rng.integers(0, fold_count)) if fold_count > 1 else 0
        for offset, idx in enumerate(shuffled):
            folds[(start + offset) % fold_count].append(idx)

    for fold_idx, fold in enumerate(folds):
        if fold:
            continue
        donor = max(range(fold_count), key=lambda item: len(folds[item]))
        folds[fold_idx].append(folds[donor].pop())
    return [sorted(fold) for fold in folds]


def compute_normalizer(records: list[VideoRecord], indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    selected = [records[idx].signals for idx in indices]
    if not selected:
        selected = [record.signals for record in records]
    stacked = np.concatenate(selected, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.full_like(std, 1e-4))
    return mean, std


def normalized_signals(record: VideoRecord, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((record.signals - mean) / std).astype(np.float32)


class TemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, hidden)
        self.norm2 = nn.GroupNorm(1, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.gelu(x + residual)


class TemporalSegmentLocator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64, dropout: float = 0.1, architecture: str = "tcn"):
        super().__init__()
        if architecture not in {"tcn", "attn_tcn", "bilstm"}:
            raise ValueError(f"unsupported temporal locator architecture: {architecture}")
        self.architecture = str(architecture)
        self.input = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalBlock(hidden, dilation=1, dropout=dropout),
            TemporalBlock(hidden, dilation=2, dropout=dropout),
            TemporalBlock(hidden, dilation=4, dropout=dropout),
            TemporalBlock(hidden, dilation=8, dropout=dropout),
        )
        if self.architecture == "attn_tcn":
            n_heads = 4 if hidden % 4 == 0 else 2 if hidden % 2 == 0 else 1
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=max(hidden * 2, 32),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.attn_norm = nn.LayerNorm(hidden)
        else:
            self.attention = None
            self.attn_norm = None
        if self.architecture == "bilstm":
            lstm_hidden = max(1, hidden // 2)
            self.recurrent = nn.LSTM(
                input_size=hidden,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            output_channels = lstm_hidden * 2
            self.recurrent_norm = nn.LayerNorm(output_channels)
            self.recurrent_dropout = nn.Dropout(dropout)
        else:
            self.recurrent = None
            self.recurrent_norm = None
            self.recurrent_dropout = None
            output_channels = hidden
        self.output = nn.Conv1d(output_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, C], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = F.gelu(self.input(x))
        x = self.blocks(x)
        if self.attention is not None:
            seq = x.transpose(1, 2)
            attended = self.attention(seq)
            if self.attn_norm is not None:
                attended = self.attn_norm(attended + seq)
            x = attended.transpose(1, 2)
        if self.recurrent is not None:
            seq = x.transpose(1, 2)
            recurrent, _ = self.recurrent(seq)
            if self.recurrent_norm is not None:
                recurrent = self.recurrent_norm(recurrent)
            if self.recurrent_dropout is not None:
                recurrent = self.recurrent_dropout(recurrent)
            x = recurrent.transpose(1, 2)
        return self.output(x).squeeze(1)


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(labels.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def smooth_probabilities(probs: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return np.asarray(probs, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.asarray(probs, dtype=np.float32), kernel, mode="same")


def gaussian_smooth_probabilities(probs: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float32).reshape(-1)
    sigma = float(sigma)
    if sigma <= 0.0 or len(values) <= 1:
        return values.astype(np.float32)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= float(kernel.sum())
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def event_refine_probabilities(
    probs: np.ndarray,
    sigmas: Iterable[float] = (1.0, 2.0, 4.0),
    raw_weight: float = 0.5,
) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float32).reshape(-1)
    sigma_values = [float(sigma) for sigma in sigmas if float(sigma) > 0.0]
    if not sigma_values:
        return values.astype(np.float32)
    smoothed = [gaussian_smooth_probabilities(values, sigma) for sigma in sigma_values]
    event_context = np.maximum.reduce(smoothed)
    weight = float(np.clip(raw_weight, 0.0, 1.0))
    refined = weight * values + (1.0 - weight) * event_context
    return np.nan_to_num(refined, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def refine_prediction_records(
    model_outputs: list[dict],
    raw_weight: float,
    sigmas: Iterable[float],
) -> list[dict]:
    refined = []
    for output in model_outputs:
        refined.append(
            {
                "name": output.get("name", ""),
                "labels": np.asarray(output["labels"], dtype=np.int64).copy(),
                "probs": event_refine_probabilities(output["probs"], sigmas=sigmas, raw_weight=raw_weight),
            }
        )
    return refined


def _merge_short_gaps(binary: np.ndarray, min_gap: int) -> np.ndarray:
    if min_gap <= 0:
        return binary
    merged = binary.copy()
    segments = contiguous_segments(merged)
    for left, right in zip(segments, segments[1:]):
        gap_start = left[1] + 1
        gap_end = right[0] - 1
        if 0 <= gap_end - gap_start + 1 <= min_gap:
            merged[gap_start : gap_end + 1] = 1
    return merged


def probabilities_to_segments(
    probs: np.ndarray,
    threshold: float,
    smooth_window: int = 1,
    min_gap: int = 0,
    min_length: int = 1,
) -> list[tuple[int, int]]:
    smoothed = smooth_probabilities(probs, smooth_window)
    binary = (smoothed >= float(threshold)).astype(np.int64)
    binary = _merge_short_gaps(binary, int(min_gap))
    return [
        (start, end)
        for start, end in contiguous_segments(binary)
        if end - start + 1 >= int(min_length)
    ]


def hysteresis_probabilities_to_segments(
    probs: np.ndarray,
    high_threshold: float,
    low_threshold: float,
    smooth_window: int = 1,
    min_gap: int = 0,
    min_length: int = 1,
) -> list[tuple[int, int]]:
    smoothed = smooth_probabilities(probs, smooth_window)
    high = smoothed >= float(high_threshold)
    low = smoothed >= min(float(low_threshold), float(high_threshold))
    binary = np.zeros_like(high, dtype=np.int64)
    for start, end in contiguous_segments(high.astype(np.int64)):
        left = start
        while left > 0 and low[left - 1]:
            left -= 1
        right = end
        while right + 1 < len(low) and low[right + 1]:
            right += 1
        binary[left : right + 1] = 1
    binary = _merge_short_gaps(binary, int(min_gap))
    return [
        (start, end)
        for start, end in contiguous_segments(binary)
        if end - start + 1 >= int(min_length)
    ]


def _segments_from_params(probs: np.ndarray, params: dict) -> list[tuple[int, int]]:
    if params.get("low_threshold") is None:
        return probabilities_to_segments(
            probs,
            threshold=float(params["threshold"]),
            smooth_window=int(params["smooth_window"]),
            min_gap=int(params["min_gap"]),
            min_length=int(params["min_length"]),
        )
    return hysteresis_probabilities_to_segments(
        probs,
        high_threshold=float(params["threshold"]),
        low_threshold=float(params["low_threshold"]),
        smooth_window=int(params["smooth_window"]),
        min_gap=int(params["min_gap"]),
        min_length=int(params["min_length"]),
    )


def _segment_intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0, end - start + 1)


def _segment_length(segment: tuple[int, int]) -> int:
    return max(0, int(segment[1]) - int(segment[0]) + 1)


def _max_candidate_coverage(candidate: tuple[int, int], segments: list[tuple[int, int]]) -> float:
    length = max(1, _segment_length(candidate))
    return max((_segment_intersection(candidate, segment) / length for segment in segments), default=0.0)


def _expand_segment(segment: tuple[int, int], n_frames: int, pad: int, min_length: int) -> tuple[int, int]:
    start = max(0, int(segment[0]) - int(pad))
    end = min(int(n_frames) - 1, int(segment[1]) + int(pad))
    target = max(1, int(min_length))
    while end - start + 1 < target and (start > 0 or end + 1 < n_frames):
        if start > 0:
            start -= 1
        if end - start + 1 >= target:
            break
        if end + 1 < n_frames:
            end += 1
    return start, end


def _candidate_contrast(values: np.ndarray, segment: tuple[int, int]) -> float:
    start, end = segment
    segment_values = values[start : end + 1]
    if len(segment_values) == 0:
        return 0.0
    if start <= 0 and end + 1 >= len(values):
        background = values
    elif start <= 0:
        background = values[end + 1 :]
    elif end + 1 >= len(values):
        background = values[:start]
    else:
        background = np.concatenate([values[:start], values[end + 1 :]])
    if len(background) == 0:
        background = values
    return float(np.max(segment_values) - np.median(background))


def _auxiliary_candidates(
    values: np.ndarray,
    threshold: float,
    smooth_window: int,
    min_gap: int,
    min_length: int,
    pad: int,
    min_contrast: float,
) -> list[dict]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(values) == 0:
        return []
    smoothed = smooth_probabilities(values, int(smooth_window))
    binary = (smoothed >= float(threshold)).astype(np.int64)
    binary = _merge_short_gaps(binary, int(min_gap))
    candidates: list[dict] = []
    for start, end in contiguous_segments(binary):
        if end - start + 1 < int(min_length):
            continue
        segment = _expand_segment((start, end), len(values), pad=int(pad), min_length=int(min_length))
        contrast = _candidate_contrast(smoothed, segment)
        if contrast < float(min_contrast):
            continue
        seg_values = smoothed[segment[0] : segment[1] + 1]
        candidates.append(
            {
                "segment": segment,
                "score": float(np.max(seg_values) + 0.1 * np.mean(seg_values) + 0.01 * contrast),
                "mean": float(np.mean(seg_values)),
                "max": float(np.max(seg_values)),
                "contrast": float(contrast),
            }
        )
    return candidates


def apply_recall_repair_to_records(
    model_outputs: list[dict],
    records: list[VideoRecord],
    indices: Iterable[int],
    base_params: dict,
    aux_channels: Iterable[int],
    aux_threshold: float = 0.85,
    aux_smooth_window: int = 1,
    aux_min_gap: int = 0,
    aux_min_length: int = 1,
    pad: int = 0,
    min_contrast: float = 0.0,
    max_existing_coverage: float = 0.25,
    nms_iou: float = 0.3,
    max_rescues_per_video: int = 3,
    rescue_value: float | None = None,
) -> list[dict]:
    idx_list = list(indices)
    if len(model_outputs) != len(idx_list):
        raise ValueError(f"outputs/indices mismatch: {len(model_outputs)} vs {len(idx_list)}")
    channel_list = [int(channel) for channel in aux_channels]
    repaired: list[dict] = []
    threshold = float(base_params.get("threshold", 0.5))
    write_value = float(rescue_value) if rescue_value is not None else max(1.0, threshold + 1e-3)
    for output, record_idx in zip(model_outputs, idx_list):
        record = records[int(record_idx)]
        probs = np.asarray(output["probs"], dtype=np.float32).reshape(-1)
        labels = np.asarray(output.get("labels", record.labels), dtype=np.int64).reshape(-1)
        n = min(len(probs), len(labels), len(record.signals))
        probs = np.nan_to_num(probs[:n], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        labels = labels[:n].copy()
        base_segments = _segments_from_params(probs, base_params)

        candidates: list[dict] = []
        for channel in channel_list:
            if channel < 0 or channel >= record.signals.shape[1]:
                raise IndexError(f"aux_channel {channel} outside signal shape {record.signals.shape}")
            for candidate in _auxiliary_candidates(
                record.signals[:n, channel],
                threshold=aux_threshold,
                smooth_window=aux_smooth_window,
                min_gap=aux_min_gap,
                min_length=aux_min_length,
                pad=pad,
                min_contrast=min_contrast,
            ):
                candidate = dict(candidate)
                candidate["channel"] = channel
                candidates.append(candidate)

        candidates.sort(key=lambda item: (float(item["score"]), _segment_length(item["segment"])), reverse=True)
        accepted: list[dict] = []
        accepted_segments: list[tuple[int, int]] = []
        for candidate in candidates:
            segment = candidate["segment"]
            if _max_candidate_coverage(segment, base_segments) > float(max_existing_coverage):
                continue
            if any(segment_iou(segment, accepted_segment) > float(nms_iou) for accepted_segment in accepted_segments):
                continue
            accepted.append(candidate)
            accepted_segments.append(segment)
            if len(accepted) >= int(max_rescues_per_video):
                break

        repaired_probs = probs.copy()
        final_min_length = max(int(base_params.get("min_length", 1)), int(aux_min_length))
        rescue_segments: list[tuple[int, int]] = []
        rescue_details: list[dict] = []
        for candidate in accepted:
            start, end = _expand_segment(candidate["segment"], n, pad=0, min_length=final_min_length)
            repaired_probs[start : end + 1] = np.maximum(repaired_probs[start : end + 1], write_value)
            rescue_segments.append((start, end))
            rescue_details.append({**candidate, "segment": [start, end]})

        repaired.append(
            {
                "name": output.get("name", record.name),
                "labels": labels,
                "probs": repaired_probs.astype(np.float32),
                "rescue_segments": rescue_segments,
                "rescue_details": rescue_details,
            }
        )
    return repaired


def _evidence_islands(
    evidence: np.ndarray,
    parent: tuple[int, int],
    evidence_threshold: float,
    smooth_window: int,
    island_min_gap: int,
    island_min_length: int,
    pad: int,
) -> list[tuple[int, int]]:
    start, end = parent
    local = np.asarray(evidence[start : end + 1], dtype=np.float32)
    if len(local) == 0:
        return []
    smoothed = smooth_probabilities(local, int(smooth_window))
    binary = (smoothed >= float(evidence_threshold)).astype(np.int64)
    binary = _merge_short_gaps(binary, int(island_min_gap))
    islands: list[tuple[int, int]] = []
    for local_start, local_end in contiguous_segments(binary):
        if local_end - local_start + 1 < int(island_min_length):
            continue
        island = _expand_segment(
            (start + local_start, start + local_end),
            len(evidence),
            pad=int(pad),
            min_length=int(island_min_length),
        )
        island = (max(start, island[0]), min(end, island[1]))
        if _segment_length(island) >= int(island_min_length):
            islands.append(island)
    return islands


def apply_valley_split_to_records(
    model_outputs: list[dict],
    records: list[VideoRecord],
    indices: Iterable[int],
    base_params: dict,
    aux_channels: Iterable[int],
    evidence_threshold: float = 0.7,
    evidence_smooth_window: int = 1,
    island_min_gap: int = 0,
    island_min_length: int = 1,
    pad: int = 0,
    parent_min_length: int = 24,
    split_value: float | None = None,
    peak_value: float | None = None,
) -> list[dict]:
    idx_list = list(indices)
    if len(model_outputs) != len(idx_list):
        raise ValueError(f"outputs/indices mismatch: {len(model_outputs)} vs {len(idx_list)}")
    channel_list = [int(channel) for channel in aux_channels]
    if not channel_list:
        return [
            {
                "name": output.get("name", ""),
                "labels": np.asarray(output["labels"], dtype=np.int64).copy(),
                "probs": np.asarray(output["probs"], dtype=np.float32).copy(),
                "valley_split_segments": [],
            }
            for output in model_outputs
        ]

    threshold = float(base_params.get("threshold", 0.5))
    low_threshold = float(base_params.get("low_threshold", threshold))
    valley_value = float(split_value) if split_value is not None else max(0.0, min(threshold, low_threshold) - 1e-3)
    write_value = float(peak_value) if peak_value is not None else max(1.0, threshold + 1e-3)
    split_records: list[dict] = []
    for output, record_idx in zip(model_outputs, idx_list):
        record = records[int(record_idx)]
        probs = np.asarray(output["probs"], dtype=np.float32).reshape(-1)
        labels = np.asarray(output.get("labels", record.labels), dtype=np.int64).reshape(-1)
        n = min(len(probs), len(labels), len(record.signals))
        probs = np.nan_to_num(probs[:n], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        labels = labels[:n].copy()
        for channel in channel_list:
            if channel < 0 or channel >= record.signals.shape[1]:
                raise IndexError(f"aux_channel {channel} outside signal shape {record.signals.shape}")
        evidence_values = np.maximum.reduce(
            [np.asarray(record.signals[:n, channel], dtype=np.float32).reshape(-1) for channel in channel_list]
        )
        evidence_values = np.nan_to_num(evidence_values, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        base_segments = _segments_from_params(probs, base_params)
        updated = probs.copy()
        split_segments: list[tuple[int, int]] = []
        split_details: list[dict] = []
        for parent in base_segments:
            if _segment_length(parent) < int(parent_min_length):
                continue
            islands = _evidence_islands(
                evidence_values,
                parent=parent,
                evidence_threshold=evidence_threshold,
                smooth_window=evidence_smooth_window,
                island_min_gap=island_min_gap,
                island_min_length=island_min_length,
                pad=pad,
            )
            if len(islands) < 2:
                continue
            parent_start, parent_end = parent
            updated[parent_start : parent_end + 1] = np.minimum(updated[parent_start : parent_end + 1], valley_value)
            for island_start, island_end in islands:
                updated[island_start : island_end + 1] = np.maximum(updated[island_start : island_end + 1], write_value)
                split_segments.append((island_start, island_end))
            split_details.append({"parent": [int(parent_start), int(parent_end)], "islands": [[int(s), int(e)] for s, e in islands]})

        split_records.append(
            {
                "name": output.get("name", record.name),
                "labels": labels,
                "probs": updated.astype(np.float32),
                "valley_split_segments": split_segments,
                "valley_split_details": split_details,
            }
        )
    return split_records


def segment_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    inter = max(0, inter_end - inter_start + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / max(1, union)


def _binary_frame_metrics(labels: np.ndarray, pred: np.ndarray) -> dict:
    truth = labels.astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _frame_metrics(labels: np.ndarray, probs: np.ndarray, params: dict) -> dict:
    smoothed = smooth_probabilities(probs, int(params["smooth_window"]))
    pred = smoothed >= float(params["threshold"])
    return _binary_frame_metrics(labels, pred)


def _segment_counts(predicted: list[tuple[int, int]], target: list[tuple[int, int]], iou_threshold: float) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = 0
    for pred in predicted:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(target):
            if idx in matched:
                continue
            iou = segment_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.add(best_idx)
            tp += 1
    fp = len(predicted) - tp
    fn = len(target) - tp
    return tp, fp, fn


def evaluate_records(records: list[dict], params: dict, iou_threshold: float = 0.3) -> dict:
    frame_tp = frame_fp = frame_fn = 0
    seg_tp = seg_fp = seg_fn = 0
    for record in records:
        labels = np.asarray(record["labels"], dtype=np.int64)
        probs = np.asarray(record["probs"], dtype=np.float32)
        pred_segments = _segments_from_params(probs, params)
        if params.get("low_threshold") is None:
            frame = _frame_metrics(labels, probs, params)
        else:
            pred_binary = np.zeros_like(labels, dtype=np.int64)
            for start, end in pred_segments:
                pred_binary[start : end + 1] = 1
            frame = _binary_frame_metrics(labels, pred_binary)
        frame_tp += frame["tp"]
        frame_fp += frame["fp"]
        frame_fn += frame["fn"]

        true_segments = contiguous_segments(labels)
        tp, fp, fn = _segment_counts(pred_segments, true_segments, iou_threshold)
        seg_tp += tp
        seg_fp += fp
        seg_fn += fn

    frame_precision = frame_tp / (frame_tp + frame_fp + 1e-6)
    frame_recall = frame_tp / (frame_tp + frame_fn + 1e-6)
    frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall + 1e-6)
    seg_precision = seg_tp / (seg_tp + seg_fp + 1e-6)
    seg_recall = seg_tp / (seg_tp + seg_fn + 1e-6)
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall + 1e-6)
    return {
        "frame": {
            "precision": frame_precision,
            "recall": frame_recall,
            "f1": frame_f1,
            "tp": frame_tp,
            "fp": frame_fp,
            "fn": frame_fn,
        },
        "segment": {
            "precision": seg_precision,
            "recall": seg_recall,
            "f1": seg_f1,
            "tp": seg_tp,
            "fp": seg_fp,
            "fn": seg_fn,
        },
    }


def select_postprocess_params(
    records: list[dict],
    thresholds: Iterable[float] | None = None,
    smooth_windows: Iterable[int] | None = None,
    min_gaps: Iterable[int] | None = None,
    min_lengths: Iterable[int] | None = None,
    low_thresholds: Iterable[float | None] | None = None,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict]:
    thresholds = list(thresholds if thresholds is not None else np.linspace(0.2, 0.8, 13))
    smooth_windows = list(smooth_windows if smooth_windows is not None else [1, 3, 5])
    min_gaps = list(min_gaps if min_gaps is not None else [0, 1, 2])
    min_lengths = list(min_lengths if min_lengths is not None else [1, 2, 4])
    low_thresholds = list(low_thresholds if low_thresholds is not None else [None, 0.2, 0.3, 0.4, 0.5, 0.6])

    best_params: dict | None = None
    best_metrics: dict | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold in thresholds:
        for smooth_window in smooth_windows:
            for min_gap in min_gaps:
                for min_length in min_lengths:
                    for low_threshold in low_thresholds:
                        if low_threshold is not None and float(low_threshold) > float(threshold):
                            continue
                        params = {
                            "threshold": float(threshold),
                            "smooth_window": int(smooth_window),
                            "min_gap": int(min_gap),
                            "min_length": int(min_length),
                        }
                        if low_threshold is not None and float(low_threshold) < float(threshold):
                            params["low_threshold"] = float(low_threshold)
                        metrics = evaluate_records(records, params, iou_threshold=iou_threshold)
                        key = (
                            float(metrics["segment"]["f1"]),
                            float(metrics["frame"]["f1"]),
                            -abs(float(threshold) - 0.5),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_params = params
                            best_metrics = metrics
    assert best_params is not None and best_metrics is not None
    return best_params, best_metrics


def _segments_for_tensor(labels: torch.Tensor) -> list[tuple[int, int]]:
    return contiguous_segments(labels.detach().cpu().numpy().astype(np.int64))


def soft_dice_loss(probs: torch.Tensor, labels: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = probs.float()
    labels = labels.float()
    if probs.shape != labels.shape:
        raise ValueError(f"probs shape {tuple(probs.shape)} != labels shape {tuple(labels.shape)}")
    reduce_dims = tuple(range(1, probs.ndim))
    intersection = (probs * labels).sum(dim=reduce_dims)
    denominator = probs.sum(dim=reduce_dims) + labels.sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice).mean()


def boundary_transition_loss(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = probs.float()
    labels = labels.float()
    if probs.shape != labels.shape:
        raise ValueError(f"probs shape {tuple(probs.shape)} != labels shape {tuple(labels.shape)}")
    if probs.shape[1] <= 1:
        return probs.new_tensor(0.0)
    pred_delta = torch.abs(probs[:, 1:] - probs[:, :-1])
    label_delta = torch.abs(labels[:, 1:] - labels[:, :-1])
    return F.binary_cross_entropy(pred_delta.clamp(1e-6, 1.0 - 1e-6), label_delta)


def locator_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float,
    frame_weights: torch.Tensor | None = None,
    lambda_smooth: float = 0.05,
    lambda_segment: float = 0.25,
    lambda_dice: float = 0.0,
    lambda_boundary: float = 0.0,
) -> torch.Tensor:
    bce_per_frame = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=torch.tensor(float(pos_weight), device=logits.device),
        reduction="none",
    )
    if frame_weights is None:
        bce = bce_per_frame.mean()
    else:
        weights = torch.as_tensor(frame_weights, dtype=bce_per_frame.dtype, device=logits.device)
        if weights.shape != bce_per_frame.shape:
            raise ValueError(f"frame_weights shape {tuple(weights.shape)} != logits shape {tuple(logits.shape)}")
        weights = torch.clamp(weights, min=0.0)
        bce = (bce_per_frame * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
    probs = torch.sigmoid(logits)
    if probs.shape[1] > 1:
        smooth = torch.abs(probs[:, 1:] - probs[:, :-1]).mean()
    else:
        smooth = probs.new_tensor(0.0)

    seg_losses: list[torch.Tensor] = []
    for batch_idx in range(labels.shape[0]):
        for start, end in _segments_for_tensor(labels[batch_idx]):
            seg_logits = logits[batch_idx, start : end + 1]
            seg_losses.append(F.binary_cross_entropy_with_logits(seg_logits.mean(), seg_logits.new_tensor(1.0)))
    segment = torch.stack(seg_losses).mean() if seg_losses else probs.new_tensor(0.0)
    dice = soft_dice_loss(probs, labels.float()) if lambda_dice else probs.new_tensor(0.0)
    boundary = boundary_transition_loss(probs, labels.float()) if lambda_boundary else probs.new_tensor(0.0)
    return bce + lambda_smooth * smooth + lambda_segment * segment + lambda_dice * dice + lambda_boundary * boundary


def compute_pos_weight(records: list[VideoRecord], indices: Iterable[int]) -> float:
    pos = 0
    total = 0
    for idx in indices:
        labels = records[idx].labels
        pos += int(labels.sum())
        total += len(labels)
    neg = total - pos
    if pos == 0:
        return 1.0
    return float(min(50.0, max(1.0, neg / max(1, pos))))


def predict_records(
    model: nn.Module,
    records: list[VideoRecord],
    indices: Iterable[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> list[dict]:
    model.eval()
    outputs: list[dict] = []
    with torch.no_grad():
        for idx in indices:
            record = records[idx]
            x = torch.from_numpy(normalized_signals(record, mean, std)).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
            outputs.append({"name": record.name, "labels": record.labels.copy(), "probs": probs})
    return outputs


def blend_prediction_records(
    model_outputs: list[dict],
    records: list[VideoRecord],
    indices: Iterable[int],
    aux_channel: int,
    alpha_model: float,
) -> list[dict]:
    idx_list = list(indices)
    if len(model_outputs) != len(idx_list):
        raise ValueError(f"outputs/indices mismatch: {len(model_outputs)} vs {len(idx_list)}")
    alpha = float(alpha_model)
    blended: list[dict] = []
    for output, record_idx in zip(model_outputs, idx_list):
        record = records[int(record_idx)]
        if aux_channel < 0 or aux_channel >= record.signals.shape[1]:
            raise IndexError(f"aux_channel {aux_channel} outside signal shape {record.signals.shape}")
        model_probs = np.asarray(output["probs"], dtype=np.float32).reshape(-1)
        aux_probs = np.asarray(record.signals[:, aux_channel], dtype=np.float32).reshape(-1)
        labels = np.asarray(output.get("labels", record.labels), dtype=np.int64).reshape(-1)
        n = min(len(model_probs), len(aux_probs), len(labels))
        probs = alpha * model_probs[:n] + (1.0 - alpha) * aux_probs[:n]
        blended.append(
            {
                "name": output.get("name", record.name),
                "labels": labels[:n].copy(),
                "probs": np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
            }
        )
    return blended


def select_blended_postprocess_params(
    model_outputs: list[dict],
    records: list[VideoRecord],
    indices: Iterable[int],
    aux_channel: int,
    alphas: Iterable[float] | None = None,
    thresholds: Iterable[float] | None = None,
    smooth_windows: Iterable[int] | None = None,
    min_gaps: Iterable[int] | None = None,
    min_lengths: Iterable[int] | None = None,
    iou_threshold: float = 0.3,
) -> tuple[float, dict, dict]:
    alpha_values = list(alphas if alphas is not None else np.linspace(0.0, 1.0, 11))
    best_alpha = 1.0
    best_params: dict | None = None
    best_metrics: dict | None = None
    best_key: tuple[float, float, float] | None = None
    idx_list = list(indices)
    for alpha in alpha_values:
        blended = blend_prediction_records(model_outputs, records, idx_list, aux_channel, float(alpha))
        params, metrics = select_postprocess_params(
            blended,
            thresholds=thresholds,
            smooth_windows=smooth_windows,
            min_gaps=min_gaps,
            min_lengths=min_lengths,
            iou_threshold=iou_threshold,
        )
        key = (
            float(metrics["segment"]["f1"]),
            float(metrics["frame"]["f1"]),
            float(alpha),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_alpha = float(alpha)
            best_params = params
            best_metrics = metrics
    assert best_params is not None and best_metrics is not None
    return best_alpha, best_params, best_metrics


def select_event_refined_postprocess_params(
    model_outputs: list[dict],
    raw_weights: Iterable[float] | None = None,
    sigma_sets: Iterable[Iterable[float]] | None = None,
    thresholds: Iterable[float] | None = None,
    smooth_windows: Iterable[int] | None = None,
    min_gaps: Iterable[int] | None = None,
    min_lengths: Iterable[int] | None = None,
    low_thresholds: Iterable[float | None] | None = None,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, dict]:
    raw_weights = list(raw_weights if raw_weights is not None else [1.0, 0.75, 0.5, 0.25, 0.0])
    sigma_sets = list(sigma_sets if sigma_sets is not None else [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 4.0]])
    best_config: dict | None = None
    best_params: dict | None = None
    best_metrics: dict | None = None
    best_key: tuple[float, float, float] | None = None
    for raw_weight in raw_weights:
        for sigmas in sigma_sets:
            refined = refine_prediction_records(model_outputs, raw_weight=float(raw_weight), sigmas=sigmas)
            params, metrics = select_postprocess_params(
                refined,
                thresholds=thresholds,
                smooth_windows=smooth_windows,
                min_gaps=min_gaps,
                min_lengths=min_lengths,
                low_thresholds=low_thresholds,
                iou_threshold=iou_threshold,
            )
            key = (
                float(metrics["segment"]["f1"]),
                float(metrics["frame"]["f1"]),
                float(raw_weight),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_config = {"raw_weight": float(raw_weight), "sigmas": [float(x) for x in sigmas]}
                best_params = params
                best_metrics = metrics
    assert best_config is not None and best_params is not None and best_metrics is not None
    return best_config, best_params, best_metrics


def _count_rescue_segments(records: list[dict]) -> int:
    return int(sum(len(record.get("rescue_segments", [])) for record in records))


def _count_valley_split_segments(records: list[dict]) -> int:
    return int(sum(len(record.get("valley_split_segments", [])) for record in records))


def select_recall_repair_params(
    model_outputs: list[dict],
    records: list[VideoRecord],
    indices: Iterable[int],
    base_params: dict,
    aux_channels: Iterable[int],
    aux_thresholds: Iterable[float] | None = None,
    aux_smooth_windows: Iterable[int] | None = None,
    aux_min_gaps: Iterable[int] | None = None,
    aux_min_lengths: Iterable[int] | None = None,
    pads: Iterable[int] | None = None,
    min_contrasts: Iterable[float] | None = None,
    max_existing_coverages: Iterable[float] | None = None,
    max_rescues_per_videos: Iterable[int] | None = None,
    iou_threshold: float = 0.3,
) -> tuple[dict, list[dict], dict]:
    idx_list = list(indices)
    channel_list = [int(channel) for channel in aux_channels]
    if not channel_list:
        metrics = evaluate_records(model_outputs, base_params, iou_threshold=iou_threshold)
        return {"enabled": False, "reason": "no_aux_channels"}, model_outputs, metrics

    aux_thresholds = list(aux_thresholds if aux_thresholds is not None else [0.55, 0.6, 0.7, 0.8, 0.9])
    aux_smooth_windows = list(aux_smooth_windows if aux_smooth_windows is not None else [1, 3])
    aux_min_gaps = list(aux_min_gaps if aux_min_gaps is not None else [0, 1])
    aux_min_lengths = list(aux_min_lengths if aux_min_lengths is not None else [1, 2, 4])
    pads = list(pads if pads is not None else [0, 1, 2])
    min_contrasts = list(min_contrasts if min_contrasts is not None else [0.0, 0.1, 0.2])
    max_existing_coverages = list(max_existing_coverages if max_existing_coverages is not None else [0.25, 0.5])
    max_rescues_per_videos = list(max_rescues_per_videos if max_rescues_per_videos is not None else [1, 2, 3])

    base_metrics = evaluate_records(model_outputs, base_params, iou_threshold=iou_threshold)
    best_config: dict = {
        "enabled": False,
        "aux_channels": channel_list,
        "rescue_segments": 0,
        "base_metrics": base_metrics,
    }
    best_records = model_outputs
    best_metrics = base_metrics
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )

    for aux_threshold in aux_thresholds:
        for aux_smooth_window in aux_smooth_windows:
            for aux_min_gap in aux_min_gaps:
                for aux_min_length in aux_min_lengths:
                    for pad in pads:
                        for min_contrast in min_contrasts:
                            for max_existing_coverage in max_existing_coverages:
                                for max_rescues_per_video in max_rescues_per_videos:
                                    repaired = apply_recall_repair_to_records(
                                        model_outputs,
                                        records,
                                        idx_list,
                                        base_params=base_params,
                                        aux_channels=channel_list,
                                        aux_threshold=float(aux_threshold),
                                        aux_smooth_window=int(aux_smooth_window),
                                        aux_min_gap=int(aux_min_gap),
                                        aux_min_length=int(aux_min_length),
                                        pad=int(pad),
                                        min_contrast=float(min_contrast),
                                        max_existing_coverage=float(max_existing_coverage),
                                        max_rescues_per_video=int(max_rescues_per_video),
                                    )
                                    rescue_count = _count_rescue_segments(repaired)
                                    if rescue_count <= 0:
                                        continue
                                    metrics = evaluate_records(repaired, base_params, iou_threshold=iou_threshold)
                                    key = (
                                        float(metrics["segment"]["f1"]),
                                        float(metrics["segment"]["precision"]),
                                        float(metrics["segment"]["recall"]),
                                        float(metrics["frame"]["f1"]),
                                        -float(rescue_count),
                                    )
                                    if key > best_key:
                                        best_key = key
                                        best_records = repaired
                                        best_metrics = metrics
                                        best_config = {
                                            "enabled": True,
                                            "aux_channels": channel_list,
                                            "aux_threshold": float(aux_threshold),
                                            "aux_smooth_window": int(aux_smooth_window),
                                            "aux_min_gap": int(aux_min_gap),
                                            "aux_min_length": int(aux_min_length),
                                            "pad": int(pad),
                                            "min_contrast": float(min_contrast),
                                            "max_existing_coverage": float(max_existing_coverage),
                                            "max_rescues_per_video": int(max_rescues_per_video),
                                            "rescue_segments": rescue_count,
                                            "base_metrics": base_metrics,
                                        }
    return best_config, best_records, best_metrics


def select_valley_split_params(
    model_outputs: list[dict],
    records: list[VideoRecord],
    indices: Iterable[int],
    base_params: dict,
    aux_channels: Iterable[int],
    evidence_thresholds: Iterable[float] | None = None,
    evidence_smooth_windows: Iterable[int] | None = None,
    island_min_gaps: Iterable[int] | None = None,
    island_min_lengths: Iterable[int] | None = None,
    pads: Iterable[int] | None = None,
    parent_min_lengths: Iterable[int] | None = None,
    thresholds: Iterable[float] | None = None,
    smooth_windows: Iterable[int] | None = None,
    min_gaps: Iterable[int] | None = None,
    min_lengths: Iterable[int] | None = None,
    low_thresholds: Iterable[float | None] | None = None,
    iou_threshold: float = 0.3,
) -> tuple[dict, list[dict], dict]:
    idx_list = list(indices)
    channel_list = [int(channel) for channel in aux_channels]
    if not channel_list:
        metrics = evaluate_records(model_outputs, base_params, iou_threshold=iou_threshold)
        return {"enabled": False, "reason": "no_aux_channels"}, model_outputs, metrics

    evidence_thresholds = list(evidence_thresholds if evidence_thresholds is not None else [0.45, 0.55, 0.65])
    evidence_smooth_windows = list(evidence_smooth_windows if evidence_smooth_windows is not None else [1, 3])
    island_min_gaps = list(island_min_gaps if island_min_gaps is not None else [0, 1])
    island_min_lengths = list(island_min_lengths if island_min_lengths is not None else [1, 2, 4])
    pads = list(pads if pads is not None else [0, 1])
    parent_min_lengths = list(parent_min_lengths if parent_min_lengths is not None else [24, 48])
    thresholds = list(thresholds if thresholds is not None else [float(base_params.get("threshold", 0.5))])
    smooth_windows = list(
        smooth_windows
        if smooth_windows is not None
        else sorted({1, int(base_params.get("smooth_window", 1))})
    )
    min_gaps = list(min_gaps if min_gaps is not None else sorted({0, int(base_params.get("min_gap", 0))}))
    min_lengths = list(min_lengths if min_lengths is not None else sorted({1, int(base_params.get("min_length", 1))}))
    if low_thresholds is None:
        low_thresholds = [float(base_params["low_threshold"])] if base_params.get("low_threshold") is not None else [None]
    else:
        low_thresholds = list(low_thresholds)

    base_metrics = evaluate_records(model_outputs, base_params, iou_threshold=iou_threshold)
    best_config: dict = {
        "enabled": False,
        "aux_channels": channel_list,
        "split_segments": 0,
        "params": dict(base_params),
        "base_params": dict(base_params),
        "base_metrics": base_metrics,
    }
    best_records = model_outputs
    best_metrics = base_metrics
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )

    for evidence_threshold in evidence_thresholds:
        for evidence_smooth_window in evidence_smooth_windows:
            for island_min_gap in island_min_gaps:
                for island_min_length in island_min_lengths:
                    for pad in pads:
                        for parent_min_length in parent_min_lengths:
                            split = apply_valley_split_to_records(
                                model_outputs,
                                records,
                                idx_list,
                                base_params=base_params,
                                aux_channels=channel_list,
                                evidence_threshold=float(evidence_threshold),
                                evidence_smooth_window=int(evidence_smooth_window),
                                island_min_gap=int(island_min_gap),
                                island_min_length=int(island_min_length),
                                pad=int(pad),
                                parent_min_length=int(parent_min_length),
                            )
                            split_count = _count_valley_split_segments(split)
                            if split_count <= 0:
                                continue
                            split_params, metrics = select_postprocess_params(
                                split,
                                thresholds=thresholds,
                                smooth_windows=smooth_windows,
                                min_gaps=min_gaps,
                                min_lengths=min_lengths,
                                low_thresholds=low_thresholds,
                                iou_threshold=iou_threshold,
                            )
                            key = (
                                float(metrics["segment"]["f1"]),
                                float(metrics["segment"]["precision"]),
                                float(metrics["segment"]["recall"]),
                                float(metrics["frame"]["f1"]),
                                -float(split_count),
                            )
                            if key > best_key:
                                best_key = key
                                best_records = split
                                best_metrics = metrics
                                best_config = {
                                    "enabled": True,
                                    "aux_channels": channel_list,
                                    "evidence_threshold": float(evidence_threshold),
                                    "evidence_smooth_window": int(evidence_smooth_window),
                                    "island_min_gap": int(island_min_gap),
                                    "island_min_length": int(island_min_length),
                                    "pad": int(pad),
                                    "parent_min_length": int(parent_min_length),
                                    "split_segments": split_count,
                                    "params": split_params,
                                    "base_params": dict(base_params),
                                    "base_metrics": base_metrics,
                                }
    return best_config, best_records, best_metrics


def _load_feature_names(data_dir: str | Path) -> list[str]:
    summary_path = Path(data_dir) / "summary.json"
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    names = data.get("feature_names", [])
    return [str(name) for name in names] if isinstance(names, list) else []


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_name_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _feature_indices(feature_names: list[str], selected_names: Iterable[str], context: str) -> list[int]:
    indices: list[int] = []
    for name in selected_names:
        if name not in feature_names:
            raise ValueError(f"{context} feature {name!r} not found in summary.json")
        indices.append(feature_names.index(name))
    return indices


def train_model(
    records: list[VideoRecord],
    train_idx: list[int],
    val_idx: list[int],
    epochs: int,
    lr: float,
    hidden: int,
    dropout: float,
    patience: int,
    device: str,
    seed: int,
    lambda_dice: float = 0.0,
    lambda_boundary: float = 0.0,
    architecture: str = "tcn",
    frame_weights_by_idx: dict[int, np.ndarray] | None = None,
) -> tuple[TemporalSegmentLocator, dict, np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mean, std = compute_normalizer(records, train_idx)
    in_channels = int(records[0].signals.shape[1])
    model = TemporalSegmentLocator(
        in_channels=in_channels,
        hidden=hidden,
        dropout=dropout,
        architecture=architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = compute_pos_weight(records, train_idx)

    best_score = -math.inf
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metrics: dict = {}
    no_improve = 0

    for epoch in range(int(epochs)):
        model.train()
        order = train_idx[:]
        random.shuffle(order)
        losses: list[float] = []
        for idx in order:
            record = records[idx]
            x = torch.from_numpy(normalized_signals(record, mean, std)).unsqueeze(0).to(device)
            y = torch.from_numpy(record.labels).unsqueeze(0).to(device)
            frame_weights = None
            if frame_weights_by_idx is not None and int(idx) in frame_weights_by_idx:
                weights_np = np.asarray(frame_weights_by_idx[int(idx)], dtype=np.float32).reshape(-1)
                n = min(len(weights_np), len(record.labels))
                weights_np = weights_np[:n]
                if n < len(record.labels):
                    weights_np = np.pad(weights_np, (0, len(record.labels) - n), mode="edge")
                frame_weights = torch.from_numpy(weights_np).unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = locator_loss(
                logits,
                y,
                pos_weight=pos_weight,
                frame_weights=frame_weights,
                lambda_dice=lambda_dice,
                lambda_boundary=lambda_boundary,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        eval_idx = val_idx or train_idx
        val_records = predict_records(model, records, eval_idx, mean, std, device)
        params, metrics = select_postprocess_params(val_records)
        score = float(metrics["segment"]["f1"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {"epoch": epoch, "loss": float(np.mean(losses) if losses else 0.0), "params": params, **metrics}
            no_improve = 0
        else:
            no_improve += 1

        if epoch < 5 or epoch % 10 == 0 or no_improve >= patience:
            print(
                f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
                f"val_seg_f1={metrics['segment']['f1']:.3f} best={best_score:.3f}",
                flush=True,
            )
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_metrics, mean, std


def save_checkpoint(
    path: str | Path,
    model: TemporalSegmentLocator,
    mean: np.ndarray,
    std: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": {
            "in_channels": int(mean.shape[0]),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "architecture": str(args.architecture),
        },
        "normalizer": {"mean": mean.tolist(), "std": std.tolist()},
        "train_idx": train_idx,
        "val_idx": val_idx,
        "metrics": metrics,
        "signal_columns": ["true_vjepa_raw", "true_ijepa_raw", "dual_jepa_composite"],
        "task": "binary_error_segment_localization",
    }
    torch.save(checkpoint, path)


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if dataclass_isinstance(obj):
        return asdict(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def dataclass_isinstance(obj) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="/home/zzy/jepa_data/segment_locator_true_jepa.pt")
    parser.add_argument("--summary", default="")
    parser.add_argument("--test-data-dir", default="")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lambda-dice", type=float, default=0.0)
    parser.add_argument("--lambda-boundary", type=float, default=0.0)
    parser.add_argument("--architecture", choices=["tcn", "attn_tcn", "bilstm"], default="tcn")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--blend-aux-name",
        default="",
        help="Optional event feature name to blend with model probabilities during postprocess selection.",
    )
    parser.add_argument("--blend-alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--event-refine", action="store_true", help="Select multi-scale Gaussian event-completeness refinement.")
    parser.add_argument("--recall-repair", action="store_true", help="Select complementary JEPA proposal recall repair.")
    parser.add_argument(
        "--recall-repair-aux-names",
        default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank",
        help="Comma-separated event feature names used as complementary JEPA proposal sources.",
    )
    parser.add_argument("--recall-repair-thresholds", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--recall-repair-smooth-windows", default="1,3")
    parser.add_argument("--recall-repair-min-gaps", default="0,1")
    parser.add_argument("--recall-repair-min-lengths", default="1,2,4")
    parser.add_argument("--recall-repair-pads", default="0,1,2")
    parser.add_argument("--recall-repair-min-contrasts", default="0,0.1,0.2")
    parser.add_argument("--recall-repair-coverages", default="0.25,0.5")
    parser.add_argument("--recall-repair-max-per-video", default="1,2,3")
    parser.add_argument("--valley-split", action="store_true", help="Split merged long predictions by JEPA evidence valleys.")
    parser.add_argument(
        "--valley-split-aux-names",
        default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank",
        help="Comma-separated event feature names used as JEPA evidence for valley splitting.",
    )
    parser.add_argument("--valley-split-thresholds", default="0.45,0.55,0.65")
    parser.add_argument("--valley-split-smooth-windows", default="1,3")
    parser.add_argument("--valley-split-island-min-gaps", default="0,1")
    parser.add_argument("--valley-split-island-min-lengths", default="1,2,4")
    parser.add_argument("--valley-split-pads", default="0,1")
    parser.add_argument("--valley-split-parent-min-lengths", default="24,48")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    train_idx, val_idx = train_val_split(records, args.val_ratio, args.seed)
    positives = int(sum(record.labels.sum() for record in records))
    frames = int(sum(len(record.labels) for record in records))
    print(
        f"dataset videos={len(records)} frames={frames} positives={positives} "
        f"features={records[0].signals.shape[1]} train={len(train_idx)} val={len(val_idx)} device={args.device}",
        flush=True,
    )

    model, metrics, mean, std = train_model(
        records,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        lambda_dice=args.lambda_dice,
        lambda_boundary=args.lambda_boundary,
        architecture=args.architecture,
    )

    pure_metrics = metrics
    blend_aux_channel: int | None = None
    blend_alpha: float | None = None
    if args.blend_aux_name:
        if args.blend_aux_name not in feature_names:
            raise ValueError(f"blend feature {args.blend_aux_name!r} not found in {Path(args.data_dir) / 'summary.json'}")
        blend_aux_channel = feature_names.index(args.blend_aux_name)
        val_outputs = predict_records(model, records, val_idx or train_idx, mean, std, args.device)
        blend_alpha, blend_params, blend_metrics = select_blended_postprocess_params(
            val_outputs,
            records,
            val_idx or train_idx,
            aux_channel=blend_aux_channel,
            alphas=_parse_float_list(args.blend_alphas),
        )
        metrics = {
            "epoch": pure_metrics.get("epoch"),
            "loss": pure_metrics.get("loss"),
            "params": blend_params,
            **blend_metrics,
            "hybrid": {
                "aux_name": args.blend_aux_name,
                "aux_channel": blend_aux_channel,
                "alpha_model": blend_alpha,
                "pure_validation": pure_metrics,
            },
        }
    if args.event_refine:
        val_outputs = predict_records(model, records, val_idx or train_idx, mean, std, args.device)
        if blend_aux_channel is not None and blend_alpha is not None:
            val_outputs = blend_prediction_records(
                val_outputs,
                records,
                val_idx or train_idx,
                aux_channel=blend_aux_channel,
                alpha_model=blend_alpha,
            )
        refine_config, refine_params, refine_metrics = select_event_refined_postprocess_params(val_outputs)
        metrics = {
            "epoch": metrics.get("epoch"),
            "loss": metrics.get("loss"),
            "params": refine_params,
            **refine_metrics,
            "event_refine": {
                **refine_config,
                "pre_refine_validation": metrics,
            },
        }
    if args.valley_split:
        val_outputs = predict_records(model, records, val_idx or train_idx, mean, std, args.device)
        if blend_aux_channel is not None and blend_alpha is not None:
            val_outputs = blend_prediction_records(
                val_outputs,
                records,
                val_idx or train_idx,
                aux_channel=blend_aux_channel,
                alpha_model=blend_alpha,
            )
        if args.event_refine:
            refine_config = metrics.get("event_refine", {})
            val_outputs = refine_prediction_records(
                val_outputs,
                raw_weight=float(refine_config.get("raw_weight", 1.0)),
                sigmas=refine_config.get("sigmas", []),
            )
        split_names = _parse_name_list(args.valley_split_aux_names)
        split_channels = _feature_indices(feature_names, split_names, "valley-split")
        split_config, _, split_metrics = select_valley_split_params(
            val_outputs,
            records,
            val_idx or train_idx,
            base_params=metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
            aux_channels=split_channels,
            evidence_thresholds=_parse_float_list(args.valley_split_thresholds),
            evidence_smooth_windows=_parse_int_list(args.valley_split_smooth_windows),
            island_min_gaps=_parse_int_list(args.valley_split_island_min_gaps),
            island_min_lengths=_parse_int_list(args.valley_split_island_min_lengths),
            pads=_parse_int_list(args.valley_split_pads),
            parent_min_lengths=_parse_int_list(args.valley_split_parent_min_lengths),
        )
        metrics = {
            "epoch": metrics.get("epoch"),
            "loss": metrics.get("loss"),
            "params": split_config.get("params", metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1})),
            **split_metrics,
            "valley_split": {
                **split_config,
                "aux_names": split_names,
                "pre_split_validation": metrics,
            },
        }
    if args.recall_repair:
        val_outputs = predict_records(model, records, val_idx or train_idx, mean, std, args.device)
        if blend_aux_channel is not None and blend_alpha is not None:
            val_outputs = blend_prediction_records(
                val_outputs,
                records,
                val_idx or train_idx,
                aux_channel=blend_aux_channel,
                alpha_model=blend_alpha,
            )
        if args.event_refine:
            refine_config = metrics.get("event_refine", {})
            val_outputs = refine_prediction_records(
                val_outputs,
                raw_weight=float(refine_config.get("raw_weight", 1.0)),
                sigmas=refine_config.get("sigmas", []),
            )
        if args.valley_split:
            split_config = metrics.get("valley_split", {})
            if split_config.get("enabled"):
                split_names = split_config.get("aux_names", _parse_name_list(args.valley_split_aux_names))
                split_channels = _feature_indices(feature_names, split_names, "valley-split")
                val_outputs = apply_valley_split_to_records(
                    val_outputs,
                    records,
                    val_idx or train_idx,
                    base_params=split_config.get("base_params", metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1})),
                    aux_channels=split_channels,
                    evidence_threshold=float(split_config.get("evidence_threshold", 0.7)),
                    evidence_smooth_window=int(split_config.get("evidence_smooth_window", 1)),
                    island_min_gap=int(split_config.get("island_min_gap", 0)),
                    island_min_length=int(split_config.get("island_min_length", 1)),
                    pad=int(split_config.get("pad", 0)),
                    parent_min_length=int(split_config.get("parent_min_length", 24)),
                )
        repair_names = _parse_name_list(args.recall_repair_aux_names)
        repair_channels = _feature_indices(feature_names, repair_names, "recall-repair")
        repair_config, _, repair_metrics = select_recall_repair_params(
            val_outputs,
            records,
            val_idx or train_idx,
            base_params=metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
            aux_channels=repair_channels,
            aux_thresholds=_parse_float_list(args.recall_repair_thresholds),
            aux_smooth_windows=_parse_int_list(args.recall_repair_smooth_windows),
            aux_min_gaps=_parse_int_list(args.recall_repair_min_gaps),
            aux_min_lengths=_parse_int_list(args.recall_repair_min_lengths),
            pads=_parse_int_list(args.recall_repair_pads),
            min_contrasts=_parse_float_list(args.recall_repair_min_contrasts),
            max_existing_coverages=_parse_float_list(args.recall_repair_coverages),
            max_rescues_per_videos=_parse_int_list(args.recall_repair_max_per_video),
        )
        metrics = {
            "epoch": metrics.get("epoch"),
            "loss": metrics.get("loss"),
            "params": metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}),
            **repair_metrics,
            "recall_repair": {
                **repair_config,
                "aux_names": repair_names,
                "pre_repair_validation": metrics,
            },
        }
    save_checkpoint(args.output, model, mean, std, train_idx, val_idx, metrics, args)

    summary = {
        "data_dir": args.data_dir,
        "output": args.output,
        "n_videos": len(records),
        "frames": frames,
        "positive_frames": positives,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_names": [records[idx].name for idx in train_idx],
        "val_names": [records[idx].name for idx in val_idx],
        "validation": metrics,
        "config": {
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "lambda_dice": args.lambda_dice,
            "lambda_boundary": args.lambda_boundary,
            "architecture": args.architecture,
        },
    }
    if args.blend_aux_name:
        summary["pure_validation"] = pure_metrics

    if args.test_data_dir:
        test_records = load_signal_dataset(args.test_data_dir)
        checkpoint_params = metrics.get("params", {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1})
        test_outputs = predict_records(model, test_records, range(len(test_records)), mean, std, args.device)
        if blend_aux_channel is not None and blend_alpha is not None:
            test_outputs = blend_prediction_records(
                test_outputs,
                test_records,
                range(len(test_records)),
                aux_channel=blend_aux_channel,
                alpha_model=blend_alpha,
            )
        if args.event_refine:
            refine_config = metrics.get("event_refine", {})
            test_outputs = refine_prediction_records(
                test_outputs,
                raw_weight=float(refine_config.get("raw_weight", 1.0)),
                sigmas=refine_config.get("sigmas", []),
            )
        if args.valley_split:
            split_config = metrics.get("valley_split", {})
            if split_config.get("enabled"):
                split_names = split_config.get("aux_names", _parse_name_list(args.valley_split_aux_names))
                split_channels = _feature_indices(feature_names, split_names, "valley-split")
                test_outputs = apply_valley_split_to_records(
                    test_outputs,
                    test_records,
                    range(len(test_records)),
                    base_params=split_config.get("base_params", checkpoint_params),
                    aux_channels=split_channels,
                    evidence_threshold=float(split_config.get("evidence_threshold", 0.7)),
                    evidence_smooth_window=int(split_config.get("evidence_smooth_window", 1)),
                    island_min_gap=int(split_config.get("island_min_gap", 0)),
                    island_min_length=int(split_config.get("island_min_length", 1)),
                    pad=int(split_config.get("pad", 0)),
                    parent_min_length=int(split_config.get("parent_min_length", 24)),
                )
        if args.recall_repair:
            repair_config = metrics.get("recall_repair", {})
            if repair_config.get("enabled"):
                repair_names = repair_config.get("aux_names", _parse_name_list(args.recall_repair_aux_names))
                repair_channels = _feature_indices(feature_names, repair_names, "recall-repair")
                test_outputs = apply_recall_repair_to_records(
                    test_outputs,
                    test_records,
                    range(len(test_records)),
                    base_params=checkpoint_params,
                    aux_channels=repair_channels,
                    aux_threshold=float(repair_config.get("aux_threshold", 0.85)),
                    aux_smooth_window=int(repair_config.get("aux_smooth_window", 1)),
                    aux_min_gap=int(repair_config.get("aux_min_gap", 0)),
                    aux_min_length=int(repair_config.get("aux_min_length", 1)),
                    pad=int(repair_config.get("pad", 0)),
                    min_contrast=float(repair_config.get("min_contrast", 0.0)),
                    max_existing_coverage=float(repair_config.get("max_existing_coverage", 0.25)),
                    max_rescues_per_video=int(repair_config.get("max_rescues_per_video", 3)),
                )
        summary["test"] = evaluate_records(test_outputs, checkpoint_params)

    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
    print(json.dumps(summary["validation"], indent=2, default=_jsonable), flush=True)
    print(f"saved {args.output}", flush=True)
    print(f"summary {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
