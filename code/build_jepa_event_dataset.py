#!/usr/bin/env python3
"""
Build multi-scale temporal event features from true V/I-JEPA scalar signals.

Input dataset is the three-channel output of extract_true_jepa_segment_signals:
  [true_vjepa_raw, true_ijepa_dense_raw, dual_jepa_composite]

Output keeps the same labels and video order but expands signals to [T, D].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_segment_locator import load_signal_dataset


BASE_NAMES = ["true_vjepa_raw", "true_ijepa_dense_raw", "dual_jepa_composite"]
DEFAULT_WINDOWS = [3, 5, 9, 15]


def event_feature_names(windows: list[int] | tuple[int, ...] = DEFAULT_WINDOWS) -> list[str]:
    names = BASE_NAMES[:]
    for channel in BASE_NAMES:
        names.extend([f"{channel}_delta", f"{channel}_abs_delta"])
    for window in windows:
        for channel in BASE_NAMES:
            names.extend(
                [
                    f"{channel}_mean_w{window}",
                    f"{channel}_max_w{window}",
                    f"{channel}_std_w{window}",
                    f"{channel}_peak_w{window}",
                ]
            )
    names.extend(
        [
            "vi_min",
            "vi_max",
            "vi_agreement",
            "vi_abs_gap",
            "composite_minus_vjepa",
            "composite_minus_ijepa",
        ]
    )
    for channel in BASE_NAMES:
        names.extend([f"{channel}_robust_z", f"{channel}_rank"])
    for channel in BASE_NAMES:
        names.extend([f"{channel}_second_delta", f"{channel}_abs_second_delta"])
    for channel in BASE_NAMES:
        names.extend([f"{channel}_long_contrast_w31", f"{channel}_abs_long_contrast_w31"])
    names.extend(["vi_product", "composite_times_vi_agreement", "composite_times_vi_gap"])
    return names


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.asarray([padded[i : i + window].max() for i in range(len(values))], dtype=np.float32)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.zeros_like(values, dtype=np.float32)
    mean = _rolling_mean(values, window)
    mean_sq = _rolling_mean(values * values, window)
    return np.sqrt(np.maximum(0.0, mean_sq - mean * mean)).astype(np.float32)


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if scale < 1e-6:
        scale = float(values.std())
    if scale < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - median) / scale, -6.0, 6.0).astype(np.float32)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = int(values.shape[0])
    if n <= 1:
        return np.full_like(values, 0.5, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float32)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = ((start + end - 1) / 2.0) / float(n - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks.astype(np.float32)


def make_event_features(signals: np.ndarray, windows: list[int] | tuple[int, ...] = DEFAULT_WINDOWS) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2 or signals.shape[1] < 3:
        raise ValueError(f"expected [T, C>=3], got {signals.shape}")
    base = np.nan_to_num(signals[:, :3], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    features = [base]

    deltas = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(base, axis=0)])
    delta_features = []
    for channel in range(3):
        delta_features.extend([deltas[:, channel : channel + 1], np.abs(deltas[:, channel : channel + 1])])
    features.append(np.concatenate(delta_features, axis=1).astype(np.float32))

    for window in windows:
        per_window = []
        for channel in range(3):
            values = base[:, channel]
            mean = _rolling_mean(values, int(window))
            max_values = _rolling_max(values, int(window))
            std = _rolling_std(values, int(window))
            peak = values - mean
            per_window.extend([mean[:, None], max_values[:, None], std[:, None], peak[:, None]])
        features.append(np.concatenate(per_window, axis=1))

    v = base[:, 0]
    i = base[:, 1]
    c = base[:, 2]
    vi_min = np.minimum(v, i)
    vi_max = np.maximum(v, i)
    agreement = vi_min / (vi_max + 1e-6)
    cross = np.stack(
        [
            vi_min,
            vi_max,
            agreement,
            np.abs(v - i),
            c - v,
            c - i,
        ],
        axis=1,
    ).astype(np.float32)
    features.append(cross)

    video_internal = []
    for channel in range(3):
        values = base[:, channel]
        video_internal.extend([_robust_z(values)[:, None], _percentile_rank(values)[:, None]])
    features.append(np.concatenate(video_internal, axis=1))

    second_deltas = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(deltas, axis=0)])
    second_delta_features = []
    for channel in range(3):
        second_delta_features.extend(
            [
                second_deltas[:, channel : channel + 1],
                np.abs(second_deltas[:, channel : channel + 1]),
            ]
        )
    features.append(np.concatenate(second_delta_features, axis=1).astype(np.float32))

    long_contrast = []
    for channel in range(3):
        values = base[:, channel]
        contrast = values - _rolling_mean(values, 31)
        long_contrast.extend([contrast[:, None], np.abs(contrast)[:, None]])
    features.append(np.concatenate(long_contrast, axis=1).astype(np.float32))

    vi_gap = np.abs(v - i)
    features.append(
        np.stack(
            [
                v * i,
                c * agreement,
                c * vi_gap,
            ],
            axis=1,
        ).astype(np.float32)
    )
    return np.nan_to_num(np.concatenate(features, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_event_dataset(input_dir: str | Path, output_dir: str | Path, windows: list[int]) -> dict:
    records = load_signal_dataset(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_signals = [make_event_features(record.signals, windows=windows) for record in records]
    labels = [record.labels for record in records]
    names = [record.name for record in records]
    feature_names = event_feature_names(windows)

    np.savez_compressed(out / "signals.npz", *event_signals)
    np.savez_compressed(out / "labels.npz", *labels)
    (out / "video_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    summary = {
        "source": str(input_dir),
        "n_videos": len(records),
        "frames": int(sum(len(record.labels) for record in records)),
        "positive_frames": int(sum(record.labels.sum() for record in records)),
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "task": "binary_error_segment_localization",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", default="3,5,9,15")
    args = parser.parse_args()
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    summary = build_event_dataset(args.input, args.output, windows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
