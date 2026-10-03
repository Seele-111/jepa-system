#!/usr/bin/env python3
"""Merge JEPA event-token datasets along the video axis.

This is a protocol utility for clean-anchor expansion. It preserves per-video
arrays exactly and rejects schema drift instead of silently coercing fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _npz_arrays(path: Path) -> list[np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return [np.asarray(archive[key]) for key in archive.files]


def _load_summary(root: Path) -> dict:
    path = root / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_names(root: Path) -> list[str]:
    path = root / "video_names.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return [str(name) for name in json.loads(path.read_text(encoding="utf-8"))]


def merge_event_datasets(input_dirs: list[str | Path], output_dir: str | Path) -> dict:
    if not input_dirs:
        raise ValueError("input_dirs must not be empty")

    out = Path(output_dir)
    all_signals: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_names: list[str] = []
    sources: list[str] = []
    reference_feature_names: list[str] | None = None
    seen_names: set[str] = set()

    for input_dir in input_dirs:
        root = Path(input_dir)
        sources.append(str(root))
        summary = _load_summary(root)
        feature_names = [str(name) for name in summary.get("feature_names", [])]
        if not feature_names:
            raise ValueError(f"missing feature_names in {root / 'summary.json'}")
        if reference_feature_names is None:
            reference_feature_names = feature_names
        elif feature_names != reference_feature_names:
            raise ValueError(f"feature_names mismatch for {root}")

        signals = _npz_arrays(root / "signals.npz")
        labels = [(np.asarray(item).reshape(-1) > 0).astype(np.int64) for item in _npz_arrays(root / "labels.npz")]
        names = _load_names(root)
        if len(signals) != len(labels) or len(signals) != len(names):
            raise ValueError(f"record count mismatch in {root}: signals={len(signals)} labels={len(labels)} names={len(names)}")

        for idx, (name, signal, label) in enumerate(zip(names, signals, labels)):
            if name in seen_names:
                raise ValueError(f"duplicate video name: {name}")
            seen_names.add(name)
            signal = np.asarray(signal, dtype=np.float32)
            if signal.ndim != 2:
                raise ValueError(f"{root} record {idx} signal must be [T, D], got {signal.shape}")
            if signal.shape[1] != len(reference_feature_names):
                raise ValueError(
                    f"{root} record {idx} feature_dim mismatch: {signal.shape[1]} vs {len(reference_feature_names)}"
                )
            if len(signal) != len(label):
                raise ValueError(f"{root} record {idx} length mismatch: signals={len(signal)} labels={len(label)}")
            all_names.append(name)
            all_signals.append(np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            all_labels.append(label.astype(np.int64))

    assert reference_feature_names is not None
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "signals.npz", *all_signals)
    np.savez_compressed(out / "labels.npz", *all_labels)
    (out / "video_names.json").write_text(json.dumps(all_names, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "sources": sources,
        "n_videos": int(len(all_names)),
        "frames": int(sum(len(label) for label in all_labels)),
        "positive_frames": int(sum(label.sum() for label in all_labels)),
        "feature_dim": int(len(reference_feature_names)),
        "feature_names": reference_feature_names,
        "task": "binary_error_segment_localization",
        "protocol_note": "Merged clean-anchor dataset. Use for calibration/diagnostics until protocol is frozen.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = merge_event_datasets(args.inputs, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
