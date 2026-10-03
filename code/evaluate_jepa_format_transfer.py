#!/usr/bin/env python3
"""Bidirectional held-format transfer diagnostic for JEPA signal TCNs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cross_validate_jepa_ablation import _segments_from_params, subset_records, train_ablation_model
from train_segment_locator import (
    evaluate_records,
    load_signal_dataset,
    predict_records,
    select_postprocess_params,
    train_val_split,
)


def load_format_groups(path: Path) -> dict[str, str]:
    card = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for row in card.get("videos", []):
        name = str(row["video_name"])
        fps = str(row.get("fps", "unknown"))
        resolution = str(row.get("resolution", "unknown"))
        output[name] = f"fps_{fps}_resolution_{resolution}"
    return output


def split_source_calibration(records, source_idx: list[int], ratio: float, seed: int) -> tuple[list[int], list[int]]:
    local_fit, local_cal = train_val_split([records[idx] for idx in source_idx], ratio, seed)
    fit_idx = [source_idx[idx] for idx in local_fit]
    cal_idx = [source_idx[idx] for idx in local_cal]
    if set(fit_idx) & set(cal_idx) or set(fit_idx) | set(cal_idx) != set(source_idx):
        raise RuntimeError("source fit/calibration split is incomplete or overlapping")
    return fit_idx, cal_idx


def _mode_channels(mode: str, feature_dim: int) -> list[int]:
    if mode == "vjepa":
        return [0]
    if mode == "full":
        return list(range(feature_dim))
    raise ValueError(f"unsupported mode: {mode}")


def run_direction(records, source_group, target_group, groups, mode, seeds, args):
    channels = _mode_channels(mode, records[0].signals.shape[1])
    view = subset_records(records, channels)
    source_idx = [idx for idx, record in enumerate(view) if groups[record.name] == source_group]
    target_idx = [idx for idx, record in enumerate(view) if groups[record.name] == target_group]
    if not source_idx or not target_idx:
        raise ValueError(f"empty source/target group: {source_group} -> {target_group}")
    runs = []
    for seed in seeds:
        fit_idx, cal_idx = split_source_calibration(view, source_idx, args.calibration_ratio, seed)
        model, mean, std, training = train_ablation_model(view, fit_idx, cal_idx, args, seed)
        calibration = predict_records(model, view, cal_idx, mean, std, args.device)
        params, calibration_metrics = select_postprocess_params(calibration, iou_threshold=0.3)
        target = predict_records(model, view, target_idx, mean, std, args.device)
        metrics_03 = evaluate_records(target, params, iou_threshold=0.3)
        metrics_05 = evaluate_records(target, params, iou_threshold=0.5)
        runs.append(
            {
                "seed": seed,
                "fit_names": [view[idx].name for idx in fit_idx],
                "calibration_names": [view[idx].name for idx in cal_idx],
                "target_names": [view[idx].name for idx in target_idx],
                "training": training,
                "postprocess_params_selected_at_iou_0.3": params,
                "calibration_iou_0.3": calibration_metrics,
                "target_iou_0.3": metrics_03,
                "target_iou_0.5": metrics_05,
                "target_predictions": [[list(segment) for segment in _segments_from_params(row["probs"], params)] for row in target],
                "target_labels": [row["labels"].astype(int).tolist() for row in target],
            }
        )
        print(
            f"mode={mode} source={source_group} target={target_group} seed={seed} "
            f"f1@0.3={metrics_03['segment']['f1']:.4f} f1@0.5={metrics_05['segment']['f1']:.4f}",
            flush=True,
        )

    aggregate = {}
    for threshold in ("0.3", "0.5"):
        key = f"target_iou_{threshold}"
        aggregate[threshold] = {}
        for section in ("frame", "segment"):
            aggregate[threshold][section] = {}
            for metric in ("precision", "recall", "f1"):
                values = np.asarray([run[key][section][metric] for run in runs], dtype=np.float64)
                aggregate[threshold][section][f"{metric}_mean"] = float(values.mean())
                aggregate[threshold][section][f"{metric}_std"] = float(values.std(ddof=0))
    return {
        "mode": mode,
        "channels": channels,
        "source_group": source_group,
        "target_group": target_group,
        "source_videos": len(source_idx),
        "target_videos": len(target_idx),
        "aggregate_across_training_seeds": aggregate,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-card", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", default="vjepa,full")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    groups = load_format_groups(args.dataset_card)
    missing = sorted(record.name for record in records if record.name not in groups)
    if missing:
        raise ValueError(f"dataset card is missing {len(missing)} records, first={missing[0]!r}")
    counts = Counter(groups[record.name] for record in records)
    if len(counts) != 2:
        raise ValueError(f"expected exactly two format groups, got {dict(counts)}")
    group_names = sorted(counts)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    train_args = SimpleNamespace(**vars(args))
    reports = []
    for mode in modes:
        for source_group, target_group in ((group_names[0], group_names[1]), (group_names[1], group_names[0])):
            reports.append(run_direction(records, source_group, target_group, groups, mode, seeds, train_args))

    output = {
        "schema_version": "round2-jepa-format-transfer-v1",
        "protocol": "train/calibrate on one FPS-resolution group and evaluate on the other",
        "interpretation_limit": (
            "FPS and resolution are perfectly confounded in Full215; generator identity is unknown. "
            "This is a held-format-combination diagnostic, not an isolated FPS, resolution, or generator test."
        ),
        "data_dir": args.data_dir,
        "dataset_card": str(args.dataset_card),
        "format_counts": dict(counts),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reports": len(reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
