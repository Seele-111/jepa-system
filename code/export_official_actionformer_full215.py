#!/usr/bin/env python3
"""Export Full215 features and strict folds for the official ActionFormer code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from train_segment_locator import contiguous_segments, load_signal_dataset


def _safe_video_id(index: int) -> str:
    return f"full215_{index:04d}"


def _next_power_of_two(value: int) -> int:
    return 1 << max(6, int(value - 1).bit_length())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_actionformer_dataset(
    data_dir: Path,
    split_summary: Path,
    output_dir: Path,
    normalize_fit_per_fold: bool = False,
) -> dict:
    records = load_signal_dataset(data_dir)
    summary = json.loads(split_summary.read_text(encoding="utf-8"))
    folds = summary.get("folds", [])
    if not folds:
        raise ValueError("split summary has no folds")

    by_name = {record.name: record for record in records}
    if len(by_name) != len(records):
        raise ValueError("video names must be unique")
    expected_names = set(by_name)
    feature_dim = int(records[0].signals.shape[1])
    if any(record.signals.shape[1] != feature_dim for record in records):
        raise ValueError("all records must have the same feature dimension")

    features_dir = output_dir / "features"
    annotations_dir = output_dir / "annotations"
    configs_dir = output_dir / "configs"
    features_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    name_to_id = {}
    id_to_record = {}
    for index, record in enumerate(records):
        video_id = _safe_video_id(index)
        name_to_id[record.name] = video_id
        id_to_record[video_id] = record
        if not normalize_fit_per_fold:
            np.save(features_dir / f"{video_id}.npy", np.asarray(record.signals, dtype=np.float32), allow_pickle=False)

    max_seq_len = _next_power_of_two(max(len(record.labels) for record in records))
    fold_manifests = []
    for fold_index, fold in enumerate(folds):
        fit_names = list(fold.get("fit_names", []))
        calibration_names = list(fold.get("calibration_names", []))
        outer_names = list(fold.get("val_names", fold.get("outer_names", [])))
        split_sets = {
            "training": set(fit_names),
            "calibration": set(calibration_names),
            "validation": set(outer_names),
        }
        if any(split_sets[left] & split_sets[right] for left, right in (("training", "calibration"), ("training", "validation"), ("calibration", "validation"))):
            raise ValueError(f"fold {fold_index} contains overlapping splits")
        observed_names = set().union(*split_sets.values())
        if observed_names != expected_names:
            missing = sorted(expected_names - observed_names)
            extra = sorted(observed_names - expected_names)
            raise ValueError(f"fold {fold_index} does not partition the dataset; missing={missing}, extra={extra}")

        fold_features_dir = features_dir
        normalization = {"kind": "none"}
        if normalize_fit_per_fold:
            fold_features_dir = output_dir / f"features_fold_{fold_index}"
            fold_features_dir.mkdir(parents=True, exist_ok=True)
            fit_values = np.concatenate([by_name[name].signals for name in fit_names], axis=0).astype(np.float32)
            mean = fit_values.mean(axis=0).astype(np.float32)
            std = np.maximum(fit_values.std(axis=0), 1e-4).astype(np.float32)
            for video_id, record in id_to_record.items():
                normalized = ((record.signals - mean) / std).astype(np.float32)
                np.save(fold_features_dir / f"{video_id}.npy", normalized, allow_pickle=False)
            normalization = {
                "kind": "fit-only channel z-score",
                "fit_frames": int(len(fit_values)),
                "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
                "std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
            }

        database = {}
        event_count = 0
        for subset, names in split_sets.items():
            for name in sorted(names):
                video_id = name_to_id[name]
                record = id_to_record[video_id]
                annotations = []
                for start, end in contiguous_segments(record.labels):
                    annotations.append(
                        {
                            "segment": [float(start), float(end + 1)],
                            "label": "error",
                            "label_id": 0,
                        }
                    )
                event_count += len(annotations)
                database[video_id] = {
                    "subset": subset,
                    "duration": float(len(record.labels)),
                    "fps": 1.0,
                    "annotations": annotations,
                }

        annotation_path = annotations_dir / f"fold_{fold_index}.json"
        annotation_path.write_text(json.dumps({"database": database}, indent=2), encoding="utf-8")
        common_config = {
            "init_rand_seed": 42 + fold_index,
            "dataset_name": "anet",
            "devices": [0],
            "train_split": ["training"],
            "dataset": {
                "json_file": str(annotation_path.resolve()),
                "feat_folder": str(fold_features_dir.resolve()),
                "file_prefix": None,
                "file_ext": ".npy",
                "num_classes": 1,
                "input_dim": feature_dim,
                "feat_stride": 1,
                "num_frames": 1,
                "default_fps": 1,
                "downsample_rate": 1,
                "max_seq_len": max_seq_len,
                "trunc_thresh": 0.5,
                "crop_ratio": [0.9, 1.0],
                "force_upsampling": False,
            },
            # Official make_data_loader enables persistent_workers unconditionally.
            "loader": {"batch_size": 8, "num_workers": 1},
            "model": {"fpn_type": "identity", "n_head": 4},
            "opt": {"learning_rate": 0.001, "epochs": 10, "weight_decay": 0.05},
            "train_cfg": {
                "init_loss_norm": 200,
                "clip_grad_l2norm": 1.0,
                "cls_prior_prob": 0.01,
                "center_sample": "radius",
                "center_sample_radius": 1.5,
                "label_smoothing": 0.1,
                "droppath": 0.1,
                "loss_weight": 2.0,
            },
            "test_cfg": {
                "pre_nms_topk": 2000,
                "max_seg_num": 100,
                "min_score": 0.001,
                "multiclass_nms": False,
                "nms_sigma": 0.75,
                "ext_score_file": None,
                "duration_thresh": 0.001,
            },
            "output_folder": str((output_dir / "checkpoints").resolve()),
        }
        config_paths = {}
        for role in ("calibration", "validation"):
            config = {**common_config, "val_split": [role]}
            config_path = configs_dir / f"fold_{fold_index}_{role}.yaml"
            # JSON is valid YAML and avoids hand-built configuration text.
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            config_paths[role] = str(config_path)

        fold_manifests.append(
            {
                "fold": fold_index,
                "fit_videos": len(fit_names),
                "calibration_videos": len(calibration_names),
                "outer_videos": len(outer_names),
                "events": event_count,
                "annotation": str(annotation_path),
                "annotation_sha256": _sha256(annotation_path),
                "configs": config_paths,
                "normalization": normalization,
            }
        )

    manifest = {
        "schema_version": "official-actionformer-full215-export-v1",
        "source_data_dir": str(data_dir),
        "source_split_summary": str(split_summary),
        "videos": len(records),
        "feature_dim": feature_dim,
        "max_seq_len": max_seq_len,
        "timeline": "one feature per annotation frame; fps=feat_stride=num_frames=1",
        "segment_convention": "half-open [start, end+1] timestamps",
        "normalization": "fit-only channel z-score per fold" if normalize_fit_per_fold else "none",
        "name_to_id": name_to_id,
        "folds": fold_manifests,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--normalize-fit-per-fold", action="store_true")
    args = parser.parse_args()
    manifest = export_actionformer_dataset(
        args.data_dir,
        args.split_summary,
        args.output_dir,
        normalize_fit_per_fold=args.normalize_fit_per_fold,
    )
    print(json.dumps({"videos": manifest["videos"], "folds": len(manifest["folds"]), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
