#!/usr/bin/env python3
"""Diagnose clean-anchor frame-teacher scores on a target JEPA event dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_clean_anchor_rectification import CleanAnchorRectifyConfig, _fit_anchor_teacher, _predict_teacher_probability
from train_segment_locator import _load_feature_names, contiguous_segments, load_signal_dataset


def _parse_names(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return {"n": 0}
    qs = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        **{f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in qs},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-data-dir", default="/home/zzy/jepa_data/clean79_event_jepa_v2")
    parser.add_argument("--target-data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--teacher-model", choices=["prototype", "gbdt", "rf", "extratrees", "logreg"], default="extratrees")
    parser.add_argument(
        "--evidence-names",
        default=(
            "true_vjepa_raw,true_ijepa_dense_raw,dual_jepa_composite,"
            "true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank,"
            "vi_min,vi_max,vi_agreement,vi_abs_gap,vi_product,composite_times_vi_agreement,"
            "true_vjepa_raw_abs_delta,true_ijepa_dense_raw_abs_delta,dual_jepa_composite_abs_delta"
        ),
    )
    parser.add_argument("--output", default="/home/zzy/jepa_data/clean_anchor_teacher_score_diagnostics.json")
    args = parser.parse_args()

    evidence_names = _parse_names(args.evidence_names)
    clean_feature_names = _load_feature_names(args.clean_data_dir)
    anchor = _fit_anchor_teacher(args.clean_data_dir, evidence_names, exclude_video_names=[], teacher_model=args.teacher_model)
    target_feature_names = _load_feature_names(args.target_data_dir)
    target_channels = [target_feature_names.index(name) for name in evidence_names if name in target_feature_names]
    if not target_channels:
        raise ValueError("no evidence channels found in target data")
    target_records = load_signal_dataset(args.target_data_dir)

    pos_scores = []
    neg_scores = []
    video_rows = []
    for record in target_records:
        teacher = _predict_teacher_probability(record.signals, target_channels, anchor)
        labels = (record.labels > 0).astype(np.int64)
        n = min(len(teacher), len(labels))
        teacher = teacher[:n]
        labels = labels[:n]
        pos = teacher[labels > 0]
        neg = teacher[labels <= 0]
        pos_scores.append(pos)
        neg_scores.append(neg)
        low_pos = int(np.sum((labels > 0) & (teacher <= 0.3)))
        very_low_pos = int(np.sum((labels > 0) & (teacher <= 0.15)))
        high_neg = int(np.sum((labels <= 0) & (teacher >= 0.7)))
        very_high_neg = int(np.sum((labels <= 0) & (teacher >= 0.85)))
        if low_pos or high_neg:
            video_rows.append(
                {
                    "video": record.name,
                    "frames": int(n),
                    "positive_frames": int(labels.sum()),
                    "low_positive_frames_le_0.30": low_pos,
                    "very_low_positive_frames_le_0.15": very_low_pos,
                    "high_negative_frames_ge_0.70": high_neg,
                    "very_high_negative_frames_ge_0.85": very_high_neg,
                    "teacher_mean": float(teacher.mean()) if n else 0.0,
                }
            )
    pos_all = np.concatenate([item for item in pos_scores if len(item)], axis=0)
    neg_all = np.concatenate([item for item in neg_scores if len(item)], axis=0)
    thresholds = {}
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        thresholds[f"pos_le_{threshold:.2f}"] = int(np.sum(pos_all <= threshold))
        thresholds[f"neg_ge_{threshold:.2f}"] = int(np.sum(neg_all >= threshold))
    result = {
        "clean_data_dir": args.clean_data_dir,
        "target_data_dir": args.target_data_dir,
        "teacher_model": anchor.get("teacher_model"),
        "evidence_names": evidence_names,
        "clean_feature_count": int(len(clean_feature_names)),
        "target_feature_count": int(len(target_feature_names)),
        "positive_scores": _quantiles(pos_all),
        "negative_scores": _quantiles(neg_all),
        "threshold_counts": thresholds,
        "videos_with_potential_changes": sorted(
            video_rows,
            key=lambda row: (
                row["low_positive_frames_le_0.30"] + row["high_negative_frames_ge_0.70"],
                row["positive_frames"],
            ),
            reverse=True,
        )[:50],
        "target_segments": int(sum(len(contiguous_segments(record.labels)) for record in target_records)),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in result.items() if key != "videos_with_potential_changes"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
