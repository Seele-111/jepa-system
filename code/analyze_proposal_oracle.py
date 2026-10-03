#!/usr/bin/env python3
"""Oracle coverage for multi-channel JEPA temporal proposals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train_proposal_calibrator import _segments_from_scores, evaluate_segment_predictions
from train_segment_locator import contiguous_segments, load_signal_dataset, segment_iou


DATA_DIR = Path("/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
MAIN_SUMMARY = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.summary.json")
ERROR_ANALYSIS = Path("/home/zzy/jepa_data/segment_train_full_merge_error_analysis.json")
OUT_PATH = Path("/home/zzy/jepa_data/segment_train_full_proposal_oracle.json")


CHANNEL_NAMES = [
    "true_vjepa_raw_rank",
    "true_ijepa_dense_raw_rank",
    "dual_jepa_composite_rank",
    "true_vjepa_raw",
    "true_ijepa_dense_raw",
    "dual_jepa_composite",
    "vi_max",
    "vi_product",
    "composite_times_vi_agreement",
    "true_vjepa_raw_abs_delta",
    "true_ijepa_dense_raw_abs_delta",
    "dual_jepa_composite_abs_delta",
]


def _thresholds_for(values: np.ndarray, name: str) -> list[float]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if "rank" in name or name in {"vi_max", "vi_product", "composite_times_vi_agreement"}:
        return [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    percentiles = np.percentile(values, [50, 60, 70, 80, 90, 95])
    return sorted({float(x) for x in percentiles if np.isfinite(x)})


def generate_multichannel_candidates(signals: np.ndarray, feature_names: list[str]) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for name in CHANNEL_NAMES:
        if name not in feature_names:
            continue
        values = np.asarray(signals[:, feature_names.index(name)], dtype=np.float32)
        for threshold in _thresholds_for(values, name):
            for min_gap in [0, 1, 2, 4, 8]:
                for min_length in [1, 2, 4, 8]:
                    candidates.update(_segments_from_scores(values, threshold, min_gap=min_gap, min_length=min_length))
    return sorted(candidates)


def oracle_predictions(candidates: list[tuple[int, int]], labels: np.ndarray, iou_threshold: float) -> list[tuple[int, int]]:
    gt_segments = contiguous_segments(labels)
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for gt in gt_segments:
        best_idx = -1
        best_iou = 0.0
        for idx, candidate in enumerate(candidates):
            if idx in used:
                continue
            iou = segment_iou(candidate, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            used.add(best_idx)
            selected.append(candidates[best_idx])
    return sorted(selected)


def main() -> int:
    records = load_signal_dataset(DATA_DIR)
    dataset_summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    feature_names = [str(name) for name in dataset_summary["feature_names"]]
    main_summary = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    val_idx = [int(idx) for idx in main_summary["val_idx"]]
    missed = json.loads(ERROR_ANALYSIS.read_text(encoding="utf-8")).get("details", [])
    missed_keys = {(item["video"], tuple(item["gt"])) for item in missed}

    all_candidates = {}
    oracle_preds = []
    labels = []
    detail_rows = []
    for idx in val_idx:
        record = records[idx]
        candidates = generate_multichannel_candidates(record.signals, feature_names)
        all_candidates[record.name] = candidates
        labels.append(record.labels)
        oracle_preds.append(oracle_predictions(candidates, record.labels, iou_threshold=0.3))
        for gt in contiguous_segments(record.labels):
            best_iou = max((segment_iou(candidate, gt) for candidate in candidates), default=0.0)
            row = {
                "video": record.name,
                "gt": [int(gt[0]), int(gt[1])],
                "length": int(gt[1] - gt[0] + 1),
                "best_candidate_iou": float(best_iou),
                "n_candidates": len(candidates),
                "was_mainline_missed": (record.name, tuple(gt)) in missed_keys,
            }
            detail_rows.append(row)

    oracle_metrics = evaluate_segment_predictions(oracle_preds, labels, iou_threshold=0.3)
    gt_count = len(detail_rows)
    covered = sum(1 for row in detail_rows if row["best_candidate_iou"] >= 0.3)
    missed_rows = [row for row in detail_rows if row["was_mainline_missed"]]
    missed_covered = sum(1 for row in missed_rows if row["best_candidate_iou"] >= 0.3)
    result = {
        "channels": [name for name in CHANNEL_NAMES if name in feature_names],
        "val_videos": len(val_idx),
        "gt_segments": gt_count,
        "oracle_covered": covered,
        "oracle_coverage": covered / max(1, gt_count),
        "mainline_missed_segments": len(missed_rows),
        "mainline_missed_oracle_covered": missed_covered,
        "mainline_missed_oracle_coverage": missed_covered / max(1, len(missed_rows)),
        "oracle_segment_metrics": oracle_metrics,
        "candidate_count_mean": float(np.mean([len(items) for items in all_candidates.values()])),
        "candidate_count_max": int(max((len(items) for items in all_candidates.values()), default=0)),
        "details": detail_rows,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "oracle_coverage": result["oracle_coverage"],
                "missed_oracle_coverage": result["mainline_missed_oracle_coverage"],
                "oracle_segment_metrics": oracle_metrics,
                "candidate_count_mean": result["candidate_count_mean"],
                "out": str(OUT_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
