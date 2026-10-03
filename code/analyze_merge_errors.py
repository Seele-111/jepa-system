#!/usr/bin/env python3
"""Diagnose whether missed validation segments come from merged predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train_segment_locator import (
    TemporalSegmentLocator,
    _segments_from_params,
    blend_prediction_records,
    contiguous_segments,
    load_signal_dataset,
    predict_records,
    segment_iou,
)


DATA_DIR = Path("/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
SUMMARY_PATH = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.summary.json")
CKPT_PATH = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.pt")
OUT_PATH = Path("/home/zzy/jepa_data/segment_train_full_merge_error_analysis.json")


def main() -> int:
    records = load_signal_dataset(DATA_DIR)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    data_summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    feature_names = data_summary["feature_names"]
    feature_idx = {
        "vjepa_rank": feature_names.index("true_vjepa_raw_rank"),
        "ijepa_rank": feature_names.index("true_ijepa_dense_raw_rank"),
        "dual_rank": feature_names.index("dual_jepa_composite_rank"),
    }

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model = TemporalSegmentLocator(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32)
    val_idx = [int(idx) for idx in summary["val_idx"]]
    outputs = predict_records(model, records, val_idx, mean, std, "cpu")
    hybrid = summary["validation"].get("hybrid", {})
    if hybrid:
        outputs = blend_prediction_records(
            outputs,
            records,
            val_idx,
            aux_channel=int(hybrid["aux_channel"]),
            alpha_model=float(hybrid["alpha_model"]),
        )

    params = summary["validation"]["params"]
    missed = []
    merged_misses = 0
    for output, record_idx in zip(outputs, val_idx):
        record = records[record_idx]
        preds = _segments_from_params(output["probs"], params)
        gts = contiguous_segments(record.labels)
        for gt in gts:
            best_pred = None
            best_iou = 0.0
            for pred in preds:
                iou = segment_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
            if best_iou >= 0.3:
                continue
            overlapping_gts = []
            if best_pred is not None:
                for other in gts:
                    if segment_iou(best_pred, other) > 0.0:
                        overlapping_gts.append(other)
            if len(overlapping_gts) > 1:
                merged_misses += 1
            start, end = gt
            row = {
                "video": record.name,
                "gt": [int(start), int(end)],
                "length": int(end - start + 1),
                "best_pred": list(best_pred) if best_pred is not None else None,
                "best_iou": float(best_iou),
                "pred_length": int(best_pred[1] - best_pred[0] + 1) if best_pred is not None else 0,
                "overlapping_gt_count": len(overlapping_gts),
                "overlapping_gts": [[int(s), int(e)] for s, e in overlapping_gts],
            }
            for name, idx in feature_idx.items():
                values = record.signals[start : end + 1, idx]
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_max"] = float(values.max())
            if best_pred is not None:
                ps, pe = best_pred
                for name, idx in feature_idx.items():
                    values = record.signals[ps : pe + 1, idx]
                    row[f"pred_{name}_min"] = float(values.min())
                    row[f"pred_{name}_p10"] = float(np.percentile(values, 10))
                    row[f"pred_{name}_p50"] = float(np.percentile(values, 50))
            missed.append(row)

    result = {
        "missed": len(missed),
        "merged_misses": merged_misses,
        "merged_miss_ratio": merged_misses / max(1, len(missed)),
        "details": missed,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"missed": len(missed), "merged_misses": merged_misses, "out": str(OUT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
