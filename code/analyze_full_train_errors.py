#!/usr/bin/env python3
"""Analyze missed validation segments for the full-train JEPA locator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train_segment_locator import (
    TemporalSegmentLocator,
    blend_prediction_records,
    contiguous_segments,
    hysteresis_probabilities_to_segments,
    load_signal_dataset,
    predict_records,
    probabilities_to_segments,
    segment_iou,
)


def _predicted_segments(probs: np.ndarray, params: dict) -> list[tuple[int, int]]:
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


def main() -> int:
    data_dir = Path("/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    summary_path = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.summary.json")
    ckpt_path = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.pt")
    out_path = Path("/home/zzy/jepa_data/segment_train_full_error_analysis.json")

    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    feature_names = data_summary["feature_names"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(data_dir)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = TemporalSegmentLocator(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32)
    val_idx = summary["val_idx"]
    outputs = predict_records(model, records, val_idx, mean, std, "cpu")
    hybrid = summary["validation"].get("hybrid", {})
    if hybrid:
        outputs = blend_prediction_records(
            outputs,
            records,
            val_idx,
            int(hybrid["aux_channel"]),
            float(hybrid["alpha_model"]),
        )
    params = summary["validation"]["params"]

    indices = {
        "vjepa_rank": feature_names.index("true_vjepa_raw_rank"),
        "ijepa_rank": feature_names.index("true_ijepa_dense_raw_rank"),
        "dual_rank": feature_names.index("dual_jepa_composite_rank"),
        "vjepa_robust": feature_names.index("true_vjepa_raw_robust_z"),
        "ijepa_robust": feature_names.index("true_ijepa_dense_raw_robust_z"),
        "dual_robust": feature_names.index("dual_jepa_composite_robust_z"),
    }

    missed = []
    matched = 0
    for output, record_idx in zip(outputs, val_idx):
        record = records[record_idx]
        predicted = _predicted_segments(output["probs"], params)
        for gt in contiguous_segments(record.labels):
            best_iou = max((segment_iou(gt, pred) for pred in predicted), default=0.0)
            start, end = gt
            values = record.signals[start : end + 1]
            if best_iou >= 0.3:
                matched += 1
                continue
            item = {
                "video": record.name,
                "gt": [int(start), int(end)],
                "length": int(end - start + 1),
                "best_iou": float(best_iou),
                "score_mean": float(np.mean(output["probs"][start : end + 1])),
                "score_max": float(np.max(output["probs"][start : end + 1])),
            }
            for name, idx in indices.items():
                item[f"{name}_mean"] = float(values[:, idx].mean())
                item[f"{name}_max"] = float(values[:, idx].max())
            missed.append(item)

    aggregate = {}
    if missed:
        for key in missed[0]:
            if key in {"video", "gt"}:
                continue
            aggregate[key] = float(np.mean([item[key] for item in missed]))
    result = {"matched": matched, "missed": missed, "missed_aggregate": aggregate}
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"matched": matched, "missed": len(missed), "missed_aggregate": aggregate}, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
