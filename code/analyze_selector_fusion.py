#!/usr/bin/env python3
"""Analyze complementarity between mainline and proposal set selectors."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from train_proposal_calibrator import evaluate_segment_predictions, interval_iou
from train_proposal_set_selector import predict_video_segments
from train_proposal_set_selector import QualityModelWrapper
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
MAIN_SUMMARY = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.summary.json")
MAIN_CKPT = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_hybrid_hyst.pt")
SELECTORS = {
    "binary": Path("/home/zzy/jepa_data/proposal_set_selector_full_extratrees.pkl"),
    "quality": Path("/home/zzy/jepa_data/proposal_set_selector_full_quality_extratrees.pkl"),
}
OUT_PATH = Path("/home/zzy/jepa_data/selector_fusion_analysis.json")


# Older selector checkpoints were created by running train_proposal_set_selector.py
# as a script, so pickle recorded the wrapper as __main__.QualityModelWrapper.
setattr(sys.modules["__main__"], "QualityModelWrapper", QualityModelWrapper)


def _load_mainline(records, val_idx):
    summary = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    checkpoint = torch.load(MAIN_CKPT, map_location="cpu")
    model = TemporalSegmentLocator(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32)
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
    return [_segments_from_params(output["probs"], params) for output in outputs]


def _load_selector_predictions(path: Path, records, val_idx):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    classifier = payload["classifier"]
    feature_names = payload["feature_names"]
    channel_names = payload["channel_names"]
    thresholds = payload["thresholds"]
    min_gaps = payload["min_gaps"]
    min_lengths = payload["min_lengths"]
    params = payload["params"]
    predictions = []
    for idx in val_idx:
        predictions.append(
            predict_video_segments(
                classifier,
                records[idx],
                feature_names,
                channel_names,
                thresholds,
                min_gaps,
                min_lengths,
                prob_threshold=float(params["prob_threshold"]),
                length_penalty=float(params["length_penalty"]),
            )
        )
    return predictions


def _nms_segments(segments: list[tuple[int, int]], iou_threshold: float) -> list[tuple[int, int]]:
    ordered = sorted(segments, key=lambda seg: (seg[1] - seg[0] + 1), reverse=False)
    kept: list[tuple[int, int]] = []
    for segment in ordered:
        if all(interval_iou(segment, existing) <= iou_threshold for existing in kept):
            kept.append(segment)
    return sorted(kept)


def _union_predictions(a, b, nms_iou: float | None = None):
    out = []
    for left, right in zip(a, b):
        merged = list(left) + list(right)
        out.append(_nms_segments(merged, nms_iou) if nms_iou is not None else sorted(set(merged)))
    return out


def _matched_gt_keys(predictions, records, val_idx, iou_threshold: float = 0.3):
    keys = set()
    for pred_segments, idx in zip(predictions, val_idx):
        record = records[idx]
        for gt in contiguous_segments(record.labels):
            if max((segment_iou(pred, gt) for pred in pred_segments), default=0.0) >= iou_threshold:
                keys.add((record.name, int(gt[0]), int(gt[1])))
    return keys


def main() -> int:
    records = load_signal_dataset(DATA_DIR)
    main_summary = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    val_idx = [int(idx) for idx in main_summary["val_idx"]]
    labels = [records[idx].labels for idx in val_idx]
    main_predictions = _load_mainline(records, val_idx)
    results = {
        "mainline": evaluate_segment_predictions(main_predictions, labels, iou_threshold=0.3),
        "selectors": {},
        "fusion": {},
    }
    main_keys = _matched_gt_keys(main_predictions, records, val_idx)
    all_gt = {
        (records[idx].name, int(gt[0]), int(gt[1]))
        for idx in val_idx
        for gt in contiguous_segments(records[idx].labels)
    }
    selector_predictions = {}
    for name, path in SELECTORS.items():
        if not path.exists():
            continue
        predictions = _load_selector_predictions(path, records, val_idx)
        selector_predictions[name] = predictions
        keys = _matched_gt_keys(predictions, records, val_idx)
        results["selectors"][name] = {
            "metrics": evaluate_segment_predictions(predictions, labels, iou_threshold=0.3),
            "matched_not_mainline": len(keys - main_keys),
            "missed_by_both": len(all_gt - (keys | main_keys)),
        }
        for nms_iou in [None, 0.1, 0.3, 0.5]:
            fused = _union_predictions(main_predictions, predictions, nms_iou=nms_iou)
            key = f"{name}_union" if nms_iou is None else f"{name}_union_nms{nms_iou}"
            results["fusion"][key] = evaluate_segment_predictions(fused, labels, iou_threshold=0.3)

    if set(selector_predictions) >= {"binary", "quality"}:
        fused_selectors = _union_predictions(selector_predictions["binary"], selector_predictions["quality"], nms_iou=None)
        results["fusion"]["binary_quality_union"] = evaluate_segment_predictions(fused_selectors, labels, iou_threshold=0.3)
        fused_all = _union_predictions(main_predictions, fused_selectors, nms_iou=None)
        results["fusion"]["main_binary_quality_union"] = evaluate_segment_predictions(fused_all, labels, iou_threshold=0.3)

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
