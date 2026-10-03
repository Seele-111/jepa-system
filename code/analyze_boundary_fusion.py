#!/usr/bin/env python3
"""Compare mainline and boundary JEPA locators on the full-train validation split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train_boundary_event_locator import (
    TemporalBoundaryLocator,
    predict_records as predict_boundary_records,
    refine_segments_with_boundaries,
)
from train_segment_locator import (
    TemporalSegmentLocator,
    _segment_counts,
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
BOUNDARY_SUMMARY = Path("/home/zzy/jepa_data/boundary_event_full_h32.summary.json")
BOUNDARY_CKPT = Path("/home/zzy/jepa_data/boundary_event_full_h32.pt")
OUT_PATH = Path("/home/zzy/jepa_data/boundary_fusion_analysis.json")


def _load_main_outputs(records, val_idx):
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
    return outputs, summary["validation"]["params"], summary


def _load_boundary_outputs(records, val_idx):
    summary = json.loads(BOUNDARY_SUMMARY.read_text(encoding="utf-8"))
    checkpoint = torch.load(BOUNDARY_CKPT, map_location="cpu")
    model = TemporalBoundaryLocator(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32)
    outputs = predict_boundary_records(model, records, val_idx, mean, std, "cpu")
    return outputs, summary["validation"]["params"], summary


def _boundary_segments(output: dict, params: dict, radius: int | None = None) -> list[tuple[int, int]]:
    params = dict(params)
    if radius is not None:
        params["refine_radius"] = int(radius)
    coarse = _segments_from_params(output["probs"], params)
    return refine_segments_with_boundaries(
        coarse,
        np.asarray(output["start_probs"], dtype=np.float32),
        np.asarray(output["end_probs"], dtype=np.float32),
        int(params.get("refine_radius", 0)),
    )


def _metrics_from_segments(per_video_segments, labels, iou_threshold: float = 0.3) -> dict:
    tp = fp = fn = 0
    for pred, lab in zip(per_video_segments, labels):
        gt = contiguous_segments(np.asarray(lab, dtype=np.int64))
        local_tp, local_fp, local_fn = _segment_counts(pred, gt, iou_threshold=iou_threshold)
        tp += local_tp
        fp += local_fp
        fn += local_fn
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _nms_segments(segments: list[tuple[int, int]], scores: list[float], iou_threshold: float) -> list[tuple[int, int]]:
    order = sorted(range(len(segments)), key=lambda idx: (scores[idx], segments[idx][1] - segments[idx][0]), reverse=True)
    kept: list[tuple[int, int]] = []
    for idx in order:
        segment = segments[idx]
        if all(segment_iou(segment, existing) <= iou_threshold for existing in kept):
            kept.append(segment)
    return sorted(kept)


def _segment_score(probs: np.ndarray, segment: tuple[int, int]) -> float:
    start, end = segment
    values = np.asarray(probs[start : end + 1], dtype=np.float32)
    if len(values) == 0:
        return 0.0
    return float(np.max(values) + 0.1 * np.mean(values))


def main() -> int:
    records = load_signal_dataset(DATA_DIR)
    main_summary = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    boundary_summary = json.loads(BOUNDARY_SUMMARY.read_text(encoding="utf-8"))
    val_idx = [int(idx) for idx in main_summary["val_idx"]]
    if val_idx != [int(idx) for idx in boundary_summary["val_idx"]]:
        raise ValueError("mainline and boundary summaries use different validation splits")

    main_outputs, main_params, _ = _load_main_outputs(records, val_idx)
    boundary_outputs, boundary_params, _ = _load_boundary_outputs(records, val_idx)
    labels = [records[idx].labels for idx in val_idx]
    base_segments = [_segments_from_params(output["probs"], main_params) for output in main_outputs]
    boundary_segments = [_boundary_segments(output, boundary_params) for output in boundary_outputs]

    variants: dict[str, dict] = {
        "mainline": _metrics_from_segments(base_segments, labels),
        "boundary": _metrics_from_segments(boundary_segments, labels),
    }

    for radius in [2, 4, 8, 12, 16, 24]:
        refined = [
            refine_segments_with_boundaries(
                segments,
                boundary_output["start_probs"],
                boundary_output["end_probs"],
                radius,
            )
            for segments, boundary_output in zip(base_segments, boundary_outputs)
        ]
        variants[f"main_refined_by_boundary_r{radius}"] = _metrics_from_segments(refined, labels)

    for pad in [1, 2, 4, 8, 12]:
        expanded = []
        for segments, label in zip(base_segments, labels):
            n = len(label)
            expanded.append([(max(0, s - pad), min(n - 1, e + pad)) for s, e in segments])
        variants[f"main_expanded_p{pad}"] = _metrics_from_segments(expanded, labels)

    for nms_iou in [0.1, 0.3, 0.5]:
        union_segments = []
        for main_output, boundary_output, main_items, boundary_items in zip(
            main_outputs, boundary_outputs, base_segments, boundary_segments
        ):
            segments = list(main_items) + list(boundary_items)
            scores = [_segment_score(main_output["probs"], seg) for seg in main_items]
            scores.extend(_segment_score(boundary_output["probs"], seg) for seg in boundary_items)
            union_segments.append(_nms_segments(segments, scores, iou_threshold=nms_iou))
        variants[f"main_boundary_union_nms{nms_iou}"] = _metrics_from_segments(union_segments, labels)

    missed_details = []
    for record_idx, main_output, boundary_output, main_items, boundary_items in zip(
        val_idx, main_outputs, boundary_outputs, base_segments, boundary_segments
    ):
        record = records[record_idx]
        for gt in contiguous_segments(record.labels):
            main_iou = max((segment_iou(gt, pred) for pred in main_items), default=0.0)
            if main_iou >= 0.3:
                continue
            boundary_iou = max((segment_iou(gt, pred) for pred in boundary_items), default=0.0)
            refined_iou = {}
            for radius in [2, 4, 8, 12, 16, 24]:
                refined = refine_segments_with_boundaries(
                    main_items,
                    boundary_output["start_probs"],
                    boundary_output["end_probs"],
                    radius,
                )
                refined_iou[f"r{radius}"] = max((segment_iou(gt, pred) for pred in refined), default=0.0)
            missed_details.append(
                {
                    "video": record.name,
                    "gt": [int(gt[0]), int(gt[1])],
                    "length": int(gt[1] - gt[0] + 1),
                    "main_best_iou": float(main_iou),
                    "boundary_best_iou": float(boundary_iou),
                    "refined_best_iou": refined_iou,
                }
            )

    result = {
        "main_summary": str(MAIN_SUMMARY),
        "boundary_summary": str(BOUNDARY_SUMMARY),
        "val_videos": len(val_idx),
        "variants": variants,
        "missed_details": missed_details,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"variants": variants, "missed": len(missed_details), "out": str(OUT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
