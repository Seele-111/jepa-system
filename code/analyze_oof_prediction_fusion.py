#!/usr/bin/env python3
"""Analyze strict OOF prediction fusion outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train_proposal_calibrator import evaluate_segment_predictions, interval_iou
from train_segment_locator import contiguous_segments


BASE = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.predictions.json")
RECALL = Path("/home/zzy/jepa_data/segment_train_full_event_v2_clean_anchor_soft_weights_graph_cv5.predictions.json")
FUSION = Path("/home/zzy/jepa_data/segment_train_full_event_v2_protected_oof_prediction_fusion.summary.json")


def _segments(raw):
    return [tuple(map(int, item)) for item in raw]


def _max_iou(segment, others):
    return max((interval_iou(segment, other) for other in others), default=0.0)


def _matched_gt(predictions, labels, iou_threshold=0.3):
    gt = [tuple(item) for item in contiguous_segments(np.asarray(labels))]
    matched = set()
    for pred in predictions:
        best_idx = -1
        best_iou = 0.0
        for idx, item in enumerate(gt):
            if idx in matched:
                continue
            iou = interval_iou(pred, item)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.add(best_idx)
    return matched, gt


def main() -> int:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    recall = json.loads(RECALL.read_text(encoding="utf-8"))
    fusion = json.loads(FUSION.read_text(encoding="utf-8"))
    print("fusion aggregate", json.dumps(fusion["aggregate"]["segment"], indent=2))
    for fold in fusion["folds"]:
        print("fusion fold", fold["fold"], "enabled", fold["config"].get("enabled"), fold["validation"]["segment"])

    additions = []
    extra_tp = 0
    extra_fp = 0
    missing_base_tp_recovered = 0
    for bf, rf in zip(base["folds"], recall["folds"]):
        for name, bpred_raw, rpred_raw, labels in zip(
            bf["val_names"],
            bf["predictions"],
            rf["predictions"],
            bf["labels"],
        ):
            bpred = _segments(bpred_raw)
            rpred = _segments(rpred_raw)
            bmatched, gt = _matched_gt(bpred, labels)
            for segment in rpred:
                if segment in bpred:
                    continue
                base_iou = _max_iou(segment, bpred)
                gt_iou = _max_iou(segment, gt)
                is_tp = gt_iou >= 0.3
                additions.append((name, segment, base_iou, gt_iou, is_tp))
                if is_tp:
                    extra_tp += 1
                    best_gt = max(range(len(gt)), key=lambda idx: interval_iou(segment, gt[idx])) if gt else -1
                    if best_gt not in bmatched:
                        missing_base_tp_recovered += 1
                else:
                    extra_fp += 1
    print(
        "recall_export_extra_segments",
        {
            "count": len(additions),
            "gt_iou>=0.3": extra_tp,
            "gt_iou<0.3": extra_fp,
            "base_missed_gt_recovered": missing_base_tp_recovered,
        },
    )
    buckets = [(0.0, 0.0), (0.0001, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 1.0)]
    for lo, hi in buckets:
        selected = [item for item in additions if lo <= item[2] <= hi]
        print(
            "base_iou_bucket",
            (lo, hi),
            {
                "count": len(selected),
                "tp_like": sum(1 for item in selected if item[4]),
                "fp_like": sum(1 for item in selected if not item[4]),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
