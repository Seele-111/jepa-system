#!/usr/bin/env python3
"""Run single-split fusion of the mainline locator and a JEPA proposal selector."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from selector_fusion import (
    evaluate_fused_predictions,
    fuse_protected_segment_predictions,
    fuse_segment_predictions,
    select_fusion_nms_iou,
    select_protected_fusion_params,
)
from train_proposal_calibrator import evaluate_segment_predictions
from train_proposal_set_selector import QualityModelWrapper, predict_video_segments
from train_segment_locator import (
    TemporalSegmentLocator,
    _segments_from_params,
    blend_prediction_records,
    load_signal_dataset,
    predict_records,
)


setattr(sys.modules["__main__"], "QualityModelWrapper", QualityModelWrapper)


def _parse_nms_candidates(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else float(item))
    if not values:
        raise ValueError("at least one NMS candidate is required")
    return values


def load_mainline_predictions(
    records,
    val_idx: list[int],
    summary_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
) -> tuple[list[list[tuple[int, int]]], list[dict], dict]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TemporalSegmentLocator(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32)
    outputs = predict_records(model, records, val_idx, mean, std, device)
    validation = summary["validation"]
    hybrid = validation.get("hybrid", {})
    if hybrid:
        outputs = blend_prediction_records(
            outputs,
            records,
            val_idx,
            aux_channel=int(hybrid["aux_channel"]),
            alpha_model=float(hybrid["alpha_model"]),
        )
    params = validation["params"]
    predictions = [_segments_from_params(output["probs"], params) for output in outputs]
    return predictions, outputs, validation


def load_selector_predictions(path: str | Path, records, val_idx: list[int]) -> tuple[list[list[tuple[int, int]]], dict]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    predictions = [
        predict_video_segments(
            payload["classifier"],
            records[idx],
            payload["feature_names"],
            payload["channel_names"],
            payload["thresholds"],
            payload["min_gaps"],
            payload["min_lengths"],
            prob_threshold=float(payload["params"]["prob_threshold"]),
            length_penalty=float(payload["params"]["length_penalty"]),
        )
        for idx in val_idx
    ]
    return predictions, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--main-summary", required=True)
    parser.add_argument("--main-checkpoint", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--nms-candidates", default="none,0.1,0.3,0.5")
    parser.add_argument("--protected-mainline-iou-candidates", default="0,0.1,0.25,0.5")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    main_summary = json.loads(Path(args.main_summary).read_text(encoding="utf-8"))
    val_idx = [int(idx) for idx in main_summary["val_idx"]]
    labels = [records[idx].labels for idx in val_idx]
    main_predictions, _, main_validation = load_mainline_predictions(
        records,
        val_idx,
        args.main_summary,
        args.main_checkpoint,
        args.device,
    )
    selector_predictions, selector_payload = load_selector_predictions(args.selector, records, val_idx)
    nms_candidates = _parse_nms_candidates(args.nms_candidates)
    best_nms, best_segment_metrics = select_fusion_nms_iou(
        main_predictions,
        selector_predictions,
        labels,
        candidates=nms_candidates,
        iou_threshold=args.iou_threshold,
    )
    fused_predictions = fuse_segment_predictions(main_predictions, selector_predictions, nms_iou=best_nms)
    protected_config, protected_metrics = select_protected_fusion_params(
        main_predictions,
        selector_predictions,
        labels,
        max_mainline_iou_candidates=[float(item) for item in args.protected_mainline_iou_candidates.split(",") if item.strip()],
        selector_nms_iou_candidates=nms_candidates,
        iou_threshold=args.iou_threshold,
    )
    if protected_config.get("enabled"):
        protected_predictions = fuse_protected_segment_predictions(
            main_predictions,
            selector_predictions,
            max_mainline_iou=protected_config["max_mainline_iou"],
            selector_nms_iou=protected_config["selector_nms_iou"],
        )
    else:
        protected_predictions = main_predictions
    per_nms = {}
    for candidate in nms_candidates:
        key = "none" if candidate is None else str(candidate)
        per_nms[key] = evaluate_fused_predictions(
            fuse_segment_predictions(main_predictions, selector_predictions, nms_iou=candidate),
            labels,
            iou_threshold=args.iou_threshold,
        )

    summary = {
        "data_dir": args.data_dir,
        "main_summary": args.main_summary,
        "main_checkpoint": args.main_checkpoint,
        "selector": args.selector,
        "n_videos": len(records),
        "val_idx": val_idx,
        "val_names": [records[idx].name for idx in val_idx],
        "mainline": evaluate_fused_predictions(main_predictions, labels, iou_threshold=args.iou_threshold),
        "mainline_validation_from_summary": main_validation,
        "selector": {
            "target": selector_payload.get("target", "binary"),
            "params": selector_payload.get("params", {}),
            "metrics": evaluate_segment_predictions(selector_predictions, labels, iou_threshold=args.iou_threshold),
        },
        "fusion": {
            "best_nms_iou": best_nms,
            "best_segment_metrics": best_segment_metrics,
            "per_nms": per_nms,
        },
        "protected_fusion": {
            "config": protected_config,
            "metrics": protected_metrics,
        },
        "validation": evaluate_fused_predictions(protected_predictions, labels, iou_threshold=args.iou_threshold),
        "unprotected_validation": evaluate_fused_predictions(fused_predictions, labels, iou_threshold=args.iou_threshold),
        "task": "binary_error_segment_localization_mainline_selector_fusion",
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["validation"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
