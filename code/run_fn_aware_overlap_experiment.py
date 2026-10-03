#!/usr/bin/env python3
"""Fast FN-aware overlap rescue experiment on cached full-train CV folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fn_candidate_attribution import replay_fold
from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_fn_aware_reranker import (
    build_fn_aware_candidate_records,
    fit_fn_aware_reranker,
    score_fn_aware_candidates,
    select_fn_aware_reranker_params,
)
from train_segment_locator import load_signal_dataset


DEFAULT_DATA_DIR = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2"
DEFAULT_SOURCE_SUMMARY = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json"
DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_fn_aware_overlap_fast_cv2.json"


def _parse_float_or_none_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(None if stripped.lower() in {"none", "null"} else float(stripped))
    return values


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=2)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp", "prototype"], default="extratrees")
    parser.add_argument("--train-max-per-video", type=int, default=160)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--max-base-ious", default="0,0.05,0.1,0.25,none")
    parser.add_argument("--nms-ious", default="none,0.1,0.3,0.5")
    parser.add_argument("--length-penalties", default="0,0.005,0.01,0.02")
    parser.add_argument("--max-per-video", default="1,2,3,4")
    args = parser.parse_args()

    source = json.loads(Path(args.source_summary).read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [
        str(name)
        for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]
    ]
    channel_names = [name for name in source["config"].get("selector_channel_names", []) if name in feature_names]
    folds = list(source["folds"])
    if args.fold_limit > 0:
        folds = folds[: int(args.fold_limit)]

    replays: list[dict] = []
    for fold in folds:
        replays.append(
            replay_fold(
                records,
                feature_names,
                source["config"],
                fold,
                device=str(args.device),
                iou_threshold=float(args.iou_threshold),
            )
        )

    fold_outputs: list[dict] = []
    for replay in replays:
        val_idx = [int(idx) for idx in replay["val_idx"]]
        train_replays = [other for other in replays if int(other["fold"]) != int(replay["fold"])]
        train_candidates_by_idx = {
            int(idx): candidates
            for other in train_replays
            for idx, candidates in zip(other["val_idx"], other["raw_candidates"])
        }
        train_scores_by_idx = {
            int(idx): scores
            for other in train_replays
            for idx, scores in zip(other["val_idx"], other["raw_scores"])
        }
        train_base_by_idx = {
            int(idx): segments
            for other in train_replays
            for idx, segments in zip(other["val_idx"], other["fused_predictions"])
        }
        if not train_candidates_by_idx:
            replay["fn_aware_overlap"] = {"enabled": False, "reason": "no_oof_meta_train_folds"}
            fold_outputs.append(replay)
            continue

        candidate_rows = build_fn_aware_candidate_records(
            records,
            list(train_candidates_by_idx.keys()),
            candidate_predictions_by_idx=train_candidates_by_idx,
            candidate_scores_by_idx=train_scores_by_idx,
            base_segments_by_idx=train_base_by_idx,
            feature_names=feature_names,
            channel_names=channel_names,
            iou_threshold=float(args.iou_threshold),
            max_records_per_video=int(args.train_max_per_video),
        )
        if not candidate_rows:
            replay["fn_aware_overlap"] = {"enabled": False, "reason": "no_train_candidates"}
            fold_outputs.append(replay)
            continue
        x_train = np.stack([row.features for row in candidate_rows])
        y_train = np.asarray([row.label for row in candidate_rows], dtype=np.int64)
        if len(np.unique(y_train)) < 2:
            replay["fn_aware_overlap"] = {
                "enabled": False,
                "reason": "single_class_train_candidates",
                "n_train_candidates": len(candidate_rows),
                "n_positive_train_candidates": int(y_train.sum()),
            }
            fold_outputs.append(replay)
            continue
        model = fit_fn_aware_reranker(
            x_train,
            y_train,
            seed=9000 + int(replay["fold"]),
            model_name=str(args.model),
            device=str(args.device),
            epochs=120,
            batch_size=512,
        )
        rescue_scores = [
            score_fn_aware_candidates(
                model,
                records[int(idx)],
                candidates,
                scores,
                feature_names,
                channel_names,
                base_segments,
            )
            for idx, candidates, scores, base_segments in zip(
                val_idx,
                replay["raw_candidates"],
                replay["raw_scores"],
                replay["fused_predictions"],
            )
        ]
        labels = [records[int(idx)].labels for idx in val_idx]
        config, metrics, predictions = select_fn_aware_reranker_params(
            replay["fused_predictions"],
            replay["raw_candidates"],
            rescue_scores,
            labels,
            thresholds=_parse_float_list(args.thresholds),
            max_base_ious=_parse_float_or_none_list(args.max_base_ious),
            nms_ious=_parse_float_or_none_list(args.nms_ious),
            length_penalties=_parse_float_list(args.length_penalties),
            max_rescues_per_videos=_parse_int_list(args.max_per_video),
            iou_threshold=float(args.iou_threshold),
        )
        replay["fn_aware_overlap"] = {
            "enabled": bool(config.get("enabled")),
            "config": config,
            "metrics": metrics,
            "n_train_candidates": len(candidate_rows),
            "n_positive_train_candidates": int(y_train.sum()),
            "model": str(args.model),
        }
        replay["validation"] = metrics if config.get("enabled") else replay["validation"]
        replay["fused_predictions"] = predictions if config.get("enabled") else replay["fused_predictions"]
        fold_outputs.append(replay)

    summary_folds = [
        {
            "fold": int(fold["fold"]),
            "validation": fold["validation"],
            "base_validation": fold.get("fn_aware_overlap", {}).get("config", {}).get("base_metrics", fold["validation"]),
            "fn_aware_overlap": fold.get("fn_aware_overlap", {}),
        }
        for fold in fold_outputs
    ]
    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(args.source_summary),
        "fold_limit": int(args.fold_limit),
        "config": vars(args),
        "folds": summary_folds,
        "aggregate": aggregate_fold_metrics(summary_folds),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate_segment": result["aggregate"]["segment"], "out": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
