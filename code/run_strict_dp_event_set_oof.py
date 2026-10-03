#!/usr/bin/env python3
"""Strict OOF DP event-set selection from exported base predictions.

The base predictions are assumed to be OOF exports. For each export fold, this
script trains the JEPA proposal scorer on that fold's training videos, scores
raw JEPA candidates for the held-out videos, then selects DP parameters using
only the other folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa_dp_event_set_selector import run_fold_heldout_dp_event_set
from jepa_prediction_set_switcher import _as_segments
from train_proposal_set_selector import DEFAULT_CHANNEL_NAMES, score_video_candidates
from train_segment_locator import (
    _load_feature_names,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    load_signal_dataset,
)
from cross_validate_selector_fusion import train_selector_for_fold


def _parse_optional_int(text: str) -> int | None:
    value = str(text).strip().lower()
    return None if value in {"none", "null"} else int(value)


def _load_base_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for fold in data.get("folds", []):
        fold["predictions"] = [_as_segments(video) for video in fold.get("predictions", [])]
        if "labels" in fold:
            fold["labels"] = [np.asarray(item, dtype=np.int64) for item in fold["labels"]]
    return data


def build_dp_folds_from_base_export(
    base_export: dict,
    records,
    raw_by_video: dict[int, tuple[list[tuple[int, int]], np.ndarray]],
) -> list[dict]:
    folds: list[dict] = []
    all_indices = set(range(len(records)))
    for fold in base_export.get("folds", []):
        val_idx = [int(idx) for idx in fold.get("val_idx", [])]
        train_idx = [int(idx) for idx in fold.get("train_idx", [])]
        if not train_idx:
            train_idx = sorted(all_indices - set(val_idx))
        base_predictions = [[(int(s), int(e)) for s, e in video] for video in fold.get("predictions", [])]
        if len(base_predictions) != len(val_idx):
            raise ValueError(f"fold={fold.get('fold')} prediction/val_idx mismatch")
        labels_raw = fold.get("labels")
        labels = (
            [np.asarray(item, dtype=np.int64) for item in labels_raw]
            if labels_raw is not None
            else [records[idx].labels for idx in val_idx]
        )
        candidates = []
        scores = []
        for idx in val_idx:
            video_candidates, video_scores = raw_by_video[int(idx)]
            candidates.append([(int(s), int(e)) for s, e in video_candidates])
            scores.append(np.asarray(video_scores, dtype=np.float32))
        folds.append(
            {
                "fold": int(fold.get("fold", len(folds))),
                "train_idx": train_idx,
                "val_idx": val_idx,
                "base": base_predictions,
                "candidates": candidates,
                "scores": scores,
                "labels": labels,
            }
        )
    return folds


def build_fold_local_raw_candidates(
    records,
    base_export: dict,
    feature_names: list[str],
    channel_names: list[str],
    selector_thresholds: list[float],
    selector_min_gaps: list[int],
    selector_min_lengths: list[int],
    selector_model: str,
    selector_target: str,
    iou_threshold: float,
    seed: int,
    device: str,
    selector_epochs: int,
    selector_batch_size: int,
) -> tuple[dict[int, tuple[list[tuple[int, int]], np.ndarray]], list[dict]]:
    raw_by_video: dict[int, tuple[list[tuple[int, int]], np.ndarray]] = {}
    selector_diagnostics: list[dict] = []
    all_indices = set(range(len(records)))
    for fold_pos, fold in enumerate(base_export.get("folds", [])):
        train_idx = [int(idx) for idx in fold.get("train_idx", [])]
        val_idx = [int(idx) for idx in fold.get("val_idx", [])]
        if not train_idx:
            train_idx = sorted(all_indices - set(val_idx))
        if not train_idx or not val_idx:
            raise ValueError(f"fold={fold.get('fold', fold_pos)} needs train_idx and val_idx")
        classifier, params, metrics, n_candidates, n_positive = train_selector_for_fold(
            records,
            train_idx,
            val_idx,
            feature_names,
            channel_names,
            selector_thresholds,
            selector_min_gaps,
            selector_min_lengths,
            iou_threshold=float(iou_threshold),
            seed=int(seed) + 1000 + int(fold_pos),
            model_name=selector_model,
            target=selector_target,
            device=device,
            selector_epochs=int(selector_epochs),
            selector_batch_size=int(selector_batch_size),
        )
        selector_diagnostics.append(
            {
                "fold": int(fold.get("fold", fold_pos)),
                "n_train_candidates": int(n_candidates),
                "n_positive_train_candidates": int(n_positive),
                "selector_params": params,
                "selector_validation": metrics,
            }
        )
        for idx in val_idx:
            raw_by_video[int(idx)] = score_video_candidates(
                classifier,
                records[int(idx)],
                feature_names,
                channel_names,
                selector_thresholds,
                selector_min_gaps,
                selector_min_lengths,
            )
    return raw_by_video, selector_diagnostics


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--selector-model", choices=["gbdt", "rf", "extratrees", "logreg", "mlp"], default="extratrees")
    parser.add_argument("--selector-target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--selector-channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--selector-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selector-min-gaps", default="0,1,2,4,8")
    parser.add_argument("--selector-min-lengths", default="1,2,4,8")
    parser.add_argument("--selector-epochs", type=int, default=120)
    parser.add_argument("--selector-batch-size", type=int, default=512)
    parser.add_argument("--dp-base-keep-scores", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--dp-candidate-score-weights", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--dp-min-candidate-scores", default="0,0.1,0.2,0.3")
    parser.add_argument("--dp-length-penalties", default="0,0.005,0.01")
    parser.add_argument("--dp-base-overlap-penalties", default="0,0.1,0.25")
    parser.add_argument("--dp-max-fp-increase", default="0")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.selector_channel_names) if name in feature_names]
    if not channel_names:
        raise ValueError("no selector channels found in event dataset summary")
    base_export = _load_base_export(args.base_predictions)
    raw_by_video, selector_diagnostics = build_fold_local_raw_candidates(
        records,
        base_export,
        feature_names,
        channel_names,
        _parse_float_list(args.selector_thresholds),
        _parse_int_list(args.selector_min_gaps),
        _parse_int_list(args.selector_min_lengths),
        selector_model=args.selector_model,
        selector_target=args.selector_target,
        iou_threshold=float(args.iou_threshold),
        seed=int(args.seed),
        device=args.device,
        selector_epochs=int(args.selector_epochs),
        selector_batch_size=int(args.selector_batch_size),
    )
    folds = build_dp_folds_from_base_export(base_export, records, raw_by_video)
    result = run_fold_heldout_dp_event_set(
        folds,
        base_keep_scores=_parse_float_list(args.dp_base_keep_scores),
        candidate_score_weights=_parse_float_list(args.dp_candidate_score_weights),
        min_candidate_scores=_parse_float_list(args.dp_min_candidate_scores),
        length_penalties=_parse_float_list(args.dp_length_penalties),
        base_overlap_penalties=_parse_float_list(args.dp_base_overlap_penalties),
        max_fp_increase=_parse_optional_int(args.dp_max_fp_increase),
        iou_threshold=float(args.iou_threshold),
    )
    result.update(
        {
            "data_dir": args.data_dir,
            "base_predictions": args.base_predictions,
            "selector_model": args.selector_model,
            "selector_target": args.selector_target,
            "selector_channel_names": channel_names,
            "selector_thresholds": _parse_float_list(args.selector_thresholds),
            "selector_min_gaps": _parse_int_list(args.selector_min_gaps),
            "selector_min_lengths": _parse_int_list(args.selector_min_lengths),
            "selector_diagnostics": selector_diagnostics,
            "dp_base_keep_scores": _parse_float_list(args.dp_base_keep_scores),
            "dp_candidate_score_weights": _parse_float_list(args.dp_candidate_score_weights),
            "dp_min_candidate_scores": _parse_float_list(args.dp_min_candidate_scores),
            "dp_length_penalties": _parse_float_list(args.dp_length_penalties),
            "dp_base_overlap_penalties": _parse_float_list(args.dp_base_overlap_penalties),
            "dp_max_fp_increase": _parse_optional_int(args.dp_max_fp_increase),
            "iou_threshold": float(args.iou_threshold),
        }
    )
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_jsonable), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
