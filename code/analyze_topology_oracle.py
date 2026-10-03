#!/usr/bin/env python3
"""Oracle headroom for replacing broad predictions with JEPA child events.

This diagnostic estimates whether the remaining topology errors are solvable
by the current V/I/dual-JEPA candidate pool. It uses labels only for analysis
on full-train CV folds and must not be used for held-out test inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fn_candidate_attribution import replay_fold
from jepa_event_topology_splitter import _coverage, _normalise_segment, _segment_length
from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou
from train_segment_locator import contiguous_segments, load_signal_dataset


DEFAULT_DATA_DIR = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2"
DEFAULT_SOURCE_SUMMARY = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json"
DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_topology_oracle.json"


Segment = tuple[int, int]


def _greedy_matches(predictions: list[tuple[int, int]], labels: np.ndarray, iou_threshold: float) -> tuple[set[int], dict[int, Segment]]:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched: set[int] = set()
    pred_to_gt: dict[int, Segment] = {}
    for pred_idx, pred_raw in enumerate(predictions):
        pred = _normalise_segment(pred_raw)
        best_idx = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_segments):
            if gt_idx in matched:
                continue
            iou = float(interval_iou(pred, gt))
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
            pred_to_gt[pred_idx] = gt_segments[best_idx]
    return matched, pred_to_gt


def _best_candidate_for_gt(
    gt: Segment,
    parent: Segment,
    candidates: list[Segment],
    used: set[Segment],
    iou_threshold: float,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
) -> Segment | None:
    best: Segment | None = None
    best_key: tuple[float, float, float, float] | None = None
    parent_length = _segment_length(parent)
    for candidate in candidates:
        if candidate in used:
            continue
        if _coverage(candidate, parent) < float(min_child_parent_coverage):
            continue
        if _segment_length(candidate) / max(1, parent_length) > float(max_child_parent_ratio):
            continue
        iou = float(interval_iou(candidate, gt))
        if iou < float(iou_threshold):
            continue
        length_ratio = min(_segment_length(candidate), _segment_length(gt)) / max(_segment_length(candidate), _segment_length(gt))
        center_gap = abs((candidate[0] + candidate[1]) - (gt[0] + gt[1])) / 2.0
        key = (iou, length_ratio, -center_gap, -float(_segment_length(candidate)))
        if best_key is None or key > best_key:
            best_key = key
            best = candidate
    return best


def oracle_topology_replace_predictions(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    iou_threshold: float,
    parent_min_length: int,
    min_parent_gt_count: int,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_replaced_parents_per_video: int,
) -> tuple[list[list[Segment]], list[dict]]:
    replaced_all: list[list[Segment]] = []
    events: list[dict] = []
    for video_idx, (base_raw, candidates_raw, label_array) in enumerate(zip(base_predictions, candidate_predictions, labels)):
        base = [_normalise_segment(item) for item in base_raw]
        candidates = sorted({_normalise_segment(item) for item in candidates_raw})
        gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(label_array, dtype=np.int64))]
        matched, pred_to_gt = _greedy_matches(base, label_array, iou_threshold=float(iou_threshold))
        replacements: list[tuple[int, list[Segment], list[Segment]]] = []
        for pred_idx, parent in enumerate(base):
            if pred_idx not in pred_to_gt:
                continue
            if _segment_length(parent) < int(parent_min_length):
                continue
            inside_gt_indices = [
                gt_idx
                for gt_idx, gt in enumerate(gt_segments)
                if _coverage(gt, parent) >= 0.8
            ]
            if len(inside_gt_indices) < int(min_parent_gt_count):
                continue
            used_candidates: set[Segment] = set()
            child_segments: list[Segment] = []
            covered_gts: list[Segment] = []
            for gt_idx in inside_gt_indices:
                gt = gt_segments[gt_idx]
                child = _best_candidate_for_gt(
                    gt,
                    parent,
                    candidates,
                    used_candidates,
                    iou_threshold=float(iou_threshold),
                    min_child_parent_coverage=float(min_child_parent_coverage),
                    max_child_parent_ratio=float(max_child_parent_ratio),
                )
                if child is not None:
                    used_candidates.add(child)
                    child_segments.append(child)
                    covered_gts.append(gt)
            if len(child_segments) >= int(min_parent_gt_count):
                replacements.append((pred_idx, child_segments, covered_gts))
        replacements = replacements[: int(max_replaced_parents_per_video)]
        replace_indices = {idx for idx, _, _ in replacements}
        out_segments = [segment for idx, segment in enumerate(base) if idx not in replace_indices]
        for pred_idx, child_segments, covered_gts in replacements:
            out_segments.extend(child_segments)
            events.append(
                {
                    "video_local_idx": int(video_idx),
                    "parent": list(base[pred_idx]),
                    "children": [list(item) for item in child_segments],
                    "covered_gts": [list(item) for item in covered_gts],
                }
            )
        replaced_all.append(sorted(set(out_segments)))
    return replaced_all, events


def evaluate_topology_oracle(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    iou_threshold: float = 0.3,
    parent_min_lengths: tuple[int, ...] = (12, 24, 48),
    min_parent_gt_counts: tuple[int, ...] = (2,),
    min_child_parent_coverages: tuple[float, ...] = (0.5, 0.7, 0.9),
    max_child_parent_ratios: tuple[float, ...] = (0.5, 0.7, 1.0),
    max_replaced_parents_per_videos: tuple[int, ...] = (1, 2),
) -> dict:
    base_metrics = evaluate_fused_predictions(base_predictions, labels, iou_threshold=iou_threshold)
    grid: list[dict] = []
    for parent_min_length in parent_min_lengths:
        for min_parent_gt_count in min_parent_gt_counts:
            for min_child_parent_coverage in min_child_parent_coverages:
                for max_child_parent_ratio in max_child_parent_ratios:
                    for max_replaced_parents_per_video in max_replaced_parents_per_videos:
                        predictions, events = oracle_topology_replace_predictions(
                            base_predictions,
                            candidate_predictions,
                            labels,
                            iou_threshold=float(iou_threshold),
                            parent_min_length=int(parent_min_length),
                            min_parent_gt_count=int(min_parent_gt_count),
                            min_child_parent_coverage=float(min_child_parent_coverage),
                            max_child_parent_ratio=float(max_child_parent_ratio),
                            max_replaced_parents_per_video=int(max_replaced_parents_per_video),
                        )
                        metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=iou_threshold)
                        grid.append(
                            {
                                "config": {
                                    "parent_min_length": int(parent_min_length),
                                    "min_parent_gt_count": int(min_parent_gt_count),
                                    "min_child_parent_coverage": float(min_child_parent_coverage),
                                    "max_child_parent_ratio": float(max_child_parent_ratio),
                                    "max_replaced_parents_per_video": int(max_replaced_parents_per_video),
                                },
                                "metrics": metrics,
                                "tp_delta": int(metrics["segment"]["tp"]) - int(base_metrics["segment"]["tp"]),
                                "fp_delta": int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"]),
                                "fn_delta": int(metrics["segment"]["fn"]) - int(base_metrics["segment"]["fn"]),
                                "replacement_events": int(len(events)),
                            }
                        )
    best = max(
        grid,
        key=lambda item: (
            float(item["metrics"]["segment"]["f1"]),
            float(item["metrics"]["segment"]["recall"]),
            float(item["metrics"]["segment"]["precision"]),
            -float(item["replacement_events"]),
        ),
    )
    no_fp = [item for item in grid if int(item["fp_delta"]) <= 0]
    best_no_fp = max(
        no_fp,
        key=lambda item: (
            float(item["metrics"]["segment"]["f1"]),
            float(item["metrics"]["segment"]["recall"]),
            float(item["metrics"]["segment"]["precision"]),
            -float(item["replacement_events"]),
        ),
    )
    return {"base_metrics": base_metrics, "best": best, "best_no_fp_increase": best_no_fp, "grid": grid}


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--parent-min-lengths", default="12,24,48")
    parser.add_argument("--min-parent-gt-counts", default="2")
    parser.add_argument("--min-child-parent-coverages", default="0.5,0.7,0.9")
    parser.add_argument("--max-child-parent-ratios", default="0.5,0.7,1.0")
    parser.add_argument("--max-replaced-parents-per-video", default="1,2")
    args = parser.parse_args()

    source_path = Path(args.source_summary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [str(name) for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]]
    folds = list(source["folds"])
    if int(args.fold_limit) > 0:
        folds = folds[: int(args.fold_limit)]

    fold_results = [
        replay_fold(
            records,
            feature_names,
            config=source["config"],
            fold_summary=fold,
            device=str(args.device),
            iou_threshold=float(args.iou_threshold),
        )
        for fold in folds
    ]
    base_predictions = [video for fold in fold_results for video in fold["fused_predictions"]]
    candidate_predictions = [video for fold in fold_results for video in fold["raw_candidates"]]
    labels = [np.asarray(video, dtype=np.int64) for fold in fold_results for video in fold["labels"]]
    oracle = evaluate_topology_oracle(
        base_predictions,
        candidate_predictions,
        labels,
        iou_threshold=float(args.iou_threshold),
        parent_min_lengths=_parse_int_tuple(args.parent_min_lengths),
        min_parent_gt_counts=_parse_int_tuple(args.min_parent_gt_counts),
        min_child_parent_coverages=_parse_float_tuple(args.min_child_parent_coverages),
        max_child_parent_ratios=_parse_float_tuple(args.max_child_parent_ratios),
        max_replaced_parents_per_videos=_parse_int_tuple(args.max_replaced_parents_per_video),
    )
    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(source_path),
        "fold_limit": int(args.fold_limit),
        "iou_threshold": float(args.iou_threshold),
        "device": str(args.device),
        "oracle": oracle,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "base_segment": oracle["base_metrics"]["segment"],
                "best": {
                    "config": oracle["best"]["config"],
                    "segment": oracle["best"]["metrics"]["segment"],
                    "tp_delta": oracle["best"]["tp_delta"],
                    "fp_delta": oracle["best"]["fp_delta"],
                    "fn_delta": oracle["best"]["fn_delta"],
                    "replacement_events": oracle["best"]["replacement_events"],
                },
                "best_no_fp_increase": {
                    "config": oracle["best_no_fp_increase"]["config"],
                    "segment": oracle["best_no_fp_increase"]["metrics"]["segment"],
                    "tp_delta": oracle["best_no_fp_increase"]["tp_delta"],
                    "fp_delta": oracle["best_no_fp_increase"]["fp_delta"],
                    "fn_delta": oracle["best_no_fp_increase"]["fn_delta"],
                    "replacement_events": oracle["best_no_fp_increase"]["replacement_events"],
                },
                "out": str(out_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
