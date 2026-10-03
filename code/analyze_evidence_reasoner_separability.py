#!/usr/bin/env python3
"""Analyze whether JEPA evidence-set scores separate missed positives from FPs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fn_candidate_attribution import replay_fold
from jepa_evidence_set_reasoner import precompute_reasoner_stats, reasoner_scores_from_stats
from train_proposal_calibrator import interval_iou
from train_segment_locator import contiguous_segments, load_signal_dataset


DEFAULT_DATA_DIR = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2"
DEFAULT_SOURCE_SUMMARY = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json"
DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_evidence_reasoner_separability.json"


def _normalise_segment(segment: tuple[int, int]) -> tuple[int, int]:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _matched_gt_indices(predictions: list[tuple[int, int]], labels: np.ndarray, iou_threshold: float) -> set[int]:
    matched: set[int] = set()
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    for pred_raw in predictions:
        pred = _normalise_segment(pred_raw)
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_segments):
            if idx in matched:
                continue
            iou = float(interval_iou(pred, gt))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
    return matched


def _candidate_label(candidate: tuple[int, int], base: list[tuple[int, int]], labels: np.ndarray, iou_threshold: float) -> int:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched = _matched_gt_indices(base, labels, iou_threshold=iou_threshold)
    best_idx = -1
    best_iou = 0.0
    for idx, gt in enumerate(gt_segments):
        iou = float(interval_iou(_normalise_segment(candidate), gt))
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return int(best_idx >= 0 and best_idx not in matched and best_iou >= float(iou_threshold))


def _evidence_matrix(record, feature_names: list[str], evidence_names: list[str]) -> np.ndarray:
    arrays = []
    for name in evidence_names:
        if name not in feature_names:
            continue
        arrays.append(np.nan_to_num(record.signals[:, feature_names.index(name)], nan=0.0, posinf=1.0, neginf=0.0))
    if not arrays:
        raise ValueError(f"none of evidence channels were found: {evidence_names}")
    return np.stack(arrays, axis=1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", type=int, default=2)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--evidence-names", default="true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank")
    parser.add_argument("--evidence-threshold", type=float, default=0.6)
    parser.add_argument("--support-iou", type=float, default=0.3)
    parser.add_argument("--prefilter-top-k", type=int, default=80)
    args = parser.parse_args()

    source = json.loads(Path(args.source_summary).read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [str(name) for name in json.loads((Path(args.data_dir) / "summary.json").read_text(encoding="utf-8"))["feature_names"]]
    evidence_names = [item.strip() for item in args.evidence_names.split(",") if item.strip()]
    folds = list(source["folds"])
    if args.fold_limit > 0:
        folds = folds[: int(args.fold_limit)]
    fold_results = [
        replay_fold(records, feature_names, source["config"], fold, device=args.device, iou_threshold=args.iou_threshold)
        for fold in folds
    ]

    rows: list[dict] = []
    for fold in fold_results:
        for local_idx, record_idx in enumerate(fold["val_idx"]):
            record = records[int(record_idx)]
            base = fold["fused_predictions"][local_idx]
            candidates = fold["raw_candidates"][local_idx]
            selector_scores = fold["raw_scores"][local_idx]
            evidence = _evidence_matrix(record, feature_names, evidence_names)
            stats = precompute_reasoner_stats(
                base,
                candidates,
                selector_scores,
                evidence,
                evidence_threshold=float(args.evidence_threshold),
                support_iou=float(args.support_iou),
                prefilter_top_k=int(args.prefilter_top_k),
            )
            scores = reasoner_scores_from_stats(
                stats,
                selector_weight=0.0,
                contrast_weight=0.5,
                active_fraction_weight=0.5,
                consensus_weight=0.5,
                support_count_weight=0.0,
                base_iou_penalty=0.0,
                length_penalty=0.0,
            )
            candidate_indices = np.asarray(stats["candidate_indices"], dtype=np.int64)
            for idx, score in enumerate(scores):
                candidate = candidates[int(candidate_indices[idx])]
                rows.append(
                    {
                        "fold": int(fold["fold"]),
                        "video_idx": int(record_idx),
                        "video": record.name,
                        "candidate": list(_normalise_segment(candidate)),
                        "label": _candidate_label(candidate, base, record.labels, iou_threshold=float(args.iou_threshold)),
                        "reasoner_score": float(score),
                        "selector_score": float(stats["selector_scores"][idx]),
                        "evidence_mean": float(stats["evidence_mean"][idx]),
                        "active_fraction": float(stats["active_fraction"][idx]),
                        "contrast": float(stats["contrast"][idx]),
                        "consensus": float(stats["consensus"][idx]),
                        "base_iou": float(stats["base_iou"][idx]),
                        "support_count": float(stats["support_count"][idx]),
                    }
                )
    pos = [row for row in rows if row["label"] == 1]
    neg = [row for row in rows if row["label"] == 0]
    thresholds = [float(x) for x in np.percentile([row["reasoner_score"] for row in rows], [50, 60, 70, 80, 90, 95, 98])] if rows else []
    sweep = []
    for threshold in thresholds:
        tp = sum(1 for row in pos if row["reasoner_score"] >= threshold)
        fp = sum(1 for row in neg if row["reasoner_score"] >= threshold)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, len(pos))
        sweep.append({"threshold": threshold, "candidate_precision": precision, "candidate_recall": recall, "tp": tp, "fp": fp})
    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(args.source_summary),
        "fold_limit": int(args.fold_limit),
        "n_rows": len(rows),
        "n_positive_candidates": len(pos),
        "n_negative_candidates": len(neg),
        "positive_score_percentiles": np.percentile([row["reasoner_score"] for row in pos], [0, 25, 50, 75, 90, 100]).tolist() if pos else [],
        "negative_score_percentiles": np.percentile([row["reasoner_score"] for row in neg], [0, 25, 50, 75, 90, 100]).tolist() if neg else [],
        "sweep": sweep,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["n_rows", "n_positive_candidates", "n_negative_candidates", "positive_score_percentiles", "negative_score_percentiles", "sweep"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
