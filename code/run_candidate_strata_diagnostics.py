#!/usr/bin/env python3
"""Replay fold predictions and summarize JEPA candidate strata.

This script is diagnostic only. It does not evaluate held-out test data; it
uses the existing full-train CV summary to replay validation folds and asks
whether useful oracle-rescue candidates are separable from false positives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_fn_candidate_attribution import replay_fold
from cross_validate_segment_locator import aggregate_fold_metrics
from jepa_candidate_strata_diagnostics import classify_candidate_strata, summarize_strata_scores
from jepa_graph_reranker import graph_rerank_candidate_scores
from jepa_ot_event_set_matcher import score_ot_candidates
from train_segment_locator import _load_feature_names, load_signal_dataset


DEFAULT_DATA_DIR = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2"
DEFAULT_SOURCE_SUMMARY = "/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.summary.json"
DEFAULT_OUT = "/home/zzy/jepa_data/segment_train_full_event_v2_candidate_strata_diagnostics.json"
DEFAULT_OT_EVIDENCE_NAMES = "true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank"


Segment = tuple[int, int]


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    if len(clean) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-clean, kind="mergesort")
    ranks = np.empty(len(clean), dtype=np.int64)
    ranks[order] = np.arange(1, len(clean) + 1, dtype=np.int64)
    return ranks


def _parse_name_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _event_evidence(record, feature_names: list[str], evidence_names: list[str], reducer: str) -> np.ndarray:
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    channels = [name_to_idx[name] for name in evidence_names if name in name_to_idx]
    if not channels:
        raise ValueError(f"none of evidence channels were found: {evidence_names}")
    values = np.nan_to_num(record.signals[:, channels], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if reducer == "stack":
        return values
    if reducer == "max":
        return values.max(axis=1).astype(np.float32)
    if reducer == "mean":
        return values.mean(axis=1).astype(np.float32)
    raise ValueError(f"unsupported reducer: {reducer}")


def build_strata_rows_for_video(
    fold: int,
    video_idx: int,
    video_name: str,
    base_predictions: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    raw_scores: np.ndarray,
    active_scores: np.ndarray | None,
    ot_scores: np.ndarray | None,
    labels: np.ndarray,
    iou_threshold: float = 0.3,
    base_overlap_iou: float = 0.5,
) -> list[dict]:
    """Classify one video's candidates and attach score/rank diagnostics."""
    norm_candidates = [_normalise_segment(item) for item in candidates]
    raw = np.nan_to_num(np.asarray(raw_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    if len(norm_candidates) != len(raw):
        raise ValueError(f"candidate/raw-score mismatch: {len(norm_candidates)} vs {len(raw)}")
    active = raw if active_scores is None else np.nan_to_num(np.asarray(active_scores, dtype=np.float32).reshape(-1), nan=0.0)
    if len(active) != len(raw):
        raise ValueError(f"candidate/active-score mismatch: {len(norm_candidates)} vs {len(active)}")
    ot = None if ot_scores is None else np.nan_to_num(np.asarray(ot_scores, dtype=np.float32).reshape(-1), nan=0.0)
    if ot is not None and len(ot) != len(raw):
        raise ValueError(f"candidate/ot-score mismatch: {len(norm_candidates)} vs {len(ot)}")

    raw_ranks = _rank_desc(raw)
    active_ranks = _rank_desc(active)
    ot_ranks = None if ot is None else _rank_desc(ot)
    rows = classify_candidate_strata(
        [_normalise_segment(item) for item in base_predictions],
        norm_candidates,
        labels,
        iou_threshold=float(iou_threshold),
        base_overlap_iou=float(base_overlap_iou),
    )
    for idx, row in enumerate(rows):
        row.update(
            {
                "fold": int(fold),
                "video_idx": int(video_idx),
                "video": str(video_name),
                "raw_score": float(raw[idx]),
                "active_score": float(active[idx]),
                "raw_rank": int(raw_ranks[idx]),
                "active_rank": int(active_ranks[idx]),
                "candidate_length": int(norm_candidates[idx][1] - norm_candidates[idx][0] + 1),
            }
        )
        if ot is not None and ot_ranks is not None:
            row["ot_score"] = float(ot[idx])
            row["ot_rank"] = int(ot_ranks[idx])
    return rows


def summarize_all_strata(rows: list[dict]) -> dict:
    return summarize_strata_scores(
        rows,
        score_keys=[
            "raw_score",
            "active_score",
            "ot_score",
            "raw_rank",
            "active_rank",
            "ot_rank",
            "best_iou",
            "max_base_iou",
            "candidate_length",
        ],
    )


def _graph_adjusted_scores(fold_summary: dict, mainline_predictions, raw_candidates, raw_scores):
    graph_summary = fold_summary.get("fusion", {}).get("graph_reranker", {})
    graph_config = graph_summary.get("config", {})
    graph_enabled = bool(graph_summary.get("enabled") or graph_config.get("enabled"))
    if not graph_enabled:
        return raw_scores
    return [
        graph_rerank_candidate_scores(
            mainline,
            candidates,
            scores,
            support_iou=float(graph_config["support_iou"]),
            support_weight=float(graph_config["support_weight"]),
            support_count_weight=float(graph_config["support_count_weight"]),
            split_penalty=float(graph_config["split_penalty"]),
            mainline_overlap_penalty=float(graph_config["mainline_overlap_penalty"]),
        )
        for mainline, candidates, scores in zip(mainline_predictions, raw_candidates, raw_scores)
    ]


def _ot_scores_for_fold(
    records,
    val_idx: list[int],
    feature_names: list[str],
    evidence_names: list[str],
    reducer: str,
    base_predictions,
    raw_candidates,
    active_scores,
    evidence_threshold: float,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for record_idx, base, candidates, scores in zip(val_idx, base_predictions, raw_candidates, active_scores):
        evidence = _event_evidence(records[record_idx], feature_names, evidence_names, reducer)
        out.append(
            score_ot_candidates(
                base,
                candidates,
                scores,
                evidence,
                evidence_threshold=float(evidence_threshold),
                selector_weight=0.1,
                contrast_weight=0.5,
                active_fraction_weight=0.5,
                agreement_weight=0.5,
                island_coverage_weight=1.0,
                peak_alignment_weight=0.5,
                base_overlap_penalty=0.0,
                length_penalty=0.0,
                smooth_window=1,
                min_evidence_island_length=1,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-summary", default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-limit", "--folds", dest="fold_limit", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--base-overlap-iou", type=float, default=0.5)
    parser.add_argument("--ot-evidence-names", default=DEFAULT_OT_EVIDENCE_NAMES)
    parser.add_argument("--ot-reducer", choices=["max", "mean", "stack"], default="stack")
    parser.add_argument("--ot-evidence-threshold", type=float, default=0.55)
    args = parser.parse_args()

    source_path = Path(args.source_summary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = load_signal_dataset(args.data_dir)
    feature_names = [str(name) for name in _load_feature_names(args.data_dir)]
    evidence_names = _parse_name_list(args.ot_evidence_names)
    folds = list(source["folds"])
    if int(args.fold_limit) > 0:
        folds = folds[: int(args.fold_limit)]

    fold_results = []
    all_rows: list[dict] = []
    for fold_summary in folds:
        replay = replay_fold(
            records,
            feature_names,
            config=source["config"],
            fold_summary=fold_summary,
            device=str(args.device),
            iou_threshold=float(args.iou_threshold),
        )
        active_scores = _graph_adjusted_scores(
            fold_summary,
            replay["mainline_predictions"],
            replay["raw_candidates"],
            replay["raw_scores"],
        )
        ot_scores = _ot_scores_for_fold(
            records,
            replay["val_idx"],
            feature_names,
            evidence_names,
            str(args.ot_reducer),
            replay["mainline_predictions"],
            replay["raw_candidates"],
            active_scores,
            evidence_threshold=float(args.ot_evidence_threshold),
        )
        rows_for_fold: list[dict] = []
        for local_idx, record_idx in enumerate(replay["val_idx"]):
            rows = build_strata_rows_for_video(
                fold=int(replay["fold"]),
                video_idx=int(record_idx),
                video_name=records[record_idx].name,
                base_predictions=replay["mainline_predictions"][local_idx],
                candidates=replay["raw_candidates"][local_idx],
                raw_scores=replay["raw_scores"][local_idx],
                active_scores=active_scores[local_idx],
                ot_scores=ot_scores[local_idx],
                labels=replay["labels"][local_idx],
                iou_threshold=float(args.iou_threshold),
                base_overlap_iou=float(args.base_overlap_iou),
            )
            rows_for_fold.extend(rows)
        all_rows.extend(rows_for_fold)
        fold_results.append(
            {
                "fold": int(replay["fold"]),
                "validation": replay["validation"],
                "mainline_validation": replay["mainline_validation"],
                "selector_validation": replay["selector_validation"],
                "strata": summarize_all_strata(rows_for_fold),
                "n_candidates": int(len(rows_for_fold)),
            }
        )

    result = {
        "data_dir": str(args.data_dir),
        "source_summary": str(source_path),
        "fold_limit": int(args.fold_limit),
        "iou_threshold": float(args.iou_threshold),
        "base_overlap_iou": float(args.base_overlap_iou),
        "device": str(args.device),
        "ot_evidence_names": evidence_names,
        "ot_reducer": str(args.ot_reducer),
        "ot_evidence_threshold": float(args.ot_evidence_threshold),
        "aggregate": aggregate_fold_metrics([{"validation": fold["validation"]} for fold in fold_results]),
        "source_aggregate": source.get("aggregate"),
        "strata_summary": summarize_all_strata(all_rows),
        "folds": fold_results,
        "rows": all_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate_segment": result["aggregate"]["segment"],
                "strata_summary": result["strata_summary"],
                "output": str(out_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
