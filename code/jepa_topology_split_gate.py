#!/usr/bin/env python3
"""Learned gate for JEPA topology split replacement.

The oracle diagnostics show that many remaining misses are solvable by
replacing a broad prediction with a set of compact JEPA child proposals, but
hand-written evidence thresholds are too brittle. This module learns a
parent-level gate from training folds: should this broad parent be replaced by
its JEPA child set?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from selector_fusion import evaluate_fused_predictions
from train_proposal_calibrator import interval_iou
from train_proposal_set_selector import fit_classifier
from train_segment_locator import contiguous_segments


Segment = tuple[int, int]


@dataclass(frozen=True)
class SplitGateRecord:
    video_idx: int
    parent: Segment
    children: list[Segment]
    features: np.ndarray
    label: int
    base_tp: int
    split_tp: int
    base_fp: int
    split_fp: int


def _normalise_segment(segment: tuple[int, int]) -> Segment:
    start, end = int(segment[0]), int(segment[1])
    return (start, end) if start <= end else (end, start)


def _segment_length(segment: tuple[int, int]) -> int:
    start, end = _normalise_segment(segment)
    return max(1, end - start + 1)


def _intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    a = _normalise_segment(a)
    b = _normalise_segment(b)
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _coverage(child: tuple[int, int], parent: tuple[int, int]) -> float:
    return float(_intersection(child, parent) / max(1, _segment_length(child)))


def _rank_desc(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def _matrix(evidence: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(evidence, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values.astype(np.float32)


def _segment_evidence_mean(evidence: np.ndarray, segment: tuple[int, int]) -> float:
    values = _matrix(evidence)
    if len(values) == 0:
        return 0.0
    start, end = _normalise_segment(segment)
    start = max(0, min(start, len(values) - 1))
    end = max(start, min(end, len(values) - 1))
    return float(values[start : end + 1].mean())


def _greedy_segment_metrics(predictions: list[Segment], labels: np.ndarray, iou_threshold: float) -> tuple[int, int, int]:
    gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(labels, dtype=np.int64))]
    matched: set[int] = set()
    tp = fp = 0
    for pred in predictions:
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
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt_segments) - len(matched)


def _candidate_key(candidate: Segment, score: float, rank: int, parent: Segment, evidence: np.ndarray) -> tuple[float, float, float, float]:
    evidence_mean = _segment_evidence_mean(evidence, candidate)
    length_ratio = _segment_length(candidate) / max(1, _segment_length(parent))
    return (float(score), evidence_mean, -float(rank), -length_ratio)


def _select_child_set(
    parent: Segment,
    candidates: list[Segment],
    scores: np.ndarray,
    ranks: np.ndarray,
    evidence: np.ndarray,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_children_per_parent: int,
) -> list[Segment]:
    items: list[tuple[Segment, tuple[float, float, float, float]]] = []
    parent_length = _segment_length(parent)
    for candidate, score, rank in zip(candidates, scores, ranks):
        child = _normalise_segment(candidate)
        child_length = _segment_length(child)
        if child == parent or child_length >= parent_length:
            continue
        if _coverage(child, parent) < float(min_child_parent_coverage):
            continue
        if child_length / max(1, parent_length) > float(max_child_parent_ratio):
            continue
        items.append((child, _candidate_key(child, float(score), int(rank), parent, evidence)))
    items.sort(key=lambda item: item[1], reverse=True)
    selected: list[Segment] = []
    for child, _ in items:
        if any(interval_iou(child, kept) > 0.3 for kept in selected):
            continue
        selected.append(child)
        if len(selected) >= int(max_children_per_parent):
            break
    return sorted(selected)


def _features_for_parent(
    parent: Segment,
    children: list[Segment],
    candidates: list[Segment],
    scores: np.ndarray,
    ranks: np.ndarray,
    evidence: np.ndarray,
) -> np.ndarray:
    parent_len = _segment_length(parent)
    child_scores = []
    child_ranks = []
    child_lengths = []
    child_evidence = []
    child_coverages = []
    for child in children:
        matching = [idx for idx, candidate in enumerate(candidates) if candidate == child]
        if matching:
            idx = int(matching[0])
            child_scores.append(float(scores[idx]))
            child_ranks.append(float(ranks[idx]))
        else:
            child_scores.append(0.0)
            child_ranks.append(float(len(candidates) + 1))
        child_lengths.append(float(_segment_length(child)))
        child_evidence.append(float(_segment_evidence_mean(evidence, child)))
        child_coverages.append(float(_coverage(child, parent)))
    parent_evidence = _segment_evidence_mean(evidence, parent)
    child_count = len(children)
    gaps = []
    for left, right in zip(children, children[1:]):
        gaps.append(float(max(0, right[0] - left[1] - 1)))
    values = [
        float(parent_len),
        float(np.log1p(parent_len)),
        float(child_count),
        float(sum(child_lengths) / max(1.0, parent_len)),
        float(np.mean(child_lengths) / max(1.0, parent_len)) if child_lengths else 0.0,
        float(np.max(child_lengths) / max(1.0, parent_len)) if child_lengths else 0.0,
        float(np.mean(gaps) / max(1.0, parent_len)) if gaps else 0.0,
        float(np.max(gaps) / max(1.0, parent_len)) if gaps else 0.0,
        float(parent_evidence),
        float(np.mean(child_evidence)) if child_evidence else 0.0,
        float(np.max(child_evidence)) if child_evidence else 0.0,
        float(np.mean(child_evidence) - parent_evidence) if child_evidence else 0.0,
        float(np.mean(child_scores)) if child_scores else 0.0,
        float(np.max(child_scores)) if child_scores else 0.0,
        float(np.min(child_scores)) if child_scores else 0.0,
        float(np.mean(child_ranks)) if child_ranks else 0.0,
        float(np.min(child_ranks)) if child_ranks else 0.0,
        float(np.mean(child_coverages)) if child_coverages else 0.0,
        float(len(candidates)),
    ]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_split_gate_records(
    base_predictions: list[list[tuple[int, int]]],
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    iou_threshold: float,
    parent_min_length: int,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_children_per_parent: int,
) -> list[SplitGateRecord]:
    if not (
        len(base_predictions)
        == len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
        == len(labels)
    ):
        raise ValueError("base/candidate/score/evidence/label video counts must match")
    records: list[SplitGateRecord] = []
    for video_idx, (base_raw, candidates_raw, scores_raw, evidence, label_array) in enumerate(
        zip(base_predictions, candidate_predictions, candidate_scores, evidence_arrays, labels)
    ):
        base = [_normalise_segment(item) for item in base_raw]
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        ranks = _rank_desc(scores)
        for parent in base:
            if _segment_length(parent) < int(parent_min_length):
                continue
            children = _select_child_set(
                parent,
                candidates,
                scores,
                ranks,
                evidence,
                min_child_parent_coverage=float(min_child_parent_coverage),
                max_child_parent_ratio=float(max_child_parent_ratio),
                max_children_per_parent=int(max_children_per_parent),
            )
            if len(children) < 2:
                continue
            base_tp, base_fp, _ = _greedy_segment_metrics([parent], label_array, iou_threshold=float(iou_threshold))
            split_tp, split_fp, _ = _greedy_segment_metrics(children, label_array, iou_threshold=float(iou_threshold))
            label = int(split_tp > base_tp and split_fp <= base_fp)
            records.append(
                SplitGateRecord(
                    video_idx=int(video_idx),
                    parent=parent,
                    children=children,
                    features=_features_for_parent(parent, children, candidates, scores, ranks, evidence),
                    label=label,
                    base_tp=int(base_tp),
                    split_tp=int(split_tp),
                    base_fp=int(base_fp),
                    split_fp=int(split_fp),
                )
            )
    return records


def build_synthetic_split_gate_records(
    candidate_predictions: list[list[tuple[int, int]]],
    candidate_scores: list[np.ndarray],
    evidence_arrays: list[np.ndarray],
    labels: list[np.ndarray],
    iou_threshold: float,
    max_gt_gap: int,
    parent_pad: int,
    min_child_parent_coverage: float,
    max_child_parent_ratio: float,
    max_children_per_parent: int,
) -> list[SplitGateRecord]:
    """Create positive split-gate samples by merging nearby training GT events."""
    if not (
        len(candidate_predictions)
        == len(candidate_scores)
        == len(evidence_arrays)
        == len(labels)
    ):
        raise ValueError("candidate/score/evidence/label video counts must match")
    records: list[SplitGateRecord] = []
    for video_idx, (candidates_raw, scores_raw, evidence, label_array) in enumerate(
        zip(candidate_predictions, candidate_scores, evidence_arrays, labels)
    ):
        gt_segments = [_normalise_segment(item) for item in contiguous_segments(np.asarray(label_array, dtype=np.int64))]
        if len(gt_segments) < 2:
            continue
        candidates = [_normalise_segment(item) for item in candidates_raw]
        scores = np.nan_to_num(np.asarray(scores_raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        if len(candidates) != len(scores):
            raise ValueError(f"candidate/score count mismatch: {len(candidates)} vs {len(scores)}")
        ranks = _rank_desc(scores)
        for left, right in zip(gt_segments, gt_segments[1:]):
            gap = right[0] - left[1] - 1
            if gap < 0 or gap > int(max_gt_gap):
                continue
            parent = (
                max(0, left[0] - int(parent_pad)),
                min(len(label_array) - 1, right[1] + int(parent_pad)),
            )
            children = _select_child_set(
                parent,
                candidates,
                scores,
                ranks,
                evidence,
                min_child_parent_coverage=float(min_child_parent_coverage),
                max_child_parent_ratio=float(max_child_parent_ratio),
                max_children_per_parent=int(max_children_per_parent),
            )
            if len(children) < 2:
                continue
            base_tp, base_fp, _ = _greedy_segment_metrics([parent], label_array, iou_threshold=float(iou_threshold))
            split_tp, split_fp, _ = _greedy_segment_metrics(children, label_array, iou_threshold=float(iou_threshold))
            if split_tp <= base_tp or split_fp > base_fp:
                continue
            records.append(
                SplitGateRecord(
                    video_idx=int(video_idx),
                    parent=parent,
                    children=children,
                    features=_features_for_parent(parent, children, candidates, scores, ranks, evidence),
                    label=1,
                    base_tp=int(base_tp),
                    split_tp=int(split_tp),
                    base_fp=int(base_fp),
                    split_fp=int(split_fp),
                )
            )
    return records


def _fit_gate_model(x: np.ndarray, y: np.ndarray, model_name: str, seed: int):
    if model_name == "logreg" or len(np.unique(y)) < 2:
        return fit_classifier(x, y, seed=seed, model_name="logreg")
    return fit_classifier(x, y, seed=seed, model_name=model_name)


def _predict_gate_prob(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    return np.asarray(probs[:, 1], dtype=np.float32)


def _apply_gate_records(
    base_predictions: list[list[tuple[int, int]]],
    records: list[SplitGateRecord],
    probabilities: np.ndarray,
    threshold: float,
    max_replaced_parents_per_video: int,
) -> list[list[Segment]]:
    out = [[_normalise_segment(item) for item in video] for video in base_predictions]
    by_video: dict[int, list[tuple[SplitGateRecord, float]]] = {}
    for record, probability in zip(records, probabilities):
        if float(probability) < float(threshold):
            continue
        by_video.setdefault(record.video_idx, []).append((record, float(probability)))
    for video_idx, items in by_video.items():
        items.sort(key=lambda item: item[1], reverse=True)
        if max_replaced_parents_per_video > 0:
            items = items[: int(max_replaced_parents_per_video)]
        replace_parents = {record.parent for record, _ in items}
        children = [child for record, _ in items for child in record.children]
        out[video_idx] = sorted(set([segment for segment in out[video_idx] if segment not in replace_parents] + children))
    return out


def select_topology_split_gate_params(
    train_base_predictions: list[list[tuple[int, int]]],
    train_candidate_predictions: list[list[tuple[int, int]]],
    train_candidate_scores: list[np.ndarray],
    train_evidence_arrays: list[np.ndarray],
    train_labels: list[np.ndarray],
    val_base_predictions: list[list[tuple[int, int]]],
    val_candidate_predictions: list[list[tuple[int, int]]],
    val_candidate_scores: list[np.ndarray],
    val_evidence_arrays: list[np.ndarray],
    val_labels: list[np.ndarray],
    model_name: str,
    seed: int,
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
    parent_min_lengths: Iterable[int] = (12, 24, 48),
    min_child_parent_coverages: Iterable[float] = (0.5, 0.7),
    max_child_parent_ratios: Iterable[float] = (0.5, 0.7),
    max_children_per_parents: Iterable[int] = (2, 3),
    max_replaced_parents_per_videos: Iterable[int] = (1,),
    max_fp_increase: int | None = 0,
    use_synthetic_positives: bool = False,
    synthetic_max_gt_gap: int = 24,
    synthetic_parent_pad: int = 0,
    iou_threshold: float = 0.3,
) -> tuple[dict, dict, list[list[Segment]]]:
    base = [[_normalise_segment(item) for item in video] for video in val_base_predictions]
    base_metrics = evaluate_fused_predictions(base, val_labels, iou_threshold=iou_threshold)
    best_config: dict = {"enabled": False, "base_metrics": base_metrics}
    best_metrics = base_metrics
    best_predictions = base
    best_key = (
        float(base_metrics["segment"]["f1"]),
        float(base_metrics["segment"]["precision"]),
        float(base_metrics["segment"]["recall"]),
        float(base_metrics["frame"]["f1"]),
        0.0,
    )
    for parent_min_length in parent_min_lengths:
        for min_child_parent_coverage in min_child_parent_coverages:
            for max_child_parent_ratio in max_child_parent_ratios:
                for max_children_per_parent in max_children_per_parents:
                    train_records = build_split_gate_records(
                        train_base_predictions,
                        train_candidate_predictions,
                        train_candidate_scores,
                        train_evidence_arrays,
                        train_labels,
                        iou_threshold=float(iou_threshold),
                        parent_min_length=int(parent_min_length),
                        min_child_parent_coverage=float(min_child_parent_coverage),
                        max_child_parent_ratio=float(max_child_parent_ratio),
                        max_children_per_parent=int(max_children_per_parent),
                    )
                    if use_synthetic_positives:
                        train_records = train_records + build_synthetic_split_gate_records(
                            train_candidate_predictions,
                            train_candidate_scores,
                            train_evidence_arrays,
                            train_labels,
                            iou_threshold=float(iou_threshold),
                            max_gt_gap=int(synthetic_max_gt_gap),
                            parent_pad=int(synthetic_parent_pad),
                            min_child_parent_coverage=float(min_child_parent_coverage),
                            max_child_parent_ratio=float(max_child_parent_ratio),
                            max_children_per_parent=int(max_children_per_parent),
                        )
                    val_records = build_split_gate_records(
                        val_base_predictions,
                        val_candidate_predictions,
                        val_candidate_scores,
                        val_evidence_arrays,
                        val_labels,
                        iou_threshold=float(iou_threshold),
                        parent_min_length=int(parent_min_length),
                        min_child_parent_coverage=float(min_child_parent_coverage),
                        max_child_parent_ratio=float(max_child_parent_ratio),
                        max_children_per_parent=int(max_children_per_parent),
                    )
                    if not train_records or not val_records:
                        continue
                    x_train = np.stack([record.features for record in train_records])
                    y_train = np.asarray([record.label for record in train_records], dtype=np.int64)
                    if len(np.unique(y_train)) < 2:
                        continue
                    model = _fit_gate_model(x_train, y_train, model_name=model_name, seed=int(seed))
                    x_val = np.stack([record.features for record in val_records])
                    probabilities = _predict_gate_prob(model, x_val)
                    for threshold in thresholds:
                        for max_replaced_parents_per_video in max_replaced_parents_per_videos:
                            predictions = _apply_gate_records(
                                base,
                                val_records,
                                probabilities,
                                threshold=float(threshold),
                                max_replaced_parents_per_video=int(max_replaced_parents_per_video),
                            )
                            metrics = evaluate_fused_predictions(predictions, val_labels, iou_threshold=iou_threshold)
                            if max_fp_increase is not None:
                                fp_delta = int(metrics["segment"]["fp"]) - int(base_metrics["segment"]["fp"])
                                if fp_delta > int(max_fp_increase):
                                    continue
                            changed = sum(int(sorted(a) != sorted(b)) for a, b in zip(base, predictions))
                            key = (
                                float(metrics["segment"]["f1"]),
                                float(metrics["segment"]["precision"]),
                                float(metrics["segment"]["recall"]),
                                float(metrics["frame"]["f1"]),
                                -float(changed),
                            )
                            if key > best_key:
                                best_key = key
                                best_metrics = metrics
                                best_predictions = predictions
                                best_config = {
                                    "enabled": True,
                                    "model": str(model_name),
                                    "threshold": float(threshold),
                                    "parent_min_length": int(parent_min_length),
                                    "min_child_parent_coverage": float(min_child_parent_coverage),
                                    "max_child_parent_ratio": float(max_child_parent_ratio),
                                    "max_children_per_parent": int(max_children_per_parent),
                                    "max_replaced_parents_per_video": int(max_replaced_parents_per_video),
                                    "max_fp_increase": max_fp_increase,
                                    "use_synthetic_positives": bool(use_synthetic_positives),
                                    "synthetic_max_gt_gap": int(synthetic_max_gt_gap),
                                    "synthetic_parent_pad": int(synthetic_parent_pad),
                                    "n_train_records": int(len(train_records)),
                                    "n_positive_train_records": int(y_train.sum()),
                                    "n_val_records": int(len(val_records)),
                                    "changed_videos": int(changed),
                                    "base_metrics": base_metrics,
                                }
    return best_config, best_metrics, best_predictions
