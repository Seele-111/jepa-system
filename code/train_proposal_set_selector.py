#!/usr/bin/env python3
"""
Train a multi-channel JEPA proposal set selector.

This is still binary error-segment localization only. The module generates
temporal proposals from complementary V/I/dual-JEPA evidence channels, scores
each proposal, then selects a non-overlapping event set so one long proposal
does not swallow several shorter error events.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_proposal_calibrator import (
    NumpyLogisticClassifier,
    evaluate_segment_predictions,
    interval_iou,
)
from train_segment_locator import VideoRecord, contiguous_segments, load_signal_dataset, train_val_split


DEFAULT_CHANNEL_NAMES = [
    "true_vjepa_raw_rank",
    "true_ijepa_dense_raw_rank",
    "dual_jepa_composite_rank",
    "true_vjepa_raw",
    "true_ijepa_dense_raw",
    "dual_jepa_composite",
    "vi_max",
    "vi_product",
    "composite_times_vi_agreement",
    "true_vjepa_raw_abs_delta",
    "true_ijepa_dense_raw_abs_delta",
    "dual_jepa_composite_abs_delta",
]


@dataclass
class CandidateRecord:
    video_idx: int
    segment: tuple[int, int]
    features: np.ndarray
    label: int
    best_iou: float


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_name_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _segments_from_scores(scores: np.ndarray, threshold: float, min_gap: int, min_length: int) -> list[tuple[int, int]]:
    binary = (np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0) >= float(threshold)).astype(np.int64)
    segments = contiguous_segments(binary)
    if int(min_gap) > 0 and len(segments) > 1:
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end - 1 <= int(min_gap):
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        segments = merged
    return [(int(start), int(end)) for start, end in segments if end - start + 1 >= int(min_length)]


def generate_multichannel_candidates(
    signals: np.ndarray,
    feature_names: list[str],
    channel_names: Iterable[str],
    thresholds: Iterable[float],
    min_gaps: Iterable[int] = (0, 1, 2, 4),
    min_lengths: Iterable[int] = (1, 2, 4),
) -> list[tuple[int, int]]:
    signals = np.asarray(signals, dtype=np.float32)
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    candidates: set[tuple[int, int]] = set()
    for name in channel_names:
        if name not in name_to_idx:
            continue
        values = signals[:, name_to_idx[name]]
        finite_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        local_thresholds = list(thresholds)
        if not local_thresholds:
            local_thresholds = [float(x) for x in np.percentile(finite_values, [50, 60, 70, 80, 90, 95])]
        for threshold in local_thresholds:
            for min_gap in min_gaps:
                for min_length in min_lengths:
                    candidates.update(_segments_from_scores(finite_values, threshold, int(min_gap), int(min_length)))
    return sorted(candidates)


def _topk_mean(values: np.ndarray, fraction: float = 0.25) -> float:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return 0.0
    k = max(1, int(round(len(values) * float(fraction))))
    return float(np.sort(values)[-k:].mean())


def _outside_values(values: np.ndarray, start: int, end: int) -> np.ndarray:
    length = end - start + 1
    left = values[max(0, start - length) : start]
    right = values[end + 1 : min(len(values), end + 1 + length)]
    if len(left) == 0 and len(right) == 0:
        return np.asarray([], dtype=np.float32)
    return np.concatenate([left, right])


def extract_event_topology_features(values: np.ndarray) -> dict[str, float]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
        return {"island_count": 0, "compactness": 0.0, "valley_depth": 0.0, "active_fraction": 0.0}

    finite_min = float(values.min())
    finite_max = float(values.max())
    if finite_max <= finite_min + 1e-6:
        return {"island_count": 0, "compactness": 0.0, "valley_depth": 0.0, "active_fraction": 0.0}

    median = float(np.median(values))
    high_quantile = float(np.percentile(values, 75))
    threshold = max(high_quantile, median + 0.25 * (finite_max - median))
    active = (values >= threshold).astype(np.int64)
    islands = contiguous_segments(active)
    island_count = len(islands)
    active_frames = int(active.sum())
    longest = max((end - start + 1 for start, end in islands), default=0)
    compactness = float(longest / max(1, active_frames))
    active_fraction = float(active_frames / max(1, len(values)))

    valley_depth = 0.0
    if island_count >= 2:
        depths = []
        for left, right in zip(islands, islands[1:]):
            gap_start = left[1] + 1
            gap_end = right[0] - 1
            if gap_start > gap_end:
                continue
            left_peak = float(values[left[0] : left[1] + 1].max())
            right_peak = float(values[right[0] : right[1] + 1].max())
            valley = float(values[gap_start : gap_end + 1].min())
            depths.append(max(0.0, min(left_peak, right_peak) - valley))
        valley_depth = float(max(depths, default=0.0))

    return {
        "island_count": int(island_count),
        "compactness": compactness,
        "valley_depth": valley_depth,
        "active_fraction": active_fraction,
    }


def extract_set_proposal_features(
    signals: np.ndarray,
    segment: tuple[int, int],
    channel_indices: Iterable[int],
) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float32)
    n = len(signals)
    start, end = segment
    start = max(0, min(int(start), n - 1))
    end = max(start, min(int(end), n - 1))
    duration = end - start + 1
    features: list[float] = [
        float(duration),
        float(math.log1p(duration)),
        float(start / max(1, n - 1)),
        float(end / max(1, n - 1)),
        float((start + end) / max(1, 2 * (n - 1))),
    ]
    per_channel_means = []
    per_channel_maxes = []
    per_channel_islands = []
    per_channel_compactness = []
    per_channel_valleys = []
    for idx in channel_indices:
        values = np.nan_to_num(signals[:, int(idx)], nan=0.0, posinf=0.0, neginf=0.0)
        inside = values[start : end + 1]
        outside = _outside_values(values, start, end)
        contrast = float(inside.mean() - outside.mean()) if len(outside) else 0.0
        slope = float(inside[-1] - inside[0]) / max(1, duration - 1)
        topology = extract_event_topology_features(inside)
        per_channel_means.append(float(inside.mean()))
        per_channel_maxes.append(float(inside.max()))
        per_channel_islands.append(float(topology["island_count"]))
        per_channel_compactness.append(float(topology["compactness"]))
        per_channel_valleys.append(float(topology["valley_depth"]))
        features.extend(
            [
                float(inside.mean()),
                float(inside.max()),
                _topk_mean(inside),
                float(inside.std()),
                float(inside.max() - inside.min()),
                slope,
                contrast,
                float(topology["island_count"]),
                float(topology["compactness"]),
                float(topology["valley_depth"]),
                float(topology["active_fraction"]),
            ]
        )
    if per_channel_means:
        features.extend(
            [
                float(np.mean(per_channel_means)),
                float(np.max(per_channel_means)),
                float(np.std(per_channel_means)),
                float(np.mean(per_channel_maxes)),
                float(np.max(per_channel_maxes)),
                float(np.mean(per_channel_islands)),
                float(np.max(per_channel_islands)),
                float(np.mean(per_channel_compactness)),
                float(np.min(per_channel_compactness)),
                float(np.mean(per_channel_valleys)),
                float(np.max(per_channel_valleys)),
            ]
        )
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def label_candidates(
    candidates: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt_segments = contiguous_segments(np.asarray(labels, dtype=np.int64))
    y = np.zeros(len(candidates), dtype=np.int64)
    best = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        if gt_segments:
            best[idx] = max(interval_iou(candidate, gt) for gt in gt_segments)
        y[idx] = int(best[idx] >= float(iou_threshold))
    return y, best


def build_candidate_records(
    records: list[VideoRecord],
    indices: Iterable[int],
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    iou_threshold: float,
) -> list[CandidateRecord]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    candidates_out: list[CandidateRecord] = []
    for video_idx in indices:
        record = records[int(video_idx)]
        candidates = generate_multichannel_candidates(
            record.signals,
            feature_names=feature_names,
            channel_names=channel_names,
            thresholds=thresholds,
            min_gaps=min_gaps,
            min_lengths=min_lengths,
        )
        labels, best_iou = label_candidates(candidates, record.labels, iou_threshold=iou_threshold)
        for segment, label, iou in zip(candidates, labels, best_iou):
            candidates_out.append(
                CandidateRecord(
                    video_idx=int(video_idx),
                    segment=segment,
                    features=extract_set_proposal_features(record.signals, segment, channel_indices),
                    label=int(label),
                    best_iou=float(iou),
                )
            )
    return candidates_out


def select_weighted_proposal_set(
    candidates: list[tuple[int, int]],
    scores: Iterable[float],
    overlap_iou: float = 0.0,
    length_penalty: float = 0.0,
) -> list[tuple[int, int]]:
    del overlap_iou  # Set selection is exact for non-overlapping temporal events.
    score_values = [float(score) for score in scores]
    items = []
    for idx, (segment, score) in enumerate(zip(candidates, score_values)):
        start, end = int(segment[0]), int(segment[1])
        weight = score - float(length_penalty) * math.log1p(max(1, end - start + 1))
        if weight <= 0.0:
            continue
        items.append((start, end, weight, idx))
    if not items:
        return []
    items.sort(key=lambda item: (item[1], item[0], item[3]))
    ends = [item[1] for item in items]
    previous = []
    for item in items:
        start = item[0]
        lo, hi = 0, len(ends)
        while lo < hi:
            mid = (lo + hi) // 2
            if ends[mid] < start:
                lo = mid + 1
            else:
                hi = mid
        previous.append(lo - 1)

    n = len(items)
    dp = np.zeros(n + 1, dtype=np.float64)
    take = np.zeros(n, dtype=bool)
    for idx, item in enumerate(items, start=1):
        take_score = item[2] + dp[previous[idx - 1] + 1]
        skip_score = dp[idx - 1]
        if take_score > skip_score:
            dp[idx] = take_score
            take[idx - 1] = True
        else:
            dp[idx] = skip_score

    selected: list[tuple[int, int]] = []
    idx = n
    while idx > 0:
        if take[idx - 1] and items[idx - 1][2] + dp[previous[idx - 1] + 1] >= dp[idx - 1]:
            selected.append((items[idx - 1][0], items[idx - 1][1]))
            idx = previous[idx - 1] + 1
        else:
            idx -= 1
    return sorted(selected)


def candidate_quality_targets(best_iou: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(best_iou, dtype=np.float32), 0.0, 1.0)
    values = np.where(values >= 0.3, values, 0.0)
    return np.square(values).astype(np.float32)


class QualityModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            quality = self.model.predict(x)
        else:
            quality = self.model.predict_proba(x)[:, 1]
        quality = np.clip(np.asarray(quality, dtype=np.float32), 0.0, 1.0)
        return np.stack([1.0 - quality, quality], axis=1)


QualityModelWrapper.__module__ = "train_proposal_set_selector"


class _ProposalMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchMLPClassifier:
    def __init__(self, model: _ProposalMLP, mean: np.ndarray, std: np.ndarray, device: str):
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.device = str(device)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        normed = ((values - self.mean) / self.std).astype(np.float32)
        self.model.eval()
        probs_out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(normed), 4096):
                batch = torch.from_numpy(normed[start : start + 4096]).to(self.device)
                probs = torch.sigmoid(self.model(batch)).detach().cpu().numpy().astype(np.float32)
                probs_out.append(probs)
        if not probs_out:
            return np.zeros((0, 2), dtype=np.float32)
        positive = np.concatenate(probs_out, axis=0)
        return np.stack([1.0 - positive, positive], axis=1)


TorchMLPClassifier.__module__ = "train_proposal_set_selector"


def _fit_torch_mlp_classifier(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    hidden: int,
    lr: float,
) -> TorchMLPClassifier:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0).astype(np.float32), 1e-4)
    x_norm = ((x - mean) / std).astype(np.float32)
    device_name = str(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = _ProposalMLP(in_features=x.shape[1], hidden=int(hidden), dropout=0.1).to(device_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(len(y) - y.sum()))
    pos_weight = torch.tensor(min(50.0, neg / pos), dtype=torch.float32, device=device_name)
    rng = np.random.default_rng(seed)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    no_improve = 0
    patience = max(8, min(20, int(epochs) // 4))
    for _ in range(int(epochs)):
        order = rng.permutation(len(x_norm))
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = torch.from_numpy(x_norm[idx]).to(device_name)
            yb = torch.from_numpy(y[idx]).to(device_name)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses) if losses else 0.0)
        if mean_loss + 1e-5 < best_loss:
            best_loss = mean_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    return TorchMLPClassifier(model, mean=mean, std=std, device=device_name)


def fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    model_name: str,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    if model_name == "mlp":
        return _fit_torch_mlp_classifier(
            x,
            y,
            seed=seed,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            hidden=96,
            lr=1e-3,
        )
    if model_name == "logreg":
        return NumpyLogisticClassifier(lr=0.03, steps=1000, l2=1e-3).fit(x, y)
    try:
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    except Exception:
        return NumpyLogisticClassifier(lr=0.03, steps=1000, l2=1e-3).fit(x, y)

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        pos = max(1, int(y.sum()))
        neg = max(1, int(len(y) - y.sum()))
        sample_weight = np.where(y > 0, neg / pos, 1.0)
        clf = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        )
        clf.fit(x, y, sample_weight=sample_weight)
        return clf
    clf.fit(x, y)
    return clf


def fit_quality_model(x: np.ndarray, best_iou: np.ndarray, seed: int, model_name: str):
    target = candidate_quality_targets(best_iou)
    if model_name == "logreg":
        return NumpyLogisticClassifier(lr=0.03, steps=1200, l2=1e-3).fit(x, target)
    try:
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    except Exception:
        return NumpyLogisticClassifier(lr=0.03, steps=1200, l2=1e-3).fit(x, target)

    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        )
    model.fit(x, target)
    return QualityModelWrapper(model)


def score_video_candidates(
    classifier,
    record: VideoRecord,
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    candidates = generate_multichannel_candidates(
        record.signals,
        feature_names=feature_names,
        channel_names=channel_names,
        thresholds=thresholds,
        min_gaps=min_gaps,
        min_lengths=min_lengths,
    )
    if not candidates:
        return [], np.zeros(0, dtype=np.float32)
    x = np.stack([extract_set_proposal_features(record.signals, segment, channel_indices) for segment in candidates])
    probabilities = classifier.predict_proba(x)[:, 1].astype(np.float32)
    return candidates, probabilities


def predict_video_segments(
    classifier,
    record: VideoRecord,
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    prob_threshold: float,
    length_penalty: float,
) -> list[tuple[int, int]]:
    candidates, probabilities = score_video_candidates(classifier, record, feature_names, channel_names, thresholds, min_gaps, min_lengths)
    filtered_candidates = []
    filtered_scores = []
    for segment, probability in zip(candidates, probabilities):
        if float(probability) >= float(prob_threshold):
            filtered_candidates.append(segment)
            filtered_scores.append(float(probability))
    return select_weighted_proposal_set(filtered_candidates, filtered_scores, length_penalty=float(length_penalty))


def select_inference_params(
    classifier,
    records: list[VideoRecord],
    val_idx: list[int],
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    iou_threshold: float,
) -> tuple[dict, dict]:
    cache = {
        idx: score_video_candidates(classifier, records[idx], feature_names, channel_names, thresholds, min_gaps, min_lengths)
        for idx in val_idx
    }
    best_params = None
    best_metrics = None
    best_key = None
    for prob_threshold in np.linspace(0.15, 0.9, 16):
        for length_penalty in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]:
            predictions = []
            for idx in val_idx:
                candidates, probabilities = cache[idx]
                filtered_candidates = []
                filtered_scores = []
                for segment, probability in zip(candidates, probabilities):
                    if float(probability) >= float(prob_threshold):
                        filtered_candidates.append(segment)
                        filtered_scores.append(float(probability))
                predictions.append(select_weighted_proposal_set(filtered_candidates, filtered_scores, length_penalty=float(length_penalty)))
            metrics = evaluate_segment_predictions(predictions, [records[idx].labels for idx in val_idx], iou_threshold=iou_threshold)
            key = (float(metrics["f1"]), float(metrics["precision"]), float(metrics["recall"]), -float(metrics["fp"]))
            if best_key is None or key > best_key:
                best_key = key
                best_params = {"prob_threshold": float(prob_threshold), "length_penalty": float(length_penalty)}
                best_metrics = metrics
    assert best_params is not None and best_metrics is not None
    return best_params, best_metrics


def _load_feature_names(data_dir: str | Path) -> list[str]:
    data = json.loads((Path(data_dir) / "summary.json").read_text(encoding="utf-8"))
    return [str(name) for name in data.get("feature_names", [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="/home/zzy/jepa_data/proposal_set_selector.pkl")
    parser.add_argument("--summary", default="")
    parser.add_argument("--model", choices=["gbdt", "rf", "extratrees", "logreg"], default="extratrees")
    parser.add_argument("--target", choices=["binary", "quality"], default="binary")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--min-gaps", default="0,1,2,4,8")
    parser.add_argument("--min-lengths", default="1,2,4,8")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.channel_names) if name in feature_names]
    if not channel_names:
        raise ValueError("no selected channel names were found in dataset summary")
    thresholds = _parse_float_list(args.thresholds)
    min_gaps = _parse_int_list(args.min_gaps)
    min_lengths = _parse_int_list(args.min_lengths)
    train_idx, val_idx = train_val_split(records, args.val_ratio, args.seed)
    candidates = build_candidate_records(
        records,
        train_idx,
        feature_names=feature_names,
        channel_names=channel_names,
        thresholds=thresholds,
        min_gaps=min_gaps,
        min_lengths=min_lengths,
        iou_threshold=args.iou_threshold,
    )
    if not candidates:
        raise RuntimeError("no training candidates generated")
    x_train = np.stack([item.features for item in candidates])
    y_train = np.asarray([item.label for item in candidates], dtype=np.int64)
    best_iou_train = np.asarray([item.best_iou for item in candidates], dtype=np.float32)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"candidate labels need both classes, got positives={int(y_train.sum())}/{len(y_train)}")

    if args.target == "quality":
        classifier = fit_quality_model(x_train, best_iou_train, args.seed, args.model)
    else:
        classifier = fit_classifier(x_train, y_train, args.seed, args.model)
    params, val_metrics = select_inference_params(
        classifier,
        records,
        val_idx,
        feature_names,
        channel_names,
        thresholds,
        min_gaps,
        min_lengths,
        args.iou_threshold,
    )
    train_predictions = [
        predict_video_segments(
            classifier,
            records[idx],
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
            prob_threshold=params["prob_threshold"],
            length_penalty=params["length_penalty"],
        )
        for idx in train_idx
    ]
    train_metrics = evaluate_segment_predictions(train_predictions, [records[idx].labels for idx in train_idx], args.iou_threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(
            {
                "classifier": classifier,
                "feature_names": feature_names,
                "channel_names": channel_names,
                "thresholds": thresholds,
                "min_gaps": min_gaps,
                "min_lengths": min_lengths,
                "params": params,
                "iou_threshold": args.iou_threshold,
                "target": args.target,
                "task": "binary_error_segment_proposal_set_selection",
            },
            handle,
        )

    summary = {
        "data_dir": args.data_dir,
        "output": str(output),
        "model": args.model,
        "target": args.target,
        "n_videos": len(records),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_names": [records[idx].name for idx in train_idx],
        "val_names": [records[idx].name for idx in val_idx],
        "channel_names": channel_names,
        "thresholds": thresholds,
        "min_gaps": min_gaps,
        "min_lengths": min_lengths,
        "n_train_candidates": len(candidates),
        "n_positive_train_candidates": int(y_train.sum()),
        "train_quality_target_mean": float(candidate_quality_targets(best_iou_train).mean()),
        "params": params,
        "train": train_metrics,
        "validation": val_metrics,
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
