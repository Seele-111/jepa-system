#!/usr/bin/env python3
"""Train a complementary JEPA proposal rescuer.

The rescuer is intentionally narrow: it only decides whether a JEPA proposal
should be added to a high-precision mainline prediction set. It does not
predict error type, score quality, or severity.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_prototype_rescue import fit_prototype_rescuer
from selector_fusion import evaluate_fused_predictions, fuse_protected_segment_predictions, select_protected_fusion_params
from train_proposal_calibrator import NumpyLogisticClassifier, interval_iou
from train_proposal_set_selector import (
    DEFAULT_CHANNEL_NAMES,
    extract_set_proposal_features,
    generate_multichannel_candidates,
    score_video_candidates,
    select_weighted_proposal_set,
)
from train_segment_locator import VideoRecord, contiguous_segments, load_signal_dataset


@dataclass
class RescueCandidateRecord:
    video_idx: int
    segment: tuple[int, int]
    features: np.ndarray
    label: int
    best_iou: float
    selector_score: float


def matched_ground_truth_indices(
    predicted_segments: list[tuple[int, int]],
    labels: np.ndarray,
    iou_threshold: float,
) -> set[int]:
    gt_segments = contiguous_segments(np.asarray(labels, dtype=np.int64))
    matched: set[int] = set()
    for pred in predicted_segments:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_segments):
            if idx in matched:
                continue
            iou = interval_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_idx)
    return matched


def label_rescue_candidates(
    candidates: list[tuple[int, int]],
    labels: np.ndarray,
    mainline_segments: list[tuple[int, int]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt_segments = contiguous_segments(np.asarray(labels, dtype=np.int64))
    already_matched = matched_ground_truth_indices(mainline_segments, labels, iou_threshold=iou_threshold)
    y = np.zeros(len(candidates), dtype=np.int64)
    best = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        best_idx = -1
        for gt_idx, gt in enumerate(gt_segments):
            iou = interval_iou(candidate, gt)
            if iou > float(best[idx]):
                best[idx] = float(iou)
                best_idx = gt_idx
        y[idx] = int(best_idx >= 0 and best_idx not in already_matched and best[idx] >= float(iou_threshold))
    return y, best


def _outside_values(values: np.ndarray, start: int, end: int) -> np.ndarray:
    length = end - start + 1
    left = values[max(0, start - length) : start]
    right = values[end + 1 : min(len(values), end + 1 + length)]
    if len(left) == 0 and len(right) == 0:
        return np.asarray([], dtype=np.float32)
    return np.concatenate([left, right])


def _distance_to_segments(segment: tuple[int, int], segments: list[tuple[int, int]], n_frames: int) -> float:
    if not segments:
        return 1.0
    start, end = segment
    distances = []
    for left, right in segments:
        if end < left:
            distances.append(left - end)
        elif right < start:
            distances.append(start - right)
        else:
            distances.append(0)
    return float(min(distances) / max(1, n_frames))


def _max_iou(segment: tuple[int, int], segments: list[tuple[int, int]]) -> float:
    return max((interval_iou(segment, other) for other in segments), default=0.0)


def extract_rescue_features(
    signals: np.ndarray,
    segment: tuple[int, int],
    channel_indices: Iterable[int],
    selector_score: float,
    mainline_probs: np.ndarray,
    mainline_segments: list[tuple[int, int]],
) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float32)
    mainline_probs = np.nan_to_num(np.asarray(mainline_probs, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    n = min(len(signals), len(mainline_probs))
    if n <= 0:
        return np.zeros(1, dtype=np.float32)
    start, end = int(segment[0]), int(segment[1])
    start = max(0, min(start, n - 1))
    end = max(start, min(end, n - 1))
    duration = end - start + 1
    base = extract_set_proposal_features(signals[:n], (start, end), channel_indices)
    inside = mainline_probs[start : end + 1]
    outside = _outside_values(mainline_probs[:n], start, end)
    outside_mean = float(outside.mean()) if len(outside) else 0.0
    context = np.asarray(
        [
            float(selector_score),
            float(inside.mean()),
            float(inside.max()),
            float(inside.std()),
            float(inside.max() - outside_mean),
            float(_max_iou((start, end), mainline_segments)),
            float(_distance_to_segments((start, end), mainline_segments, n)),
            float(duration / max(1, n)),
            float(math.log1p(duration)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(np.concatenate([base, context]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def score_rescue_candidates(
    selector,
    rescuer,
    record: VideoRecord,
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    mainline_probs: np.ndarray,
    mainline_segments: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    candidates, selector_scores = score_video_candidates(
        selector,
        record,
        feature_names,
        channel_names,
        thresholds,
        min_gaps,
        min_lengths,
    )
    if not candidates:
        return [], np.zeros(0, dtype=np.float32)
    features = np.stack(
        [
            extract_rescue_features(
                record.signals,
                segment,
                channel_indices,
                selector_score=float(score),
                mainline_probs=mainline_probs,
                mainline_segments=mainline_segments,
            )
            for segment, score in zip(candidates, selector_scores)
        ]
    )
    rescue_scores = rescuer.predict_proba(features)[:, 1].astype(np.float32)
    return candidates, rescue_scores


def predict_rescue_segments(
    selector,
    rescuer,
    record: VideoRecord,
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    mainline_probs: np.ndarray,
    mainline_segments: list[tuple[int, int]],
    prob_threshold: float,
    length_penalty: float,
    max_mainline_iou: float,
) -> list[tuple[int, int]]:
    candidates, scores = score_rescue_candidates(
        selector,
        rescuer,
        record,
        feature_names,
        channel_names,
        thresholds,
        min_gaps,
        min_lengths,
        mainline_probs,
        mainline_segments,
    )
    filtered_candidates = []
    filtered_scores = []
    for segment, score in zip(candidates, scores):
        if float(score) < float(prob_threshold):
            continue
        if _max_iou(segment, mainline_segments) > float(max_mainline_iou):
            continue
        filtered_candidates.append(segment)
        filtered_scores.append(float(score))
    return select_weighted_proposal_set(filtered_candidates, filtered_scores, length_penalty=float(length_penalty))


class _RescueMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.1):
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


class TorchMLPRescuer:
    def __init__(self, model: _RescueMLP, mean: np.ndarray, std: np.ndarray, device: str):
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.device = str(device)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        normed = (values - self.mean) / self.std
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


TorchMLPRescuer.__module__ = "train_proposal_rescuer"


def _fit_torch_mlp_rescuer(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    hidden: int,
    lr: float,
) -> TorchMLPRescuer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0).astype(np.float32), 1e-4)
    x_norm = ((x - mean) / std).astype(np.float32)
    device_name = str(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = _RescueMLP(in_features=x.shape[1], hidden=int(hidden), dropout=0.1).to(device_name)
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
    return TorchMLPRescuer(model, mean=mean, std=std, device=device_name)


def fit_rescuer(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    model_name: str,
    device: str = "cpu",
    epochs: int = 120,
    batch_size: int = 512,
):
    if model_name == "prototype":
        return fit_prototype_rescuer(x, y)
    if model_name == "mlp":
        return _fit_torch_mlp_rescuer(
            x,
            y,
            seed=seed,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            hidden=64,
            lr=1e-3,
        )
    if model_name == "logreg":
        return NumpyLogisticClassifier(lr=0.03, steps=1200, l2=1e-3).fit(x, y)
    try:
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    except Exception:
        return NumpyLogisticClassifier(lr=0.03, steps=1200, l2=1e-3).fit(x, y)

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "gbdt":
        pos = max(1, int(y.sum()))
        neg = max(1, int(len(y) - y.sum()))
        sample_weight = np.where(y > 0, neg / pos, 1.0)
        model = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        )
        model.fit(x, y, sample_weight=sample_weight)
        return model
    else:
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    model.fit(x, y)
    return model


def build_rescue_candidate_records(
    records: list[VideoRecord],
    indices: Iterable[int],
    selector,
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    mainline_probs_by_idx: dict[int, np.ndarray],
    mainline_segments_by_idx: dict[int, list[tuple[int, int]]],
    iou_threshold: float,
) -> list[RescueCandidateRecord]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    channel_indices = [name_to_idx[name] for name in channel_names if name in name_to_idx]
    out: list[RescueCandidateRecord] = []
    for video_idx in indices:
        record = records[int(video_idx)]
        candidates, selector_scores = score_video_candidates(
            selector,
            record,
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
        )
        if not candidates:
            continue
        labels, best_iou = label_rescue_candidates(
            candidates,
            record.labels,
            mainline_segments_by_idx.get(int(video_idx), []),
            iou_threshold=iou_threshold,
        )
        for segment, label, iou, selector_score in zip(candidates, labels, best_iou, selector_scores):
            out.append(
                RescueCandidateRecord(
                    video_idx=int(video_idx),
                    segment=segment,
                    features=extract_rescue_features(
                        record.signals,
                        segment,
                        channel_indices,
                        selector_score=float(selector_score),
                        mainline_probs=mainline_probs_by_idx[int(video_idx)],
                        mainline_segments=mainline_segments_by_idx.get(int(video_idx), []),
                    ),
                    label=int(label),
                    best_iou=float(iou),
                    selector_score=float(selector_score),
                )
            )
    return out


def select_rescuer_params(
    selector,
    rescuer,
    records: list[VideoRecord],
    val_idx: list[int],
    feature_names: list[str],
    channel_names: list[str],
    thresholds: list[float],
    min_gaps: list[int],
    min_lengths: list[int],
    mainline_probs_by_idx: dict[int, np.ndarray],
    mainline_segments_by_idx: dict[int, list[tuple[int, int]]],
    iou_threshold: float,
    active_base_segments_by_idx: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[dict, dict, list[list[tuple[int, int]]]]:
    labels = [records[idx].labels for idx in val_idx]
    base_segments_by_idx = active_base_segments_by_idx if active_base_segments_by_idx is not None else mainline_segments_by_idx
    mainline_predictions = [base_segments_by_idx[idx] for idx in val_idx]
    scored_candidates = []
    for idx in val_idx:
        candidates, scores = score_rescue_candidates(
            selector,
            rescuer,
            records[idx],
            feature_names,
            channel_names,
            thresholds,
            min_gaps,
            min_lengths,
            mainline_probs_by_idx[idx],
            base_segments_by_idx[idx],
        )
        scored_candidates.append((candidates, scores))
    best_config, best_metrics = select_protected_fusion_params(
        mainline_predictions,
        [[] for _ in val_idx],
        labels,
        max_mainline_iou_candidates=[0.0],
        selector_nms_iou_candidates=[None],
        iou_threshold=iou_threshold,
    )
    best_predictions = mainline_predictions
    best_key = (
        float(best_metrics["segment"]["f1"]),
        float(best_metrics["segment"]["precision"]),
        float(best_metrics["segment"]["recall"]),
        float(best_metrics["frame"]["f1"]),
        0.0,
    )
    for prob_threshold in np.linspace(0.2, 0.9, 15):
        for length_penalty in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]:
            for max_mainline_iou in [0.0, 0.05, 0.1, 0.25]:
                rescue_predictions = []
                for idx, (candidates, scores) in zip(val_idx, scored_candidates):
                    filtered_candidates = []
                    filtered_scores = []
                    for segment, score in zip(candidates, scores):
                        if float(score) < float(prob_threshold):
                            continue
                        if _max_iou(segment, base_segments_by_idx[idx]) > float(max_mainline_iou):
                            continue
                        filtered_candidates.append(segment)
                        filtered_scores.append(float(score))
                    rescue_predictions.append(
                        select_weighted_proposal_set(
                            filtered_candidates,
                            filtered_scores,
                            length_penalty=float(length_penalty),
                        )
                    )
                config, metrics = select_protected_fusion_params(
                    mainline_predictions,
                    rescue_predictions,
                    labels,
                    max_mainline_iou_candidates=[float(max_mainline_iou)],
                    selector_nms_iou_candidates=[None],
                    iou_threshold=iou_threshold,
                )
                if config.get("enabled"):
                    fused = fuse_protected_segment_predictions(
                        mainline_predictions,
                        rescue_predictions,
                        max_mainline_iou=float(max_mainline_iou),
                        selector_nms_iou=None,
                    )
                else:
                    fused = mainline_predictions
                rescue_count = sum(len(items) for items in rescue_predictions)
                key = (
                    float(metrics["segment"]["f1"]),
                    float(metrics["segment"]["precision"]),
                    float(metrics["segment"]["recall"]),
                    float(metrics["frame"]["f1"]),
                    -float(rescue_count),
                )
                if key > best_key:
                    best_key = key
                    best_config = {
                        **config,
                        "prob_threshold": float(prob_threshold),
                        "length_penalty": float(length_penalty),
                        "max_mainline_iou": float(max_mainline_iou),
                        "candidate_rescue_segments": int(rescue_count),
                    }
                    best_metrics = metrics
                    best_predictions = fused
    return best_config, best_metrics, best_predictions


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_name_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _load_feature_names(data_dir: str | Path) -> list[str]:
    data = json.loads((Path(data_dir) / "summary.json").read_text(encoding="utf-8"))
    return [str(name) for name in data.get("feature_names", [])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--channel-names", default=",".join(DEFAULT_CHANNEL_NAMES))
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--min-gaps", default="0,1,2,4,8")
    parser.add_argument("--min-lengths", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = load_signal_dataset(args.data_dir)
    feature_names = _load_feature_names(args.data_dir)
    channel_names = [name for name in _parse_name_list(args.channel_names) if name in feature_names]
    summary = {
        "data_dir": args.data_dir,
        "n_videos": len(records),
        "feature_names": feature_names,
        "channel_names": channel_names,
        "thresholds": _parse_float_list(args.thresholds),
        "min_gaps": _parse_int_list(args.min_gaps),
        "min_lengths": _parse_int_list(args.min_lengths),
        "task": "binary_error_segment_proposal_rescue",
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
