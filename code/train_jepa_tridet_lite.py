#!/usr/bin/env python3
"""JEPA-TriDet-lite for binary temporal error-segment localization.

The model is intentionally task-pure: it only predicts temporal error
segments. It does not train category, severity, or score labels.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cross_validate_segment_locator import aggregate_fold_metrics
from selector_fusion import evaluate_fused_predictions
from train_segment_locator import (
    TemporalBlock,
    VideoRecord,
    compute_normalizer,
    contiguous_segments,
    load_signal_dataset,
    make_stratified_folds,
    normalized_signals,
    segment_iou,
)


Segment = tuple[int, int]


def distance_to_distribution_target(distance: float, max_bin: int) -> np.ndarray:
    """Encode a continuous boundary distance with two adjacent bins."""
    bins = int(max_bin) + 1
    if bins <= 0:
        raise ValueError("max_bin must be non-negative")
    value = float(np.clip(float(distance), 0.0, float(max_bin)))
    lower = int(math.floor(value))
    upper = min(int(max_bin), lower + 1)
    target = np.zeros(bins, dtype=np.float32)
    if lower == upper:
        target[lower] = 1.0
    else:
        upper_weight = value - float(lower)
        target[lower] = 1.0 - upper_weight
        target[upper] = upper_weight
    return target


def distribution_logits_to_distance(logits: torch.Tensor) -> torch.Tensor:
    """Decode boundary-distribution logits as their expected bin value."""
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    bins = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)


def _level_length(length: int, stride: int) -> int:
    return max(1, int(math.ceil(max(1, int(length)) / max(1, int(stride)))))


def _frame_for_level_position(position: int, stride: int, length: int) -> int:
    return min(max(0, int(length) - 1), int(position) * max(1, int(stride)))


def _best_segment_for_frame(
    frame: int,
    segments: list[Segment],
    center_sampling_radius: float,
) -> Segment | None:
    matches: list[tuple[int, Segment]] = []
    for start, end in segments:
        if frame < start or frame > end:
            continue
        center = 0.5 * (float(start) + float(end))
        if abs(float(frame) - center) > float(center_sampling_radius) * max(1.0, float(end - start + 1)):
            continue
        matches.append((int(end - start + 1), (int(start), int(end))))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def build_tridet_targets(
    labels: np.ndarray,
    strides: Iterable[int],
    max_regression_bin: int,
    center_sampling_radius: float = 1.5,
    center_target_mode: str = "inside",
) -> list[dict[str, np.ndarray]]:
    """Build anchor-free center and boundary-distribution targets."""
    labels_arr = (np.asarray(labels).reshape(-1) > 0).astype(np.int64)
    n_frames = int(len(labels_arr))
    segments = contiguous_segments(labels_arr)
    if center_target_mode not in {"inside", "event_center"}:
        raise ValueError(f"unsupported center_target_mode: {center_target_mode}")
    levels: list[dict[str, np.ndarray]] = []
    for stride_raw in strides:
        stride = max(1, int(stride_raw))
        level_len = _level_length(n_frames, stride)
        center = np.zeros(level_len, dtype=np.float32)
        valid = np.zeros(level_len, dtype=np.float32)
        left_dist = np.zeros(level_len, dtype=np.float32)
        right_dist = np.zeros(level_len, dtype=np.float32)
        left_distribution = np.zeros((level_len, int(max_regression_bin) + 1), dtype=np.float32)
        right_distribution = np.zeros((level_len, int(max_regression_bin) + 1), dtype=np.float32)

        def assign_segment(pos: int, frame: int, segment: Segment) -> None:
            start, end = segment
            left = max(0.0, float(frame - start) / float(stride))
            right = max(0.0, float(end - frame) / float(stride))
            center[pos] = 1.0
            left_dist[pos] = min(float(max_regression_bin), left)
            right_dist[pos] = min(float(max_regression_bin), right)
            left_distribution[pos] = distance_to_distribution_target(left, max_bin=int(max_regression_bin))
            right_distribution[pos] = distance_to_distribution_target(right, max_bin=int(max_regression_bin))

        for pos in range(level_len):
            frame = _frame_for_level_position(pos, stride, n_frames)
            valid[pos] = 1.0 if frame < n_frames else 0.0
        if center_target_mode == "event_center":
            assigned_lengths = np.full(level_len, np.inf, dtype=np.float32)
            for start, end in segments:
                event_center = 0.5 * (float(start) + float(end))
                pos = int(round(event_center / float(stride)))
                if pos < 0 or pos >= level_len:
                    continue
                frame = _frame_for_level_position(pos, stride, n_frames)
                length = float(end - start + 1)
                if length < float(assigned_lengths[pos]):
                    assigned_lengths[pos] = length
                    assign_segment(pos, frame, (int(start), int(end)))
        else:
            for pos in range(level_len):
                frame = _frame_for_level_position(pos, stride, n_frames)
                segment = _best_segment_for_frame(frame, segments, center_sampling_radius=float(center_sampling_radius))
                if segment is None:
                    continue
                assign_segment(pos, frame, segment)
        levels.append(
            {
                "stride": np.array(stride, dtype=np.int64),
                "valid": valid,
                "center": center,
                "left_distance": left_dist,
                "right_distance": right_dist,
                "left_distribution": left_distribution,
                "right_distribution": right_distribution,
            }
        )
    return levels


def _nms_scored_segments(items: list[tuple[Segment, float]], nms_iou: float | None, max_predictions: int | None) -> list[Segment]:
    ordered = sorted(items, key=lambda item: (float(item[1]), -(item[0][1] - item[0][0] + 1)), reverse=True)
    kept: list[tuple[Segment, float]] = []
    for segment, score in ordered:
        if nms_iou is not None and any(segment_iou(segment, existing) > float(nms_iou) for existing, _ in kept):
            continue
        kept.append((segment, float(score)))
        if max_predictions is not None and len(kept) >= int(max_predictions):
            break
    return sorted(segment for segment, _ in kept)


def decode_dense_predictions(
    outputs: list[dict[str, torch.Tensor | int]],
    lengths: list[int],
    score_threshold: float = 0.5,
    nms_iou: float | None = 0.3,
    max_predictions: int | None = None,
) -> list[list[Segment]]:
    """Decode multiscale center/boundary distributions into frame segments."""
    per_video: list[list[tuple[Segment, float]]] = [[] for _ in lengths]
    for level in outputs:
        stride = int(level["stride"])
        center_scores = torch.sigmoid(level["center_logits"]).detach().cpu().numpy()
        left_distance = distribution_logits_to_distance(level["left_logits"]).detach().cpu().numpy()
        right_distance = distribution_logits_to_distance(level["right_logits"]).detach().cpu().numpy()
        for batch_idx, length_raw in enumerate(lengths):
            length = int(length_raw)
            usable = min(center_scores.shape[1], _level_length(length, stride))
            for pos in range(usable):
                score = float(center_scores[batch_idx, pos])
                if score < float(score_threshold):
                    continue
                center_frame = _frame_for_level_position(pos, stride, length)
                left = float(left_distance[batch_idx, pos]) * float(stride)
                right = float(right_distance[batch_idx, pos]) * float(stride)
                start = max(0, min(length - 1, int(round(float(center_frame) - left))))
                end = max(start, min(length - 1, int(round(float(center_frame) + right))))
                per_video[batch_idx].append(((start, end), score))
    return [_nms_scored_segments(items, nms_iou=nms_iou, max_predictions=max_predictions) for items in per_video]


class _TriDetHead(nn.Module):
    def __init__(self, hidden: int, max_regression_bin: int, dropout: float):
        super().__init__()
        bins = int(max_regression_bin) + 1
        self.tower = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
        )
        self.center = nn.Conv1d(hidden, 1, kernel_size=1)
        self.left = nn.Conv1d(hidden, bins, kernel_size=1)
        self.right = nn.Conv1d(hidden, bins, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.tower(x)
        return {
            "center_logits": self.center(features).squeeze(1),
            "left_logits": self.left(features).transpose(1, 2),
            "right_logits": self.right(features).transpose(1, 2),
        }


class JEPATriDetLite(nn.Module):
    """A compact ActionFormer/TriDet-style dense locator over JEPA features."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 96,
        strides: Iterable[int] = (1, 2, 4),
        max_regression_bin: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        stride_values = tuple(max(1, int(stride)) for stride in strides)
        if not stride_values:
            raise ValueError("at least one stride is required")
        self.strides = stride_values
        self.max_regression_bin = int(max_regression_bin)
        self.input = nn.Conv1d(int(in_features), int(hidden), kernel_size=1)
        self.backbone = nn.Sequential(
            TemporalBlock(int(hidden), dilation=1, dropout=float(dropout)),
            TemporalBlock(int(hidden), dilation=2, dropout=float(dropout)),
            TemporalBlock(int(hidden), dilation=4, dropout=float(dropout)),
            TemporalBlock(int(hidden), dilation=8, dropout=float(dropout)),
        )
        self.heads = nn.ModuleList(
            [_TriDetHead(int(hidden), int(max_regression_bin), float(dropout)) for _ in self.strides]
        )

    def _level_features(self, features: torch.Tensor, stride: int) -> torch.Tensor:
        if int(stride) <= 1:
            return features
        return F.avg_pool1d(features, kernel_size=int(stride), stride=int(stride), ceil_mode=True, count_include_pad=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> list[dict[str, torch.Tensor | int]]:
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, C], got {tuple(x.shape)}")
        features = x.transpose(1, 2)
        features = F.gelu(self.input(features))
        features = self.backbone(features)
        outputs: list[dict[str, torch.Tensor | int]] = []
        for stride, head in zip(self.strides, self.heads):
            level = head(self._level_features(features, stride))
            level["stride"] = int(stride)
            outputs.append(level)
        return outputs


@dataclass
class DenseBatch:
    x: torch.Tensor
    mask: torch.Tensor
    targets: list[dict[str, torch.Tensor]]
    labels: list[np.ndarray]
    lengths: list[int]
    names: list[str]


def _pad_level_targets(
    per_sample: list[dict[str, np.ndarray]],
    level_index: int,
    model_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    batch_size = len(per_sample)
    bins = per_sample[0][level_index]["left_distribution"].shape[1]
    center = np.zeros((batch_size, model_length), dtype=np.float32)
    valid = np.zeros((batch_size, model_length), dtype=np.float32)
    left_distance = np.zeros((batch_size, model_length), dtype=np.float32)
    right_distance = np.zeros((batch_size, model_length), dtype=np.float32)
    left_distribution = np.zeros((batch_size, model_length, bins), dtype=np.float32)
    right_distribution = np.zeros((batch_size, model_length, bins), dtype=np.float32)
    for row, sample_levels in enumerate(per_sample):
        level = sample_levels[level_index]
        length = min(model_length, int(len(level["center"])))
        center[row, :length] = level["center"][:length]
        valid[row, :length] = level["valid"][:length]
        left_distance[row, :length] = level["left_distance"][:length]
        right_distance[row, :length] = level["right_distance"][:length]
        left_distribution[row, :length] = level["left_distribution"][:length]
        right_distribution[row, :length] = level["right_distribution"][:length]
    return {
        "center": torch.from_numpy(center).to(device),
        "valid": torch.from_numpy(valid).to(device),
        "left_distance": torch.from_numpy(left_distance).to(device),
        "right_distance": torch.from_numpy(right_distance).to(device),
        "left_distribution": torch.from_numpy(left_distribution).to(device),
        "right_distribution": torch.from_numpy(right_distribution).to(device),
    }


def make_dense_batch(
    records: list[VideoRecord],
    indices: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    strides: Iterable[int],
    max_regression_bin: int,
    center_sampling_radius: float,
    device: str,
    center_target_mode: str = "inside",
) -> DenseBatch:
    selected = [records[int(idx)] for idx in indices]
    lengths = [len(record.labels) for record in selected]
    max_len = max(lengths)
    feature_dim = selected[0].signals.shape[1]
    x_np = np.zeros((len(selected), max_len, feature_dim), dtype=np.float32)
    mask_np = np.zeros((len(selected), max_len), dtype=bool)
    labels: list[np.ndarray] = []
    names: list[str] = []
    per_sample_targets: list[list[dict[str, np.ndarray]]] = []
    stride_values = tuple(max(1, int(stride)) for stride in strides)
    for row, record in enumerate(selected):
        length = len(record.labels)
        x_np[row, :length] = normalized_signals(record, mean, std)
        mask_np[row, :length] = True
        labels.append(record.labels.copy())
        names.append(record.name)
        per_sample_targets.append(
            build_tridet_targets(
                record.labels,
                strides=stride_values,
                max_regression_bin=int(max_regression_bin),
                center_sampling_radius=float(center_sampling_radius),
                center_target_mode=str(center_target_mode),
            )
        )
    targets = [
        _pad_level_targets(
            per_sample_targets,
            level_index=level_idx,
            model_length=_level_length(max_len, stride),
            device=device,
        )
        for level_idx, stride in enumerate(stride_values)
    ]
    return DenseBatch(
        x=torch.from_numpy(x_np).to(device),
        mask=torch.from_numpy(mask_np).to(device),
        targets=targets,
        labels=labels,
        lengths=lengths,
        names=names,
    )


def tridet_loss(
    outputs: list[dict[str, torch.Tensor | int]],
    targets: list[dict[str, torch.Tensor]],
    center_weight: float = 1.0,
    distribution_weight: float = 1.0,
    iou_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    if len(outputs) != len(targets):
        raise ValueError(f"output/target level mismatch: {len(outputs)} vs {len(targets)}")
    total = None
    stats = {"center_loss": 0.0, "distribution_loss": 0.0, "iou_loss": 0.0, "positives": 0}
    for output, target in zip(outputs, targets):
        center_logits = output["center_logits"]
        left_logits = output["left_logits"]
        right_logits = output["right_logits"]
        center_target = target["center"].to(center_logits.device)
        valid = target["valid"].to(center_logits.device) > 0.5
        positives = (center_target > 0.5) & valid

        valid_values = center_target[valid]
        if valid_values.numel() == 0:
            center_loss = center_logits.sum() * 0.0
        else:
            pos = torch.clamp(valid_values.sum(), min=1.0)
            neg = torch.clamp(valid_values.numel() - valid_values.sum(), min=1.0)
            pos_weight = torch.clamp(neg / pos, min=0.25, max=20.0)
            raw = F.binary_cross_entropy_with_logits(
                center_logits,
                center_target,
                pos_weight=pos_weight,
                reduction="none",
            )
            center_loss = raw[valid].mean()

        if positives.any():
            left_log_probs = F.log_softmax(left_logits[positives], dim=-1)
            right_log_probs = F.log_softmax(right_logits[positives], dim=-1)
            left_distribution = target["left_distribution"].to(left_logits.device)[positives]
            right_distribution = target["right_distribution"].to(right_logits.device)[positives]
            distribution_loss = (
                -(left_distribution * left_log_probs).sum(dim=-1).mean()
                - (right_distribution * right_log_probs).sum(dim=-1).mean()
            )

            pred_left = distribution_logits_to_distance(left_logits[positives])
            pred_right = distribution_logits_to_distance(right_logits[positives])
            target_left = target["left_distance"].to(left_logits.device)[positives]
            target_right = target["right_distance"].to(right_logits.device)[positives]
            inter = torch.minimum(pred_left, target_left) + torch.minimum(pred_right, target_right) + 1.0
            union = torch.maximum(pred_left, target_left) + torch.maximum(pred_right, target_right) + 1.0
            boundary_iou = inter / torch.clamp(union, min=1e-6)
            iou_loss = (1.0 - boundary_iou).mean()
            positive_count = int(positives.sum().detach().cpu())
        else:
            distribution_loss = center_logits.sum() * 0.0
            iou_loss = center_logits.sum() * 0.0
            positive_count = 0

        level_loss = (
            float(center_weight) * center_loss
            + float(distribution_weight) * distribution_loss
            + float(iou_weight) * iou_loss
        )
        total = level_loss if total is None else total + level_loss
        stats["center_loss"] += float(center_loss.detach().cpu())
        stats["distribution_loss"] += float(distribution_loss.detach().cpu())
        stats["iou_loss"] += float(iou_loss.detach().cpu())
        stats["positives"] += positive_count
    assert total is not None
    stats["loss"] = float(total.detach().cpu())
    return total, stats


def predict_indices(
    model: JEPATriDetLite,
    records: list[VideoRecord],
    indices: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
    score_threshold: float,
    nms_iou: float | None,
    max_predictions: int | None,
) -> tuple[list[list[Segment]], list[np.ndarray], list[str]]:
    model.eval()
    predictions: list[list[Segment]] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            batch = make_dense_batch(
                records,
                batch_indices,
                mean,
                std,
                strides=model.strides,
                max_regression_bin=model.max_regression_bin,
                center_sampling_radius=1.0,
                device=device,
            )
            outputs = model(batch.x, batch.mask)
            predictions.extend(
                decode_dense_predictions(
                    outputs,
                    batch.lengths,
                    score_threshold=float(score_threshold),
                    nms_iou=nms_iou,
                    max_predictions=max_predictions,
                )
            )
            labels.extend(batch.labels)
            names.extend(batch.names)
    return predictions, labels, names


def select_decode_params(
    model: JEPATriDetLite,
    records: list[VideoRecord],
    val_idx: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
    thresholds: Iterable[float],
    nms_ious: Iterable[float | None],
    max_prediction_values: Iterable[int | None],
    iou_threshold: float,
) -> tuple[dict, dict, list[list[Segment]]]:
    best_params: dict | None = None
    best_metrics: dict | None = None
    best_predictions: list[list[Segment]] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        for nms_iou in nms_ious:
            for max_predictions in max_prediction_values:
                predictions, labels, _ = predict_indices(
                    model,
                    records,
                    val_idx,
                    mean,
                    std,
                    device=device,
                    batch_size=batch_size,
                    score_threshold=float(threshold),
                    nms_iou=nms_iou,
                    max_predictions=max_predictions,
                )
                metrics = evaluate_fused_predictions(predictions, labels, iou_threshold=float(iou_threshold))
                key = (
                    float(metrics["segment"]["f1"]),
                    float(metrics["segment"]["precision"]),
                    float(metrics["segment"]["recall"]),
                    float(metrics["frame"]["f1"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_params = {
                        "score_threshold": float(threshold),
                        "nms_iou": None if nms_iou is None else float(nms_iou),
                        "max_predictions": None if max_predictions is None else int(max_predictions),
                    }
                    best_metrics = metrics
                    best_predictions = predictions
    assert best_params is not None and best_metrics is not None and best_predictions is not None
    return best_params, best_metrics, best_predictions


def train_fold(
    records: list[VideoRecord],
    train_idx: list[int],
    val_idx: list[int],
    hidden: int,
    strides: tuple[int, ...],
    max_regression_bin: int,
    center_sampling_radius: float,
    center_target_mode: str,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    center_weight: float,
    distribution_weight: float,
    iou_weight: float,
    decode_thresholds: list[float],
    decode_nms_ious: list[float | None],
    decode_max_predictions: list[int | None],
    iou_threshold: float,
) -> tuple[JEPATriDetLite, dict, np.ndarray, np.ndarray]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    mean, std = compute_normalizer(records, train_idx)
    model = JEPATriDetLite(
        in_features=records[0].signals.shape[1],
        hidden=int(hidden),
        strides=strides,
        max_regression_bin=int(max_regression_bin),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(seed))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metrics: dict | None = None
    best_params: dict | None = None
    best_key: tuple[float, float, float, float] | None = None
    no_improve = 0
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train_idx)
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            batch_indices = [int(idx) for idx in order[start : start + int(batch_size)]]
            batch = make_dense_batch(
                records,
                batch_indices,
                mean,
                std,
                strides=strides,
                max_regression_bin=int(max_regression_bin),
                center_sampling_radius=float(center_sampling_radius),
                center_target_mode=str(center_target_mode),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.x, batch.mask)
            loss, _ = tridet_loss(
                outputs,
                batch.targets,
                center_weight=float(center_weight),
                distribution_weight=float(distribution_weight),
                iou_weight=float(iou_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        params, metrics, _ = select_decode_params(
            model,
            records,
            val_idx,
            mean,
            std,
            device=device,
            batch_size=batch_size,
            thresholds=decode_thresholds,
            nms_ious=decode_nms_ious,
            max_prediction_values=decode_max_predictions,
            iou_threshold=float(iou_threshold),
        )
        key = (
            float(metrics["segment"]["f1"]),
            float(metrics["segment"]["precision"]),
            float(metrics["segment"]["recall"]),
            float(metrics["frame"]["f1"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics
            best_params = params
            best_state = {key_name: value.detach().cpu().clone() for key_name, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch < 5 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} loss={float(np.mean(losses)):.4f} "
                f"val_seg_f1={metrics['segment']['f1']:.3f} best={best_key[0]:.3f}",
                flush=True,
            )
        if no_improve >= int(patience):
            break
    model.load_state_dict(best_state)
    assert best_metrics is not None and best_params is not None
    return model, {"params": best_params, **best_metrics}, mean, std


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_optional_float_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else float(item))
    return values


def _parse_optional_int_list(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null"} else int(item))
    return values


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--strides", default="1,2,4")
    parser.add_argument("--max-regression-bin", type=int, default=32)
    parser.add_argument("--center-sampling-radius", type=float, default=1.5)
    parser.add_argument("--center-target-mode", choices=["inside", "event_center"], default="event_center")
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--distribution-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=0.5)
    parser.add_argument("--decode-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--decode-nms-ious", default="0.1,0.3,0.5,none")
    parser.add_argument("--decode-max-predictions", default="none,4,8,12")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = str(args.device)
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    records = load_signal_dataset(args.data_dir)
    folds = make_stratified_folds(records, args.folds, args.seed)
    all_indices = set(range(len(records)))
    stride_values = _parse_int_tuple(args.strides)
    decode_thresholds = _parse_float_list(args.decode_thresholds)
    decode_nms_ious = _parse_optional_float_list(args.decode_nms_ious)
    decode_max_predictions = _parse_optional_int_list(args.decode_max_predictions)
    fold_summaries: list[dict] = []
    for fold_idx, val_idx in enumerate(folds):
        train_idx = sorted(all_indices - set(val_idx))
        print(f"fold={fold_idx} train={len(train_idx)} val={len(val_idx)} device={device}", flush=True)
        model, metrics, mean, std = train_fold(
            records,
            train_idx=train_idx,
            val_idx=val_idx,
            hidden=args.hidden,
            strides=stride_values,
            max_regression_bin=args.max_regression_bin,
            center_sampling_radius=args.center_sampling_radius,
            center_target_mode=args.center_target_mode,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
            seed=args.seed + fold_idx,
            center_weight=args.center_weight,
            distribution_weight=args.distribution_weight,
            iou_weight=args.iou_weight,
            decode_thresholds=decode_thresholds,
            decode_nms_ious=decode_nms_ious,
            decode_max_predictions=decode_max_predictions,
            iou_threshold=args.iou_threshold,
        )
        predictions, labels, names = predict_indices(
            model,
            records,
            val_idx,
            mean,
            std,
            device=device,
            batch_size=args.batch_size,
            score_threshold=float(metrics["params"]["score_threshold"]),
            nms_iou=metrics["params"]["nms_iou"],
            max_predictions=metrics["params"]["max_predictions"],
        )
        validation = evaluate_fused_predictions(predictions, labels, iou_threshold=args.iou_threshold)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_names": names,
                "validation": validation,
                "selected": metrics,
            }
        )
    summary = {
        "data_dir": args.data_dir,
        "n_videos": len(records),
        "frames": int(sum(len(record.labels) for record in records)),
        "positive_frames": int(sum(record.labels.sum() for record in records)),
        "folds": fold_summaries,
        "aggregate": aggregate_fold_metrics(fold_summaries),
        "config": {
            "folds": args.folds,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "strides": stride_values,
            "max_regression_bin": args.max_regression_bin,
            "center_sampling_radius": args.center_sampling_radius,
            "center_target_mode": args.center_target_mode,
            "center_weight": args.center_weight,
            "distribution_weight": args.distribution_weight,
            "iou_weight": args.iou_weight,
            "decode_thresholds": decode_thresholds,
            "decode_nms_ious": decode_nms_ious,
            "decode_max_predictions": decode_max_predictions,
            "iou_threshold": args.iou_threshold,
            "seed": args.seed,
            "device": device,
            "task": "binary_error_segment_localization_jepa_tridet_lite",
        },
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
