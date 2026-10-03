#!/usr/bin/env python3
"""GPU-native JEPA temporal set prediction for binary error localization.

This module predicts a set of temporal error intervals directly from true
V/I-JEPA event-token sequences. It ignores category, severity, and score.
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
    VideoRecord,
    compute_normalizer,
    contiguous_segments,
    load_signal_dataset,
    make_stratified_folds,
    normalized_signals,
    probabilities_to_segments,
    segment_iou,
)


Segment = tuple[int, int]


def labels_to_normalized_segments(labels: np.ndarray, max_segments: int | None = None) -> np.ndarray:
    labels_arr = (np.asarray(labels).reshape(-1) > 0).astype(np.int64)
    n = int(len(labels_arr))
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    segments = contiguous_segments(labels_arr)
    if max_segments is not None:
        segments = segments[: int(max_segments)]
    out = []
    for start, end in segments:
        length = max(1, int(end) - int(start) + 1)
        center = (float(start) + float(end) + 1.0) / 2.0 / float(n)
        width = float(length) / float(n)
        out.append([float(np.clip(center, 0.0, 1.0)), float(np.clip(width, 1.0 / n, 1.0))])
    return np.asarray(out, dtype=np.float32)


def normalized_to_bounds(segments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = segments[..., 0].clamp(0.0, 1.0)
    width = segments[..., 1].clamp(1e-4, 1.0)
    start = (center - 0.5 * width).clamp(0.0, 1.0)
    end = (center + 0.5 * width).clamp(0.0, 1.0)
    return start, end


def normalized_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_start, pred_end = normalized_to_bounds(pred)
    target_start, target_end = normalized_to_bounds(target)
    inter = (torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start)).clamp_min(0.0)
    union = (torch.maximum(pred_end, target_end) - torch.minimum(pred_start, target_start)).clamp_min(1e-6)
    return inter / union


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden))
        pe = torch.zeros(max_len, hidden, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class JEPATemporalSetPredictor(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 128,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden % int(num_heads) != 0:
            raise ValueError("hidden must be divisible by num_heads")
        self.num_queries = int(num_queries)
        self.input = nn.Sequential(
            nn.Linear(int(in_features), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.position = SinusoidalPositionEncoding(int(hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden),
            nhead=int(num_heads),
            dim_feedforward=max(int(hidden) * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.queries = nn.Parameter(torch.randn(int(num_queries), int(hidden)) * 0.02)
        self.cross_attention = nn.MultiheadAttention(int(hidden), int(num_heads), dropout=float(dropout), batch_first=True)
        self.query_norm = nn.LayerNorm(int(hidden))
        self.ffn = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
        )
        self.logit_head = nn.Linear(int(hidden), 1)
        self.frame_head = nn.Linear(int(hidden), 1)
        self.segment_head = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(x.shape)}")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        mask = mask.to(device=x.device, dtype=torch.bool)
        hidden = self.position(self.input(x))
        key_padding_mask = ~mask
        memory = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        queries = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
        query_features = self.query_norm(queries + attended)
        query_features = self.query_norm(query_features + self.ffn(query_features))
        logits = self.logit_head(query_features).squeeze(-1)
        frame_logits = self.frame_head(memory).squeeze(-1)
        segments = torch.sigmoid(self.segment_head(query_features))
        segments = torch.stack([segments[..., 0], segments[..., 1].clamp_min(1e-4)], dim=-1)
        return {"logits": logits, "segments": segments, "frame_logits": frame_logits}


def _greedy_match(cost: torch.Tensor) -> list[tuple[int, int]]:
    if cost.numel() == 0:
        return []
    pairs: list[tuple[float, int, int]] = []
    for q in range(cost.shape[0]):
        for g in range(cost.shape[1]):
            pairs.append((float(cost[q, g].detach().cpu()), q, g))
    pairs.sort(key=lambda item: item[0])
    used_q: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, q, g in pairs:
        if q in used_q or g in used_g:
            continue
        used_q.add(q)
        used_g.add(g)
        matches.append((q, g))
        if len(used_g) >= cost.shape[1]:
            break
    return matches


def set_prediction_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[torch.Tensor],
    frame_labels: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    object_weight: float = 1.0,
    l1_weight: float = 5.0,
    iou_weight: float = 2.0,
    no_object_weight: float = 0.25,
    frame_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    logits = outputs["logits"]
    pred_segments = outputs["segments"]
    if logits.ndim != 2 or pred_segments.ndim != 3:
        raise ValueError("expected logits [B,Q] and segments [B,Q,2]")
    batch_size, num_queries = logits.shape
    object_targets = torch.zeros_like(logits)
    matched_pred: list[torch.Tensor] = []
    matched_target: list[torch.Tensor] = []
    n_matched = 0
    for batch_idx in range(batch_size):
        gt = targets[batch_idx].to(device=logits.device, dtype=pred_segments.dtype)
        if gt.numel() == 0:
            continue
        gt = gt.reshape(-1, 2)
        pred = pred_segments[batch_idx]
        l1 = torch.cdist(pred, gt, p=1)
        iou = normalized_iou(pred[:, None, :], gt[None, :, :])
        cost = l1_weight * l1 + iou_weight * (1.0 - iou)
        for query_idx, gt_idx in _greedy_match(cost):
            object_targets[batch_idx, query_idx] = 1.0
            matched_pred.append(pred[query_idx])
            matched_target.append(gt[gt_idx])
            n_matched += 1
    weights = torch.where(object_targets > 0.5, torch.ones_like(object_targets), torch.full_like(object_targets, float(no_object_weight)))
    object_loss = F.binary_cross_entropy_with_logits(logits, object_targets, weight=weights)
    if matched_pred:
        pred_stack = torch.stack(matched_pred, dim=0)
        target_stack = torch.stack(matched_target, dim=0)
        l1_loss = F.l1_loss(pred_stack, target_stack)
        iou_loss = (1.0 - normalized_iou(pred_stack, target_stack)).mean()
    else:
        l1_loss = logits.new_tensor(0.0)
        iou_loss = logits.new_tensor(0.0)
    frame_loss = logits.new_tensor(0.0)
    if float(frame_weight) > 0.0 and frame_labels is not None and frame_mask is not None and "frame_logits" in outputs:
        frame_logits = outputs["frame_logits"]
        labels = frame_labels.to(device=frame_logits.device, dtype=frame_logits.dtype)
        mask = frame_mask.to(device=frame_logits.device, dtype=torch.bool)
        pos = torch.clamp(labels[mask].sum(), min=1.0)
        neg = torch.clamp(mask.sum().to(frame_logits.dtype) - labels[mask].sum(), min=1.0)
        pos_weight = torch.clamp(neg / pos, max=50.0)
        raw_frame_loss = F.binary_cross_entropy_with_logits(
            frame_logits,
            labels,
            pos_weight=pos_weight,
            reduction="none",
        )
        frame_loss = raw_frame_loss[mask].mean() if mask.any() else logits.new_tensor(0.0)
    loss = (
        float(object_weight) * object_loss
        + float(l1_weight) * l1_loss
        + float(iou_weight) * iou_loss
        + float(frame_weight) * frame_loss
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "l1_loss": float(l1_loss.detach().cpu()),
        "iou_loss": float(iou_loss.detach().cpu()),
        "frame_loss": float(frame_loss.detach().cpu()),
        "matched": int(n_matched),
    }
    return loss, stats


def _normalized_segment_to_frame(segment: np.ndarray, length: int) -> Segment:
    n = max(1, int(length))
    center = float(np.clip(segment[0], 0.0, 1.0))
    width = float(np.clip(segment[1], 1.0 / n, 1.0))
    start_f = (center - 0.5 * width) * n
    end_f = (center + 0.5 * width) * n - 1.0
    start = max(0, min(n - 1, int(math.floor(start_f))))
    end = max(start, min(n - 1, int(math.ceil(end_f))))
    return start, end


def _nms_segments(items: list[tuple[Segment, float]], nms_iou: float | None) -> list[Segment]:
    if nms_iou is None:
        return sorted({segment for segment, _ in items})
    kept: list[tuple[Segment, float]] = []
    for segment, score in sorted(items, key=lambda item: (item[1], -(item[0][1] - item[0][0] + 1)), reverse=True):
        if all(segment_iou(segment, existing) <= float(nms_iou) for existing, _ in kept):
            kept.append((segment, score))
    return sorted(segment for segment, _ in kept)


def decode_set_predictions(
    outputs: dict[str, torch.Tensor],
    lengths: list[int],
    score_threshold: float = 0.5,
    nms_iou: float | None = 0.3,
    max_segments: int | None = None,
    frame_threshold: float | None = None,
    frame_min_gap: int = 0,
    frame_min_length: int = 1,
) -> list[list[Segment]]:
    logits = outputs["logits"].detach().cpu()
    segments = outputs["segments"].detach().cpu().numpy()
    scores = torch.sigmoid(logits).numpy()
    predictions: list[list[Segment]] = []
    for batch_idx, length in enumerate(lengths):
        items: list[tuple[Segment, float]] = []
        for query_idx in range(scores.shape[1]):
            score = float(scores[batch_idx, query_idx])
            if score < float(score_threshold):
                continue
            segment = _normalized_segment_to_frame(segments[batch_idx, query_idx], int(length))
            items.append((segment, score))
        if frame_threshold is not None and "frame_logits" in outputs:
            frame_probs = torch.sigmoid(outputs["frame_logits"][batch_idx]).detach().cpu().numpy().astype(np.float32)
            frame_probs = frame_probs[: int(length)]
            for segment in probabilities_to_segments(
                frame_probs,
                threshold=float(frame_threshold),
                smooth_window=1,
                min_gap=int(frame_min_gap),
                min_length=int(frame_min_length),
            ):
                start, end = segment
                score = float(frame_probs[start : end + 1].max()) if end >= start else float(frame_threshold)
                items.append(((int(start), int(end)), score))
        selected = _nms_segments(items, nms_iou=nms_iou)
        if max_segments is not None and len(selected) > int(max_segments):
            score_by_segment = {segment: score for segment, score in items}
            selected = sorted(selected, key=lambda segment: score_by_segment.get(segment, 0.0), reverse=True)[: int(max_segments)]
            selected = sorted(selected)
        predictions.append(selected)
    return predictions


@dataclass
class Batch:
    x: torch.Tensor
    mask: torch.Tensor
    targets: list[torch.Tensor]
    label_tensor: torch.Tensor
    labels: list[np.ndarray]
    lengths: list[int]
    names: list[str]


def make_batch(
    records: list[VideoRecord],
    indices: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    max_target_segments: int | None,
    device: str,
) -> Batch:
    selected = [records[int(idx)] for idx in indices]
    lengths = [len(record.labels) for record in selected]
    max_len = max(lengths)
    feature_dim = selected[0].signals.shape[1]
    x_np = np.zeros((len(selected), max_len, feature_dim), dtype=np.float32)
    mask_np = np.zeros((len(selected), max_len), dtype=bool)
    label_np = np.zeros((len(selected), max_len), dtype=np.float32)
    targets: list[torch.Tensor] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for row, record in enumerate(selected):
        normed = normalized_signals(record, mean, std)
        length = len(record.labels)
        x_np[row, :length] = normed[:length]
        mask_np[row, :length] = True
        label_np[row, :length] = record.labels[:length].astype(np.float32)
        targets.append(torch.from_numpy(labels_to_normalized_segments(record.labels, max_segments=max_target_segments)))
        labels.append(record.labels.copy())
        names.append(record.name)
    return Batch(
        x=torch.from_numpy(x_np).to(device),
        mask=torch.from_numpy(mask_np).to(device),
        targets=[target.to(device) for target in targets],
        label_tensor=torch.from_numpy(label_np).to(device),
        labels=labels,
        lengths=lengths,
        names=names,
    )


def predict_indices(
    model: JEPATemporalSetPredictor,
    records: list[VideoRecord],
    indices: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
    score_threshold: float,
    nms_iou: float | None,
    max_segments: int | None,
    frame_threshold: float | None = None,
    frame_min_gap: int = 0,
    frame_min_length: int = 1,
) -> tuple[list[list[Segment]], list[np.ndarray], list[str]]:
    model.eval()
    predictions: list[list[Segment]] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            batch = make_batch(records, batch_indices, mean, std, max_target_segments=max_segments, device=device)
            outputs = model(batch.x, batch.mask)
            predictions.extend(
                decode_set_predictions(
                    outputs,
                    batch.lengths,
                    score_threshold=score_threshold,
                    nms_iou=nms_iou,
                    max_segments=max_segments,
                    frame_threshold=frame_threshold,
                    frame_min_gap=frame_min_gap,
                    frame_min_length=frame_min_length,
                )
            )
            labels.extend(batch.labels)
            names.extend(batch.names)
    return predictions, labels, names


def select_decode_params(
    model: JEPATemporalSetPredictor,
    records: list[VideoRecord],
    val_idx: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
    thresholds: Iterable[float],
    nms_ious: Iterable[float | None],
    frame_thresholds: Iterable[float | None],
    frame_min_gaps: Iterable[int],
    frame_min_lengths: Iterable[int],
    max_segments: int | None,
    iou_threshold: float,
) -> tuple[dict, dict, list[list[Segment]]]:
    best_params: dict | None = None
    best_metrics: dict | None = None
    best_predictions: list[list[Segment]] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        for nms_iou in nms_ious:
            for frame_threshold in frame_thresholds:
                active_min_gaps = frame_min_gaps if frame_threshold is not None else [0]
                active_min_lengths = frame_min_lengths if frame_threshold is not None else [1]
                for frame_min_gap in active_min_gaps:
                    for frame_min_length in active_min_lengths:
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
                            max_segments=max_segments,
                            frame_threshold=frame_threshold,
                            frame_min_gap=int(frame_min_gap),
                            frame_min_length=int(frame_min_length),
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
                                "frame_threshold": None if frame_threshold is None else float(frame_threshold),
                                "frame_min_gap": int(frame_min_gap),
                                "frame_min_length": int(frame_min_length),
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
    num_queries: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    seed: int,
    max_target_segments: int | None,
    object_weight: float,
    l1_weight: float,
    iou_weight: float,
    no_object_weight: float,
    frame_weight: float,
    decode_thresholds: list[float],
    decode_nms_ious: list[float | None],
    frame_thresholds: list[float | None],
    frame_min_gaps: list[int],
    frame_min_lengths: list[int],
    iou_threshold: float,
) -> tuple[JEPATemporalSetPredictor, dict, np.ndarray, np.ndarray]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    mean, std = compute_normalizer(records, train_idx)
    in_features = records[0].signals.shape[1]
    model = JEPATemporalSetPredictor(
        in_features=in_features,
        hidden=int(hidden),
        num_queries=int(num_queries),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(seed))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metrics: dict | None = None
    best_params: dict | None = None
    best_key: tuple[float, float, float, float] | None = None
    patience = max(8, min(20, int(epochs) // 3))
    no_improve = 0
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train_idx)
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            batch_indices = [int(idx) for idx in order[start : start + int(batch_size)]]
            batch = make_batch(records, batch_indices, mean, std, max_target_segments=max_target_segments, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.x, batch.mask)
            loss, _ = set_prediction_loss(
                outputs,
                batch.targets,
                frame_labels=batch.label_tensor,
                frame_mask=batch.mask,
                object_weight=float(object_weight),
                l1_weight=float(l1_weight),
                iou_weight=float(iou_weight),
                no_object_weight=float(no_object_weight),
                frame_weight=float(frame_weight),
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
            frame_thresholds=frame_thresholds,
            frame_min_gaps=frame_min_gaps,
            frame_min_lengths=frame_min_lengths,
            max_segments=max_target_segments,
            iou_threshold=iou_threshold,
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
        if no_improve >= patience:
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


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-target-segments", type=int, default=8)
    parser.add_argument("--object-weight", type=float, default=1.0)
    parser.add_argument("--l1-weight", type=float, default=5.0)
    parser.add_argument("--iou-weight", type=float, default=2.0)
    parser.add_argument("--no-object-weight", type=float, default=0.25)
    parser.add_argument("--frame-loss-weight", type=float, default=0.5)
    parser.add_argument("--decode-thresholds", default="0.25,0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--decode-nms-ious", default="0.1,0.3,none")
    parser.add_argument("--frame-thresholds", default="none,0.3,0.4,0.5,0.6")
    parser.add_argument("--frame-min-gaps", default="0,1,2")
    parser.add_argument("--frame-min-lengths", default="1,2,4")
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
    fold_summaries: list[dict] = []
    decode_thresholds = _parse_float_list(args.decode_thresholds)
    decode_nms_ious = _parse_optional_float_list(args.decode_nms_ious)
    frame_thresholds = _parse_optional_float_list(args.frame_thresholds)
    frame_min_gaps = _parse_int_list(args.frame_min_gaps)
    frame_min_lengths = _parse_int_list(args.frame_min_lengths)
    for fold_idx, val_idx in enumerate(folds):
        train_idx = sorted(all_indices - set(val_idx))
        print(f"fold={fold_idx} train={len(train_idx)} val={len(val_idx)} device={device}", flush=True)
        model, metrics, mean, std = train_fold(
            records,
            train_idx=train_idx,
            val_idx=val_idx,
            hidden=args.hidden,
            num_queries=args.num_queries,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + fold_idx,
            max_target_segments=args.max_target_segments,
            object_weight=args.object_weight,
            l1_weight=args.l1_weight,
            iou_weight=args.iou_weight,
            no_object_weight=args.no_object_weight,
            frame_weight=args.frame_loss_weight,
            decode_thresholds=decode_thresholds,
            decode_nms_ious=decode_nms_ious,
            frame_thresholds=frame_thresholds,
            frame_min_gaps=frame_min_gaps,
            frame_min_lengths=frame_min_lengths,
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
            max_segments=args.max_target_segments,
            frame_threshold=metrics["params"].get("frame_threshold"),
            frame_min_gap=int(metrics["params"].get("frame_min_gap", 0)),
            frame_min_length=int(metrics["params"].get("frame_min_length", 1)),
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
            "num_queries": args.num_queries,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "max_target_segments": args.max_target_segments,
            "object_weight": args.object_weight,
            "l1_weight": args.l1_weight,
            "iou_weight": args.iou_weight,
            "no_object_weight": args.no_object_weight,
            "frame_loss_weight": args.frame_loss_weight,
            "decode_thresholds": decode_thresholds,
            "decode_nms_ious": decode_nms_ious,
            "frame_thresholds": frame_thresholds,
            "frame_min_gaps": frame_min_gaps,
            "frame_min_lengths": frame_min_lengths,
            "iou_threshold": args.iou_threshold,
            "seed": args.seed,
            "device": device,
            "task": "binary_error_segment_localization_jepa_temporal_set_prediction",
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
