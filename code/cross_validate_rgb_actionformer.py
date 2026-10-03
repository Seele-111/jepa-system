#!/usr/bin/env python3
"""ActionFormer-style RGB temporal localization with nested video CV.

This is an independent, compact adapter over frozen Kinetics R3D-18 features.
It implements local self-attention at three temporal scales and dense frame
scoring. It is not the authors' official ActionFormer implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cross_validate_rgb_baseline import aggregate, nested_indices
from train_segment_locator import (
    _segments_from_params,
    compute_normalizer,
    evaluate_records,
    load_signal_dataset,
    make_stratified_folds,
    select_postprocess_params,
)


class LocalTransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, window: int, dropout: float):
        super().__init__()
        self.window = int(window)
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        length = x.shape[1]
        positions = torch.arange(length, device=x.device)
        local_mask = (positions[:, None] - positions[None, :]).abs() > self.window
        z = self.norm1(x)
        attended, _ = self.attn(z, z, z, attn_mask=local_mask, key_padding_mask=padding_mask, need_weights=False)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class RGBActionFormer(nn.Module):
    def __init__(self, in_features: int = 512, hidden: int = 64, heads: int = 4, window: int = 9, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        self.levels = nn.ModuleList([LocalTransformerBlock(hidden, heads, max(2, window // stride), dropout) for stride in (1, 2, 4)])
        self.fuse = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, 1)
        self.occupancy_head = nn.Linear(hidden * 2, 1)
        self.event_count_head = nn.Linear(hidden * 2, 3)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, return_aux: bool = False):
        base = self.proj(x)
        target_length = base.shape[1]
        outputs = []
        for level, stride in zip(self.levels, (1, 2, 4)):
            features = base if stride == 1 else F.avg_pool1d(base.transpose(1, 2), stride, stride, ceil_mode=True).transpose(1, 2)
            level_valid = valid if stride == 1 else F.max_pool1d(valid.float().unsqueeze(1), stride, stride, ceil_mode=True).squeeze(1).bool()
            features = level(features, padding_mask=~level_valid)
            if stride != 1:
                features = F.interpolate(features.transpose(1, 2), size=target_length, mode="linear", align_corners=False).transpose(1, 2)
            outputs.append(features)
        fused = self.fuse(torch.cat(outputs, dim=-1))
        frame_logits = self.head(fused).squeeze(-1)
        if not return_aux:
            return frame_logits
        mask = valid.unsqueeze(-1)
        masked_fused = fused.masked_fill(~mask, 0.0)
        pooled_mean = masked_fused.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled_max = fused.masked_fill(~mask, torch.finfo(fused.dtype).min).max(dim=1).values
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        occupancy = torch.sigmoid(self.occupancy_head(pooled).squeeze(-1))
        event_count_logits = self.event_count_head(pooled)
        return frame_logits, occupancy, event_count_logits


def auxiliary_targets(labels: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    binary = (labels > 0.5) & valid
    occupancy = binary.sum(dim=1).float() / valid.sum(dim=1).clamp_min(1).float()
    starts = binary[:, 0].long()
    if binary.shape[1] > 1:
        starts = starts + (binary[:, 1:] & ~binary[:, :-1]).sum(dim=1)
    return occupancy, starts.clamp_max(2)


def collate(records, indices: list[int], mean: np.ndarray, std: np.ndarray, device: str):
    length = max(len(records[i].labels) for i in indices)
    features = np.zeros((len(indices), length, records[indices[0]].signals.shape[1]), np.float32)
    labels = np.zeros((len(indices), length), np.float32)
    valid = np.zeros((len(indices), length), bool)
    for row, idx in enumerate(indices):
        n = len(records[idx].labels)
        features[row, :n] = (records[idx].signals - mean) / std
        labels[row, :n] = records[idx].labels
        valid[row, :n] = True
    return (torch.from_numpy(features).to(device), torch.from_numpy(labels).to(device), torch.from_numpy(valid).to(device))


@torch.inference_mode()
def predict(model, records, indices: list[int], mean: np.ndarray, std: np.ndarray, device: str, batch_size: int) -> list[dict]:
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        x, _, valid = collate(records, batch_indices, mean, std, device)
        logits, occupancy, event_count_logits = model(x, valid, return_aux=True)
        probs = torch.sigmoid(logits).cpu().numpy()
        occupancy = occupancy.cpu().numpy()
        event_count_probs = torch.softmax(event_count_logits, dim=-1).cpu().numpy()
        for row, idx in enumerate(batch_indices):
            n = len(records[idx].labels)
            rows.append(
                {
                    "name": records[idx].name,
                    "labels": records[idx].labels,
                    "probs": probs[row, :n].astype(np.float32),
                    "predicted_occupancy": float(occupancy[row]),
                    "event_count_probs": event_count_probs[row].astype(np.float32),
                }
            )
    return rows


def train_fold(records, fit_idx: list[int], cal_idx: list[int], args, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mean, std = compute_normalizer(records, fit_idx)
    model = RGBActionFormer(records[0].signals.shape[1], args.hidden, args.heads, args.window, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y = np.concatenate([records[i].labels for i in fit_idx])
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())], device=args.device, dtype=torch.float32)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    rng = np.random.default_rng(seed)
    for epoch in range(args.epochs):
        model.train()
        order = np.asarray(fit_idx)
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            x, labels, valid = collate(records, order[start : start + args.batch_size].tolist(), mean, std, args.device)
            logits, occupancy, event_count_logits = model(x, valid, return_aux=True)
            target_occupancy, target_event_count = auxiliary_targets(labels, valid)
            frame_loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], pos_weight=pos_weight)
            occupancy_loss = F.smooth_l1_loss(occupancy, target_occupancy)
            event_count_loss = F.cross_entropy(event_count_logits, target_event_count)
            loss = (
                frame_loss
                + args.occupancy_loss_weight * occupancy_loss
                + args.event_count_loss_weight * event_count_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        calibration = predict(model, records, cal_idx, mean, std, args.device, args.batch_size)
        cal_labels = np.concatenate([row["labels"] for row in calibration])
        cal_probs = np.concatenate([row["probs"] for row in calibration])
        eps = 1e-6
        frame_cal_loss = float(-np.mean(cal_labels * np.log(cal_probs + eps) + (1 - cal_labels) * np.log(1 - cal_probs + eps)))
        occupancy_targets = np.asarray([np.mean(row["labels"]) for row in calibration], dtype=np.float32)
        occupancy_predictions = np.asarray([row["predicted_occupancy"] for row in calibration], dtype=np.float32)
        occupancy_cal_loss = float(np.mean(np.abs(occupancy_targets - occupancy_predictions)))
        count_targets = np.asarray(
            [min(2, len(_segments_from_params(row["labels"], {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}))) for row in calibration],
            dtype=np.int64,
        )
        count_probs = np.stack([row["event_count_probs"] for row in calibration])
        count_cal_loss = float(-np.mean(np.log(count_probs[np.arange(len(count_targets)), count_targets] + eps)))
        cal_loss = (
            frame_cal_loss
            + args.occupancy_loss_weight * occupancy_cal_loss
            + args.event_count_loss_weight * count_cal_loss
        )
        if cal_loss < best_loss - 1e-4:
            best_loss = cal_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std, {"best_calibration_objective": best_loss, "epochs_ran": epoch + 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--predictions-out", required=True)
    ap.add_argument("--checkpoints-dir", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibration-ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--occupancy-loss-weight", type=float, default=0.0)
    ap.add_argument("--event-count-loss-weight", type=float, default=0.0)
    ap.add_argument("--require-calibration-over-full-span", action="store_true")
    ap.add_argument("--calibration-min-gain", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--method", default="Compact ActionFormer-style local-attention pyramid on frozen R3D-18")
    ap.add_argument("--task", default="rgb_temporal_localization")
    ap.add_argument("--input-description", default="frozen RGB features")
    args = ap.parse_args()
    records = load_signal_dataset(args.data_dir)
    outer_folds = make_stratified_folds(records, args.folds, args.seed)
    fold_rows, bundle_folds = [], []
    for fold, outer_val in enumerate(outer_folds):
        fit_idx, cal_idx = nested_indices(records, outer_val, args.calibration_ratio, args.seed + fold)
        model, mean, std, training = train_fold(records, fit_idx, cal_idx, args, args.seed + fold)
        calibration = predict(model, records, cal_idx, mean, std, args.device, args.batch_size)
        params, cal_metrics = select_postprocess_params(calibration, iou_threshold=0.3)
        validation = predict(model, records, outer_val, mean, std, args.device, args.batch_size)
        full_span_params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1}
        full_span_calibration = [{**row, "probs": np.ones_like(row["probs"])} for row in calibration]
        full_span_metrics = evaluate_records(full_span_calibration, full_span_params, iou_threshold=0.3)
        selected_source = "model"
        selected_validation = validation
        selected_params = params
        if args.require_calibration_over_full_span and (
            cal_metrics["segment"]["f1"]
            <= full_span_metrics["segment"]["f1"] + args.calibration_min_gain
        ):
            selected_source = "full_span"
            selected_validation = [{**row, "probs": np.ones_like(row["probs"])} for row in validation]
            selected_params = full_span_params
        metrics = evaluate_records(selected_validation, selected_params, iou_threshold=0.3)
        checkpoint = None
        checkpoint_sha256 = None
        if args.checkpoints_dir:
            checkpoint_path = Path(args.checkpoints_dir) / f"fold_{fold}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "params": selected_params, "config": vars(args)}, checkpoint_path)
            checkpoint = str(checkpoint_path)
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        fold_rows.append({"fold": fold, "fit_names": [records[i].name for i in fit_idx], "calibration_names": [records[i].name for i in cal_idx], "val_names": [records[i].name for i in outer_val], "training": training, "params": selected_params, "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256, "calibration": cal_metrics, "full_span_calibration": full_span_metrics, "selected_source": selected_source, "metrics": metrics})
        bundle_folds.append({"fold": fold, "val_names": [r["name"] for r in selected_validation], "predictions": [[list(s) for s in _segments_from_params(r["probs"], selected_params)] for r in selected_validation], "labels": [r["labels"].astype(int).tolist() for r in selected_validation]})
        print(f"fold={fold} selected={selected_source} f1={metrics['segment']['f1']:.4f}", flush=True)
    method = args.method
    summary = {"method": method, "implementation": "independent adapter, not official ActionFormer", "input": args.input_description, "protocol": "nested video-level CV", "config": vars(args), "folds": fold_rows, "aggregate": {"frame": aggregate(fold_rows, "frame"), "segment": aggregate(fold_rows, "segment")}}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bundle = {"schema_version": "round2-prediction-bundle-v1", "task": args.task, "method": method, "folds": bundle_folds}
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions_out).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
