#!/usr/bin/env python3
"""
Noise-robust MIL temporal locator for JEPA event-token features.

Binary-only objective:
  - soft frame BCE for coarse supervision
  - positive segment top-k MIL
  - negative interval top-k suppression
  - temporal smoothness
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_segment_locator import (
    TemporalSegmentLocator,
    VideoRecord,
    compute_normalizer,
    compute_pos_weight,
    contiguous_segments,
    load_signal_dataset,
    normalized_signals,
    predict_records,
    select_postprocess_params,
    train_val_split,
)


def build_soft_frame_targets(labels: np.ndarray, margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    labels = (np.asarray(labels).reshape(-1) > 0).astype(np.float32)
    target = labels.copy()
    weight = np.ones_like(labels, dtype=np.float32)
    for start, end in contiguous_segments(labels.astype(np.int64)):
        for offset in range(1, margin + 1):
            softness = 0.25 * (1.0 - (offset - 1) / max(1, margin))
            left = start - offset
            right = end + offset
            if 0 <= left < len(labels):
                target[left] = max(target[left], softness)
                weight[left] = min(weight[left], 0.35)
            if 0 <= right < len(labels):
                target[right] = max(target[right], softness)
                weight[right] = min(weight[right], 0.35)
        boundary = list(range(start, min(end + 1, start + margin))) + list(range(max(start, end - margin + 1), end + 1))
        for idx in boundary:
            weight[idx] = min(weight[idx], 0.65)
    return target.astype(np.float32), weight.astype(np.float32)


def _negative_segments(labels: torch.Tensor) -> list[tuple[int, int]]:
    inv = (labels.detach().cpu().numpy().astype(np.int64) == 0).astype(np.int64)
    return contiguous_segments(inv)


def segment_topk_mil_loss(logits: torch.Tensor, labels: torch.Tensor, topk_fraction: float = 0.3) -> torch.Tensor:
    if logits.ndim != 1:
        logits = logits.reshape(-1)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    losses: list[torch.Tensor] = []
    label_np = labels.detach().cpu().numpy().astype(np.int64)
    for start, end in contiguous_segments(label_np):
        seg_logits = logits[start : end + 1]
        k = max(1, int(round(len(seg_logits) * topk_fraction)))
        top = torch.topk(seg_logits, k=k).values.mean()
        losses.append(F.binary_cross_entropy_with_logits(top, top.new_tensor(1.0)))
    for start, end in _negative_segments(labels):
        seg_logits = logits[start : end + 1]
        if len(seg_logits) < 2:
            continue
        k = max(1, int(round(len(seg_logits) * topk_fraction)))
        top = torch.topk(seg_logits, k=k).values.mean()
        losses.append(F.binary_cross_entropy_with_logits(top, top.new_tensor(0.0)))
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def robust_mil_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float,
    margin: int,
    topk_fraction: float,
) -> torch.Tensor:
    frame_losses = []
    mil_losses = []
    for batch_idx in range(logits.shape[0]):
        target_np, weight_np = build_soft_frame_targets(labels[batch_idx].detach().cpu().numpy(), margin=margin)
        target = torch.from_numpy(target_np).to(logits.device)
        weight = torch.from_numpy(weight_np).to(logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits[batch_idx],
            target,
            reduction="none",
            pos_weight=torch.tensor(float(pos_weight), device=logits.device),
        )
        frame_losses.append((bce * weight).mean())
        mil_losses.append(segment_topk_mil_loss(logits[batch_idx], labels[batch_idx], topk_fraction=topk_fraction))
    probs = torch.sigmoid(logits)
    smooth = torch.abs(probs[:, 1:] - probs[:, :-1]).mean() if probs.shape[1] > 1 else probs.new_tensor(0.0)
    return torch.stack(frame_losses).mean() + 0.65 * torch.stack(mil_losses).mean() + 0.05 * smooth


def train_model(records: list[VideoRecord], train_idx: list[int], val_idx: list[int], args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mean, std = compute_normalizer(records, train_idx)
    model = TemporalSegmentLocator(records[0].signals.shape[1], hidden=args.hidden, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = compute_pos_weight(records, train_idx)
    best_score = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metrics = {}
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        order = train_idx[:]
        random.shuffle(order)
        losses = []
        for idx in order:
            record = records[idx]
            x = torch.from_numpy(normalized_signals(record, mean, std)).unsqueeze(0).to(args.device)
            y = torch.from_numpy(record.labels).unsqueeze(0).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = robust_mil_loss(logits, y, pos_weight, args.margin, args.topk_fraction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_records = predict_records(model, records, val_idx or train_idx, mean, std, args.device)
        params, metrics = select_postprocess_params(val_records, iou_threshold=args.iou_threshold)
        score = float(metrics["segment"]["f1"])
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, "loss": float(np.mean(losses)), "params": params, **metrics}
            no_improve = 0
        else:
            no_improve += 1
        if epoch < 5 or epoch % 10 == 0 or no_improve >= args.patience:
            print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_seg_f1={score:.3f} best={best_score:.3f}", flush=True)
        if no_improve >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_metrics, mean, std


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="/home/zzy/jepa_data/mil_event_locator.pt")
    parser.add_argument("--summary", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--topk-fraction", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_signal_dataset(args.data_dir)
    train_idx, val_idx = train_val_split(records, args.val_ratio, args.seed)
    print(
        f"dataset videos={len(records)} frames={sum(len(r.labels) for r in records)} "
        f"features={records[0].signals.shape[1]} train={len(train_idx)} val={len(val_idx)} device={args.device}",
        flush=True,
    )
    model, metrics, mean, std = train_model(records, train_idx, val_idx, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"in_channels": records[0].signals.shape[1], "hidden": args.hidden, "dropout": args.dropout},
            "normalizer": {"mean": mean.tolist(), "std": std.tolist()},
            "metrics": metrics,
            "task": "binary_error_segment_mil_localization",
        },
        output,
    )
    summary = {
        "data_dir": args.data_dir,
        "output": str(output),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_names": [records[idx].name for idx in train_idx],
        "val_names": [records[idx].name for idx in val_idx],
        "validation": metrics,
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
