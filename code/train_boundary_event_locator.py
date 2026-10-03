#!/usr/bin/env python3
"""
Train a binary temporal boundary locator on JEPA event-token features.

The model predicts only error segments:
  - objectness: frame is inside an error segment
  - start boundary likelihood
  - end boundary likelihood

No category or severity target is used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_segment_locator import (
    TemporalBlock,
    VideoRecord,
    _parse_float_list,
    _parse_int_list,
    compute_normalizer,
    compute_pos_weight,
    contiguous_segments,
    evaluate_records,
    load_signal_dataset,
    normalized_signals,
    probabilities_to_segments,
    select_postprocess_params,
    train_val_split,
)


class TemporalBoundaryLocator(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalBlock(hidden, dilation=1, dropout=dropout),
            TemporalBlock(hidden, dilation=2, dropout=dropout),
            TemporalBlock(hidden, dilation=4, dropout=dropout),
            TemporalBlock(hidden, dilation=8, dropout=dropout),
        )
        self.output = nn.Conv1d(hidden, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, C], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = F.gelu(self.input(x))
        x = self.blocks(x)
        return self.output(x).transpose(1, 2)


def build_boundary_targets(labels: np.ndarray, boundary_radius: int = 3) -> dict[str, np.ndarray]:
    labels = (np.asarray(labels).reshape(-1) > 0).astype(np.float32)
    start = np.zeros_like(labels, dtype=np.float32)
    end = np.zeros_like(labels, dtype=np.float32)
    radius = max(0, int(boundary_radius))
    sigma = max(1.0, radius / 2.0)
    for seg_start, seg_end in contiguous_segments(labels.astype(np.int64)):
        for offset in range(-radius, radius + 1):
            weight = math.exp(-(offset * offset) / (2.0 * sigma * sigma))
            s_idx = seg_start + offset
            e_idx = seg_end + offset
            if 0 <= s_idx < len(labels):
                start[s_idx] = max(start[s_idx], weight)
            if 0 <= e_idx < len(labels):
                end[e_idx] = max(end[e_idx], weight)
    return {"object": labels.astype(np.float32), "start": start, "end": end}


def refine_segments_with_boundaries(
    coarse_segments: list[tuple[int, int]],
    start_probs: np.ndarray,
    end_probs: np.ndarray,
    radius: int,
) -> list[tuple[int, int]]:
    refined: list[tuple[int, int]] = []
    n = len(start_probs)
    for start, end in coarse_segments:
        s0 = max(0, start - radius)
        s1 = min(n - 1, start + radius)
        e0 = max(0, end - radius)
        e1 = min(n - 1, end + radius)
        new_start = int(s0 + np.argmax(start_probs[s0 : s1 + 1]))
        new_end = int(e0 + np.argmax(end_probs[e0 : e1 + 1]))
        if new_end < new_start:
            new_start, new_end = start, end
        refined.append((new_start, new_end))
    return refined


def boundary_loss(logits: torch.Tensor, targets: dict[str, torch.Tensor], pos_weight: float) -> torch.Tensor:
    object_logits = logits[..., 0]
    start_logits = logits[..., 1]
    end_logits = logits[..., 2]
    obj = targets["object"]
    start = targets["start"]
    end = targets["end"]

    obj_loss = F.binary_cross_entropy_with_logits(
        object_logits,
        obj,
        pos_weight=torch.tensor(float(pos_weight), device=logits.device),
    )
    boundary_weight = max(1.0, float((start.numel() - start.sum().item()) / max(1.0, start.sum().item())))
    start_loss = F.binary_cross_entropy_with_logits(
        start_logits,
        start,
        pos_weight=torch.tensor(boundary_weight, device=logits.device),
    )
    end_loss = F.binary_cross_entropy_with_logits(
        end_logits,
        end,
        pos_weight=torch.tensor(boundary_weight, device=logits.device),
    )
    obj_probs = torch.sigmoid(object_logits)
    smooth = torch.abs(obj_probs[:, 1:] - obj_probs[:, :-1]).mean() if obj_probs.shape[1] > 1 else obj_probs.new_tensor(0.0)
    return obj_loss + 0.35 * (start_loss + end_loss) + 0.05 * smooth


def predict_records(
    model: nn.Module,
    records: list[VideoRecord],
    indices: Iterable[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> list[dict]:
    model.eval()
    outputs = []
    with torch.no_grad():
        for idx in indices:
            record = records[idx]
            x = torch.from_numpy(normalized_signals(record, mean, std)).unsqueeze(0).to(device)
            probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy().astype(np.float32)
            outputs.append(
                {
                    "name": record.name,
                    "labels": record.labels.copy(),
                    "probs": probs[:, 0],
                    "start_probs": probs[:, 1],
                    "end_probs": probs[:, 2],
                }
            )
    return outputs


def evaluate_boundary_records(records: list[dict], params: dict, iou_threshold: float = 0.3) -> dict:
    frame_records = [{"labels": r["labels"], "probs": r["probs"]} for r in records]
    frame_and_coarse = evaluate_records(frame_records, params, iou_threshold=iou_threshold)
    tp = fp = fn = 0
    refined_predictions = []
    for record in records:
        coarse = probabilities_to_segments(
            record["probs"],
            threshold=params["threshold"],
            smooth_window=params["smooth_window"],
            min_gap=params["min_gap"],
            min_length=params["min_length"],
        )
        refined = refine_segments_with_boundaries(
            coarse,
            record["start_probs"],
            record["end_probs"],
            int(params.get("refine_radius", 0)),
        )
        refined_predictions.append(refined)
        gt = contiguous_segments(record["labels"])
        matched: set[int] = set()
        local_tp = 0
        for pred in refined:
            best_idx = -1
            best_iou = 0.0
            for idx, target in enumerate(gt):
                if idx in matched:
                    continue
                start = max(pred[0], target[0])
                end = min(pred[1], target[1])
                inter = max(0, end - start + 1)
                union = max(pred[1], target[1]) - min(pred[0], target[0]) + 1
                iou = inter / max(1, union)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched.add(best_idx)
                local_tp += 1
        tp += local_tp
        fp += len(refined) - local_tp
        fn += len(gt) - local_tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    result = dict(frame_and_coarse)
    result["segment"] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    result["predictions"] = refined_predictions
    return result


def select_boundary_params(
    records: list[dict],
    iou_threshold: float = 0.3,
    thresholds: Iterable[float] | None = None,
    smooth_windows: Iterable[int] | None = None,
    min_gaps: Iterable[int] | None = None,
    min_lengths: Iterable[int] | None = None,
    refine_radii: Iterable[int] | None = None,
) -> tuple[dict, dict]:
    base_records = [{"labels": r["labels"], "probs": r["probs"]} for r in records]
    best_params, _ = select_postprocess_params(base_records, iou_threshold=iou_threshold)
    thresholds = list(thresholds if thresholds is not None else np.linspace(0.2, 0.8, 13))
    smooth_windows = list(smooth_windows if smooth_windows is not None else [1, 3, 5])
    min_gaps = list(min_gaps if min_gaps is not None else [0, 1, 2])
    min_lengths = list(min_lengths if min_lengths is not None else [1, 2, 4])
    refine_radii = list(refine_radii if refine_radii is not None else [0, 2, 4, 8, 12])
    best_metrics = None
    best_key = None
    selected = None
    for threshold in thresholds:
        for smooth_window in smooth_windows:
            for min_gap in min_gaps:
                for min_length in min_lengths:
                    for refine_radius in refine_radii:
                        params = {
                            "threshold": float(threshold),
                            "smooth_window": int(smooth_window),
                            "min_gap": int(min_gap),
                            "min_length": int(min_length),
                            "refine_radius": int(refine_radius),
                        }
                        metrics = evaluate_boundary_records(records, params, iou_threshold=iou_threshold)
                        key = (metrics["segment"]["f1"], metrics["segment"]["recall"], metrics["frame"]["f1"])
                        if best_key is None or key > best_key:
                            best_key = key
                            best_metrics = metrics
                            selected = params
    if selected is None:
        selected = {**best_params, "refine_radius": 0}
        best_metrics = evaluate_boundary_records(records, selected, iou_threshold=iou_threshold)
    best_metrics = {k: v for k, v in best_metrics.items() if k != "predictions"}
    return selected, best_metrics


def train_model(
    records: list[VideoRecord],
    train_idx: list[int],
    val_idx: list[int],
    args,
) -> tuple[TemporalBoundaryLocator, dict, np.ndarray, np.ndarray]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mean, std = compute_normalizer(records, train_idx)
    model = TemporalBoundaryLocator(records[0].signals.shape[1], hidden=args.hidden, dropout=args.dropout).to(args.device)
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
            target_np = build_boundary_targets(record.labels, boundary_radius=args.boundary_radius)
            targets = {k: torch.from_numpy(v).unsqueeze(0).to(args.device) for k, v in target_np.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = boundary_loss(logits, targets, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_records = predict_records(model, records, val_idx or train_idx, mean, std, args.device)
        params, metrics = select_boundary_params(
            val_records,
            iou_threshold=args.iou_threshold,
            thresholds=_parse_float_list(args.thresholds),
            smooth_windows=_parse_int_list(args.smooth_windows),
            min_gaps=_parse_int_list(args.min_gaps),
            min_lengths=_parse_int_list(args.min_lengths),
            refine_radii=_parse_int_list(args.refine_radii),
        )
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
    parser.add_argument("--output", default="/home/zzy/jepa_data/boundary_event_locator.pt")
    parser.add_argument("--summary", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--smooth-windows", default="1,3")
    parser.add_argument("--min-gaps", default="0,1,2")
    parser.add_argument("--min-lengths", default="1,2,4")
    parser.add_argument("--refine-radii", default="0,2,4,8")
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
            "task": "binary_error_segment_boundary_localization",
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
    print(f"saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
