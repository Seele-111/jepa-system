#!/usr/bin/env python3
"""Summarize frozen fresh-candidate localization errors without tuning."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_segment_locator import contiguous_segments


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    left, right = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, right - left + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union else 0.0


def analyze(bundle: dict, annotation_dir: Path, thresholds=(0.3, 0.5)) -> list[dict]:
    rows = []
    fold = bundle["folds"][0]
    for name, pred, labels in zip(fold["val_names"], fold["predictions"], fold["labels"]):
        y = np.asarray(labels, dtype=np.int64)
        gt = [tuple(map(int, x)) for x in contiguous_segments(y)]
        prediction = [tuple(map(int, x)) for x in pred]
        meta_path = annotation_dir / f"{Path(name).stem}_annotations.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        base = {
            "video_name": name,
            "fps": meta.get("fps"),
            "total_frames": int(meta.get("total_frames", len(y))),
            "duration_sec": (len(y) / float(meta["fps"])) if meta.get("fps") else None,
            "gt_segments": len(gt),
            "pred_segments": len(prediction),
            "positive_frames": int(y.sum()),
            "positive_fraction": float(y.mean()) if len(y) else 0.0,
            "gt_lengths": [b - a + 1 for a, b in gt],
            "pred_lengths": [b - a + 1 for a, b in prediction],
        }
        for threshold in thresholds:
            matched_pred, matched_gt = set(), set()
            pairs = sorted(
                ((iou(p, g), pi, gi) for pi, p in enumerate(prediction) for gi, g in enumerate(gt) if iou(p, g) >= threshold),
                reverse=True,
            )
            for _, pi, gi in pairs:
                if pi not in matched_pred and gi not in matched_gt:
                    matched_pred.add(pi)
                    matched_gt.add(gi)
            fp_segments = [prediction[i] for i in range(len(prediction)) if i not in matched_pred]
            fn_segments = [gt[i] for i in range(len(gt)) if i not in matched_gt]
            pred_mask = np.zeros(len(y), dtype=bool)
            for a, b in prediction:
                pred_mask[max(0, a) : min(len(y), b + 1)] = True
            gt_mask = y.astype(bool)
            base[f"iou_{threshold:.1f}"] = {
                "matched_segments": len(matched_pred),
                "fp_segments": len(fp_segments),
                "fn_segments": len(fn_segments),
                "fp_frames": int(np.logical_and(pred_mask, ~gt_mask).sum()),
                "fn_frames": int(np.logical_and(~pred_mask, gt_mask).sum()),
                "fp_segment_lengths": [b - a + 1 for a, b in fp_segments],
                "fn_segment_lengths": [b - a + 1 for a, b in fn_segments],
            }
        rows.append(base)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    rows = analyze(json.loads(args.bundle.read_text(encoding="utf-8")), args.annotations)
    def grouped(lo: float, hi: float) -> dict:
        subset = [r for r in rows if r["fps"] is not None and lo <= float(r["fps"]) < hi]
        return {
            "videos": len(subset),
            "fp_frames_iou_0.3": int(sum(r["iou_0.3"]["fp_frames"] for r in subset)),
            "fn_frames_iou_0.3": int(sum(r["iou_0.3"]["fn_frames"] for r in subset)),
            "fp_segments_iou_0.3": int(sum(r["iou_0.3"]["fp_segments"] for r in subset)),
            "positive_frames": int(sum(r["positive_frames"] for r in subset)),
        }

    summary = {
        "schema_version": "round2-fresh-error-analysis-v1",
        "protocol": "post-hoc descriptive analysis of frozen predictions",
        "videos": len(rows),
        "rows": rows,
        "aggregate": {
            "mean_positive_fraction": float(np.mean([r["positive_fraction"] for r in rows])),
            "mean_pred_segments": float(np.mean([r["pred_segments"] for r in rows])),
            "mean_fps": float(np.mean([r["fps"] for r in rows if r["fps"] is not None])),
            "fps_bins": {"8-9": grouped(0, 10), "10-15": grouped(10, 16), "16-24": grouped(16, 25)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = ["video_name", "fps", "total_frames", "duration_sec", "gt_segments", "pred_segments", "positive_frames", "positive_fraction"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    print(json.dumps(summary["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
