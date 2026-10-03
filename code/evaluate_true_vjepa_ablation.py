#!/usr/bin/env python3
"""
Evaluate encoder-proxy V-JEPA against true masked-predictor V-JEPA.

This script deliberately evaluates the actual report pipeline twice per video:
once with the historical encoder proxy, and once with --use-true-vjepa. The
result is a reproducible ablation table for claims about JEPA objective use.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


DEFAULT_PYTHON = "/home/zzy/vjepa2-main/vjepa-env/bin/python"
DEFAULT_PIPELINE = "/home/zzy/jepa_data/detect_and_report_v4.py"
DEFAULT_ANNOTATIONS = "/home/zzy/jepa_data/clean_test_dataset/annotations"


@dataclass(frozen=True)
class SegmentMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class FrameMetrics:
    precision: float
    recall: float
    f1: float
    auroc: float | None
    auprc: float | None
    tp: int
    fp: int
    fn: int


def labels_from_annotations(total_frames: int, annotations: list[dict]) -> np.ndarray:
    labels = np.zeros(max(0, int(total_frames)), dtype=np.int64)
    for ann in annotations:
        start = max(0, int(ann.get("start_frame", 0)))
        end = min(len(labels) - 1, int(ann.get("end_frame", start)))
        if end >= start:
            labels[start : end + 1] = 1
    return labels


def segments_from_binary(values: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0
    for idx, value in enumerate(values.astype(bool)):
        if value and not in_segment:
            start = idx
            in_segment = True
        elif not value and in_segment:
            segments.append((start, idx - 1))
            in_segment = False
    if in_segment:
        segments.append((start, len(values) - 1))
    return segments


def segment_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0, end - start + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / max(1, union)


def segment_metrics(
    predicted: list[tuple[int, int]],
    target: list[tuple[int, int]],
    iou_threshold: float = 0.3,
) -> SegmentMetrics:
    matched: set[int] = set()
    tp = 0
    for pred in predicted:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(target):
            if idx in matched:
                continue
            iou = segment_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.add(best_idx)
            tp += 1
    fp = len(predicted) - tp
    fn = len(target) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return SegmentMetrics(precision, recall, f1, tp, fp, fn)


def binary_frame_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float, float, int, int, int]:
    pred = scores >= threshold
    truth = labels.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1, tp, fp, fn


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, weights=ranks)
    avg_ranks = rank_sums / counts
    tied_ranks = avg_ranks[inverse]
    pos_rank_sum = tied_ranks[: len(pos)].sum()
    return float((pos_rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int((labels == 1).sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * (y == 1)).sum() / positives)


def frame_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> FrameMetrics:
    precision, recall, f1, tp, fp, fn = binary_frame_metrics(labels, scores, threshold)
    return FrameMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=_rank_auc(labels, scores),
        auprc=_average_precision(labels, scores),
        tp=tp,
        fp=fp,
        fn=fn,
    )


def extract_vjepa_scores(timeseries: dict) -> np.ndarray:
    """Return JEPA objective scores, preferring raw scores over post-processed sigmoid scores."""
    key = "physics_raw" if "physics_raw" in timeseries else "physics_sig"
    return np.asarray(timeseries[key], dtype=np.float64)


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(scores, kernel, mode="same")


def select_threshold(labels: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    best_threshold = float(grid[0])
    best_f1 = -1.0
    for threshold in grid:
        _, _, f1, *_ = binary_frame_metrics(labels, scores, float(threshold))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline(
    video_path: str,
    output_dir: Path,
    use_true_vjepa: bool,
    python_bin: str,
    pipeline_path: str,
    threshold: float,
    max_frames: int,
    max_keyframes: int,
    timeout: int,
) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        pipeline_path,
        "--video",
        video_path,
        "--output",
        str(output_dir),
        "--threshold",
        str(threshold),
        "--min-gap",
        "1",
        "--min-length",
        "1",
        "--max-frames",
        str(max_frames),
        "--max-keyframes",
        str(max_keyframes),
    ]
    if use_true_vjepa:
        cmd.append("--use-true-vjepa")

    proc = subprocess.run(cmd, cwd=str(Path(pipeline_path).parent), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:] or proc.stdout[-4000:] or f"exit {proc.returncode}")

    report = _read_json(output_dir / "report.json")
    timeseries = _read_json(output_dir / "vjepa_timeseries.json")
    return {"report": report, "timeseries": timeseries}


def evaluate_method(records: list[dict], score_key: str, threshold: float, iou_threshold: float) -> dict:
    labels = np.concatenate([r["labels"] for r in records])
    scores = np.concatenate([r[score_key] for r in records])
    fm = frame_metrics(labels, scores, threshold)

    tp = fp = fn = 0
    for record in records:
        pred = segments_from_binary(record[score_key] >= threshold)
        gt = segments_from_binary(record["labels"])
        sm = segment_metrics(pred, gt, iou_threshold=iou_threshold)
        tp += sm.tp
        fp += sm.fp
        fn += sm.fn
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "threshold": threshold,
        "frame": asdict(fm),
        "segment": asdict(SegmentMetrics(precision, recall, f1, tp, fp, fn)),
    }


def _infer_video_roots(annotation_dir: Path) -> list[Path]:
    roots: list[Path] = [annotation_dir.parent]
    for parent in [annotation_dir, *annotation_dir.parents]:
        if parent.name == "JEPA-data":
            parts = {p.name for p in annotation_dir.parents}
            if "testdata" in parts:
                roots.extend([parent / "filtered_videos_test", parent / "filtered_videos_train"])
            else:
                roots.extend([parent / "filtered_videos_train", parent / "filtered_videos_test"])
            break
    return roots


def _resolve_video_path(data: dict, annotation_dir: Path, video_roots: list[Path] | None = None) -> str | None:
    raw_path = data.get("video_path") or data.get("video")
    candidates: list[Path] = []
    if raw_path:
        candidates.append(Path(raw_path))
    video_name = data.get("video_name")
    if video_name:
        candidates.append(annotation_dir.parent / video_name)
        for root in (video_roots or _infer_video_roots(annotation_dir)):
            candidates.append(root / video_name)
    if raw_path:
        candidates.append(annotation_dir.parent / Path(raw_path).name)
        for root in (video_roots or _infer_video_roots(annotation_dir)):
            candidates.append(root / Path(raw_path).name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_annotations(path: Path, limit: int | None, video_roots: list[Path] | None = None) -> list[dict]:
    files = [
        file
        for file in sorted(path.glob("*.json"))
        if not file.name.startswith(".") and "backup" not in file.name.lower()
    ]
    if limit is not None:
        files = files[:limit]
    annotations = []
    for file in files:
        data = _read_json(file)
        if not isinstance(data, dict):
            continue
        video_path = _resolve_video_path(data, path, video_roots=video_roots)
        if video_path:
            data["video_path"] = video_path
            annotations.append(data)
    return annotations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--video-roots", default="",
                        help="Comma-separated directories used to resolve video_name when JSON paths are stale")
    parser.add_argument("--output", default="/home/zzy/jepa_data/true_vjepa_ablation_results")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    video_roots = [Path(item) for item in args.video_roots.split(",") if item.strip()]
    annos = load_annotations(Path(args.annotations), args.limit, video_roots=video_roots or None)
    records: list[dict] = []
    errors: list[dict] = []

    for idx, anno in enumerate(annos, 1):
        stem = Path(anno.get("video_name", f"video_{idx:04d}")).stem
        labels = labels_from_annotations(int(anno.get("total_frames", 0)), anno.get("annotations", []))
        print(f"[{idx}/{len(annos)}] {stem} labels={int(labels.sum())}/{len(labels)}", flush=True)

        try:
            proxy = run_pipeline(
                anno["video_path"],
                out_root / "runs" / stem / "proxy",
                False,
                args.python,
                args.pipeline,
                args.threshold,
                args.max_frames,
                args.max_keyframes,
                args.timeout,
            )
            true = run_pipeline(
                anno["video_path"],
                out_root / "runs" / stem / "true_vjepa",
                True,
                args.python,
                args.pipeline,
                args.threshold,
                args.max_frames,
                args.max_keyframes,
                args.timeout,
            )

            proxy_scores = smooth_scores(extract_vjepa_scores(proxy["timeseries"]), args.smooth_window)
            true_scores = smooth_scores(extract_vjepa_scores(true["timeseries"]), args.smooth_window)
            n = min(len(labels), len(proxy_scores), len(true_scores))
            record = {
                "video_name": anno.get("video_name", stem),
                "video_path": anno["video_path"],
                "labels": labels[:n],
                "proxy_scores": proxy_scores[:n],
                "true_scores": true_scores[:n],
                "proxy_usage": proxy["timeseries"].get("jepa_usage"),
                "true_usage": true["timeseries"].get("jepa_usage"),
            }
            records.append(record)
            print(
                f"  proxy={record['proxy_usage']} true={record['true_usage']} "
                f"n={n} max_proxy={proxy_scores.max():.3f} max_true={true_scores.max():.3f}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"video_name": anno.get("video_name", stem), "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)

    if not records:
        (out_root / "summary.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
        return 2

    all_labels = np.concatenate([r["labels"] for r in records])
    threshold_grid = np.linspace(0.05, 0.95, 37)
    proxy_scores_all = np.concatenate([r["proxy_scores"] for r in records])
    true_scores_all = np.concatenate([r["true_scores"] for r in records])
    proxy_threshold, proxy_val_f1 = select_threshold(all_labels, proxy_scores_all, threshold_grid)
    true_threshold, true_val_f1 = select_threshold(all_labels, true_scores_all, threshold_grid)

    summary = {
        "n_videos": len(records),
        "n_errors": len(errors),
        "selection_note": "Thresholds selected on this run only. For paper results, select thresholds on validation split and report test once.",
        "smooth_window": args.smooth_window,
        "proxy": evaluate_method(records, "proxy_scores", proxy_threshold, args.iou_threshold),
        "true_vjepa": evaluate_method(records, "true_scores", true_threshold, args.iou_threshold),
        "threshold_selection": {
            "proxy": {"threshold": proxy_threshold, "frame_f1": proxy_val_f1},
            "true_vjepa": {"threshold": true_threshold, "frame_f1": true_val_f1},
        },
        "usage_expectation": {
            "proxy": "encoder_proxy",
            "true_vjepa": "true_vjepa_predictor",
            "ijepa": "encoder_proxy",
        },
        "errors": errors,
    }

    per_video = []
    for record in records:
        per_video.append(
            {
                "video_name": record["video_name"],
                "video_path": record["video_path"],
                "n_frames": int(len(record["labels"])),
                "positive_frames": int(record["labels"].sum()),
                "proxy_usage": record["proxy_usage"],
                "true_usage": record["true_usage"],
                "proxy_score_max": float(record["proxy_scores"].max()),
                "true_score_max": float(record["true_scores"].max()),
            }
        )

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "per_video.json").write_text(json.dumps(per_video, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
