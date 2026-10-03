#!/usr/bin/env python3
"""
Build segment-localization training data from true V/I-JEPA detector runs.

Output is binary-only:
  signals: [T, 3] = physics_raw, corruption_raw, composite
  labels:  [T]    = 1 inside any annotated error segment, else 0
No category/severity targets are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from evaluate_true_vjepa_ablation import labels_from_annotations, load_annotations


PYTHON_BIN = "/home/zzy/vjepa2-main/vjepa-env/bin/python"
PIPELINE = "/home/zzy/jepa_data/detect_and_report_v4.py"


def _safe_id(video_path: str) -> str:
    stem = Path(video_path).stem[:80]
    digest = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def _run_detector(anno: dict, run_dir: Path, max_frames: int, max_keyframes: int, timeout: int) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN,
        PIPELINE,
        "--video",
        anno["video_path"],
        "--output",
        str(run_dir),
        "--max-frames",
        str(max_frames),
        "--max-keyframes",
        str(max_keyframes),
        "--use-true-vjepa",
        "--use-true-ijepa",
        "--threshold",
        "0.4",
        "--min-gap",
        "1",
        "--min-length",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=str(Path(PIPELINE).parent), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:] or proc.stdout[-4000:] or f"exit {proc.returncode}")
    return json.loads((run_dir / "vjepa_timeseries.json").read_text(encoding="utf-8"))


def _signals_from_timeseries(timeseries: dict) -> np.ndarray:
    physics = np.asarray(timeseries["physics_raw"], dtype=np.float32)
    corruption = np.asarray(timeseries["corruption_raw"], dtype=np.float32)
    composite = np.asarray(timeseries["composite"], dtype=np.float32)
    n = min(len(physics), len(corruption), len(composite))
    return np.stack([physics[:n], corruption[:n], composite[:n]], axis=1).astype(np.float32)


def build_dataset(args) -> int:
    out = Path(args.output)
    cache_dir = out / "cache"
    runs_dir = out / "runs"
    out.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_runs:
        runs_dir.mkdir(parents=True, exist_ok=True)

    video_roots = [Path(value.strip()) for value in str(getattr(args, "video_roots", "")).split(",") if value.strip()]
    annotations = load_annotations(Path(args.annotations), args.limit, video_roots=video_roots or None)
    all_signals: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    names: list[str] = []
    errors: list[dict] = []

    for idx, anno in enumerate(annotations, 1):
        sid = _safe_id(anno["video_path"])
        cache_file = cache_dir / f"{sid}.npz"
        print(f"[{idx}/{len(annotations)}] {anno.get('video_name', sid)}", flush=True)
        try:
            if cache_file.exists() and not args.rebuild:
                item = np.load(cache_file, allow_pickle=True)
                signals = item["signals"]
                labels = item["labels"]
            else:
                run_dir = runs_dir / sid if args.keep_runs else out / "_latest_run"
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                timeseries = _run_detector(anno, run_dir, args.max_frames, args.max_keyframes, args.timeout)
                signals = _signals_from_timeseries(timeseries)
                labels = labels_from_annotations(int(anno.get("total_frames", len(signals))), anno.get("annotations", []))
                n = min(len(signals), len(labels))
                signals = signals[:n]
                labels = labels[:n].astype(np.int64)
                np.savez_compressed(cache_file, signals=signals, labels=labels, video_path=anno["video_path"])
                if not args.keep_runs and run_dir.exists():
                    shutil.rmtree(run_dir)

            all_signals.append(signals)
            all_labels.append(labels)
            names.append(anno.get("video_name", Path(anno["video_path"]).name))
            print(f"  frames={len(labels)} positives={int(labels.sum())} signal_shape={signals.shape}", flush=True)
        except Exception as exc:
            errors.append({"video": anno.get("video_name", anno.get("video_path")), "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)

    if not all_signals:
        (out / "summary.json").write_text(json.dumps({"errors": errors}, indent=2), encoding="utf-8")
        return 2

    np.savez_compressed(out / "signals.npz", *all_signals)
    np.savez_compressed(out / "labels.npz", *all_labels)
    (out / "video_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    summary = {
        "n_videos": len(all_signals),
        "n_errors": len(errors),
        "frames": int(sum(len(x) for x in all_labels)),
        "positive_frames": int(sum(x.sum() for x in all_labels)),
        "segments_ignored_categories": True,
        "signal_columns": ["true_vjepa_raw", "true_ijepa_raw", "dual_jepa_composite"],
        "errors": errors,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-roots", default="", help="Comma-separated directories used to resolve stale video paths")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()
    return build_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
