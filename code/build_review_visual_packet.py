#!/usr/bin/env python3
"""Export contact sheets for high-priority JEPA review items."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np


Segment = list[int] | tuple[int, int] | None
ProbeDuration = Callable[[Path], float]
ExtractFrame = Callable[[Path, float, Path], bool]


def _npz_arrays(path: Path) -> list[np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return [np.asarray(archive[key]) for key in archive.files]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_segment_frames(segment: Segment, label_length: int, count: int = 3) -> list[int]:
    if segment is None:
        return []
    start = int(segment[0])
    end = int(segment[1])
    if start > end:
        start, end = end, start
    n = max(1, int(label_length))
    start = max(0, min(n - 1, start))
    end = max(0, min(n - 1, end))
    if start == end:
        return [start]
    if count <= 1:
        return [start]
    values = np.linspace(start, end, int(count))
    frames = sorted({int(round(value)) for value in values})
    return frames


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return 0.0
    try:
        return max(0.0, float(proc.stdout.strip()))
    except ValueError:
        return 0.0


def _run_ffmpeg_extract(video_path: Path, timestamp: float, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, float(timestamp)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and output_path.exists()


def ffmpeg_extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    if _run_ffmpeg_extract(video_path, timestamp, output_path):
        return True
    fallback_timestamp = max(0.0, float(timestamp) - 0.08)
    if fallback_timestamp != float(timestamp):
        return _run_ffmpeg_extract(video_path, fallback_timestamp, output_path)
    return False


def frame_to_time(frame_idx: int, label_length: int, duration: float) -> float:
    if duration <= 0.0:
        return 0.0
    n = max(1, int(label_length))
    timestamp = (float(frame_idx) + 0.5) / float(n) * float(duration)
    tail_margin = min(0.12, max(0.02, float(duration) * 0.02))
    return max(0.0, min(float(timestamp), float(duration) - tail_margin))


def _segment_fields(item: dict[str, Any]) -> list[tuple[str, Segment]]:
    return [
        ("target", item.get("target_segment")),
        ("current", item.get("current_prediction")),
        ("candidate", item.get("best_jepa_candidate")),
    ]


def _load_label_lengths(data_dir: Path) -> dict[int, int]:
    labels = _npz_arrays(data_dir / "labels.npz")
    return {idx: int(len(item)) for idx, item in enumerate(labels)}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120]


def build_visual_packet(
    review_json: str | Path,
    output_dir: str | Path,
    data_dir: str | Path = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2",
    top_k: int = 12,
    frames_per_segment: int = 3,
    probe_duration: ProbeDuration = ffprobe_duration,
    extract_frame: ExtractFrame = ffmpeg_extract_frame,
) -> dict[str, Any]:
    review_path = Path(review_json)
    out = Path(output_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    data = _load_json(review_path)
    items = data.get("items", data.get("rows", [])) if isinstance(data, dict) else data
    if not isinstance(items, list):
        items = []
    selected = [dict(item) for item in items[: int(top_k)] if isinstance(item, dict)]
    lengths = _load_label_lengths(Path(data_dir))

    manifest_items: list[dict[str, Any]] = []
    for item_idx, item in enumerate(selected, 1):
        video_path_raw = item.get("video_path")
        video_path = Path(video_path_raw) if video_path_raw else None
        duration = probe_duration(video_path) if video_path and video_path.exists() else 0.0
        video_idx = int(item.get("video_idx") or 0)
        label_length = lengths.get(video_idx, 1)
        captures: list[dict[str, Any]] = []
        for field_name, segment in _segment_fields(item):
            for frame_idx in sample_segment_frames(segment, label_length=label_length, count=frames_per_segment):
                timestamp = frame_to_time(frame_idx, label_length=label_length, duration=duration)
                stem = _safe_name(f"{item_idx:03d}_{field_name}_{frame_idx}_{item.get('video') or video_idx}")
                rel_path = Path("frames") / f"{stem}.jpg"
                output_path = out / rel_path
                ok = bool(video_path and video_path.exists() and extract_frame(video_path, timestamp, output_path))
                captures.append(
                    {
                        "field": field_name,
                        "frame": int(frame_idx),
                        "timestamp": float(timestamp),
                        "image": str(rel_path) if ok else None,
                        "ok": bool(ok),
                    }
                )
        manifest_items.append({**item, "label_length": label_length, "duration": duration, "captures": captures})

    result = {
        "source_review_json": str(review_path),
        "data_dir": str(data_dir),
        "output_dir": str(out),
        "n_items": len(manifest_items),
        "items": manifest_items,
    }
    (out / "visual_review_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_html(out / "visual_review.html", result)
    return result


def _write_html(path: Path, result: dict[str, Any]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>JEPA Visual Review</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#111}"
        ".item{background:white;border:1px solid #ddd;padding:16px;margin:16px 0}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}"
        "img{max-width:100%;border:1px solid #ccc}.meta{font-size:13px;color:#444}</style>",
        "</head><body>",
        "<h1>JEPA Visual Review</h1>",
        f"<p>Items: {int(result['n_items'])}</p>",
    ]
    for idx, item in enumerate(result["items"], 1):
        parts.extend(
            [
                "<div class='item'>",
                f"<h2>{idx}. {html.escape(str(item.get('video')))} "
                f"{html.escape(str(item.get('kind')))} priority={html.escape(str(item.get('priority')))}</h2>",
                f"<p class='meta'>action={html.escape(str(item.get('action')))} "
                f"reason={html.escape(str(item.get('reason')))}</p>",
                f"<p class='meta'>target={html.escape(str(item.get('target_segment')))} "
                f"current={html.escape(str(item.get('current_prediction')))} "
                f"candidate={html.escape(str(item.get('best_jepa_candidate')))}</p>",
                f"<p class='meta'>video={html.escape(str(item.get('video_path')))}</p>",
                "<div class='grid'>",
            ]
        )
        for capture in item.get("captures", []):
            label = f"{capture.get('field')} frame={capture.get('frame')} t={float(capture.get('timestamp') or 0.0):.2f}s"
            parts.append("<div>")
            parts.append(f"<p class='meta'>{html.escape(label)}</p>")
            if capture.get("image"):
                parts.append(f"<img src='{html.escape(str(capture['image']))}' alt='{html.escape(label)}'>")
            else:
                parts.append("<p class='meta'>frame extraction failed</p>")
            parts.append("</div>")
        parts.extend(["</div>", "</div>"])
    parts.extend(["</body></html>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-json", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_packet/review_packet.json")
    parser.add_argument("--output-dir", default="/home/zzy/jepa_data/segment_train_full_event_v2_visual_review_packet")
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--frames-per-segment", type=int, default=3)
    args = parser.parse_args()
    result = build_visual_packet(
        review_json=args.review_json,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        top_k=args.top_k,
        frames_per_segment=args.frames_per_segment,
    )
    print(
        json.dumps(
            {"out": args.output_dir, "n_items": result["n_items"]},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
