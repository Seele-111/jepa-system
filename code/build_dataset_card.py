#!/usr/bin/env python3
"""Build an auditable dataset card from annotation/event-token directories.

The card intentionally preserves unknown metadata as ``unknown``. It never
guesses a generator from a filename.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"annotation must be an object: {path}")
    return value


def _annotation_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.json") if not p.name.startswith("."))


def _find_events(value: dict[str, Any]) -> list[dict[str, Any]]:
    events = value.get("annotations", value.get("events", []))
    return [item for item in events if isinstance(item, dict)] if isinstance(events, list) else []


def _video_name(value: dict[str, Any], path: Path) -> str:
    return str(value.get("video_name") or value.get("name") or path.stem)


def _metadata_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row.get("video_name") or row.get("name") or row.get("filename") or ""): {
                str(k): str(v) for k, v in row.items() if v not in (None, "")
            }
            for row in rows
        }


def _load_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return set()
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return {Path(str(item)).name for item in value}
    except json.JSONDecodeError:
        pass
    return {Path(line.strip()).name for line in text.splitlines() if line.strip()}


def _probe_video(video_dir: Path | None, name: str) -> dict[str, str]:
    if video_dir is None:
        return {}
    path = video_dir / name
    if not path.exists():
        return {}
    try:
        import cv2  # type: ignore
        capture = cv2.VideoCapture(str(path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        result: dict[str, str] = {}
        if width > 0 and height > 0:
            result["resolution"] = f"{width}x{height}"
        if fps > 0:
            result["fps"] = f"{fps:g}"
        return result
    except Exception:
        try:
            completed = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            stream = json.loads(completed.stdout).get("streams", [{}])[0]
            result = {}
            if stream.get("width") and stream.get("height"):
                result["resolution"] = f"{int(stream['width'])}x{int(stream['height'])}"
            rate = str(stream.get("r_frame_rate", ""))
            if "/" in rate:
                numerator, denominator = rate.split("/", 1)
                if float(denominator):
                    result["fps"] = f"{float(numerator) / float(denominator):g}"
            return result
        except Exception:
            return {}


def _merge_binary_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def build_card(
    annotation_dir: Path,
    role: str,
    metadata_csv: Path | None = None,
    names_file: Path | None = None,
    video_dir: Path | None = None,
) -> dict[str, Any]:
    metadata = _metadata_index(metadata_csv)
    allowed_names = _load_names(names_file)
    videos: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    total_frames = total_positive = total_events = total_raw_annotations = 0
    for path in _annotation_files(annotation_dir):
        if names_file is not None and path.resolve() == names_file.resolve():
            continue
        value = _read_json(path)
        name = _video_name(value, path)
        if allowed_names is not None and name not in allowed_names:
            continue
        events = _find_events(value)
        frame_count = value.get("total_frames", value.get("num_frames", "unknown"))
        frame_count_int = int(frame_count) if isinstance(frame_count, (int, float)) else None
        intervals: list[tuple[int, int]] = []
        for event in events:
            try:
                start = int(event.get("start_frame", event.get("start", 0)))
                end = int(event.get("end_frame", event.get("end", -1)))
            except (TypeError, ValueError):
                continue
            if frame_count_int is not None:
                start = max(0, min(start, frame_count_int - 1))
                end = max(0, min(end, frame_count_int - 1))
            if end >= start:
                intervals.append((start, end))
            categories[str(event.get("category", "unknown"))] += 1
        binary_events = _merge_binary_intervals(intervals)
        positive = sum(end - start + 1 for start, end in binary_events)
        total_events += len(binary_events)
        total_raw_annotations += len(events)
        if frame_count_int is not None:
            total_frames += frame_count_int
            total_positive += min(frame_count_int, positive)
        row = {
            "video_name": name,
            "role": role,
            "frame_count": frame_count_int if frame_count_int is not None else "unknown",
            "event_count": len(binary_events),
            "raw_annotation_count": len(events),
            "positive_frames": positive,
            "positive_frame_ratio": (positive / frame_count_int) if frame_count_int else "unknown",
            "generator": "unknown",
            "resolution": "unknown",
            "fps": str(value.get("fps", "unknown")),
            "prompt_id": "unknown",
            "split": role,
            "annotation_file": str(path),
        }
        row.update(_probe_video(video_dir, name))
        for key in ("generator", "resolution", "fps", "prompt_id", "split"):
            if key in metadata.get(name, {}):
                row[key] = metadata[name][key]
        videos.append(row)
    return {
        "schema_version": "round2-dataset-card-v1",
        "role": role,
        "annotation_dir": str(annotation_dir),
        "videos": videos,
        "summary": {
            "videos": len(videos),
            "frames": total_frames,
            "positive_frames": total_positive,
            "events": total_events,
            "raw_annotations": total_raw_annotations,
            "positive_frame_ratio": total_positive / total_frames if total_frames else "unknown",
            "categories": dict(sorted(categories.items())),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("full215", "clean79_diagnostic", "fresh_blind", "auxiliary"))
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--names-file", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    card = build_card(args.annotation_dir, args.role, args.metadata_csv, args.names_file, args.video_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output), "summary": card["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
