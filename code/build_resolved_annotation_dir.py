#!/usr/bin/env python3
"""Build a copy of annotation JSON files with resolved video paths.

This utility does not edit labels. It only rewrites stale video_path fields so
downstream binary error-segment localization pipelines can find the videos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _candidate_paths(data: dict, annotation_dir: Path, video_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    raw_path = data.get("video_path") or data.get("video")
    video_name = data.get("video_name")
    if raw_path:
        raw = Path(str(raw_path))
        candidates.append(raw)
        candidates.append(annotation_dir.parent / raw.name)
        for root in video_roots:
            candidates.append(root / raw.name)
    if video_name:
        name = Path(str(video_name)).name
        candidates.append(annotation_dir.parent / name)
        for root in video_roots:
            candidates.append(root / name)
    return candidates


def resolve_annotations(input_dir: Path, output_dir: Path, video_roots: list[Path], limit: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in input_dir.glob("*.json") if not path.name.startswith("."))
    if limit is not None:
        files = files[: int(limit)]

    written = 0
    unresolved: list[dict] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            unresolved.append({"file": str(path), "reason": "not_object"})
            continue
        resolved = None
        for candidate in _candidate_paths(data, input_dir, video_roots):
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            unresolved.append(
                {
                    "file": str(path),
                    "video_name": data.get("video_name"),
                    "video_path": data.get("video_path") or data.get("video"),
                    "reason": "video_not_found",
                }
            )
            continue
        data["video_path"] = str(resolved)
        if "video_name" not in data:
            data["video_name"] = resolved.name
        (output_dir / path.name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "video_roots": [str(path) for path in video_roots],
        "input_files": len(files),
        "resolved_files": written,
        "unresolved_files": len(unresolved),
        "unresolved": unresolved,
        "task": "binary_error_segment_localization_path_resolution",
    }
    (output_dir / "resolution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-roots", required=True, help="Comma-separated video root directories.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = resolve_annotations(
        Path(args.input_dir),
        Path(args.output_dir),
        [Path(item) for item in args.video_roots.split(",") if item.strip()],
        args.limit,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["resolved_files"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
