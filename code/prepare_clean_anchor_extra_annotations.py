#!/usr/bin/env python3
"""Prepare annotation-only desktop videos for clean-anchor JEPA extraction."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _annotation_filename(video_name: str) -> str:
    return f"{Path(video_name).stem}_annotations.json"


def prepare_extra_annotations(
    manifest_path: str | Path,
    source_annotations_dir: str | Path,
    output_annotations_dir: str | Path,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    video_names = [str(name) for name in manifest.get("annotation_only_sample", [])]
    source_root = Path(source_annotations_dir)
    out_root = Path(output_annotations_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for video_name in video_names:
        src = source_root / _annotation_filename(video_name)
        if not src.exists():
            missing.append(video_name)
            continue
        dst = out_root / src.name
        shutil.copy2(src, dst)
        copied.append(video_name)
    summary = {
        "manifest_path": str(manifest_path),
        "source_annotations_dir": str(source_annotations_dir),
        "output_annotations_dir": str(output_annotations_dir),
        "requested": int(len(video_names)),
        "copied": int(len(copied)),
        "missing": missing,
        "copied_video_names": copied,
        "next_step": (
            "Run extract_true_jepa_segment_signals.py on output_annotations_dir, then build_jepa_event_dataset.py "
            "to create an expanded clean-anchor event-token dataset."
        ),
    }
    (out_root / "extra_annotations_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/home/zzy/jepa_data/clean69_desktop_clean_anchor_manifest.json")
    parser.add_argument("--source-annotations-dir", default="/mnt/c/Users/admin/Desktop/测试/annotations")
    parser.add_argument("--output-annotations-dir", default="/home/zzy/jepa_data/clean_anchor_extra10_annotations")
    args = parser.parse_args()
    summary = prepare_extra_annotations(
        manifest_path=args.manifest,
        source_annotations_dir=args.source_annotations_dir,
        output_annotations_dir=args.output_annotations_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not summary["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
