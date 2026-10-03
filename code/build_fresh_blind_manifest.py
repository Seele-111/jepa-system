#!/usr/bin/env python3
"""Select and card a zero-overlap fresh-blind candidate from test annotations."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from audit_dataset_overlap import load_names
from build_dataset_card import build_card


def bucket(row: dict) -> str:
    ratio_value = row.get("positive_frame_ratio")
    if ratio_value in (None, "unknown"):
        raise ValueError("row has unknown positive_frame_ratio")
    ratio = float(ratio_value)
    length = int(row.get("frame_count") or 0)
    ratio_key = "zero" if ratio == 0 else "low" if ratio <= 0.25 else "mid" if ratio <= 0.6 else "high"
    length_key = "short" if length <= 49 else "medium" if length <= 61 else "long"
    return f"{ratio_key}_{length_key}"


def select_names(card: dict, excluded: set[str], count: int, seed: int) -> tuple[list[str], dict[str, int]]:
    candidates = [row for row in card.get("videos", []) if row["video_name"] not in excluded and row.get("frame_count") != "unknown" and row.get("positive_frame_ratio") not in (None, "unknown")]
    groups: dict[str, list[str]] = {}
    for row in candidates:
        groups.setdefault(bucket(row), []).append(str(row["video_name"]))
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[str] = []
    selected_counts = {key: 0 for key in groups}
    # Round-robin prevents the blind set from being dominated by long/high-ratio clips.
    while len(selected) < min(count, len(candidates)):
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                selected_counts[key] += 1
                progressed = True
        if not progressed:
            break
    return sorted(selected), selected_counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-dir", required=True, type=Path)
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--exclude", action="append", required=True, type=Path, help="video_names.json or newline-separated names")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    card = build_card(args.annotation_dir, "fresh_blind_candidate", video_dir=args.video_dir)
    excluded = set().union(*(load_names(path) for path in args.exclude))
    (args.output_dir / "excluded_names.json").write_text(json.dumps(sorted(excluded), indent=2), encoding="utf-8")
    names, bucket_counts = select_names(card, excluded, args.count, args.seed)
    names_path = args.output_dir / "video_names.json"
    names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    selected_card = build_card(args.annotation_dir, "fresh_blind_candidate", names_file=names_path, video_dir=args.video_dir)
    (args.output_dir / "dataset_card.json").write_text(json.dumps(selected_card, indent=2), encoding="utf-8")
    report = {"seed": args.seed, "requested": args.count, "selected": len(names), "excluded_count": len(excluded), "bucket_counts": bucket_counts, "excluded_sources": [str(path) for path in args.exclude], "status": "candidate_only_not_evaluated"}
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
