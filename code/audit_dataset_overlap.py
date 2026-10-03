#!/usr/bin/env python3
"""Audit exact video-name overlap between evaluation datasets."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def load_names(path: Path) -> set[str]:
    if path.is_dir():
        path = path / "video_names.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"expected a JSON list: {path}")
    return {str(value) for value in values}


def audit(datasets: dict[str, set[str]]) -> dict:
    pairs = []
    for left, right in combinations(sorted(datasets), 2):
        overlap = sorted(datasets[left] & datasets[right])
        pairs.append({"left": left, "right": right, "overlap_count": len(overlap), "overlap": overlap})
    return {"datasets": {name: len(values) for name, values in sorted(datasets.items())}, "pairs": pairs, "zero_overlap": all(row["overlap_count"] == 0 for row in pairs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="NAME=directory-or-video_names.json")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    datasets = {}
    for item in args.dataset:
        name, sep, path = item.partition("=")
        if not sep or not name:
            raise ValueError(f"invalid --dataset value: {item!r}")
        datasets[name] = load_names(Path(path))
    report = audit(datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
