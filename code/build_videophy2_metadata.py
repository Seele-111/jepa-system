#!/usr/bin/env python3
"""Recover verified VideoPhy2 metadata by exact source filename matching."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse


def source_filename(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def build_index(source_csv: Path) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    with source_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = source_filename(str(row.get("video_url", "")))
            if name:
                index[name].append({str(k): str(v) for k, v in row.items() if k and v not in (None, "")})
    return index


def recover_rows(source_csv: Path, names: list[str]) -> tuple[list[dict[str, str]], dict[str, object]]:
    index = build_index(source_csv)
    output: list[dict[str, str]] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for name in names:
        matches = index.get(Path(name).name, [])
        if len(matches) != 1:
            (ambiguous if len(matches) > 1 else missing).append(name)
            continue
        source = matches[0]
        caption = source.get("caption", "")
        output.append(
            {
                "video_name": Path(name).name,
                "generator": source.get("model_name", "unknown"),
                "prompt_id": hashlib.sha256(caption.encode("utf-8")).hexdigest()[:16] if caption else "unknown",
                "caption": caption,
                "source_url": source.get("video_url", ""),
                "source_category": source.get("category", "unknown"),
            }
        )
    return output, {
        "requested": len(names),
        "matched": len(output),
        "missing": len(missing),
        "ambiguous": len(ambiguous),
        "missing_names": missing,
        "ambiguous_names": ambiguous,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--dataset-card", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    card = json.loads(args.dataset_card.read_text(encoding="utf-8"))
    names = [str(row["video_name"]) for row in card.get("videos", [])]
    rows, report = recover_rows(args.source_csv, names)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["video_name", "generator", "prompt_id", "caption", "source_url", "source_category"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report["generator_counts"] = {
        generator: sum(row["generator"] == generator for row in rows)
        for generator in sorted({row["generator"] for row in rows})
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("requested", "matched", "missing", "ambiguous", "generator_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
