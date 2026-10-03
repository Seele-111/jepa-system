#!/usr/bin/env python3
"""Apply explicit binary label corrections from a JEPA review packet CSV."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from build_review_packet import parse_segment


APPLY_DECISION = "replace_video_segments"


def _npz_arrays(path: Path) -> list[np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return [np.asarray(archive[key]) for key in archive.files]


def parse_segments_field(value: Any) -> list[tuple[int, int]]:
    if value is None or value == "":
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid corrected_error_segments: {value!r}") from exc
    if parsed is None or parsed == "":
        return []
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2 and not isinstance(parsed[0], (list, tuple)):
        segment = parse_segment(parsed)
        return [] if segment is None else [segment]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"corrected_error_segments must be a segment list, got {type(parsed).__name__}")
    segments: list[tuple[int, int]] = []
    for item in parsed:
        segment = parse_segment(item)
        if segment is None:
            raise ValueError(f"invalid corrected segment: {item!r}")
        segments.append(segment)
    return segments


def _segments_to_labels(segments: list[tuple[int, int]], length: int) -> np.ndarray:
    labels = np.zeros(int(length), dtype=np.int64)
    for start, end in segments:
        left = max(0, int(start))
        right = min(int(length) - 1, int(end))
        if right >= left:
            labels[left : right + 1] = 1
    return labels


def _load_names(data_dir: Path, n: int) -> list[str]:
    path = data_dir / "video_names.json"
    if not path.exists():
        return [f"video_{idx:04d}" for idx in range(n)]
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list):
        raise ValueError(f"video_names.json must contain a list: {path}")
    return [str(names[idx]) if idx < len(names) else f"video_{idx:04d}" for idx in range(n)]


def _read_review_replacements(review_csv: Path, names: list[str]) -> tuple[dict[int, list[tuple[int, int]]], int]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    replacements: dict[int, list[tuple[int, int]]] = {}
    n_rows = 0
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n_rows += 1
            if str(row.get("review_decision", "")).strip() != APPLY_DECISION:
                continue
            video = str(row.get("video", "")).strip()
            if video and video in name_to_idx:
                idx = name_to_idx[video]
            else:
                try:
                    idx = int(str(row.get("video_idx", "")).strip())
                except ValueError as exc:
                    raise ValueError(f"review row cannot be mapped to a dataset video: {row}") from exc
            if idx < 0 or idx >= len(names):
                raise ValueError(f"review row video_idx is out of range: {idx}")
            segments = parse_segments_field(row.get("corrected_error_segments", ""))
            if idx in replacements and replacements[idx] != segments:
                raise ValueError(f"conflicting replacements for video {names[idx]}: {replacements[idx]} vs {segments}")
            replacements[idx] = segments
    return replacements, n_rows


def apply_review_corrections(data_dir: str | Path, review_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    data_root = Path(data_dir)
    review_path = Path(review_csv)
    out = Path(output_dir)
    signals = _npz_arrays(data_root / "signals.npz")
    labels = [(np.asarray(item).reshape(-1) > 0).astype(np.int64) for item in _npz_arrays(data_root / "labels.npz")]
    if len(signals) != len(labels):
        raise ValueError(f"signals/labels count mismatch: {len(signals)} vs {len(labels)}")
    names = _load_names(data_root, len(labels))
    replacements, n_rows = _read_review_replacements(review_path, names)

    corrected_labels = [item.copy() for item in labels]
    changed: list[dict[str, Any]] = []
    for idx, segments in sorted(replacements.items()):
        new_labels = _segments_to_labels(segments, len(corrected_labels[idx]))
        if not np.array_equal(corrected_labels[idx], new_labels):
            changed.append(
                {
                    "video_idx": int(idx),
                    "video": names[idx],
                    "old_positive_frames": int(corrected_labels[idx].sum()),
                    "new_positive_frames": int(new_labels.sum()),
                    "corrected_error_segments": [list(segment) for segment in segments],
                }
            )
        corrected_labels[idx] = new_labels

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "signals.npz", *signals)
    np.savez_compressed(out / "labels.npz", *corrected_labels)
    (out / "video_names.json").write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")
    for filename in ["summary.json"]:
        source = data_root / filename
        if source.exists():
            try:
                summary = json.loads(source.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                shutil.copy2(source, out / filename)
                continue
            summary["review_corrections"] = {
                "source_data_dir": str(data_root),
                "review_csv": str(review_path),
                "apply_decision": APPLY_DECISION,
                "n_review_rows": int(n_rows),
                "n_replacement_decisions": int(len(replacements)),
                "n_changed_videos": int(len(changed)),
                "changed_videos": changed,
            }
            (out / filename).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            (out / "summary.json").write_text(
                json.dumps(
                    {
                        "source_data_dir": str(data_root),
                        "review_corrections": {
                            "review_csv": str(review_path),
                            "apply_decision": APPLY_DECISION,
                            "n_review_rows": int(n_rows),
                            "n_replacement_decisions": int(len(replacements)),
                            "n_changed_videos": int(len(changed)),
                            "changed_videos": changed,
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    result = {
        "output_dir": str(out),
        "n_review_rows": int(n_rows),
        "n_replacement_decisions": int(len(replacements)),
        "n_changed_videos": int(len(changed)),
        "changed_videos": changed,
    }
    (out / "review_corrections_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--review-csv", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_packet/review_packet.csv")
    parser.add_argument("--output-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2_review_corrected")
    args = parser.parse_args()
    result = apply_review_corrections(args.data_dir, args.review_csv, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
