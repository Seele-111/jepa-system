#!/usr/bin/env python3
"""Validate a filled JEPA binary review CSV before applying corrections."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from apply_review_corrections import APPLY_DECISION, parse_segments_field


ALLOWED_DECISIONS = {
    "",
    "keep_original",
    "model_fp",
    "annotation_missing",
    "replace_video_segments",
    "uncertain",
}


def _npz_arrays(path: Path) -> list[np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return [np.asarray(archive[key]) for key in archive.files]


def _load_video_names(data_dir: Path, n: int) -> list[str]:
    path = data_dir / "video_names.json"
    if not path.exists():
        return [f"video_{idx:04d}" for idx in range(n)]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(data[idx]) if idx < len(data) else f"video_{idx:04d}" for idx in range(n)]


def _row_video_idx(row: dict[str, str], name_to_idx: dict[str, int]) -> int | None:
    video = str(row.get("video", "")).strip()
    if video in name_to_idx:
        return name_to_idx[video]
    try:
        return int(str(row.get("video_idx", "")).strip())
    except ValueError:
        return None


def _error(row_number: int, video: str, message: str) -> dict[str, Any]:
    return {"row": int(row_number), "video": video, "message": message}


def validate_review_csv(review_csv: str | Path, data_dir: str | Path) -> dict[str, Any]:
    data_root = Path(data_dir)
    labels = [(np.asarray(item).reshape(-1) > 0).astype(np.int64) for item in _npz_arrays(data_root / "labels.npz")]
    names = _load_video_names(data_root, len(labels))
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    errors: list[dict[str, Any]] = []
    decision_rows = 0
    replacement_rows = 0
    review_path = Path(review_csv)
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, 2):
            video = str(row.get("video", "")).strip()
            decision = str(row.get("review_decision", "")).strip()
            if decision:
                decision_rows += 1
            if decision not in ALLOWED_DECISIONS:
                errors.append(_error(row_number, video, f"invalid review_decision: {decision!r}"))
                continue
            if decision == APPLY_DECISION:
                replacement_rows += 1
                raw_segments = str(row.get("corrected_error_segments", "")).strip()
                if raw_segments == "":
                    errors.append(_error(row_number, video, "replace_video_segments requires corrected_error_segments"))
                    continue
                idx = _row_video_idx(row, name_to_idx)
                if idx is None or idx < 0 or idx >= len(labels):
                    errors.append(_error(row_number, video, "row cannot be mapped to a dataset video"))
                    continue
                try:
                    segments = parse_segments_field(raw_segments)
                except ValueError as exc:
                    errors.append(_error(row_number, video, str(exc)))
                    continue
                length = int(len(labels[idx]))
                for start, end in segments:
                    if start < 0 or end < 0 or start >= length or end >= length:
                        errors.append(
                            _error(
                                row_number,
                                video,
                                f"segment {list((start, end))} out of range for label length {length}",
                            )
                        )
    return {
        "review_csv": str(review_path),
        "data_dir": str(data_root),
        "rows_checked": max(0, row_number - 1) if "row_number" in locals() else 0,
        "decision_rows": int(decision_rows),
        "replacement_rows": int(replacement_rows),
        "errors": errors,
        "ok": len(errors) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_packet/review_packet.csv")
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = validate_review_csv(args.review_csv, args.data_dir)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
