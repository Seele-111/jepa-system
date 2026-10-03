#!/usr/bin/env python3
"""Build a clean-anchor manifest from desktop annotations and JEPA event data.

The manifest is a protocol artifact, not a new final test. It records which
cleaner desktop annotations align with an existing JEPA event dataset and how
much the binary labels differ.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DesktopAnnotation:
    video_name: str
    labels: np.ndarray
    fps: float | None
    total_frames: int
    n_segments: int
    source_json: str


def _npz_arrays(path: Path) -> list[np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return [np.asarray(archive[key]) for key in archive.files]


def annotation_to_labels(annotation: dict) -> np.ndarray:
    total_frames = int(annotation.get("total_frames") or 0)
    if total_frames <= 0:
        raise ValueError("annotation total_frames must be positive")
    labels = np.zeros(total_frames, dtype=np.int64)
    for segment in annotation.get("annotations", []) or []:
        start = int(segment.get("start_frame", 0))
        end = int(segment.get("end_frame", start))
        if end < start:
            start, end = end, start
        start = max(0, min(total_frames - 1, start))
        end = max(0, min(total_frames - 1, end))
        labels[start : end + 1] = 1
    return labels


def load_desktop_annotations(annotations_dir: str | Path) -> dict[str, DesktopAnnotation]:
    root = Path(annotations_dir)
    if not root.exists():
        raise FileNotFoundError(f"missing annotations directory: {root}")
    out: dict[str, DesktopAnnotation] = {}
    for path in sorted(root.glob("*_annotations.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        video_name = str(data.get("video_name") or data.get("video") or path.name.replace("_annotations.json", ".mp4"))
        labels = annotation_to_labels(data)
        out[video_name] = DesktopAnnotation(
            video_name=video_name,
            labels=labels,
            fps=None if data.get("fps") is None else float(data.get("fps")),
            total_frames=int(len(labels)),
            n_segments=int(len(data.get("annotations", []) or [])),
            source_json=str(path),
        )
    return out


def _load_event_labels(event_data_dir: str | Path) -> tuple[list[str], list[np.ndarray]]:
    root = Path(event_data_dir)
    names_path = root / "video_names.json"
    labels_path = root / "labels.npz"
    if not names_path.exists():
        raise FileNotFoundError(f"missing {names_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing {labels_path}")
    names = [str(name) for name in json.loads(names_path.read_text(encoding="utf-8"))]
    labels = [(np.asarray(item).reshape(-1) > 0).astype(np.int64) for item in _npz_arrays(labels_path)]
    if len(names) != len(labels):
        raise ValueError(f"video_names/labels count mismatch: {len(names)} vs {len(labels)}")
    return names, labels


def _compare_labels(annotation_labels: np.ndarray, dataset_labels: np.ndarray) -> dict:
    n = min(int(len(annotation_labels)), int(len(dataset_labels)))
    ann = (np.asarray(annotation_labels[:n]).reshape(-1) > 0).astype(np.int64)
    data = (np.asarray(dataset_labels[:n]).reshape(-1) > 0).astype(np.int64)
    disagreement = int(np.not_equal(ann, data).sum())
    ann_pos = int(ann.sum())
    data_pos = int(data.sum())
    intersection = int(np.logical_and(ann > 0, data > 0).sum())
    union = int(np.logical_or(ann > 0, data > 0).sum())
    return {
        "aligned_frames": int(n),
        "annotation_frames": int(len(annotation_labels)),
        "dataset_frames": int(len(dataset_labels)),
        "annotation_positive_frames": ann_pos,
        "dataset_positive_frames": data_pos,
        "label_disagreement_frames": disagreement,
        "label_agreement_rate": float(1.0 - disagreement / max(1, n)),
        "positive_iou": float(intersection / max(1, union)),
    }


def build_clean_anchor_manifest(
    annotations_dir: str | Path,
    event_data_dir: str | Path,
    output_path: str | Path,
) -> dict:
    annotations = load_desktop_annotations(annotations_dir)
    names, dataset_labels = _load_event_labels(event_data_dir)
    dataset_by_name = {name: idx for idx, name in enumerate(names)}
    rows = []
    matched_names = sorted(set(annotations) & set(dataset_by_name))
    for name in matched_names:
        annotation = annotations[name]
        dataset_idx = int(dataset_by_name[name])
        compare = _compare_labels(annotation.labels, dataset_labels[dataset_idx])
        rows.append(
            {
                "video_name": name,
                "dataset_index": dataset_idx,
                "fps": annotation.fps,
                "annotation_segments": annotation.n_segments,
                "annotation_json": annotation.source_json,
                **compare,
            }
        )
    annotation_only = sorted(set(annotations) - set(dataset_by_name))
    dataset_only = sorted(set(dataset_by_name) - set(annotations))
    summary = {
        "annotations_dir": str(annotations_dir),
        "event_data_dir": str(event_data_dir),
        "total_annotation_videos": int(len(annotations)),
        "total_dataset_videos": int(len(names)),
        "matched_videos": int(len(rows)),
        "annotation_only_videos": int(len(annotation_only)),
        "dataset_only_videos": int(len(dataset_only)),
        "matched_positive_frames_annotation": int(sum(row["annotation_positive_frames"] for row in rows)),
        "matched_positive_frames_dataset": int(sum(row["dataset_positive_frames"] for row in rows)),
        "matched_label_disagreement_frames": int(sum(row["label_disagreement_frames"] for row in rows)),
        "matched_aligned_frames": int(sum(row["aligned_frames"] for row in rows)),
        "mean_label_agreement_rate": float(np.mean([row["label_agreement_rate"] for row in rows])) if rows else 0.0,
        "mean_positive_iou": float(np.mean([row["positive_iou"] for row in rows])) if rows else 0.0,
        "annotation_only_sample": annotation_only[:20],
        "dataset_only_sample": dataset_only[:20],
        "matched": rows,
        "protocol_note": (
            "Use as cleaner calibration/diagnostic anchor only. Do not treat it as held-out test "
            "unless the method and evaluation protocol are frozen beforehand."
        ),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", default="/mnt/c/Users/admin/Desktop/测试/annotations")
    parser.add_argument("--event-data-dir", default="/home/zzy/jepa_data/clean69_event_jepa_v2")
    parser.add_argument("--output", default="/home/zzy/jepa_data/clean_anchor_manifest.json")
    args = parser.parse_args()
    manifest = build_clean_anchor_manifest(args.annotations_dir, args.event_data_dir, args.output)
    printable = {key: value for key, value in manifest.items() if key != "matched"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
