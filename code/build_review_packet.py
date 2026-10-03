#!/usr/bin/env python3
"""Build a reviewer-facing packet for the current JEPA localization errors.

The packet is intentionally binary-only: it reviews missed or extra error
fragments and never introduces category, score, or severity targets.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Iterable


Segment = tuple[int, int]


DEFAULT_VIDEO_ROOTS = [
    "/mnt/e/jepa-label/JEPA-data/filtered_videos_train",
    "/mnt/e/jepa-label/JEPA-data/filtered_videos_test",
    "/mnt/c/Users/admin/Desktop/测试",
]


def parse_segment(value: Any) -> Segment | None:
    if value is None or value == "":
        return None
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        return None
    try:
        start = int(parsed[0])
        end = int(parsed[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if start <= end else (end, start)


def _segment_to_list(segment: Segment | None) -> list[int] | None:
    return None if segment is None else [int(segment[0]), int(segment[1])]


def _segment_len(segment: Segment | None) -> int:
    if segment is None:
        return 0
    return max(0, int(segment[1]) - int(segment[0]) + 1)


def infer_review_action(row: dict[str, Any]) -> str:
    kind = str(row.get("kind", "")).upper()
    reason = str(row.get("reason", ""))
    best_prediction = parse_segment(row.get("best_prediction"))
    best_candidate = parse_segment(row.get("best_candidate"))
    candidate_iou = float(row.get("best_candidate_iou") or 0.0)
    parent_len = _segment_len(best_prediction)
    child_len = _segment_len(best_candidate)

    if kind == "FP":
        return "verify_false_positive_or_missing_label"
    if reason == "wide_prediction_matching_conflict":
        if best_candidate is not None and candidate_iou >= 0.5 and parent_len >= max(20, child_len * 2):
            return "review_split_or_relabel_parent"
        return "review_parent_boundary_and_matching"
    if reason == "candidate_scored_too_low":
        return "review_candidate_threshold_or_teacher_weight"
    if reason == "candidate_generation_miss":
        return "inspect_jepa_signal_or_missing_candidate"
    if reason == "event_set_selection_conflict":
        return "review_topology_set_selection"
    if reason == "oracle_safe_parent_split":
        return "review_oracle_safe_split"
    return "manual_review"


def _load_rows(priority_path: Path) -> list[dict[str, Any]]:
    data = json.loads(priority_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("rows", data.get("items", []))
    else:
        rows = []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_video_names(data_dir: str | Path | None) -> list[str]:
    if not data_dir:
        return []
    path = Path(data_dir) / "video_names.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(item) for item in data] if isinstance(data, list) else []


def _video_name_for_row(row: dict[str, Any], video_names: list[str]) -> str | None:
    video = row.get("video")
    if video:
        return str(video)
    try:
        idx = int(row.get("video_idx"))
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(video_names):
        return video_names[idx]
    return None


def _build_video_index(video_roots: Iterable[str | Path]) -> dict[str, str]:
    index: dict[str, str] = {}
    for root_raw in video_roots:
        root = Path(root_raw)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                continue
            index.setdefault(path.name, str(path))
    return index


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_item(row: dict[str, Any], video_names: list[str], video_index: dict[str, str]) -> dict[str, Any]:
    video_name = _video_name_for_row(row, video_names)
    segment = parse_segment(row.get("segment"))
    best_prediction = parse_segment(row.get("best_prediction"))
    best_candidate = parse_segment(row.get("best_candidate"))
    action = infer_review_action(row)
    item = {
        "kind": str(row.get("kind", "")),
        "priority": _int_or_none(row.get("priority")) or 0,
        "fold": _int_or_none(row.get("fold")),
        "video_idx": _int_or_none(row.get("video_idx")),
        "video": video_name,
        "video_path": video_index.get(video_name or ""),
        "target_segment": _segment_to_list(segment),
        "current_prediction": _segment_to_list(best_prediction),
        "best_jepa_candidate": _segment_to_list(best_candidate),
        "best_prediction_iou": _float_or_none(row.get("best_prediction_iou")),
        "best_candidate_iou": _float_or_none(row.get("best_candidate_iou")),
        "best_candidate_rank": _int_or_none(row.get("best_candidate_rank")),
        "covering_candidates": _int_or_none(row.get("covering_candidates")),
        "reason": row.get("reason"),
        "action": action,
        "review_decision": "",
        "corrected_error_segments": "",
        "notes": "",
    }
    if item["kind"].upper() == "FN":
        item["review_question"] = (
            "这个视频里 current_prediction 是否吞掉了多个真实错误片段？"
            "如果是，请用 corrected_error_segments 写出应保留的二值错误片段。"
        )
    elif item["kind"].upper() == "FP":
        item["review_question"] = (
            "current_prediction 是否真的是错误片段？如果是漏标，请补为错误片段；"
            "如果不是，保持空并记录为假阳性。"
        )
    else:
        item["review_question"] = "核查该 oracle/topology 候选是否能无副作用替换当前宽预测。"
    return item


def _write_csv(path: Path, items: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind",
        "priority",
        "fold",
        "video_idx",
        "video",
        "video_path",
        "target_segment",
        "current_prediction",
        "best_jepa_candidate",
        "best_prediction_iou",
        "best_candidate_iou",
        "best_candidate_rank",
        "covering_candidates",
        "reason",
        "action",
        "review_question",
        "review_decision",
        "corrected_error_segments",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = item.copy()
            for key in ("target_segment", "current_prediction", "best_jepa_candidate"):
                row[key] = "" if row[key] is None else json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def _write_markdown(path: Path, result: dict[str, Any], top_k: int) -> None:
    lines = [
        "# JEPA Error Review Packet",
        "",
        f"- Source: `{result['source_priority_path']}`",
        f"- Items exported: `{result['n_items']}`",
        f"- Counts: `{json.dumps(result['counts'], ensure_ascii=False)}`",
        f"- Missing local video paths: `{result['missing_video_paths']}`",
        "",
        "## Top Items",
        "",
    ]
    for idx, item in enumerate(result["items"][:top_k], 1):
        lines.extend(
            [
                f"### {idx}. {item['kind']} priority={item['priority']} {item.get('video') or '<unknown video>'}",
                "",
                f"- Action: `{item['action']}`",
                f"- Reason: `{item.get('reason')}`",
                f"- Video path: `{item.get('video_path') or ''}`",
                f"- GT / target segment: `{item.get('target_segment')}`",
                f"- Current prediction: `{item.get('current_prediction')}`",
                f"- Best JEPA candidate: `{item.get('best_jepa_candidate')}`",
                f"- IoU current/candidate: `{item.get('best_prediction_iou')}` / `{item.get('best_candidate_iou')}`",
                f"- Candidate rank / covering candidates: `{item.get('best_candidate_rank')}` / `{item.get('covering_candidates')}`",
                f"- Review question: {item.get('review_question')}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_review_packet(
    priority_path: str | Path,
    output_dir: str | Path,
    video_roots: Iterable[str | Path] = DEFAULT_VIDEO_ROOTS,
    data_dir: str | Path | None = "/home/zzy/jepa_data/segment_train_full_event_jepa_v2",
    top_k: int = 50,
) -> dict[str, Any]:
    priority = Path(priority_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(priority)
    rows.sort(key=lambda row: (int(row.get("priority") or 0), str(row.get("kind")), str(row.get("video_idx"))), reverse=True)
    selected = rows[: int(top_k)] if int(top_k) > 0 else rows
    video_names = _load_video_names(data_dir)
    video_index = _build_video_index(video_roots)
    items = [_normalise_item(row, video_names, video_index) for row in selected]
    counts = {kind: sum(1 for item in items if item["kind"] == kind) for kind in sorted({item["kind"] for item in items})}
    result = {
        "source_priority_path": str(priority),
        "video_roots": [str(root) for root in video_roots],
        "data_dir": str(data_dir) if data_dir else None,
        "n_source_rows": len(rows),
        "n_items": len(items),
        "counts": counts,
        "missing_video_paths": sum(1 for item in items if item.get("video") and not item.get("video_path")),
        "items": items,
        "review_protocol": [
            "Only decide binary error-fragment boundaries.",
            "Do not add category, severity, or subjective quality scores.",
            "For wide parent conflicts, prefer explicit split/merge/boundary decisions.",
            "Freeze any corrected train labels before re-running strict CV.",
        ],
    }
    (out / "review_packet.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out / "review_packet.csv", items)
    _write_markdown(out / "review_packet.md", result, min(int(top_k), len(items)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_priority.json")
    parser.add_argument("--output-dir", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_packet")
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
    parser.add_argument("--video-roots", default=",".join(DEFAULT_VIDEO_ROOTS))
    parser.add_argument("--top-k", type=int, default=91)
    args = parser.parse_args()
    roots = [item for item in args.video_roots.split(",") if item.strip()]
    result = build_review_packet(
        priority_path=args.priority,
        output_dir=args.output_dir,
        video_roots=roots,
        data_dir=args.data_dir,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "out": args.output_dir,
                "n_source_rows": result["n_source_rows"],
                "n_items": result["n_items"],
                "counts": result["counts"],
                "missing_video_paths": result["missing_video_paths"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
