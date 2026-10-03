#!/usr/bin/env python3
"""Build a non-authoritative draft of manual JEPA review suggestions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _as_segment(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    start, end = int(value[0]), int(value[1])
    if start > end:
        start, end = end, start
    return [start, end]


def _candidate_is_useful(row: dict[str, Any], min_iou: float = 0.5) -> bool:
    candidate = _as_segment(row.get("best_jepa_candidate"))
    if candidate is None:
        return False
    try:
        candidate_iou = float(row.get("best_candidate_iou") or 0.0)
    except (TypeError, ValueError):
        candidate_iou = 0.0
    return candidate_iou >= float(min_iou)


def suggest_row(row: dict[str, Any]) -> dict[str, Any]:
    action = str(row.get("action", ""))
    candidate = _as_segment(row.get("best_jepa_candidate"))
    target = _as_segment(row.get("target_segment"))
    if action in {"review_split_or_relabel_parent", "review_parent_boundary_and_matching"} and _candidate_is_useful(row):
        return {
            "suggested_decision": "replace_video_segments",
            "suggested_error_segments": [candidate],
            "suggestion_note": "高置信 JEPA candidate；需要人工确认是否确实是二值错误片段边界。",
        }
    if action == "verify_false_positive_or_missing_label":
        return {
            "suggested_decision": "uncertain",
            "suggested_error_segments": [] if target is None else [target],
            "suggestion_note": "需要人工确认是假阳性还是原标签漏标；不要直接套用。",
        }
    return {
        "suggested_decision": "uncertain",
        "suggested_error_segments": [] if target is None else [target],
        "suggestion_note": "证据不足；请人工查看可视化包后决定。",
    }


def _load_items(review_json: Path) -> list[dict[str, Any]]:
    data = json.loads(review_json.read_text(encoding="utf-8"))
    items = data.get("items", data.get("rows", [])) if isinstance(data, dict) else data
    return [dict(item) for item in items if isinstance(item, dict)]


def build_suggestion_draft(review_json: str | Path, output_dir: str | Path, top_k: int = 37) -> dict[str, Any]:
    review_path = Path(review_json)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = _load_items(review_path)[: int(top_k)]
    rows: list[dict[str, Any]] = []
    for item in items:
        suggestion = suggest_row(item)
        rows.append(
            {
                "kind": item.get("kind"),
                "priority": item.get("priority"),
                "video": item.get("video"),
                "video_idx": item.get("video_idx"),
                "action": item.get("action"),
                "target_segment": item.get("target_segment"),
                "current_prediction": item.get("current_prediction"),
                "best_jepa_candidate": item.get("best_jepa_candidate"),
                "best_candidate_iou": item.get("best_candidate_iou"),
                "best_candidate_rank": item.get("best_candidate_rank"),
                "suggested_decision": suggestion["suggested_decision"],
                "suggested_error_segments": suggestion["suggested_error_segments"],
                "suggestion_note": suggestion["suggestion_note"],
                "human_final_decision": "",
                "human_final_segments": "",
                "human_notes": "",
            }
        )
    result = {
        "source_review_json": str(review_path),
        "n_items": len(rows),
        "warning": "Draft only. Do not apply automatically; copy confirmed decisions into review_packet.csv.",
        "rows": rows,
    }
    (out / "review_suggestion_draft.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "kind",
        "priority",
        "video",
        "video_idx",
        "action",
        "target_segment",
        "current_prediction",
        "best_jepa_candidate",
        "best_candidate_iou",
        "best_candidate_rank",
        "suggested_decision",
        "suggested_error_segments",
        "suggestion_note",
        "human_final_decision",
        "human_final_segments",
        "human_notes",
    ]
    with (out / "review_suggestion_draft.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = row.copy()
            for key in ["target_segment", "current_prediction", "best_jepa_candidate", "suggested_error_segments"]:
                encoded[key] = json.dumps(encoded[key], ensure_ascii=False)
            writer.writerow(encoded)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-json", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_packet/review_packet.json")
    parser.add_argument("--output-dir", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_suggestion_draft")
    parser.add_argument("--top-k", type=int, default=37)
    args = parser.parse_args()
    result = build_suggestion_draft(args.review_json, args.output_dir, args.top_k)
    print(json.dumps({"out": args.output_dir, "n_items": result["n_items"]}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
