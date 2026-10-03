#!/usr/bin/env python3
"""Build a compact human-review queue for JEPA localization errors."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn-attribution", default="/home/zzy/jepa_data/segment_train_full_event_v2_switcher_fn_attribution.json")
    parser.add_argument("--topology-oracle", default="/home/zzy/jepa_data/segment_train_full_event_v2_switcher_topology_oracle.json")
    parser.add_argument("--out-json", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_priority.json")
    parser.add_argument("--out-csv", default="/home/zzy/jepa_data/segment_train_full_event_v2_review_priority.csv")
    args = parser.parse_args()

    fn_data = json.loads(Path(args.fn_attribution).read_text(encoding="utf-8"))
    oracle_data = json.loads(Path(args.topology_oracle).read_text(encoding="utf-8"))
    rows = []
    for row in fn_data.get("fn_rows", []):
        priority = 100
        if row.get("reason") == "wide_prediction_matching_conflict":
            priority += 50
        priority += min(25, int(row.get("covering_candidates", 0)))
        priority += int(max(0.0, float(row.get("best_candidate_iou", 0.0))) * 25)
        rows.append(
            {
                "kind": "FN",
                "priority": int(priority),
                "fold": row.get("fold"),
                "video_idx": row.get("video_idx"),
                "video": row.get("video"),
                "segment": row.get("gt"),
                "reason": row.get("reason"),
                "best_prediction": row.get("best_prediction"),
                "best_prediction_iou": row.get("best_prediction_iou"),
                "best_candidate": row.get("best_candidate"),
                "best_candidate_iou": row.get("best_candidate_iou"),
                "best_candidate_rank": row.get("best_candidate_rank"),
                "covering_candidates": row.get("covering_candidates"),
                "note": "Check whether one broad prediction should be split into multiple binary error fragments.",
            }
        )
    for row in fn_data.get("fp_rows", []):
        rows.append(
            {
                "kind": "FP",
                "priority": 90,
                "fold": row.get("fold"),
                "video_idx": row.get("video_idx"),
                "video": None,
                "segment": row.get("prediction"),
                "reason": "current_false_positive",
                "best_prediction": row.get("prediction"),
                "best_prediction_iou": None,
                "best_candidate": None,
                "best_candidate_iou": None,
                "best_candidate_rank": None,
                "covering_candidates": None,
                "note": "Check whether current prediction is truly an error fragment or annotation missing/noisy.",
            }
        )
    oracle_events = oracle_data.get("oracle", {}).get("best_no_fp_increase", {}).get("replacement_event_details", [])
    if not isinstance(oracle_events, list):
        oracle_events = []
    for event in oracle_events:
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                "kind": "ORACLE_TOPOLOGY",
                "priority": 130,
                "fold": None,
                "video_idx": event.get("video_local_idx"),
                "video": None,
                "segment": event.get("parent"),
                "reason": "oracle_safe_parent_split",
                "best_prediction": event.get("parent"),
                "best_prediction_iou": None,
                "best_candidate": event.get("children"),
                "best_candidate_iou": None,
                "best_candidate_rank": None,
                "covering_candidates": len(event.get("children", [])),
                "note": "Oracle says this parent can be split without increasing FP under train labels.",
            }
        )
    rows.sort(key=lambda item: (int(item["priority"]), str(item.get("kind")), str(item.get("video_idx"))), reverse=True)
    result = {
        "source_fn_attribution": args.fn_attribution,
        "source_topology_oracle": args.topology_oracle,
        "n_items": len(rows),
        "counts": {kind: sum(1 for row in rows if row["kind"] == kind) for kind in sorted({row["kind"] for row in rows})},
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    out_csv = Path(args.out_csv)
    fieldnames = [
        "kind",
        "priority",
        "fold",
        "video_idx",
        "video",
        "segment",
        "reason",
        "best_prediction",
        "best_prediction_iou",
        "best_candidate",
        "best_candidate_iou",
        "best_candidate_rank",
        "covering_candidates",
        "note",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ["n_items", "counts"]}, indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
