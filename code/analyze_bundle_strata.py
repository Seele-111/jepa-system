#!/usr/bin/env python3
"""Evaluate a frozen bundle on predeclared dataset-card strata."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from selector_fusion import evaluate_fused_predictions


def strata(card: dict) -> dict[str, set[str]]:
    groups = defaultdict(set)
    for row in card["videos"]:
        name = str(row["video_name"])
        occ = float(row.get("positive_frame_ratio", 0.0))
        events = int(row.get("event_count", 0))
        fps = str(row.get("fps", "unknown"))
        resolution = str(row.get("resolution", "unknown"))
        groups["all_positive" if occ >= 1.0 else "partially_positive"].add(name)
        groups["multi_event" if events > 1 else "single_event"].add(name)
        groups["occupancy_lt_0.25" if occ < 0.25 else "occupancy_0.25_0.75" if occ <= 0.75 else "occupancy_gt_0.75"].add(name)
        groups[f"fps_{fps}_resolution_{resolution}"].add(name)
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--dataset-card", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    card = json.loads(args.dataset_card.read_text(encoding="utf-8"))
    folds = bundle["folds"]
    rows = []
    for fold in folds:
        rows.extend(zip(fold["val_names"], fold["predictions"], fold["labels"]))
    result = {"schema_version": "round2-stratified-report-v1", "source": str(args.bundle), "reports": {}}
    for group, names in strata(card).items():
        selected = [(p, np.asarray(y, dtype=np.int64)) for n, p, y in rows if n in names]
        if selected:
            metric = evaluate_fused_predictions([x[0] for x in selected], [x[1] for x in selected], iou_threshold=0.3)
            result["reports"][group] = {"videos": len(selected), "segment": metric["segment"], "frame": metric["frame"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v["segment"]["f1"] for k, v in result["reports"].items()}, indent=2))


if __name__ == "__main__":
    main()
