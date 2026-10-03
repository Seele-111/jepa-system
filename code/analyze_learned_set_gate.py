#!/usr/bin/env python3
"""Diagnostics for learned set-gate experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jepa_learned_set_gate import (
    DEFAULT_EVIDENCE_NAMES,
    build_folds_from_exports,
    build_gate_examples_from_folds,
    load_prediction_export,
)
from train_segment_locator import _load_feature_names, load_signal_dataset


DATA = Path("/home/zzy/jepa_data/segment_train_full_event_jepa_v2")
BASE = Path("/home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.predictions.json")
RECALL = Path("/home/zzy/jepa_data/segment_train_full_event_v2_clean_anchor_soft_weights_graph_cv5.predictions.json")
SUMMARIES = [
    Path("/home/zzy/jepa_data/segment_train_full_event_v2_learned_set_gate_extratrees.summary.json"),
    Path("/home/zzy/jepa_data/segment_train_full_event_v2_learned_set_gate_logreg.summary.json"),
    Path("/home/zzy/jepa_data/segment_train_full_event_v2_learned_set_gate_extratrees_maxfp1.summary.json"),
]


def main() -> int:
    records = load_signal_dataset(DATA)
    feature_names = _load_feature_names(DATA)
    folds = build_folds_from_exports(load_prediction_export(BASE), load_prediction_export(RECALL))
    examples, names = build_gate_examples_from_folds(
        folds,
        records,
        feature_names=feature_names,
        evidence_names=DEFAULT_EVIDENCE_NAMES,
        max_fp_increase=0,
        iou_threshold=0.3,
    )
    y = np.asarray([item.label for item in examples], dtype=np.int64)
    print("examples", len(examples), "positives", int(y.sum()), "positive_ratio", float(y.mean()))
    x = np.stack([item.features for item in examples])
    for name, values in zip(names, x.T):
        pos = values[y == 1]
        neg = values[y == 0]
        if len(pos) and len(neg):
            gap = float(pos.mean() - neg.mean())
            if abs(gap) > 0.05:
                print("feature_gap", name, "pos_mean", float(pos.mean()), "neg_mean", float(neg.mean()), "gap", gap)
    for path in SUMMARIES:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        print("\nsummary", path.name, data["aggregate"]["segment"])
        for fold in data["folds"]:
            print(
                "fold",
                fold["fold"],
                "enabled",
                fold["config"].get("enabled"),
                "score_mean",
                fold.get("score_mean"),
                "score_max",
                fold.get("score_max"),
                "config",
                fold["config"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
