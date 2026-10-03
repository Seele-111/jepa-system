#!/usr/bin/env python3
"""Audit held-format results against a one-full-span-per-video baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from selector_fusion import evaluate_fused_predictions


def audit_run(run: dict, iou_thresholds: tuple[float, ...] = (0.3, 0.5)) -> dict:
    labels = [np.asarray(value, dtype=np.int64) for value in run["target_labels"]]
    model_predictions = [[tuple(segment) for segment in values] for values in run["target_predictions"]]
    full_span = [[(0, len(label) - 1)] for label in labels]
    exact_full_span = sum(prediction == baseline for prediction, baseline in zip(model_predictions, full_span))
    comparisons = {}
    for threshold in iou_thresholds:
        model = evaluate_fused_predictions(model_predictions, labels, threshold)
        trivial = evaluate_fused_predictions(full_span, labels, threshold)
        comparisons[str(threshold)] = {
            "model": model,
            "trivial_full_span": trivial,
            "segment_f1_difference": float(model["segment"]["f1"] - trivial["segment"]["f1"]),
        }
    return {
        "seed": run["seed"],
        "target_videos": len(labels),
        "exact_full_span_predictions": exact_full_span,
        "exact_full_span_prediction_rate": exact_full_span / max(1, len(labels)),
        "mean_target_positive_frame_ratio": float(np.mean([label.mean() for label in labels])),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    reports = []
    for report in payload["reports"]:
        audits = [audit_run(run) for run in report["runs"]]
        reports.append(
            {
                "mode": report["mode"],
                "source_group": report["source_group"],
                "target_group": report["target_group"],
                "source_videos": report["source_videos"],
                "target_videos": report["target_videos"],
                "runs": audits,
                "mean_exact_full_span_prediction_rate": float(
                    np.mean([run["exact_full_span_prediction_rate"] for run in audits])
                ),
                "mean_f1_difference_vs_trivial": {
                    threshold: float(np.mean([run["comparisons"][threshold]["segment_f1_difference"] for run in audits]))
                    for threshold in ("0.3", "0.5")
                },
            }
        )
    output = {
        "schema_version": "round2-format-transfer-degeneracy-audit-v1",
        "source": str(args.input),
        "baseline": "one predicted segment spanning every frame of every video",
        "conclusion_rule": (
            "A held-format score is not evidence of learned transfer when it is reproduced by the trivial "
            "full-span baseline under a high-occupancy target distribution."
        ),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reports": len(reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
