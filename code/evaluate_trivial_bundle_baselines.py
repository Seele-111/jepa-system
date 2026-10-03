#!/usr/bin/env python3
"""Evaluate trivial predictors and paired improvements for an OOF bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_prediction_bundle import _flatten
from selector_fusion import evaluate_fused_predictions


def baseline_predictions(labels: list[np.ndarray], method: str) -> list[list[tuple[int, int]]]:
    if method == "empty":
        return [[] for _ in labels]
    if method == "full_span":
        return [[(0, len(label) - 1)] if len(label) else [] for label in labels]
    raise ValueError(f"unknown trivial baseline: {method}")


def paired_bootstrap(
    candidate: list[list[tuple[int, int]]],
    reference: list[list[tuple[int, int]]],
    labels: list[np.ndarray],
    threshold: float,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = [labels[idx] for idx in indices]
        candidate_f1 = evaluate_fused_predictions([candidate[idx] for idx in indices], sampled_labels, threshold)["segment"]["f1"]
        reference_f1 = evaluate_fused_predictions([reference[idx] for idx in indices], sampled_labels, threshold)["segment"]["f1"]
        differences.append(candidate_f1 - reference_f1)
    values = np.asarray(differences, dtype=np.float64)
    return {
        "samples": samples,
        "seed": seed,
        "difference": "candidate_minus_trivial",
        "difference_ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "fraction_candidate_better": float(np.mean(values > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle-key", default="")
    parser.add_argument("--iou-thresholds", default="0.3,0.5")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.bundle_key:
        try:
            payload = payload["bundles"][args.bundle_key]
        except KeyError as exc:
            raise ValueError(f"bundle key {args.bundle_key!r} not found") from exc
    rows = _flatten(payload)
    candidate = [row[1] for row in rows]
    labels = [row[2] for row in rows]
    reports = []
    for threshold_text in args.iou_thresholds.split(","):
        threshold = float(threshold_text)
        candidate_metrics = evaluate_fused_predictions(candidate, labels, threshold)
        baselines = {}
        for method in ("empty", "full_span"):
            predictions = baseline_predictions(labels, method)
            reference_metrics = evaluate_fused_predictions(predictions, labels, threshold)
            bootstrap = paired_bootstrap(candidate, predictions, labels, threshold, args.bootstrap_samples, args.seed)
            bootstrap["observed_segment_f1_difference"] = float(
                candidate_metrics["segment"]["f1"] - reference_metrics["segment"]["f1"]
            )
            baselines[method] = {"metrics": reference_metrics, "paired_bootstrap": bootstrap}
        reports.append({"iou_threshold": threshold, "candidate": candidate_metrics, "trivial_baselines": baselines})
    output = {
        "schema_version": "round2-trivial-baseline-report-v1",
        "source": str(args.predictions),
        "bundle_key": args.bundle_key or None,
        "videos": len(rows),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "videos": len(rows), "reports": len(reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
