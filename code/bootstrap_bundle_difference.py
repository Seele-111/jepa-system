#!/usr/bin/env python3
"""Paired video bootstrap for two prediction bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from selector_fusion import evaluate_fused_predictions


def rows(path: Path, bundle_key: str = ""):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if bundle_key:
        try:
            payload = payload["bundles"][bundle_key]
        except KeyError as exc:
            raise ValueError(f"bundle key {bundle_key!r} not found in {path}") from exc
    out = {}
    for fold in payload["folds"]:
        for name, pred, label in zip(fold["val_names"], fold["predictions"], fold["labels"]):
            out[str(name)] = (pred, np.asarray(label, dtype=np.int64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--reference-bundle-key", default="")
    ap.add_argument("--candidate-bundle-key", default="")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()
    ref = rows(args.reference, args.reference_bundle_key)
    cand = rows(args.candidate, args.candidate_bundle_key)
    names = sorted(set(ref) & set(cand))
    if set(ref) != set(cand):
        raise ValueError("bundles do not contain the same video names")
    for name in names:
        if not np.array_equal(ref[name][1], cand[name][1]):
            raise ValueError(f"label mismatch for video {name!r}")
    rng = np.random.default_rng(args.seed)
    differences = []
    for _ in range(args.samples):
        sample = rng.integers(0, len(names), size=len(names))
        ref_metric = evaluate_fused_predictions([ref[names[i]][0] for i in sample], [ref[names[i]][1] for i in sample], iou_threshold=args.iou)["segment"]["f1"]
        cand_metric = evaluate_fused_predictions([cand[names[i]][0] for i in sample], [cand[names[i]][1] for i in sample], iou_threshold=args.iou)["segment"]["f1"]
        differences.append(cand_metric - ref_metric)
    observed = evaluate_fused_predictions([ref[n][0] for n in names], [ref[n][1] for n in names], iou_threshold=args.iou)["segment"]["f1"]
    candidate_observed = evaluate_fused_predictions([cand[n][0] for n in names], [cand[n][1] for n in names], iou_threshold=args.iou)["segment"]["f1"]
    result = {
        "schema_version": "round2-paired-bootstrap-v1",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_bundle_key": args.reference_bundle_key or None,
        "candidate_bundle_key": args.candidate_bundle_key or None,
        "videos": len(names),
        "iou_threshold": args.iou,
        "samples": args.samples,
        "seed": args.seed,
        "reference_pooled_f1": observed,
        "candidate_pooled_f1": candidate_observed,
        "observed_difference": candidate_observed - observed,
        "difference_ci95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "fraction_candidate_better": float(np.mean(np.asarray(differences) > 0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
