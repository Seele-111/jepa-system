#!/usr/bin/env python3
"""Convert legacy frozen heldout predictions into the Round-2 bundle schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_segment_locator import load_signal_dataset


def convert(prediction_payload: dict, data_dir: str, prediction_key: str = "predictions") -> dict:
    records = load_signal_dataset(data_dir)
    labels = {record.name: record.labels.astype(int).tolist() for record in records}
    rows = prediction_payload.get(prediction_key, [])
    names, predictions, output_labels = [], [], []
    for row in rows:
        name = str(row["video_name"])
        if name not in labels:
            raise ValueError(f"prediction video {name!r} missing from dataset")
        names.append(name)
        predictions.append([[int(a), int(b)] for a, b in row.get("segments", [])])
        output_labels.append(labels[name])
    if set(names) != set(labels):
        missing = sorted(set(labels) - set(names))
        raise ValueError(f"prediction set does not cover dataset: {missing[:5]}")
    return {
        "schema_version": "round2-prediction-bundle-v1",
        "task": "frozen_cross_domain_transfer",
        "source_summary": prediction_payload.get("summary"),
        "folds": [{"fold": 0, "val_names": names, "predictions": predictions, "labels": output_labels}],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--prediction-key", default="predictions")
    args = ap.parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    bundle = convert(payload, args.data_dir, args.prediction_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
