#!/usr/bin/env python3
"""Create an auditable trivial prediction bundle from an existing labeled bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def transform_bundle(bundle: dict, method: str) -> dict:
    output = {key: value for key, value in bundle.items() if key != "folds"}
    output["schema_version"] = "round2-prediction-bundle-v1"
    output["method"] = f"trivial_{method}"
    output["source_method"] = bundle.get("method")
    output["folds"] = []
    for fold in bundle.get("folds", []):
        labels = fold.get("labels")
        if labels is None:
            raise ValueError("source bundle must include labels")
        if method == "full_span":
            predictions = [[[0, len(label) - 1]] if label else [] for label in labels]
        elif method == "empty":
            predictions = [[] for _ in labels]
        else:
            raise ValueError(f"unsupported trivial method: {method}")
        output["folds"].append(
            {
                "fold": fold.get("fold"),
                "val_names": list(fold.get("val_names", [])),
                "predictions": predictions,
                "labels": labels,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=["full_span", "empty"], default="full_span")
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    output = transform_bundle(source, args.method)
    output["source_bundle"] = str(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": len(output["folds"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
