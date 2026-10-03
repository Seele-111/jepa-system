#!/usr/bin/env python3
"""Print compact segment metrics for experiment summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"{path.name}: MISSING")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        segment = data.get("aggregate", {}).get("segment", {})
        print(
            f"{path.name}: "
            f"F1={segment.get('f1_mean')} P={segment.get('precision_mean')} "
            f"R={segment.get('recall_mean')} TP={segment.get('tp')} "
            f"FP={segment.get('fp')} FN={segment.get('fn')}"
        )
        for fold in data.get("folds", []):
            config = fold.get("config", {})
            val = fold.get("validation", {}).get("segment", {})
            print(
                f"  fold={fold.get('fold')} enabled={config.get('enabled')} "
                f"F1={val.get('f1')} P={val.get('precision')} R={val.get('recall')} "
                f"TP={val.get('tp')} FP={val.get('fp')} FN={val.get('fn')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
