#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    args = ap.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if isinstance(state, dict) and len(state) == 1 and isinstance(next(iter(state.values())), dict):
        state = next(iter(state.values()))
    sample = []
    if isinstance(state, dict):
        for key, value in list(state.items())[:8] + list(state.items())[-8:]:
            sample.append({"key": key, "shape": list(value.shape) if hasattr(value, "shape") else None, "dtype": str(value.dtype) if hasattr(value, "dtype") else type(value).__name__})
    print(json.dumps({"payload_type": type(payload).__name__, "top_keys": list(payload) if isinstance(payload, dict) else [], "state_entries": len(state) if isinstance(state, dict) else None, "sample": sample}, indent=2))


if __name__ == "__main__":
    main()
