#!/usr/bin/env python3
"""Fail closed before a one-time fresh-blind inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(candidate_dir: Path, artifacts: dict[str, Path]) -> dict:
    names_path = candidate_dir / "video_names.json"
    gate_path = candidate_dir / "fresh_overlap_gate.json"
    if not names_path.exists() or not gate_path.exists():
        raise FileNotFoundError("candidate must contain video_names.json and fresh_overlap_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("zero_overlap") is not True:
        raise RuntimeError("fresh candidate overlap gate failed")
    names = json.loads(names_path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names:
        raise ValueError("candidate name manifest is empty")
    missing = {name: str(path) for name, path in artifacts.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"missing frozen artifacts: {missing}")
    return {"status": "ready_for_one_time_inference", "candidate_videos": len(names), "candidate_names_sha256": sha256(names_path), "overlap_gate_sha256": sha256(gate_path), "artifacts": {name: {"path": str(path), "sha256": sha256(path)} for name, path in artifacts.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", required=True, type=Path)
    ap.add_argument("--artifact", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    artifacts = {}
    for item in args.artifact:
        name, sep, path = item.partition("=")
        if not sep or not name:
            raise ValueError(f"invalid --artifact: {item!r}")
        artifacts[name] = Path(path)
    report = preflight(args.candidate_dir, artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
