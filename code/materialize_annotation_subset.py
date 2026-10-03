#!/usr/bin/env python3
"""Materialize a manifest-selected annotation subset with content hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--names", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    names = [str(value) for value in json.loads(args.names.read_text(encoding="utf-8"))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        source = args.source_dir / f"{Path(name).stem}_annotations.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.output_dir / source.name
        shutil.copy2(source, target)
        rows.append({"video_name": name, "annotation": target.name, "sha256": digest(target)})
    manifest = {"count": len(rows), "names_sha256": digest(args.names), "annotations": rows}
    (args.output_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(rows), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
