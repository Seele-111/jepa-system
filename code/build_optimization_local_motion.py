#!/usr/bin/env python3
"""Frozen 77-row, label-free local-motion cache. Default writes are isolated.

Only input_path/fps/frame_count/sha (frames/sha256 compatibility aliases) are
projected from source rows. All original row fields, including labels/name, are
ignored. Source SHA is checked before AND after extraction or resume. Resume
requires exact code/runtime/profile/source contracts and checks NPZ contents
and SHA; incompatible/corrupt caches are refused, never silently overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from optimized_local_motion import (DEFAULT_PROFILE, FEATURE_NAMES,
                                    FEATURE_UNITS, VERSION, extract_video_features)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "output" / "algorithm-opt-2026-10-02-v2" / "local_motion"
SOURCE = ROOT / "output" / "algorithm-opt-2026-10-02" / "dataset_manifest.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value, *, replace=False):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if not replace:
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(content)
        return
    temp = Path(path).with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(content)
    temp.replace(path)


def source_rows(path, expected_rows=77):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))["rows"]
    if len(raw) != expected_rows:
        raise ValueError(f"expected {expected_rows} source rows, got {len(raw)}")
    rows = []
    for item in raw:
        # Do not access names, prompts, events, labels, folds, or ground truth.
        frames = item.get("frame_count", item.get("frames"))
        sha = item.get("sha", item.get("sha256"))
        fps = float(item["fps"])
        if (not isinstance(frames, int) or isinstance(frames, bool) or frames < 1 or
                not np.isfinite(fps) or fps <= 0 or not isinstance(sha, str) or
                len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("invalid allowlisted source contract")
        path_value = str(Path(item["input_path"]).resolve())
        rows.append({"input_path": path_value, "fps": fps,
                     "frame_count": frames, "sha256": sha})
    return rows


def _validate_npz(path, source):
    with np.load(path, allow_pickle=False) as data:
        x, mask = data["signals"], data["feature_valid"]
        expected = (source["frame_count"], len(FEATURE_NAMES))
        if (x.shape != expected or x.dtype != np.float32 or mask.shape != expected or
                mask.dtype != np.bool_ or not np.isfinite(x).all() or np.any(x[~mask] != 0)):
            raise ValueError("invalid NPZ shape/dtype/masks/finite data")
        if not np.isclose(float(data["fps"]), source["fps"], rtol=1e-4, atol=1e-4):
            raise ValueError("invalid NPZ FPS")
        np.testing.assert_array_equal(data["frame_ids"], np.arange(expected[0]))
        np.testing.assert_allclose(data["frame_times_seconds"], np.arange(expected[0]) / float(data["fps"]), rtol=0, atol=1e-12)
        if data["feature_names"].tolist() != list(FEATURE_NAMES):
            raise ValueError("invalid NPZ schema")


def verify_cache(out, row_id, source, profile_signature):
    target = out / f"v{row_id:03d}.npz"
    meta = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    if (meta["profile_signature"] != profile_signature or meta["source"] != source or
            meta["row_id"] != row_id or meta["status"] != "ok" or
            meta["feature_sha256"] != digest(target)):
        raise ValueError("incompatible or corrupt cache; output not overwritten")
    _validate_npz(target, source)
    return meta


def build(source_manifest=SOURCE, output_dir=OUTPUT_ROOT, *, resume=False,
          long_edge=128, expected_rows=77):
    start = time.perf_counter()
    source_manifest = Path(source_manifest).resolve()
    out = Path(output_dir).resolve()
    if not out.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("output must stay within the isolated local_motion directory")
    if long_edge not in (128, 160):
        raise ValueError("production profile long_edge must be 128 or 160")
    rows = source_rows(source_manifest, expected_rows)
    profile = dict(DEFAULT_PROFILE, long_edge=long_edge, version=VERSION,
                   extractor_sha256=digest(Path(__file__).with_name("optimized_local_motion.py")),
                   builder_sha256=digest(Path(__file__)), numpy_version=np.__version__,
                   opencv_version=cv2.__version__, feature_names=list(FEATURE_NAMES),
                   feature_units=FEATURE_UNITS)
    profile_signature = signature(profile)
    contract = {"schema_version": "local-motion-cache-v1", "profile": profile,
                "profile_signature": profile_signature, "source_projection_sha256": signature(rows),
                "source_fields": ["input_path", "fps", "frame_count|frames", "sha|sha256"],
                "source_row_count": len(rows), "label_free": True}
    header = out / "profile.json"
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError("output already exists; use --resume to verify, not overwrite")
        if json.loads(header.read_text(encoding="utf-8")) != contract:
            raise ValueError("incompatible profile/source projection; output not overwritten")
        if json.loads((out / "feature_names.json").read_text(encoding="utf-8")) != list(FEATURE_NAMES):
            raise ValueError("incompatible feature names")
    else:
        if resume:
            raise FileNotFoundError("resume requires an existing output")
        out.mkdir(parents=True, exist_ok=True)
        save_json(header, contract)
        save_json(out / "feature_names.json", list(FEATURE_NAMES))
    records, failures = [], []
    run = {**contract, "source_manifest": str(source_manifest),
           "source_manifest_sha256": digest(source_manifest), "started_at": timestamp(),
           "mode": "verified_resume" if resume else "fresh_extraction", "status": "running",
           "feature_count": len(FEATURE_NAMES), "videos": records, "failures": failures}
    save_json(out / "manifest.json", run, replace=True)
    for i, source in enumerate(rows):
        row_started = time.perf_counter()
        try:
            path = Path(source["input_path"])
            before = path.stat()
            if digest(path) != source["sha256"]:
                raise ValueError("source_sha_mismatch")
            target = out / f"v{i:03d}.npz"
            if target.exists() or target.with_suffix(".json").exists():
                if not resume:
                    raise FileExistsError("existing video artifact")
                meta = verify_cache(out, i, source, profile_signature)
                cache_status = "verified_resume"
            else:
                features = extract_video_features(path, long_edge=long_edge)
                validity = features["validity"]
                if (len(features["signals"]) != source["frame_count"] or
                        not validity["valid"] or not validity["complete_decode"] or
                        not np.isclose(features["fps"], source["fps"], rtol=1e-4, atol=1e-4)):
                    raise ValueError("decoded_frame_count/FPS/validity_mismatch")
                if features["feature_names"] != list(FEATURE_NAMES):
                    raise ValueError("feature_schema_mismatch")
                after = path.stat()
                if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or
                        digest(path) != source["sha256"]):
                    raise ValueError("source_changed_during_extraction")
                with target.open("xb") as handle:
                    np.savez_compressed(handle, signals=features["signals"],
                        feature_valid=validity["feature_valid"], feature_names=np.asarray(FEATURE_NAMES),
                        frame_ids=np.arange(len(features["signals"]), dtype=np.int64),
                        frame_times_seconds=features["frame_times_seconds"],
                        fps=np.asarray(features["fps"], dtype=np.float64))
                _validate_npz(target, source)
                meta = {"row_id": i, "file": target.name, "status": "ok", "source": source,
                        "source_sha256": source["sha256"], "extractor_sha256": profile["extractor_sha256"],
                        "profile_signature": profile_signature, "fresh_source_verified": True,
                        "source_size_bytes": before.st_size, "source_mtime_ns": before.st_mtime_ns,
                        "extracted_at": timestamp(), "feature_sha256": digest(target),
                        "elapsed_seconds": time.perf_counter() - row_started,
                        "frames": len(features["signals"]), "fps": features["fps"],
                        "validity": {k: v for k, v in validity.items() if not isinstance(v, np.ndarray)},
                        "validity_counts": {k: int(v.sum()) for k, v in validity.items()
                                            if isinstance(v, np.ndarray) and v.ndim == 1},
                        "feature_valid_counts": validity["feature_valid"].sum(axis=0).tolist(),
                        "metadata": features["metadata"]}
                save_json(target.with_suffix(".json"), meta)
                cache_status = "fresh_extraction"
            after = path.stat()
            if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or
                    digest(path) != source["sha256"]):
                raise ValueError("source_changed_during_verification")
            record = dict(meta, cache_status=cache_status, source_reverified_at=timestamp(),
                          verification_seconds=time.perf_counter() - row_started)
            records.append(record)
            print(f"{i+1:02d}/{len(rows)} v{i:03d} {cache_status} T={meta['frames']} {time.perf_counter()-row_started:.3f}s", flush=True)
        except (OSError, ValueError, KeyError, AssertionError, cv2.error) as exc:
            failures.append({"row_id": i, "error_type": type(exc).__name__, "error_code": str(exc)})
            print(f"{i+1:02d}/{len(rows)} v{i:03d} FAILED {type(exc).__name__}", flush=True)
        run.update(completed_count=len(records), failed_count=len(failures),
                   frame_count=sum(r["frames"] for r in records), elapsed_seconds=time.perf_counter() - start)
        save_json(out / "manifest.json", run, replace=True)
    run.update(status="complete" if not failures and len(records) == len(rows) else "incomplete",
               completed_at=timestamp(), elapsed_seconds=time.perf_counter() - start)
    save_json(out / "manifest.json", run, replace=True)
    print(f"{run['status'].upper()} {len(records)}/{len(rows)} {run['frame_count']} frames {run['elapsed_seconds']:.2f}s", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--long-edge", type=int, choices=(128, 160), default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = build(args.source_manifest, args.output_dir, resume=args.resume, long_edge=args.long_edge)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
