#!/usr/bin/env python3
"""Frozen local R3D-18 features; no downloads, training, labels, or annotations.

Run with the existing WSL Python. Each <video_name>.npz contains float32
features [decoded_T, 512], anchor_features [A, 512], real-FPS timing,
unclipped/clipped 16-frame indices and an explicit endpoint-padding mask.
The temporal samples cover [-0.5, +0.5] seconds, not 16 consecutive frames.
Spatial preprocessing is exactly the local extract_rgb_r3d_features.py preset.
A partial run is resumable; incompatible/corrupt outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys
import time
from typing import Any, Callable

import numpy as np

DEFAULT_CARD = Path("/mnt/e/jepa-system/data/dataset_cards/clean79_diagnostic.json")
DEFAULT_VIDEOS = Path("/mnt/c/Users/admin/Desktop/测试")
DEFAULT_OUTPUT = Path("/mnt/e/jepa-system/output/algorithm-opt-2026-10-02/rgb_features")
DEFAULT_CHECKPOINT = Path("/home/zzy/.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth")
BASELINE = Path(__file__).with_name("extract_rgb_r3d_features.py")
SCHEMA = "optimized-frozen-r3d18-v1"
FEATURE_DIM = 512
CLIP_LENGTH = 16


@dataclass(frozen=True)
class VideoSpec:
    video_name: str
    expected_frame_count: int


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    seconds: float
    started: float

    def check(self) -> None:
        if time.perf_counter() - self.started >= self.seconds:
            raise BudgetExceeded("wall-clock extraction budget exhausted")


def _validate_timing(frame_count: int, fps: float) -> None:
    if not isinstance(frame_count, (int, np.integer)) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("source FPS must be finite and positive; no guessed FPS fallback")


def anchor_frame_indices(frame_count: int, fps: float, interval_seconds: float = 0.25) -> np.ndarray:
    """Round a real-time grid to nearest frames, deduplicate, include both ends."""
    _validate_timing(frame_count, fps)
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("anchor interval must be finite and positive")
    last_time = (frame_count - 1) / fps
    count = int(np.floor(last_time / interval_seconds + 1e-9)) + 1
    grid = np.arange(count, dtype=np.float64) * interval_seconds
    centers = np.unique(np.clip(np.floor(grid * fps + 0.5).astype(np.int64), 0, frame_count - 1))
    if centers[-1] != frame_count - 1:
        centers = np.append(centers, np.int64(frame_count - 1))
    return centers


def temporal_window_indices(
    frame_count: int, fps: float, centers: np.ndarray,
    clip_length: int = CLIP_LENGTH, window_seconds: float = 1.0,
) -> dict[str, np.ndarray]:
    """Uniform inclusive 1-second windows, half-up rounding, endpoint replication."""
    _validate_timing(frame_count, fps)
    centers = np.asarray(centers)
    if centers.ndim != 1 or not len(centers) or centers.dtype.kind not in "iu":
        raise ValueError("centers must be a nonempty 1D integer array")
    if np.any(centers < 0) or np.any(centers >= frame_count):
        raise ValueError("centers must refer to decoded frames")
    if not isinstance(clip_length, int) or clip_length < 2:
        raise ValueError("clip_length must be an integer >= 2")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window duration must be finite and positive")
    offsets = np.linspace(-window_seconds / 2, window_seconds / 2, clip_length, dtype=np.float64)
    raw = np.floor(centers[:, None] + offsets[None, :] * fps + 0.5).astype(np.int64)
    padding = (raw < 0) | (raw >= frame_count)
    return {
        "sample_offsets_sec": offsets,
        "sample_frame_indices_unclipped": raw,
        "sample_frame_indices": np.clip(raw, 0, frame_count - 1),
        "padding_mask": padding,
    }


def interpolate_anchor_features(sparse: np.ndarray, centers: np.ndarray, frame_count: int) -> np.ndarray:
    """Linear interpolation on original frame positions (equivalently source time)."""
    sparse = np.asarray(sparse, dtype=np.float32)
    centers = np.asarray(centers)
    if frame_count <= 0 or centers.ndim != 1 or centers.dtype.kind not in "iu":
        raise ValueError("invalid frame count or anchor indices")
    if sparse.ndim != 2 or not len(sparse) or sparse.shape[1] == 0 or len(sparse) != len(centers):
        raise ValueError("anchor features must have shape [len(centers), D]")
    if not np.isfinite(sparse).all():
        raise ValueError("anchor features must be finite")
    if centers[0] != 0 or centers[-1] != frame_count - 1 or np.any(np.diff(centers) <= 0):
        raise ValueError("ordered unique anchors must include the first and last real frame")
    if len(centers) == 1:
        return sparse.copy()
    target = np.arange(frame_count, dtype=np.int64)
    right = np.clip(np.searchsorted(centers, target, side="right"), 1, len(centers) - 1)
    left = right - 1
    alpha = ((target - centers[left]) / (centers[right] - centers[left])).astype(np.float32)
    dense = sparse[left] + alpha[:, None] * (sparse[right] - sparse[left])
    dense[centers] = sparse
    return dense.astype(np.float32, copy=False)


def load_video_specs(card_path: Path) -> list[VideoSpec]:
    """Whitelist only names and frame counts; never open annotation/label files."""
    card = json.loads(card_path.read_text(encoding="utf-8-sig"))
    specs: list[VideoSpec] = []
    seen: set[str] = set()
    for item in card["videos"]:
        name = item["video_name"]
        count = item["frame_count"]
        if not isinstance(name, str) or not name or PureWindowsPath(name).name != name or Path(name).name != name:
            raise ValueError("data card video names must be basenames")
        if name.casefold() in seen or not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"duplicate name or invalid metadata frame count: {name}")
        seen.add(name.casefold())
        specs.append(VideoSpec(name, count))
    if not specs:
        raise ValueError("empty video inventory")
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_runtime(checkpoint: Path, device: str, threads: int) -> dict[str, Any]:
    try:
        import torch
        import torchvision
        import cv2
        from torchvision.models.video import R3D_18_Weights, r3d_18
    except Exception as exc:
        raise RuntimeError(
            "Existing Torch/torchvision/OpenCV runtime unavailable; no dependency installation or downloads: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not checkpoint.is_file():
        raise FileNotFoundError(f"existing local R3D checkpoint is required: {checkpoint}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; CPU fallback is not automatic")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(threads)
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint.name == "r3d_18-b3b3357e.pth" and not checkpoint_hash.startswith("b3b3357e"):
        raise ValueError("checkpoint SHA-256 does not match the existing R3D-18 checkpoint filename")
    model = r3d_18(weights=None)  # Must not contact torchvision's download URL.
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if model.fc.in_features != FEATURE_DIM:
        raise ValueError("R3D-18 fc input is not 512D")
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval().to(device)
    transform = R3D_18_Weights.KINETICS400_V1.transforms()
    return {
        "torch": torch, "cv2": cv2, "model": model, "transform": transform, "device": device,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "opencv": cv2.__version__, "python": sys.version.split()[0]},
        "checkpoint_sha256": checkpoint_hash,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }


def feature_profile(runtime: dict[str, Any], checkpoint: Path, interval_seconds: float) -> dict[str, Any]:
    transform = runtime["transform"]
    profile: dict[str, Any] = {
        "schema_version": SCHEMA, "encoder": "torchvision.r3d_18", "weights": "KINETICS400_V1",
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": runtime["checkpoint_sha256"],
        "feature_dim": FEATURE_DIM, "feature_layer": "avgpool.flatten(1), fc replaced by Identity",
        "frozen": True, "eval_mode": True, "inference_mode": True,
        "precision": "float32; no autocast; CUDA matmul/cudnn TF32 disabled",
        "clip_length": CLIP_LENGTH, "window_seconds": 1.0,
        "sample_offsets_seconds": np.linspace(-0.5, 0.5, CLIP_LENGTH).tolist(),
        "anchor_interval_seconds": interval_seconds,
        "sampling": "inclusive centered time grid; floor(frame_position + 0.5); deduplicated anchors include endpoints",
        "fps_source": "cv2.CAP_PROP_FPS from the source video; invalid FPS is an error",
        "padding": "clamp out-of-range rounded indices to first/last decoded frame; mask and raw indices saved",
        "alignment": "linear interpolation on source frame indices to decoded T; no frame truncation or label access",
        "preprocess": {
            "color": "OpenCV BGR -> RGB", "input": "uint8 TCHW in [0,255]",
            "resize_hw": list(transform.resize_size), "crop_hw": list(transform.crop_size),
            "resize_interpolation": transform.interpolation.value, "resize_antialias": False,
            "crop": "center", "convert_dtype": "float32 / 255",
            "mean": list(transform.mean), "std": list(transform.std),
            "implementation": "KINETICS400_V1.transforms(); spatial transform each decoded frame once",
            "baseline_path": str(BASELINE.resolve()), "baseline_sha256": _sha256(BASELINE),
        },
        "runtime_versions": runtime["versions"],
        "npz_fields": {
            "features": "float32 [T,512]", "anchor_features": "float32 [A,512]",
            "anchor_frame_indices": "int64 [A]", "anchor_times_sec": "float64 [A]",
            "frame_times_sec": "float64 [T]", "sample_offsets_sec": "float64 [16]",
            "sample_frame_indices": "int64 [A,16]", "sample_frame_indices_unclipped": "int64 [A,16]",
            "padding_mask": "bool [A,16]", "fps": "float64 scalar: source video FPS",
            "frame_count": "int64 scalar: decoded T", "expected_frame_count": "int64 scalar: card metadata",
            "metadata_json": "Unicode scalar, no pickle needed",
            "feature_profile_json": "Unicode scalar, no pickle needed",
        },
        "label_policy": "Only video_name/frame_count fields of data card are used; no labels or annotations opened",
    }
    profile["profile_id"] = hashlib.sha256(_json_text(profile).encode("utf-8")).hexdigest()
    return profile


def transform_rgb_frames(runtime: dict[str, Any], rgb_frames: np.ndarray):
    """The baseline's exact spatial preset, cached once per decoded frame."""
    if rgb_frames.dtype != np.uint8 or rgb_frames.ndim != 4 or rgb_frames.shape[-1] != 3:
        raise ValueError("expected uint8 RGB frames in THWC layout")
    torch = runtime["torch"]
    with torch.inference_mode():
        tensor = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2)
        return runtime["transform"](tensor).permute(1, 0, 2, 3).contiguous()


def prepare_video(path: Path, runtime: dict[str, Any], budget: Budget, chunk_size: int = 32):
    """Sequentially decode all real frames; never truncate to card/label length."""
    cv2 = runtime["cv2"]
    cap = cv2.VideoCapture(str(path))
    chunks, pending = [], []
    resolution = None
    try:
        if not cap.isOpened():
            raise ValueError("video cannot be opened")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        _validate_timing(1, fps)
        reported = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_count = int(round(reported)) if np.isfinite(reported) and reported > 0 else None
        while True:
            budget.check()
            ok, bgr = cap.read()
            if not ok:
                break
            if resolution is None:
                resolution = [int(bgr.shape[1]), int(bgr.shape[0])]
            pending.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(pending) == chunk_size:
                chunks.append(transform_rgb_frames(runtime, np.stack(pending)))
                pending.clear()
        if pending:
            chunks.append(transform_rgb_frames(runtime, np.stack(pending)))
    finally:
        cap.release()
    if not chunks:
        raise ValueError("no decodable frames")
    frames = runtime["torch"].cat(chunks, dim=0)
    return frames, {
        "fps": fps, "reported_frame_count": reported_count, "frame_count": int(len(frames)),
        "source_resolution_wh": resolution, "last_frame_time_sec": (len(frames) - 1) / fps,
        "duration_sec": len(frames) / fps,
    }


def extract_anchor_features(
    runtime: dict[str, Any], frames, indices: np.ndarray, batch_size: int,
    budget: Budget, report_error: Callable[[dict[str, Any]], None],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch = runtime["torch"]
    outputs, batches = [], []
    for start in range(0, len(indices), batch_size):
        budget.check()
        stop = min(start + batch_size, len(indices))
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                rows = torch.from_numpy(indices[start:stop])
                batch = frames[rows].permute(0, 2, 1, 3, 4).contiguous().to(runtime["device"])
                output = runtime["model"](batch).float().cpu().numpy()
                if output.shape != (stop - start, FEATURE_DIM) or not np.isfinite(output).all():
                    raise ValueError("invalid/nonfinite fc-before 512D encoder output")
                outputs.append(output.copy())
                del batch
            batches.append({"anchor_start": start, "anchor_stop": stop, "wall_seconds": time.perf_counter() - started})
        except Exception as exc:
            report_error({"stage": "inference_batch", "anchor_start": start, "anchor_stop": stop,
                          "error_type": type(exc).__name__, "message": str(exc)[:2000]})
            raise
    return np.concatenate(outputs).astype(np.float32, copy=False), batches


def gpu_memory(runtime: dict[str, Any]) -> dict[str, Any]:
    if runtime["device"] != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    cuda = runtime["torch"].cuda
    cuda.synchronize()
    free, total = cuda.mem_get_info()
    return {"peak_allocated_bytes": int(cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(cuda.max_memory_reserved()),
            "device_free_bytes_after": int(free), "device_total_bytes": int(total)}


def _warnings(info: dict[str, Any], expected: int) -> list[str]:
    warnings = []
    if info["frame_count"] != expected:
        warnings.append("dataset_card_frame_count_mismatch: features retain all decoded frames")
    if info["reported_frame_count"] is not None and info["reported_frame_count"] != info["frame_count"]:
        warnings.append("decoder_reported_frame_count_mismatch: possible incomplete decode")
    return warnings


def extract_one(
    spec: VideoSpec, source: Path, target: Path, runtime: dict[str, Any],
    profile: dict[str, Any], batch_size: int, budget: Budget,
    report_error: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    started = time.perf_counter()
    if target.exists():
        raise FileExistsError("output exists; use --resume to validate/reuse it, never silently overwrite")
    if runtime["device"] == "cuda":
        runtime["torch"].cuda.synchronize()
        runtime["torch"].cuda.reset_peak_memory_stats()
    frames, info = prepare_video(source, runtime, budget)
    decoded_at = time.perf_counter()
    centers = anchor_frame_indices(info["frame_count"], info["fps"], profile["anchor_interval_seconds"])
    sampling = temporal_window_indices(info["frame_count"], info["fps"], centers)
    sparse, batches = extract_anchor_features(runtime, frames, sampling["sample_frame_indices"], batch_size, budget, report_error)
    inferred_at = time.perf_counter()
    dense = interpolate_anchor_features(sparse, centers, info["frame_count"])
    interpolated_at = time.perf_counter()
    stat = source.stat()
    memory = gpu_memory(runtime)
    metadata = {
        "video_name": spec.video_name, "source_path": str(source.resolve()),
        "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "expected_frame_count": spec.expected_frame_count, **info,
        "frame_count_matches_card": info["frame_count"] == spec.expected_frame_count,
        "frame_count_matches_decoder": info["reported_frame_count"] == info["frame_count"],
        "anchor_count": len(centers), "feature_shape": list(dense.shape), "profile_id": profile["profile_id"],
        "padding_total_samples": int(sampling["padding_mask"].sum()),
        "padding_left_samples": int((sampling["sample_frame_indices_unclipped"] < 0).sum()),
        "padding_right_samples": int((sampling["sample_frame_indices_unclipped"] >= len(frames)).sum()),
        "padded_anchor_count": int(sampling["padding_mask"].any(axis=1).sum()),
        "warnings": _warnings(info, spec.expected_frame_count),
        "timing": {"decode_and_preprocess_seconds": decoded_at - started,
                   "inference_seconds": inferred_at - decoded_at,
                   "interpolation_seconds": interpolated_at - inferred_at,
                   "compute_wall_seconds": interpolated_at - started},
        "inference_batches": batches, "gpu_memory": memory,
    }
    temp = target.with_name(target.name + ".tmp")
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(
                stream, features=dense, anchor_features=sparse, anchor_frame_indices=centers,
                anchor_times_sec=centers.astype(np.float64) / info["fps"],
                frame_times_sec=np.arange(len(frames), dtype=np.float64) / info["fps"],
                fps=np.asarray(info["fps"], dtype=np.float64),
                frame_count=np.asarray(len(frames), dtype=np.int64),
                expected_frame_count=np.asarray(spec.expected_frame_count, dtype=np.int64),
                metadata_json=np.asarray(_json_text(metadata)),
                feature_profile_json=np.asarray(_json_text(profile)), **sampling,
            )
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    metadata["timing"]["save_seconds"] = time.perf_counter() - interpolated_at
    metadata["timing"]["total_wall_seconds"] = time.perf_counter() - started
    return {"status": "ok", "cached": False, "npz_path": target.name, **metadata}


def validate_cached(target: Path, source: Path, spec: VideoSpec, profile: dict[str, Any]) -> dict[str, Any]:
    """Validate shape, provenance, temporal alignment and padding before reuse."""
    with np.load(target, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        stored_profile = json.loads(str(data["feature_profile_json"].item()))
        if stored_profile != profile or metadata["profile_id"] != profile["profile_id"]:
            raise ValueError("cached feature profile is incompatible")
        stat = source.stat()
        if (metadata["video_name"] != spec.video_name or metadata["expected_frame_count"] != spec.expected_frame_count
                or metadata["source_size_bytes"] != stat.st_size or metadata["source_mtime_ns"] != stat.st_mtime_ns):
            raise ValueError("cached input provenance is incompatible")
        n, fps = metadata["frame_count"], metadata["fps"]
        centers = anchor_frame_indices(n, fps, profile["anchor_interval_seconds"])
        samples = temporal_window_indices(n, fps, centers)
        features, sparse = data["features"], data["anchor_features"]
        if (features.shape != (n, FEATURE_DIM) or sparse.shape != (len(centers), FEATURE_DIM)
                or features.dtype != np.float32 or sparse.dtype != np.float32
                or not np.isfinite(features).all() or not np.isfinite(sparse).all()):
            raise ValueError("cached feature tensor is invalid")
        expected = {"anchor_frame_indices": centers, "anchor_times_sec": centers.astype(np.float64) / fps,
                    "frame_times_sec": np.arange(n, dtype=np.float64) / fps, **samples}
        if any(not np.array_equal(data[key], value) for key, value in expected.items()):
            raise ValueError("cached timing/padding arrays are invalid")
        if not np.array_equal(features, interpolate_anchor_features(sparse, centers, n)):
            raise ValueError("cached dense features are not aligned to anchors")
    return {"status": "ok", "cached": True, "npz_path": target.name, **metadata}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-card", type=Path, default=DEFAULT_CARD)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--anchor-seconds", type=float, choices=(0.25, 0.4), default=0.25)
    ap.add_argument("--max-videos", type=int, default=0, help="first N present card videos; 0 means all")
    ap.add_argument("--time-budget-seconds", type=float, default=540.0, help="wall budget including setup, <=590")
    ap.add_argument("--resume", action="store_true", help="validate/reuse matching per-video NPZs")
    args = ap.parse_args(argv)
    if args.batch_size <= 0 or args.torch_threads <= 0 or args.max_videos < 0 or not 0 < args.time_budget_seconds <= 590:
        ap.error("positive batch/threads, nonnegative max-videos, and a wall budget in (0,590] are required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    started = time.perf_counter()
    budget = Budget(args.time_budget_seconds, started)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run: dict[str, Any] = {
        "schema_version": SCHEMA, "run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "events": [], "newly_extracted": 0, "cached": 0,
    }
    rows: list[dict[str, Any]] = []
    profile = None
    runtime = None
    exit_code = 0

    def report_error(event: dict[str, Any]) -> None:
        event = {"run_id": run_id, "utc": datetime.now(timezone.utc).isoformat(), **event}
        run["events"].append(event)
        with (out / "errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json_text(event) + "\n")

    def persist() -> None:
        elapsed = time.perf_counter() - started
        summary = {
            "card_video_count": len(rows), "present_video_count": sum(row["source_exists"] for row in rows),
            "completed_video_count": sum(row["status"] == "ok" for row in rows),
            "missing_video_count": sum(row["status"] == "missing" for row in rows),
            "failed_video_count": sum(row["status"] == "error" for row in rows),
            "pending_video_count": sum(row["status"] in ("pending", "not_selected", "budget_not_processed") for row in rows),
            "completed_frame_count": sum(row.get("frame_count", 0) for row in rows if row["status"] == "ok"),
            "completed_anchor_count": sum(row.get("anchor_count", 0) for row in rows if row["status"] == "ok"),
            "warning_video_count": sum(bool(row.get("warnings")) for row in rows),
            "wall_seconds": elapsed, "newly_extracted": run["newly_extracted"], "cached": run["cached"],
            "run_status": run["status"],
        }
        run["summary"] = summary
        run["wall_seconds"] = elapsed
        write_json(out / "manifest.json", {"schema_version": SCHEMA, "dataset_card": str(args.dataset_card.resolve()),
                                           "video_dir": str(args.video_dir.resolve()), "profile": profile,
                                           "last_run_id": run_id, "summary": summary, "videos": rows,
                                           "npz_paths_relative_to": "manifest.json parent directory",
                                           "features_by_name": {row["name"]: row["npz_path"] for row in rows if row["status"] == "ok"}})
        write_json(out / "errors.json", {"run_id": run_id, "events": run["events"],
                                         "incomplete_videos": [row["video_name"] for row in rows if row["status"] != "ok"]})
        write_json(out / "runs" / f"{run_id}.json", run)

    try:
        specs = load_video_specs(args.dataset_card)
        selected_count = 0
        for spec in specs:
            exists = (args.video_dir / spec.video_name).is_file()
            selected = exists and (not args.max_videos or selected_count < args.max_videos)
            if selected:
                selected_count += 1
            rows.append({"name": spec.video_name, "video_name": spec.video_name,
                         "npz_path": spec.video_name + ".npz" if exists else None,
                         "expected_frame_count": spec.expected_frame_count,
                         "source_exists": exists, "selected": selected,
                         "status": "pending" if selected else "not_selected" if exists else "missing"})
            if not exists:
                report_error({"stage": "inventory", "video_name": spec.video_name,
                              "error_type": "FileNotFoundError", "message": "data-card video absent from source directory"})
        persist()
        budget.check()
        runtime = load_runtime(args.checkpoint, args.device, args.torch_threads)
        profile = feature_profile(runtime, args.checkpoint, args.anchor_seconds)
        profile_path = out / "feature_profile.json"
        if profile_path.exists() and json.loads(profile_path.read_text(encoding="utf-8")) != profile:
            raise ValueError("output directory has a different feature profile; choose a separate directory")
        write_json(profile_path, profile)
        run["profile_id"] = profile["profile_id"]
        run["gpu_name"] = runtime["gpu_name"]
        for spec, row in zip(specs, rows):
            if not row["selected"]:
                continue
            try:
                budget.check()
                source = args.video_dir / spec.video_name
                target = out / (spec.video_name + ".npz")
                if args.resume and target.exists():
                    result = validate_cached(target, source, spec, profile)
                    run["cached"] += 1
                else:
                    result = extract_one(spec, source, target, runtime, profile, args.batch_size, budget,
                                         lambda event, name=spec.video_name: report_error({"video_name": name, **event}))
                    run["newly_extracted"] += 1
                row.update(result)
                print(_json_text({"video_name": spec.video_name, "status": "ok", "cached": row["cached"],
                                  "shape": row["feature_shape"], "anchors": row["anchor_count"],
                                  "padding_samples": row["padding_total_samples"], "warnings": row["warnings"],
                                  "timing": row["timing"], "gpu_memory": row["gpu_memory"]}), flush=True)
            except BudgetExceeded:
                row["status"] = "budget_not_processed"
                raise
            except Exception as exc:
                row.update({"status": "error", "error_type": type(exc).__name__, "message": str(exc)[:2000]})
                report_error({"stage": "video", "video_name": spec.video_name,
                              "error_type": type(exc).__name__, "message": str(exc)[:2000]})
                print(_json_text({"video_name": spec.video_name, "status": "error", "error_type": type(exc).__name__,
                                  "message": str(exc)[:2000]}), file=sys.stderr, flush=True)
                if runtime["device"] == "cuda":
                    runtime["torch"].cuda.empty_cache()
                exit_code = 1
            persist()
        run["status"] = "completed_with_video_errors" if exit_code else "completed_selected_inventory"
    except BudgetExceeded as exc:
        for row in rows:
            if row["status"] == "pending":
                row["status"] = "budget_not_processed"
        report_error({"stage": "batch_run", "error_type": type(exc).__name__, "message": str(exc)})
        run["status"], exit_code = "partial_budget", 3
    except (Exception, KeyboardInterrupt) as exc:
        report_error({"stage": "batch_run", "error_type": type(exc).__name__, "message": str(exc)[:2000]})
        run["status"], exit_code = "failed", 2
        print(f"Batch failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        run["ended_utc"] = datetime.now(timezone.utc).isoformat()
        run["exit_code"] = exit_code
        if runtime is not None:
            run["gpu_memory_final"] = gpu_memory(runtime)
            ok_memory = [row["gpu_memory"] for row in rows if row["status"] == "ok"]
            run["gpu_peak_allocated_bytes"] = max((item["peak_allocated_bytes"] or 0 for item in ok_memory), default=0)
            run["gpu_peak_reserved_bytes"] = max((item["peak_reserved_bytes"] or 0 for item in ok_memory), default=0)
        persist()
    print(_json_text({"manifest": str(out / "manifest.json"), **run["summary"], "exit_code": exit_code}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
