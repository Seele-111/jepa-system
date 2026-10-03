#!/usr/bin/env python3
"""Serial, lazy WSL feature worker; no dataset scan, motion, shell or downloads.

Serve: python -B optimized_model_worker.py --serve
Once:  python -B optimized_model_worker.py --video ABS --output NEW_ABS_DIR
       --channels rgb,corrected --profile-json ABS_BUNDLE_JSON

POST /infer uses sourcevideo/output/profile_json and optional channels.
The video alias is accepted only when sourcevideo is absent. Only /health is a GET.
All HTTP paths are absolute WSL filesystem paths. CLI relaxes only the output
root restriction, not video/bundle/profile checks. No models load at startup.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, fields
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import inspect
import json
from pathlib import Path
import re
import signal
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

from jepa_runtime import ROOT, settings

SCHEMA = "optimized-model-worker-v1"
HOST, PORT = "127.0.0.1", settings().worker_port
MAX_BODY_BYTES = 64 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
OUTPUT_ROOT = ROOT / "output"
VIDEO_ROOT = settings().data_root
R3D_CHECKPOINT = settings().resources["r3d_checkpoint"]
V_ENCODER = settings().resources["vjepa_encoder"]
V_PREDICTOR = settings().resources["vjepa_predictor"]
I_SLIM = settings().resources["ijepa_checkpoint"]
I_FULL = I_SLIM  # One explicit, hash-pinned checkpoint; no guessing another file.
REQUEST_FIELDS = frozenset(("sourcevideo", "video", "output", "channels", "profile_json"))
VIDEO_SUFFIXES = frozenset((".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"))
CORRECTED_NAMES = [prefix + "_" + suffix for prefix in ("v", "i") for suffix in (
    "absolute_mean", "patch_std", "patch_p90", "patch_p99", "patch_fraction",
    "direct_observation", "interpolation_range_supported")]


class WorkerError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, stage: str = "validation"):
        super().__init__(message)
        self.code, self.status, self.stage = code, status, stage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def strict_json(data: bytes | str):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON number: " + value)
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid_constant)


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _valid_sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def parse_channels(value) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if (not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value)
            or len(set(value)) != len(value) or any(item not in ("rgb", "corrected") for item in value)):
        raise WorkerError("invalid_channels", "channels must be a unique nonempty list of rgb/corrected; motion is not supported")
    return tuple(value)


@dataclass(frozen=True)
class PathPolicy:
    output_root: Path = OUTPUT_ROOT
    video_roots: tuple[Path, ...] = (OUTPUT_ROOT, VIDEO_ROOT)
    bundle_root: Path = OUTPUT_ROOT

    def validate(self, payload, http: bool) -> dict[str, Any]:
        if (not isinstance(payload, dict) or set(payload) - REQUEST_FIELDS
                or not {"output", "profile_json"}.issubset(payload)
                or (("sourcevideo" in payload) == ("video" in payload))):
            raise WorkerError("invalid_request", "required: sourcevideo (or video alias), output, profile_json; optional: channels")
        payload = {**payload, "video": payload.get("sourcevideo", payload.get("video"))}
        result: dict[str, Any] = {"channels": parse_channels(payload.get("channels", ["rgb", "corrected"]))}
        for key in ("video", "output", "profile_json"):
            value = payload[key]
            if not isinstance(value, str) or not value or "\x00" in value or not Path(value).is_absolute():
                raise WorkerError("invalid_path", key + " must be an absolute local filesystem path")
            result[key] = Path(value).resolve()
        video, output, bundle = result["video"], result["output"], result["profile_json"]
        if not any(video.is_relative_to(root.resolve()) for root in self.video_roots):
            raise WorkerError("forbidden_video", "video is outside permitted source roots", 403)
        if video.suffix.lower() not in VIDEO_SUFFIXES or not video.is_file():
            raise WorkerError("invalid_video", "source must be an existing video file")
        if not bundle.is_relative_to(self.bundle_root.resolve()) or not bundle.is_file() or bundle.suffix.lower() != ".json":
            raise WorkerError("forbidden_bundle", "profile_json must be an existing JSON file under project output", 403)
        root = self.output_root.resolve()
        if http and (output == root or not output.is_relative_to(root)):
            raise WorkerError("forbidden_output", "HTTP output must be a new subdirectory of project output", 403)
        # Reject even empty existing directories, including any saved raw directory.
        if output.exists():
            raise WorkerError("output_exists", "output already exists; saved raw/features must not be reused or overwritten", 409)
        return result


@dataclass
class ModelEntry:
    resource: Any
    identity: dict[str, Any]
    state: str = "pending"
    failure: str | None = None


class LocalAdapters:
    """Production implementation. Every import/load is lazy; tests inject fakes."""

    def __init__(self):
        self._modules: dict[str, Any] = {}
        self._module_shas: dict[str, str] = {}

    @staticmethod
    def _implementation_identity(name, module):
        path = Path(module.__file__)
        if name == "build_optimization_jepa":
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in ("patch_descriptors", "corrected_frame_features")]
            if len(functions) != 2:
                raise WorkerError("descriptor_schema_missing", "corrected descriptor functions are missing", 422)
            return hashlib.sha256(ast.dump(ast.Module(body=functions, type_ignores=[]), include_attributes=False).encode()).hexdigest()
        return sha256_file(path)

    def module(self, name):
        if name not in self._modules:
            self._modules[name] = importlib.import_module(name)
            self._module_shas[name] = self._implementation_identity(name, self._modules[name])
        elif self._implementation_identity(name, self._modules[name]) != self._module_shas[name]:
            raise WorkerError("loaded_code_changed", "loaded adapter source changed; restart worker to use the repaired implementation", 409, "model_load")
        return self._modules[name]

    def validate_profiles(self, profiles, channels):
        for channel in channels:
            if not isinstance(profiles.get(channel), dict):
                raise WorkerError("missing_bundle_profile", "bundle feature_profiles['" + channel + "'] is required", 422)
        if "rgb" in channels:
            rgb, profile = self.module("optimized_rgb_features"), profiles["rgb"]
            supplied_id = profile.get("profile_id")
            unhashed = {key: value for key, value in profile.items() if key != "profile_id"}
            if not _valid_sha(supplied_id) or hashlib.sha256(canonical_json(unhashed).encode()).hexdigest() != supplied_id:
                raise WorkerError("rgb_profile_id_mismatch", "RGB profile_id is missing or does not match its exact profile", 422)
            expected = {"schema_version": rgb.SCHEMA, "encoder": "torchvision.r3d_18", "weights": "KINETICS400_V1",
                        "feature_dim": 512, "clip_length": 16, "window_seconds": 1.0,
                        "fps_source": "cv2.CAP_PROP_FPS from the source video; invalid FPS is an error",
                        "frozen": True, "eval_mode": True, "inference_mode": True}
            if any(profile.get(key) != value for key, value in expected.items()) or profile.get("anchor_interval_seconds") not in (0.25, 0.4):
                raise WorkerError("rgb_profile_mismatch", "unsupported RGB layer/timing/FPS/frozen profile", 422)
            if profile.get("checkpoint_path") != str(R3D_CHECKPOINT.resolve()) or not _valid_sha(profile.get("checkpoint_sha256")):
                raise WorkerError("rgb_checkpoint_mismatch", "RGB profile must pin the existing local R3D checkpoint and SHA-256", 422)
            prep = profile.get("preprocess", {})
            spatial = {"color": "OpenCV BGR -> RGB", "input": "uint8 TCHW in [0,255]",
                       "resize_hw": [128, 171], "crop_hw": [112, 112], "crop": "center",
                       "resize_interpolation": "bilinear", "resize_antialias": False,
                       "convert_dtype": "float32 / 255", "mean": [0.43216, 0.394666, 0.37645],
                       "std": [0.22803, 0.22145, 0.216989], "baseline_path": str(rgb.BASELINE.resolve()),
                       "baseline_sha256": sha256_file(rgb.BASELINE)}
            if not isinstance(prep, dict) or any(prep.get(key) != value for key, value in spatial.items()):
                raise WorkerError("rgb_preprocessing_mismatch", "RGB preprocessing must match the local baseline exactly", 422)
        if "corrected" in channels:
            extractor, profile = self.module("optimized_jepa_extractor"), profiles["corrected"]
            config = profile.get("config")
            required = {field.name for field in fields(extractor.Config)}
            if not isinstance(config, dict) or set(config) != required:
                raise WorkerError("corrected_config_missing", "corrected profile must contain every Config field, without defaults or extra fields", 422)
            fixed = {"seed": 0, "max_frames": 32, "max_keyframes": 8, "bidirectional": True, "mask_passes": 1}
            if (any(type(config[key]) is not type(value) or config[key] != value for key, value in fixed.items())
                    or profile.get("name") not in ("corrected-true-jepa-absolute-patch-v1", "corrected-true-jepa-absolute-patch-v2")
                    or profile.get("legacy_cache_compatible") is not False
                    or profile.get("feature_names") != CORRECTED_NAMES
                    or profile.get("i_mask_reuse_policy") != "explicit_batch_seed_reset_and_restore"):
                raise WorkerError("corrected_profile_mismatch", "required genuine corrected profile: seed0/32frames/8keys/bidirectional/1pass, 14 absolute features; no legacy scale", 422)
            try:
                extractor.Config(**config)
            except (TypeError, ValueError) as exc:
                raise WorkerError("corrected_config_invalid", str(exc), 422) from exc
            for key, filename in (("extractor_sha256", "optimized_jepa_extractor.py"),):
                path = Path(__file__).with_name(filename)
                if not _valid_sha(profile.get(key)) or profile[key] != sha256_file(path):
                    raise WorkerError("corrected_code_mismatch", key + " does not match the current implementation", 422)
            # Capability guard only: the primary adapter owns the counter reset.
            # Never alter torch RNG or MaskCollator state in this worker.
            if (not hasattr(extractor, "seeded_mask_counter")
                    or "seeded_mask_counter" not in inspect.getsource(extractor._RealIJEPA.batch_errors)
                    or "_itr_counter" not in inspect.getsource(extractor.seeded_mask_counter)):
                raise WorkerError("ijepa_counter_policy_missing", "primary _RealIJEPA lacks the repaired seed-bound counter policy; update upstream/profile and restart worker", 422)
            if "checkpoints" in profile:
                claims = profile["checkpoints"]
                if not isinstance(claims, dict) or set(claims) != {"vjepa_encoder", "vjepa_predictor", "ijepa"}:
                    raise WorkerError("corrected_checkpoint_claims_invalid", "checkpoints must pin vjepa_encoder, vjepa_predictor, ijepa", 422)
                for claim in claims.values():
                    if not isinstance(claim, dict) or set(claim) != {"path", "sha256"} or not _valid_sha(claim["sha256"]):
                        raise WorkerError("corrected_checkpoint_claims_invalid", "checkpoint claims require exact path and sha256", 422)

    def probe(self, video):
        import cv2
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise WorkerError("video_open_failed", "video could not be opened", 422)
            count, fps = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            if not 0 < fps < float("inf") or not 1 <= count < float("inf") or abs(count - round(count)) > 1e-6:
                raise WorkerError("video_metadata_invalid", "finite positive source FPS and integral frame count are required", 422)
            return {"fps": fps, "total_frames": int(round(count))}
        finally:
            cap.release()

    def model_files(self, model, profile):
        if model == "rgb":
            return {"checkpoint": R3D_CHECKPOINT}
        if model == "vjepa":
            return {"encoder": V_ENCODER, "predictor": V_PREDICTOR}
        return {"checkpoint": I_SLIM if I_SLIM.is_file() else I_FULL}

    def check_model_claims(self, model, identity, profile):
        if model == "rgb":
            if identity["files"]["checkpoint"]["sha256"] != profile["checkpoint_sha256"]:
                raise WorkerError("rgb_checkpoint_sha_mismatch", "actual R3D checkpoint SHA differs from bundle", 422)
        elif "checkpoints" in profile:
            lookup = {"vjepa": {"encoder": "vjepa_encoder", "predictor": "vjepa_predictor"},
                      "ijepa": {"checkpoint": "ijepa"}}[model]
            for label, claim_name in lookup.items():
                actual, claim = identity["files"][label], profile["checkpoints"][claim_name]
                if claim["path"] != actual["path"] or claim["sha256"] != actual["sha256"]:
                    raise WorkerError("corrected_checkpoint_sha_mismatch", "actual JEPA checkpoint differs from bundle claim", 422)

    def load_rgb(self, profile):
        return self.module("optimized_rgb_features").load_runtime(R3D_CHECKPOINT, "cuda", 4)

    def check_loaded_rgb(self, runtime, profile):
        actual = self.module("optimized_rgb_features").feature_profile(runtime, R3D_CHECKPOINT, profile["anchor_interval_seconds"])
        if actual != profile:
            raise WorkerError("rgb_runtime_profile_mismatch", "loaded RGB checkpoint/preprocessing/runtime/FPS profile differs from bundle", 422, "model_load")

    def extract_rgb(self, runtime, video, target, profile, probe, events):
        rgb = self.module("optimized_rgb_features")
        return rgb.extract_one(rgb.VideoSpec(video.name, probe["total_frames"]), video, target, runtime,
                               profile, 4, rgb.Budget(540, time.perf_counter()), events.append)

    def config(self, profile):
        return self.module("optimized_jepa_extractor").Config(**profile["config"])

    def sample_corrected(self, video, config):
        return self.module("optimized_jepa_extractor").sample_video(video, config)

    def isolation(self, kind):
        from configured_model_loaders import isolated_legacy_module
        return isolated_legacy_module(kind)

    def make_backend(self, kind, legacy, config, identity):
        if not legacy.torch.cuda.is_available():
            raise RuntimeError("JEPA requires existing CUDA; no CPU/proxy fallback")
        expected = identity["files"]
        actual_paths = ({"encoder": Path(legacy.ENCODER_CKPT), "predictor": Path(legacy.FULL_CKPT)} if kind == "vjepa"
                        else {"checkpoint": legacy.CKPT_SLIM_BF16 if legacy.CKPT_SLIM_BF16.is_file() else legacy.CKPT_FULL})
        if any(str(path.resolve()) != expected[key]["path"] for key, path in actual_paths.items()):
            raise WorkerError("legacy_checkpoint_mismatch", "legacy loader would load an unverified checkpoint", 422, "model_load")
        extractor = self.module("optimized_jepa_extractor")
        return extractor._RealVJEPA(legacy) if kind == "vjepa" else extractor._RealIJEPA(legacy, config.seed)

    def run_backend(self, kind, backend, sampled, config):
        extractor = self.module("optimized_jepa_extractor")
        if kind == "vjepa":
            return extractor.run_vjepa(sampled.frames, sampled.frame_ids, backend, config)
        return extractor.run_ijepa(sampled.keyframes, sampled.keyframe_ids, backend, config)

    def clear_video_buffers(self, kind, backend):
        if kind == "vjepa":
            backend.video = backend.targets = backend.context_tokens = backend.context_key = None

    def build_artifacts(self, video, sampled, results, config, runtime):
        return self.module("optimized_jepa_extractor").build_artifacts(
            video, sampled, results["vjepa"], results["ijepa"], config, runtime)

    def write_raw(self, output, arrays, metadata):
        self.module("optimized_jepa_extractor").write_outputs(output, arrays, metadata)

    def corrected_features(self, arrays):
        return self.module("build_optimization_jepa").corrected_frame_features(arrays)

    def save_corrected(self, target, features, names, fps, metadata):
        import numpy as np
        features = np.asarray(features)
        if features.ndim != 2 or features.dtype != np.float32 or not np.isfinite(features).all():
            raise WorkerError("corrected_tensor_invalid", "corrected signals must be finite float32 [T,D]", 500, "inference")
        with target.open("xb") as handle:
            np.savez_compressed(handle, signals=features, names=np.asarray(names), fps=np.asarray(fps, dtype=np.float64),
                                frame_ids=np.arange(len(features), dtype=np.int64),
                                frame_count=np.asarray(len(features), dtype=np.int64),
                                metadata_json=np.asarray(canonical_json(metadata)))

    def close_rgb(self, runtime):
        torch = runtime["torch"]
        runtime.pop("model", None)
        gc.collect()
        torch.cuda.empty_cache()

    def close_backend(self, kind, backend):
        backend.close()


class FeatureWorker:
    def __init__(self, adapters=None, policy=None, retain_models=True):
        self.adapters = adapters if adapters is not None else LocalAdapters()
        self.policy = policy if policy is not None else PathPolicy()
        self.models: dict[str, ModelEntry] = {}
        self.retain_models = retain_models
        self.lock = threading.Lock()
        self.busy, self.closed = False, False
        self.active_isolation = None
        self.worker_code_sha256 = sha256_file(Path(__file__))

    def health(self):
        return {"schema": SCHEMA, "status": "closed" if self.closed else "ok",
                "worker_code_sha256": self.worker_code_sha256, "processing": "serial", "busy": self.busy,
                "project_root": str(ROOT), "runtime_config_id": settings().config_id, "retain_models": self.retain_models,
                "hot_models": [kind for kind, entry in self.models.items() if entry.state == "ready"],
                "failed_models": [kind for kind, entry in self.models.items() if entry.state == "failed"]}

    def _model_identity(self, kind, profile):
        paths = {key: path.resolve() for key, path in self.adapters.model_files(kind, profile).items()}
        existing = self.models.get(kind)
        if existing is not None and existing.state == "failed":
            raise WorkerError("model_quarantined", kind + " previously failed; restart the worker instead of claiming a cache hit", 503, "model_load")
        actual = {}
        for label, path in paths.items():
            if not path.is_file():
                raise WorkerError("checkpoint_missing", "existing local checkpoint missing: " + str(path), 500, "model_load")
            stat = path.stat()
            if existing is not None:
                saved = existing.identity["files"].get(label)
                if saved is None or any(saved[key] != value for key, value in (
                        ("path", str(path)), ("size_bytes", stat.st_size), ("mtime_ns", stat.st_mtime_ns), ("ctime_ns", stat.st_ctime_ns))):
                    raise WorkerError("hot_checkpoint_changed", "checkpoint changed while its model was hot; restart required", 409, "model_load")
                actual[label] = saved
            else:
                actual[label] = {"path": str(path), "sha256": sha256_file(path), "size_bytes": stat.st_size,
                                 "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
                after = path.stat()
                if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise WorkerError("checkpoint_changed", "checkpoint changed during hashing", 409, "model_load")
        identity = {"files": actual}
        self.adapters.check_model_claims(kind, identity, profile)
        return identity

    def _rgb(self, profile, state):
        identity = self._model_identity("rgb", profile)
        state["identity"]["models"]["rgb"] = identity
        entry = self.models.get("rgb")
        cold = entry is None
        state["model_load_status"]["rgb"] = "loading" if cold else "reusing"
        try:
            if cold:
                resource = self.adapters.load_rgb(profile)
                entry = self.models["rgb"] = ModelEntry(resource, identity)
            self.adapters.check_loaded_rgb(entry.resource, profile)
            entry.state = "ready"
            state["model_load_status"]["rgb"] = "cold_loaded" if cold else "hot_reused"
            return entry.resource
        except Exception as exc:
            if entry is not None:
                entry.state, entry.failure = "failed", type(exc).__name__
            state["model_load_status"]["rgb"] = "error"
            raise

    @contextmanager
    def _backend(self, kind, config, profile, state):
        if self.active_isolation is not None:
            raise WorkerError("isolation_overlap", "V/I legacy module isolation must never overlap", 500, "inference")
        identity = self._model_identity(kind, profile)
        state["identity"]["models"][kind] = identity
        entry = self.models.get(kind)
        cold = entry is None
        state["model_load_status"][kind] = "loading" if cold else "reusing"
        self.active_isolation = kind
        try:
            with self.adapters.isolation(kind) as legacy:
                try:
                    if cold:
                        entry = self.models[kind] = ModelEntry(self.adapters.make_backend(kind, legacy, config, identity), identity)
                    yield entry.resource
                    entry.state = "ready"
                    state["model_load_status"][kind] = "cold_loaded" if cold else "hot_reused"
                except Exception as exc:
                    if entry is not None:
                        entry.state, entry.failure = "failed", type(exc).__name__
                    state["model_load_status"][kind] = "error"
                    raise
                finally:
                    if entry is not None:
                        self.adapters.clear_video_buffers(kind, entry.resource)
        finally:
            self.active_isolation = None

    def process(self, payload, http=False) -> tuple[int, dict[str, Any]]:
        with self.lock:
            started = time.perf_counter()
            self.busy = True
            output = None
            state: dict[str, Any] = {"schema": SCHEMA, "status": "processing", "request_id": uuid.uuid4().hex,
                                     "channels": [], "paths": {}, "features": {}, "fps": None, "total_frames": None,
                                     "identity": {"worker_code_sha256": self.worker_code_sha256, "profiles": {}, "models": {}},
                                     "model_load_status": {}, "feature_cache_used": False,
                                     "channel_status": {}, "timings": {}, "batch_errors": []}
            status = 200
            try:
                if self.closed:
                    raise WorkerError("worker_closed", "worker is closed", 503)
                request = self.policy.validate(payload, http)
                channels, video, bundle_path = request["channels"], request["video"], request["profile_json"]
                state["channels"] = list(channels)
                state["sourcevideo"], state["output"], state["profile_json"] = str(video), str(request["output"]), str(bundle_path)
                if bundle_path.stat().st_size > MAX_BUNDLE_BYTES:
                    raise WorkerError("bundle_too_large", "bundle JSON exceeds 64 MiB limit", 413)
                bundle_bytes = bundle_path.read_bytes()
                if len(bundle_bytes) > MAX_BUNDLE_BYTES:
                    raise WorkerError("bundle_too_large", "bundle JSON exceeds 64 MiB limit", 413)
                try:
                    bundle = strict_json(bundle_bytes)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise WorkerError("invalid_bundle_json", "bundle must contain strict UTF-8 JSON", 422) from exc
                profiles = bundle.get("feature_profiles") if isinstance(bundle, dict) else None
                if not isinstance(profiles, dict):
                    raise WorkerError("missing_bundle_profile", "bundle feature_profiles object is required; no defaults or legacy profile fallback", 422)
                from published_models import resolve_profiles, digest_json
                declared_profile_id = digest_json(profiles)
                profiles = resolve_profiles(profiles)
                state["identity"]["declared_feature_profiles_sha256"] = declared_profile_id
                self.adapters.validate_profiles(profiles, channels)
                state["identity"]["bundle"] = {"path": str(bundle_path), "sha256": hashlib.sha256(bundle_bytes).hexdigest()}
                for channel in channels:
                    state["identity"]["profiles"][channel] = {
                        "sha256": hashlib.sha256(canonical_json(profiles[channel]).encode()).hexdigest(),
                        "profile_id": profiles[channel].get("profile_id"), "profile": profiles[channel]}
                before = video.stat()
                state["identity"]["video"] = {"path": str(video), "name": video.name,
                                              "sha256": sha256_file(video), "size_bytes": before.st_size,
                                              "mtime_ns": before.st_mtime_ns}
                probe = self.adapters.probe(video)
                if (type(probe.get("total_frames")) is not int or probe["total_frames"] < 1
                        or not isinstance(probe.get("fps"), (int, float)) or not 0 < probe["fps"] < float("inf")):
                    raise WorkerError("video_metadata_invalid", "invalid source FPS/frame count", 422)
                state["fps"], state["total_frames"] = probe["fps"], probe["total_frames"]
                request["output"].mkdir(parents=True, exist_ok=False)
                output = request["output"]
                if output.resolve() != output:
                    raise WorkerError("output_changed", "output path changed during creation", 403)
                state["alignment"] = {"status": "matched", "rgb_frame_count_source": None, "corrected_frame_count_source": None}
                # Stable order; only requested channels are loaded or computed.
                if "rgb" in channels:
                    begin = time.perf_counter()
                    runtime = self._rgb(profiles["rgb"], state)
                    try:
                        row = self.adapters.extract_rgb(runtime, video, output / "rgb.npz", profiles["rgb"], probe, state["batch_errors"])
                    except Exception:
                        self.models["rgb"].state = "failed"
                        state["model_load_status"]["rgb"] = "error"
                        raise
                    if row["frame_count"] != probe["total_frames"] or row["fps"] != probe["fps"]:
                        raise WorkerError("frame_fps_mismatch", "decoded RGB T/FPS differs from source metadata; no truncation", 422, "alignment")
                    state["paths"]["rgb_npz"] = str(output / "rgb.npz")
                    state["features"]["rgb"] = {"npz_path": str(output / "rgb.npz"), "feature_key": "features",
                                                "shape": [probe["total_frames"], 512], "fps": probe["fps"],
                                                "total_frames": probe["total_frames"], "profile_id": profiles["rgb"]["profile_id"]}
                    state["channel_status"]["rgb"] = "ok"
                    state["timings"]["rgb_seconds"] = time.perf_counter() - begin
                    state["alignment"]["rgb_frame_count_source"] = "all_decoded_frames"
                    if not self.retain_models:
                        self._release_model("rgb")
                if "corrected" in channels:
                    begin = time.perf_counter()
                    profile = profiles["corrected"]
                    config = self.adapters.config(profile)
                    sampled = self.adapters.sample_corrected(video, config)
                    if sampled.total_frames != probe["total_frames"] or sampled.fps != probe["fps"]:
                        raise WorkerError("frame_fps_mismatch", "corrected sample T/FPS differs from source metadata", 422, "alignment")
                    if len(sampled.frames) < 4 or not sampled.keyframes:
                        raise WorkerError("insufficient_evidence", "video lacks enough genuine V/I evidence; no zero/proxy fallback", 422, "inference")
                    results, model_stats = {}, {}
                    for kind in ("vjepa", "ijepa"):
                        with self._backend(kind, config, profile, state) as backend:
                            evidence = self.adapters.run_backend(kind, backend, sampled, config)
                            if evidence.metadata.get("status") != "ok" or not evidence.valid_mask.any():
                                raise WorkerError("missing_model_evidence", kind + " produced no valid genuine evidence", 500, "inference")
                            results[kind] = evidence
                            model_stats[kind] = dict(getattr(backend, "stats", {}))
                        if not self.retain_models:
                            self._release_model(kind)
                    arrays, metadata = self.adapters.build_artifacts(video, sampled, results, config,
                        {"models": model_stats, "worker_model_load_status": dict(state["model_load_status"]),
                         "processing": "serial", "hot_weights_retained": self.retain_models})
                    self.adapters.write_raw(output / "raw", arrays, metadata)
                    state["paths"].update({"raw_npz": str(output / "raw" / "signals.npz"),
                                           "raw_json": str(output / "raw" / "signals.json")})
                    features, names = self.adapters.corrected_features(arrays)
                    if names != profile["feature_names"] or tuple(features.shape) != (probe["total_frames"], len(names)):
                        raise WorkerError("corrected_feature_mismatch", "corrected feature names/T/dimension differ from bundle", 422, "alignment")
                    self.adapters.save_corrected(output / "corrected.npz", features, names, probe["fps"],
                                                 {"schema": SCHEMA, "profile": profile, "identity": state["identity"],
                                                  "frame_count_source": "reported_frame_count_not_full_decode_verified"})
                    state["paths"]["corrected_npz"] = str(output / "corrected.npz")
                    state["features"]["corrected"] = {"npz_path": str(output / "corrected.npz"), "feature_key": "signals",
                                                      "names_key": "names", "shape": list(features.shape), "fps": probe["fps"],
                                                      "total_frames": probe["total_frames"],
                                                      "profile_sha256": state["identity"]["profiles"]["corrected"]["sha256"]}
                    state["channel_status"]["corrected"] = "ok"
                    state["timings"]["corrected_seconds"] = time.perf_counter() - begin
                    state["alignment"]["corrected_frame_count_source"] = "reported_frame_count_not_full_decode_verified"
                after = video.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise WorkerError("source_changed", "source video changed during inference", 409, "alignment")
                state["status"] = "ok"
                state["elapsed_seconds"] = time.perf_counter() - started
                state["paths"]["manifest"] = str(output / "worker_manifest.json")
                state["paths"]["result"] = str(output / "worker_result.json")
                write_json_exclusive(output / "worker_manifest.json", state)
                write_json_exclusive(output / "worker_result.json", state)
            except Exception as exc:
                status = exc.status if isinstance(exc, WorkerError) else 500
                state["status"] = "error"
                state["error"] = {"code": exc.code if isinstance(exc, WorkerError) else "load_or_inference_failed",
                                  "type": type(exc).__name__, "stage": exc.stage if isinstance(exc, WorkerError) else "inference",
                                  "message": str(exc)[:2000]}
                for channel in state["channels"]:
                    state["channel_status"].setdefault(channel, "error")
                state["elapsed_seconds"] = time.perf_counter() - started
                if output is not None:
                    try:
                        state["paths"]["error"] = str(output / "worker_error.json")
                        write_json_exclusive(output / "worker_error.json", state)
                        state["paths"]["result"] = str(output / "worker_result.json")
                        # This directory was exclusively created by this request.
                        (output / "worker_result.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    except Exception as save_exc:
                        state["error"]["record_error_type"] = type(save_exc).__name__
            finally:
                self.busy = False
            return status, state

    def _release_model(self, kind):
        entry = self.models.pop(kind, None)
        if entry is None:
            return
        if kind == "rgb":
            self.adapters.close_rgb(entry.resource)
        else:
            if self.active_isolation is not None:
                raise RuntimeError("model release must not overlap an active legacy scope")
            with self.adapters.isolation(kind):
                self.adapters.close_backend(kind, entry.resource)

    def close(self):
        """Release all models once, in non-overlapping legacy scopes, at shutdown."""
        with self.lock:
            if self.closed:
                return []
            self.closed = True
            errors = []
            for kind, entry in list(self.models.items()):
                try:
                    if kind == "rgb":
                        self.adapters.close_rgb(entry.resource)
                    else:
                        if self.active_isolation is not None:
                            raise RuntimeError("shutdown isolation overlap")
                        self.active_isolation = kind
                        try:
                            with self.adapters.isolation(kind):
                                self.adapters.close_backend(kind, entry.resource)
                        finally:
                            self.active_isolation = None
                except Exception as exc:
                    errors.append({"model": kind, "type": type(exc).__name__, "message": str(exc)[:2000]})
            self.models.clear()
            return errors


def validate_local_headers(headers, port):
    hosts = headers.get_all("Host", [])
    if len(hosts) != 1:
        raise WorkerError("invalid_host", "exactly one loopback Host header is required", 403)
    try:
        host = urlsplit("http://" + hosts[0])
        if (host.hostname not in ("127.0.0.1", "localhost") or host.port != port
                or host.username is not None or host.password is not None or host.path or host.query or host.fragment):
            raise ValueError("nonlocal host")
    except ValueError as exc:
        raise WorkerError("invalid_host", "Host must be localhost/127.0.0.1 at the worker port", 403) from exc
    origins = headers.get_all("Origin", [])
    if len(origins) > 1:
        raise WorkerError("invalid_origin", "at most one Origin header is allowed", 403)
    if origins:
        try:
            origin = urlsplit(origins[0])
            if (origin.scheme not in ("http", "https") or origin.hostname not in ("127.0.0.1", "localhost")
                    or origin.username is not None or origin.password is not None
                    or origin.path or origin.query or origin.fragment
                    or (origin.port is not None and not 1 <= origin.port <= 65535)):
                raise ValueError("nonlocal origin")
        except ValueError as exc:
            raise WorkerError("invalid_origin", "Origin, when supplied, must be an HTTP(S) loopback origin", 403) from exc


class SerialHTTPServer(HTTPServer):
    def __init__(self, address, worker):
        self.worker = worker
        super().__init__(address, WorkerHandler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10.0)
        return connection, address


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # Close connections; idle keep-alive cannot monopolize serial workers.
    server_version = "LocalFeatureWorker"

    def log_message(self, format, *args):
        pass  # No request bodies, filesystem names, origins or sensitive header logs.

    def _send(self, status, body):
        encoded = canonical_json(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, exc):
        self._send(exc.status, {"schema": SCHEMA, "status": "error",
                               "error": {"code": exc.code, "type": type(exc).__name__, "stage": exc.stage, "message": str(exc)}})

    def _local(self):
        if self.client_address[0] != HOST:
            raise WorkerError("invalid_peer", "only loopback clients are accepted", 403)
        validate_local_headers(self.headers, self.server.server_port)

    def do_GET(self):
        try:
            self._local()
            if self.path != "/health":
                raise WorkerError("not_found", "only GET /health is supported", 404)
            self._send(200, self.server.worker.health())
        except WorkerError as exc:
            self._error(exc)

    def do_POST(self):
        try:
            self._local()
            if self.path != "/infer":
                raise WorkerError("not_found", "only POST /infer is supported", 404)
            if self.headers.get_all("Transfer-Encoding", []):
                raise WorkerError("invalid_transfer_encoding", "chunked/transfer-encoded requests are not accepted")
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                raise WorkerError("length_required", "Content-Length is required", 411)
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]) or len(lengths[0]) > 10:
                raise WorkerError("invalid_content_length", "one finite decimal Content-Length is required")
            length = int(lengths[0])
            if length > MAX_BODY_BYTES:
                raise WorkerError("body_too_large", "request JSON exceeds 64 KiB", 413)
            types = self.headers.get_all("Content-Type", [])
            if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
                raise WorkerError("invalid_content_type", "Content-Type: application/json is required", 415)
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("short body")
                payload = strict_json(body)
            except TimeoutError as exc:
                raise WorkerError("body_timeout", "request body timed out", 408) from exc
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise WorkerError("invalid_json", "request body must be complete strict UTF-8 JSON") from exc
            status, result = self.server.worker.process(payload, http=True)
            self._send(status, result)
        except WorkerError as exc:
            self._error(exc)

    def do_OPTIONS(self):
        self._error(WorkerError("method_not_allowed", "only GET /health and POST /infer are supported", 405))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="local runtime TOML; no resources are downloaded")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--video")
    parser.add_argument("--output")
    parser.add_argument("--channels", default="rgb,corrected")
    parser.add_argument("--profile-json")
    args = parser.parse_args(argv)
    if args.config:
        import os
        os.environ["JEPA_CONFIG"] = str(Path(args.config).resolve())
    cfg = settings()
    global PORT, VIDEO_ROOT, R3D_CHECKPOINT, V_ENCODER, V_PREDICTOR, I_SLIM, I_FULL
    PORT, VIDEO_ROOT = cfg.worker_port, cfg.data_root
    R3D_CHECKPOINT = cfg.resources["r3d_checkpoint"]
    V_ENCODER, V_PREDICTOR = cfg.resources["vjepa_encoder"], cfg.resources["vjepa_predictor"]
    I_SLIM = I_FULL = cfg.resources["ijepa_checkpoint"]
    if args.serve and any((args.video, args.output, args.profile_json)):
        parser.error("--serve cannot be combined with single-video arguments")
    if not args.serve and not all((args.video, args.output, args.profile_json)):
        parser.error("single-video CLI requires --video, --output and --profile-json")
    worker = FeatureWorker(policy=PathPolicy(output_root=OUTPUT_ROOT, video_roots=(OUTPUT_ROOT, VIDEO_ROOT), bundle_root=OUTPUT_ROOT), retain_models=cfg.retain_models)
    if args.serve:
        server = None
        def stop(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            server = SerialHTTPServer((HOST, PORT), worker)
            print(canonical_json({**worker.health(), "listening": HOST + ":" + str(PORT)}), flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if server is not None:
                server.server_close()
            errors = worker.close()
            signal.signal(signal.SIGTERM, previous)
            if errors:
                print(canonical_json({"schema": SCHEMA, "status": "error", "cleanup_errors": errors}), file=sys.stderr)
        return 1 if errors else 0
    with redirect_stdout(sys.stderr):
        status, result = worker.process({"video": args.video, "output": args.output,
                                         "channels": args.channels, "profile_json": args.profile_json})
        errors = worker.close()
    if errors:
        result["status"] = "error"
        result["cleanup_errors"] = errors
        status = 500
    # Even a CLI profile/validation rejection gets a complete result when the
    # explicitly requested absolute output is new. Never touch existing output.
    if "result" not in result["paths"] and Path(args.output).is_absolute():
        target = Path(args.output).resolve()
        try:
            target.mkdir(parents=True, exist_ok=False)
            result["paths"]["result"] = str(target / "worker_result.json")
            write_json_exclusive(target / "worker_result.json", result)
        except FileExistsError:
            pass
        except Exception as exc:
            result["result_record_error_type"] = type(exc).__name__
    if errors and "result" in result["paths"]:
        Path(result["paths"]["result"]).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {key: result[key] for key in ("schema", "status", "features", "paths", "fps", "total_frames", "elapsed_seconds", "model_load_status")}
    if "error" in result:
        summary["error"] = result["error"]
    if errors:
        summary["cleanup_errors"] = errors
    print(canonical_json(summary), flush=True)
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
