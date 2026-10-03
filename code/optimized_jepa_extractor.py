#!/usr/bin/env python3
"""Isolated, genuine V/I-JEPA masked-prediction error extraction (WSL CLI).

Example (real inference; NOT used by the synthetic tests)::
    /home/zzy/vjepa2-main/vjepa-env/bin/python -B \
        /mnt/e/jepa-system/code/optimized_jepa_extractor.py \
        --video /path/video.mp4 --output /tmp/optimized-jepa --seed 0 \
        --max-frames 32 --max-keyframes 8

Add --bidirectional for a separately encoded time-reversed pass. --mask-passes 2
uses complementary half-masks at the default --mask-ratio 0.5, reusing only the
identical *masked* context forward, never full-target features as context.

Only the legacy model loaders/preprocessing are reused, not their compute()
methods or detector/cache/report paths. Imports are lazy: importing this module,
--help, and NumPy/fake-backend tests do not import torch/cv2 or load checkpoints.

Schema v1: signals.json + signals.npz (allow_pickle=False). Raw errors are NOT
per-video scaled. Relative scores retain the legacy V-JEPA max scaling and
I-JEPA min/max scaling, restricted to observed values. NaN (NPZ) / null (JSON)
and explicit masks distinguish missing evidence from genuine zero error. Dense
arrays contain only directly sampled frame evidence, no interpolation, endpoint
extrapolation, interval filling, or uncalibrated cross-model composite. A tubelet
score is shared by its two actual decoded members, not a per-frame prediction.
V-JEPA and I-JEPA error units are different and are not directly comparable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

SCHEMA_NAME = "optimized_true_jepa_signals"
SCHEMA_VERSION = 1
VJEPA_GRID = 24
IJEPA_GRID = 14
TUBELET_SIZE = 2
VJEPA_FEATURE_DIM = 5632


@dataclass(frozen=True)
class Config:
    seed: int = 0
    max_frames: int = 32
    max_keyframes: int = 8
    bidirectional: bool = False
    context_tubelets: int = 6
    target_tubelets: int = 2
    mask_ratio: float = 0.5
    mask_passes: int = 1
    ijepa_batch_size: int = 2

    def __post_init__(self):
        for name in ("max_frames", "max_keyframes", "context_tubelets",
                     "target_tubelets", "mask_passes", "ijepa_batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_frames > 64:
            raise ValueError("max_frames > 64 exceeds the existing V-JEPA predictor capacity")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not np.isfinite(self.mask_ratio) or not 0 < self.mask_ratio <= 1:
            raise ValueError("mask_ratio must be finite and in (0, 1]")


def _frame_ids(values, length: int) -> np.ndarray:
    ids = np.asarray(values)
    if ids.size == 0 and length == 0:
        return np.empty(0, dtype=np.int64)
    if ids.ndim != 1 or len(ids) != length or ids.dtype.kind not in "iu":
        raise ValueError("frame ids must be integer, one-dimensional, and match frames")
    ids = ids.astype(np.int64)
    if np.any(ids < 0) or np.any(np.diff(ids) <= 0):
        raise ValueError("frame ids must be nonnegative, unique, and increasing")
    return ids


def uniform_frame_ids(total: int, maximum: int) -> np.ndarray:
    if total <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, total - 1, min(total, maximum), dtype=np.int64))


@dataclass
class SampledVideo:
    frames: list
    frame_ids: np.ndarray
    keyframes: list
    keyframe_ids: np.ndarray
    total_frames: int
    fps: float | None
    metadata: dict


def sample_video(video, config: Config, capture_factory=None) -> SampledVideo:
    """Decode the union once; attach each successful read to its decoder position.

    CAP_PROP_POS_FRAMES after read is the next frame index. A failed seek/read,
    invalid position, or duplicate decode is recorded, never replaced/padded.
    Unknown frame count is rejected rather than claiming prefix/full coverage.
    """
    if capture_factory is None:
        import cv2
        capture_factory = cv2.VideoCapture
    # OpenCV's public property ids; keeping these here permits NumPy-only tests.
    pos_prop, fps_prop, count_prop = 1, 5, 7
    cap = capture_factory(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("video could not be opened")
        reported_count = float(cap.get(count_prop))
        if not np.isfinite(reported_count) or reported_count < 1:
            raise ValueError("video frame count unavailable; full-video sampling is undefined")
        total = int(reported_count)
        fps_value = float(cap.get(fps_prop))
        fps = fps_value if np.isfinite(fps_value) and fps_value > 0 else None
        requested_v = uniform_frame_ids(total, config.max_frames)
        requested_i = uniform_frame_ids(total, config.max_keyframes)
        decoded, bindings, failures, remaps = {}, {}, [], []
        for requested in np.union1d(requested_v, requested_i):
            requested = int(requested)
            if not cap.set(pos_prop, requested):
                failures.append({"requested_id": requested, "reason": "seek_failed"})
                continue
            success, frame = cap.read()
            if not success or frame is None or not getattr(frame, "size", 0):
                failures.append({"requested_id": requested, "reason": "decode_failed"})
                continue
            next_pos = float(cap.get(pos_prop))
            if (not np.isfinite(next_pos) or abs(next_pos - round(next_pos)) > 1e-3
                    or not 1 <= next_pos <= total):
                failures.append({"requested_id": requested, "reason": "unverifiable_frame_id"})
                continue
            actual = int(round(next_pos)) - 1
            if actual in decoded:
                failures.append({"requested_id": requested, "actual_id": actual,
                                 "reason": "duplicate_decoded_frame"})
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                failures.append({"requested_id": requested, "reason": "invalid_bgr_frame"})
                continue
            decoded[actual] = frame
            bindings[requested] = actual
            if actual != requested:
                remaps.append({"requested_id": requested, "actual_id": actual})

        def select(requested):
            ids = np.asarray(sorted({bindings[int(i)] for i in requested
                                     if int(i) in bindings}), dtype=np.int64)
            return [decoded[int(i)] for i in ids], ids

        frames, ids = select(requested_v)
        keyframes, key_ids = select(requested_i)
        metadata = {
            "strategy": "uniform_requested_ids_decode_union_once",
            "frame_id_semantics": "zero-based decoder-reported next_position_minus_one",
            "reported_total_frames": total,
            "total_frames_verified_by_full_decode": False,
            "requested_frame_ids": requested_v.tolist(),
            "requested_keyframe_ids": requested_i.tolist(),
            "successful_frame_ids": ids.tolist(),
            "successful_keyframe_ids": key_ids.tolist(),
            "failures": failures,
            "seek_remaps": remaps,
            "padding_or_failure_replacement": False,
        }
        return SampledVideo(frames, ids, keyframes, key_ids, total, fps, metadata)
    finally:
        cap.release()


def plan_windows(n_tubelets: int, context_tubelets: int, target_tubelets: int):
    """Keep >= 1 real context and target, using actual complete tubelet count.

    Context is capped at floor((U-1)/2): short clips leave at least half their
    tubelets as targets, and forward+reverse can cover the whole tubelet axis.
    Append the last full window to score a tail that a target-sized stride misses.
    """
    if n_tubelets < 2:
        return 0, 0, []
    context = min(context_tubelets, max(1, (n_tubelets - 1) // 2))
    target = min(target_tubelets, n_tubelets - context)
    last = n_tubelets - context - target
    starts = list(range(0, last + 1, target))
    if starts[-1] != last:
        starts.append(last)
    return context, target, starts


def relative_scores(raw: np.ndarray, valid: np.ndarray, kind: str) -> np.ndarray:
    """Legacy-compatible scaling on observed scores ONLY, with missing NaN."""
    values = np.asarray(raw, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or np.any(~np.isfinite(values[valid])):
        raise ValueError("valid scores must be finite and aligned")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    if not valid.any():
        return result
    observed = values[valid]
    if kind == "vjepa":
        maximum = observed.max()
        result[valid] = observed / maximum if maximum > 0 else observed
    elif kind == "ijepa":
        low, high = observed.min(), observed.max()
        # Upstream preserves constants, including nonzero constants.
        result[valid] = ((observed - low) / (high - low + 1e-6)
                         if high > low else observed)
    else:
        raise ValueError(f"unknown score kind: {kind}")
    return result


@dataclass
class Evidence:
    kind: str
    raw_errors: np.ndarray
    relative_scores: np.ndarray
    valid_mask: np.ndarray
    raw_heatmaps: np.ndarray
    patch_valid_mask: np.ndarray
    patch_counts: np.ndarray
    metadata: dict = field(default_factory=dict)
    direction_raw_errors: np.ndarray | None = None
    direction_valid_mask: np.ndarray | None = None
    direction_counts: np.ndarray | None = None


def _scores_from_accumulators(sums, counts):
    totals = counts.reshape(len(counts), -1).sum(axis=1) if len(counts) else np.empty(0)
    error_totals = sums.reshape(len(sums), -1).sum(axis=1) if len(sums) else np.empty(0)
    valid = totals > 0
    scores = np.full(len(sums), np.nan, dtype=np.float32)
    np.divide(error_totals, totals, out=scores, where=valid, casting="unsafe")
    return scores, valid, totals.astype(np.int64)


def evidence_from_accumulators(kind, sums, counts, metadata=None) -> Evidence:
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if sums.ndim != 3 or sums.shape != counts.shape or np.any(counts < 0):
        raise ValueError("patch accumulators must have matching [N,G,G] shapes")
    valid = counts > 0
    if np.any(~np.isfinite(sums)) or np.any(sums < 0) or np.any(sums[~valid] != 0):
        raise ValueError("patch error sums must be finite, nonnegative, and counted")
    heatmaps = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=heatmaps, where=valid, casting="unsafe")
    raw, score_mask, _ = _scores_from_accumulators(sums, counts)
    meta = dict(metadata or {})
    meta.setdefault("status", "ok" if score_mask.any() else "insufficient_evidence")
    return Evidence(kind, raw, relative_scores(raw, score_mask, kind), score_mask,
                    heatmaps, valid, counts, meta)


def empty_evidence(kind, n, reason, status="insufficient_evidence", grid=None):
    grid = grid or (VJEPA_GRID if kind == "vjepa" else IJEPA_GRID)
    return evidence_from_accumulators(
        kind, np.zeros((n, grid, grid)), np.zeros((n, grid, grid), dtype=np.int64),
        {"status": status, "reason": reason})


def unwrap_prediction(result, expected_shape):
    """Select target output, not predictor context output; never truncate mismatch."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError("ambiguous predictor output list")
        result = result[0]
    if tuple(result.shape) != tuple(expected_shape):
        raise RuntimeError(f"predictor shape {tuple(result.shape)} != target {tuple(expected_shape)}")
    return result


def _checked_errors(values, length):
    errors = np.asarray(values, dtype=np.float64)
    if errors.shape != (length,) or np.any(~np.isfinite(errors)) or np.any(errors < 0):
        raise RuntimeError("predictor errors must be aligned, finite, and nonnegative")
    return errors


def run_vjepa(frames, frame_ids, backend, config: Config) -> Evidence:
    """Backend always encodes real full targets and separately masked contexts."""
    ids = _frame_ids(frame_ids, len(frames))
    n = len(frames) // TUBELET_SIZE
    grid = getattr(backend, "grid", VJEPA_GRID)
    if n < 2:
        result = empty_evidence("vjepa", n, "need_at_least_two_complete_tubelets", grid=grid)
        result.metadata.update({"tubelet_frame_ids": ids[:2*n].reshape(n, 2).tolist(),
                                "dropped_unpaired_frame_ids": ids[2*n:].tolist(),
                                "effective_context_tubelets": 0, "effective_target_tubelets": 0,
                                "directions": []})
        return result
    context, target, starts = plan_windows(n, config.context_tubelets, config.target_tubelets)
    spatial = grid * grid
    n_pred = max(1, int(spatial * config.mask_ratio))
    directions = ["forward", "reverse"] if config.bidirectional else ["forward"]
    total_sums = np.zeros((n, grid, grid), dtype=np.float64)
    total_counts = np.zeros((n, grid, grid), dtype=np.int64)
    directional_scores, directional_masks, directional_counts = [], [], []
    kept_frames = frames[:2*n]  # Drop BEFORE reversal so tubelet membership stays identical.
    for d, direction in enumerate(directions):
        oriented = kept_frames if d == 0 else kept_frames[::-1]
        effective = backend.prepare(oriented)
        if effective != n:
            raise RuntimeError(f"encoder returned {effective} tubelets; decoded mapping requires {n}")
        sums = np.zeros_like(total_sums)
        counts = np.zeros_like(total_counts)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, d]))
        for start in starts:
            context_ids = np.arange(start * spatial, (start + context) * spatial, dtype=np.int64)
            permutations = [rng.permutation(spatial) for _ in range(target)]
            for pass_id in range(config.mask_passes):
                selection = np.arange(pass_id * n_pred, (pass_id + 1) * n_pred) % spatial
                target_ids = np.concatenate([
                    (start + context + offset) * spatial + permutation[selection]
                    for offset, permutation in enumerate(permutations)
                ]).astype(np.int64)
                errors = _checked_errors(backend.predict_errors(context_ids, target_ids), len(target_ids))
                np.add.at(sums.reshape(-1), target_ids, errors)
                np.add.at(counts.reshape(-1), target_ids, 1)
        if d:
            sums, counts = sums[::-1], counts[::-1]
        raw, valid, observations = _scores_from_accumulators(sums, counts)
        directional_scores.append(raw)
        directional_masks.append(valid)
        directional_counts.append(observations)
        total_sums += sums
        total_counts += counts
    metadata = {
        "status": "ok", "tubelet_frame_ids": ids[:2*n].reshape(n, 2).tolist(),
        "dropped_unpaired_frame_ids": ids[2*n:].tolist(), "directions": directions,
        "effective_context_tubelets": context, "effective_target_tubelets": target,
        "window_starts": starts, "stride_tubelets": target,
        "predicted_patches_per_target_tubelet_per_pass": n_pred,
        "mask_passes": config.mask_passes, "mask_ratio_requested": config.mask_ratio,
        "mask_ratio_effective": n_pred / spatial,
        "context_policy": "min(requested, max(1, floor((complete_tubelets-1)/2)))",
        "direction_fusion": "token_observation_count_weighted_absolute_error_mean",
    }
    result = evidence_from_accumulators("vjepa", total_sums, total_counts, metadata)
    result.direction_raw_errors = np.stack(directional_scores)
    result.direction_valid_mask = np.stack(directional_masks)
    result.direction_counts = np.stack(directional_counts)
    return result


def run_ijepa(frames, frame_ids, backend, config: Config) -> Evidence:
    """Retain the original smooth-L1 scores BEFORE any per-video normalization."""
    ids = _frame_ids(frame_ids, len(frames))
    grid = getattr(backend, "grid", IJEPA_GRID)
    if not frames:
        return empty_evidence("ijepa", 0, "no_successfully_decoded_keyframes", grid=grid)
    sums = np.zeros((len(frames), grid, grid), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    for start in range(0, len(frames), config.ijepa_batch_size):
        batch = frames[start:start + config.ijepa_batch_size]
        patch_ids, errors = backend.batch_errors(batch, seed=(config.seed + start) % 2**63)
        patch_ids, errors = np.asarray(patch_ids), np.asarray(errors)
        if (patch_ids.ndim != 3 or patch_ids.shape != errors.shape
                or patch_ids.shape[1] != len(batch) or patch_ids.shape[0] < 1
                or patch_ids.shape[2] < 1 or patch_ids.dtype.kind not in "iu"
                or np.any(patch_ids < 0) or np.any(patch_ids >= grid * grid)):
            raise RuntimeError("I-JEPA masks/errors must match [num_masks,batch,patches]")
        for b in range(len(batch)):
            pids = patch_ids[:, b].reshape(-1)
            observed = _checked_errors(errors[:, b].reshape(-1), len(pids))
            np.add.at(sums[start+b].reshape(-1), pids, observed)
            np.add.at(counts[start+b].reshape(-1), pids, 1)
    return evidence_from_accumulators("ijepa", sums, counts, {
        "status": "ok", "keyframe_ids": ids.tolist(), "batch_size": config.ijepa_batch_size,
        "mask_seed": config.seed, "block_seed_policy": "explicit_batch_seed_reset_and_restore_independent_of_model_reuse",
        "loss": "smooth_l1_beta_1_mean_over_features",
        "target_layer_norm": True, "patch_overlap_aggregation": "observation_count_weighted_mean",
    })


@contextmanager
def isolated_legacy_module(kind):
    """Temporarily isolate upstreams' conflicting src/app packages; restore caller.

    No edits, copied source trees, cache reads/writes, checkpoint downloads, or
    global monkeypatching of upstream classes. Model use must finish in scope.
    """
    if kind not in ("vjepa", "ijepa"):
        raise ValueError("unknown legacy kind")
    roots = {"vjepa": "/home/zzy/vjepa2-main", "ijepa": "/home/zzy/ijepa-main"}
    filename = f"{kind}_predictor.py"
    path = Path(__file__).with_name(filename)
    if not path.is_file():
        path = Path("/home/zzy/jepa_data") / filename
    if not path.is_file():
        raise FileNotFoundError(f"legacy loader not found: {filename}")
    alias = f"_optimized_jepa_legacy_{kind}"

    def scoped(name):
        return (name == alias or name == "ijepa_predict"
                or name == "src" or name.startswith("src.")
                or name == "app" or name.startswith("app."))

    saved_modules = {name: module for name, module in list(sys.modules.items()) if scoped(name)}
    saved_path = sys.path[:]
    saved_bytecode = sys.dont_write_bytecode
    try:
        for name in saved_modules:
            del sys.modules[name]
        sys.path[:0] = [str(path.parent), roots[kind]]
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in list(sys.modules):
            if scoped(name):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode


class _RealVJEPA:
    grid = VJEPA_GRID

    def __init__(self, legacy):
        self.legacy = legacy
        self.torch = legacy.torch
        self.scorer = legacy.VJEPASurprise()
        self.video = self.targets = self.context_tokens = self.context_key = None
        self.stats = {"loader_source": legacy.__file__, "encoder_checkpoint": legacy.ENCODER_CKPT,
                      "predictor_checkpoint": legacy.FULL_CKPT, "device": str(legacy.DEVICE),
                      "dtype": "bfloat16", "error_reduction_dtype": "float32",
                      "target_seconds": 0.0, "context_predictor_seconds": 0.0}

    def prepare(self, frames):
        torch, device = self.torch, self.legacy.DEVICE
        self.video = self.targets = self.context_tokens = self.context_key = None
        if self.scorer.encoder is None:
            started = time.perf_counter()
            self.scorer._load_encoder()
            self.scorer._load_predictor()
            self.stats["load_seconds"] = time.perf_counter() - started
        # OpenCV supplies BGR; checkpoint preprocessing requires RGB. The legacy
        # V loader's preprocess does NOT swap channels, unlike its I counterpart.
        rgb = [frame[:, :, ::-1].copy() for frame in frames]
        self.video = self.scorer.preprocess(rgb)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.targets = self.scorer.encoder(self.video, training=True)
        torch.cuda.synchronize(device)
        self.stats["target_seconds"] += time.perf_counter() - started
        spatial = self.grid**2
        shape = tuple(self.targets.shape)
        if (len(shape) != 3 or shape[0] != 1 or shape[2] != VJEPA_FEATURE_DIM
                or shape[1] % spatial):
            raise RuntimeError(f"expected hierarchical target [1,U*576,5632], got {shape}")
        if shape[1] > self.scorer.predictor.num_patches:
            raise RuntimeError("target sequence exceeds pretrained predictor position capacity")
        return shape[1] // spatial

    def predict_errors(self, context_ids, target_ids):
        torch, device = self.torch, self.legacy.DEVICE
        started = time.perf_counter()
        key = tuple(context_ids.tolist())
        context_mask = torch.as_tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
        target_mask = torch.as_tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.context_key != key:
                self.context_tokens = self.scorer.encoder(self.video, masks=[context_mask], training=True)
                self.context_key = key
            if tuple(self.context_tokens.shape) != (1, len(context_ids), VJEPA_FEATURE_DIM):
                raise RuntimeError("masked context encoder did not preserve mask shape")
            output = self.scorer.predictor(self.context_tokens, [context_mask], [target_mask])
        predicted = unwrap_prediction(output, (1, len(target_ids), VJEPA_FEATURE_DIM))
        with torch.inference_mode():
            target = self.targets.index_select(1, target_mask[0])
            errors = (predicted.float() - target.float()).norm(dim=-1)
            values = errors[0].cpu().numpy()
        self.stats["context_predictor_seconds"] += time.perf_counter() - started
        return values

    def close(self):
        self.video = self.targets = self.context_tokens = self.context_key = None
        self.scorer._free_predictor()
        self.scorer._free_encoder()


@contextmanager
def seeded_mask_counter(collator, seed):
    """Isolate upstream's hidden block-size RNG from model reuse/request order.

    Upstream MaskCollator.step() uses a multiprocessing int32 counter as its own
    generator seed; fixing torch's global RNG alone is insufficient. Reset this
    counter to the explicit batch seed and restore it even on failure. Calls are
    serialized by the worker; this is not a cross-thread masking API.
    """
    counter=getattr(collator, "_itr_counter", None)
    if counter is None or not hasattr(counter, "get_lock"):
        raise RuntimeError("unsupported I-JEPA mask collator counter")
    with counter.get_lock():
        previous=counter.value
        counter.value=int(seed) % (2**31-1)-1
    try:
        yield
    finally:
        with counter.get_lock():counter.value=previous


class _RealIJEPA:
    grid = IJEPA_GRID

    def __init__(self, legacy, seed):
        self.legacy, self.torch = legacy, legacy.torch
        self.scorer = legacy.IJEPASurprise(mask_seed=seed)
        checkpoint = legacy.CKPT_SLIM_BF16 if legacy.CKPT_SLIM_BF16.exists() else legacy.CKPT_FULL
        self.stats = {"loader_source": legacy.__file__, "checkpoint": str(checkpoint),
                      "device": str(legacy.DEVICE), "dtype": "bfloat16",
                      "error_reduction_dtype": "float32", "inference_seconds": 0.0}

    def batch_errors(self, frames, seed):
        torch, device = self.torch, self.legacy.DEVICE
        if self.scorer.context_encoder is None:
            started = time.perf_counter()
            self.scorer._load_models()
            self.stats["load_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        batch_tensors = self.scorer._preprocess(frames)
        # The collator uses CPU RNG for locations and its own step-seeded block
        # sizes (same semantics as upstream). Restore the caller's CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            with seeded_mask_counter(self.scorer.mask_collator, seed):
                imgs, masks_enc, masks_pred = self.scorer.mask_collator(batch_tensors)
        imgs = imgs.to(device, dtype=torch.bfloat16)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]
        if len(masks_enc) != 1:
            raise RuntimeError("I-JEPA score layout requires exactly one context mask")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            target = self.scorer.target_encoder(imgs)
            target = self.legacy.F.layer_norm(target, (target.size(-1),))
            from src.masks.utils import apply_masks
            target_masked = apply_masks(target, masks_pred)
            context = self.scorer.context_encoder(imgs, masks_x=masks_enc)
            predicted = self.scorer.predictor(context, masks_enc, masks_pred)
        predicted = unwrap_prediction(predicted, target_masked.shape)
        with torch.inference_mode():
            errors = self.legacy.F.smooth_l1_loss(
                predicted.float(), target_masked.float(), reduction="none", beta=1.0).mean(dim=-1)
            values = errors.reshape(len(masks_pred), len(frames), -1).cpu().numpy()
            patch_ids = np.stack([mask.cpu().numpy() for mask in masks_pred])
        self.stats["inference_seconds"] += time.perf_counter() - started
        return patch_ids, values

    def close(self):
        self.scorer._free_models()


@contextmanager
def real_backend(kind, config):
    """Production-only lazy entry point; no fake/proxy fallback."""
    with isolated_legacy_module(kind) as legacy:
        torch, device = legacy.torch, legacy.DEVICE
        if not torch.cuda.is_available():
            raise RuntimeError("existing JEPA loaders require CUDA; no proxy or CPU fallback is used")
        torch.cuda.reset_peak_memory_stats(device)
        backend = _RealVJEPA(legacy) if kind == "vjepa" else _RealIJEPA(legacy, config.seed)
        try:
            yield backend
        finally:
            backend.stats.update({
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            })
            backend.close()


def bind_dense_evidence(total_frames, members, scores, valid):
    """Bind only explicitly observed source-frame members, never an interval."""
    members = np.asarray(members)
    scores, valid = np.asarray(scores), np.asarray(valid, dtype=bool)
    if (members.ndim != 2 or members.dtype.kind not in "iu"
            or len(members) != len(scores) or scores.shape != valid.shape
            or np.any(members < 0) or np.any(members >= total_frames)
            or np.unique(members).size != members.size):
        raise ValueError("source frame members must be aligned, unique and in bounds")
    if np.any(~np.isfinite(scores[valid])):
        raise ValueError("observed scores must be finite")
    dense = np.full(total_frames, np.nan, dtype=np.float32)
    mask = np.zeros(total_frames, dtype=bool)
    selected = members[valid].reshape(-1)
    dense[selected] = np.repeat(scores[valid], members.shape[1])
    mask[selected] = True
    return dense, mask


def _peak_rss():
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
                "scope": "whole_process_lifetime_peak_including_imports_and_model_loads"}
    except (ImportError, AttributeError):
        return {"process_peak_rss_bytes": None, "scope": "unavailable_on_this_platform"}


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _normalization_metadata(evidence):
    raw = evidence.raw_errors[evidence.valid_mask]
    low, high = (float(raw.min()), float(raw.max())) if raw.size else (None, None)
    return {
        "raw": ("L2 norm over 5632 hierarchical feature channels; mean over predicted token observations"
                if evidence.kind == "vjepa" else
                "smooth_L1(beta=1) mean over feature channels, masks and masked patch observations; target layer_norm"),
        "raw_per_video_scaling": False,
        "relative_formula": ("raw / max(valid_raw) if max > 0; otherwise preserve valid zero"
                             if evidence.kind == "vjepa" else
                             "(raw - min(valid_raw)) / (max(valid_raw) - min(valid_raw) + 1e-6) if range > 0; otherwise preserve constants"),
        "relative_scope": "per_video_after_direction_fusion_valid_values_only",
        "valid_min": low, "valid_max": high, "valid_count": int(raw.size),
        "relative_is_calibrated_probability": False,
        "constant_ijepa_may_be_outside_0_1": evidence.kind == "ijepa",
        "compatibility": "legacy score formulas, NOT legacy zero-filled masks or heatmap normalization",
    }


def _coverage(evidence):
    result = {
        "observed_positions": int(evidence.valid_mask.sum()),
        "total_positions": int(evidence.valid_mask.size),
        "position_fraction": float(evidence.valid_mask.mean()) if evidence.valid_mask.size else 0.0,
        "observed_patches": int(evidence.patch_valid_mask.sum()),
        "total_patches": int(evidence.patch_valid_mask.size),
        "patch_fraction": float(evidence.patch_valid_mask.mean()) if evidence.patch_valid_mask.size else 0.0,
        "token_observations": int(evidence.patch_counts.sum()),
    }
    if evidence.direction_valid_mask is not None:
        result["directions"] = {
            label: {"observed_tubelets": int(mask.sum()), "tubelet_fraction": float(mask.mean())}
            for label, mask in zip(evidence.metadata["directions"], evidence.direction_valid_mask)
        }
    return result


def build_artifacts(video, sampled: SampledVideo, vjepa: Evidence, ijepa: Evidence,
                    config: Config, runtime=None):
    """Create self-describing numeric arrays + strict JSON, without file I/O."""
    ids = _frame_ids(sampled.frame_ids, len(sampled.frames))
    key_ids = _frame_ids(sampled.keyframe_ids, len(sampled.keyframes))
    n = len(ids) // TUBELET_SIZE
    tubelet_ids = ids[:2*n].reshape(n, 2)
    physics, p_mask = bind_dense_evidence(sampled.total_frames, tubelet_ids,
                                         vjepa.raw_errors, vjepa.valid_mask)
    physics_relative, _ = bind_dense_evidence(sampled.total_frames, tubelet_ids,
                                             vjepa.relative_scores, vjepa.valid_mask)
    corruption, c_mask = bind_dense_evidence(sampled.total_frames, key_ids[:, None],
                                            ijepa.raw_errors, ijepa.valid_mask)
    corruption_relative, _ = bind_dense_evidence(sampled.total_frames, key_ids[:, None],
                                                ijepa.relative_scores, ijepa.valid_mask)
    timeline_ids = np.arange(sampled.total_frames, dtype=np.int64)
    timestamps = (timeline_ids.astype(np.float64) / sampled.fps if sampled.fps is not None
                  else np.full(sampled.total_frames, np.nan, dtype=np.float64))
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "frame_ids": timeline_ids, "timestamps_sec": timestamps,
        "sampled_frame_ids": ids, "keyframe_ids": key_ids, "tubelet_frame_ids": tubelet_ids,
        "dropped_unpaired_frame_ids": ids[2*n:],
        "physics_raw": physics, "physics_relative": physics_relative, "physics_valid_mask": p_mask,
        "corruption_raw": corruption, "corruption_relative": corruption_relative,
        "corruption_valid_mask": c_mask,
        "signals_raw": np.stack([physics, corruption], axis=1),
        "signals_relative": np.stack([physics_relative, corruption_relative], axis=1),
        "signals_valid_mask": np.stack([p_mask, c_mask], axis=1),
    }
    for evidence in (vjepa, ijepa):
        for attr in ("raw_errors", "relative_scores", "valid_mask", "raw_heatmaps",
                     "patch_valid_mask", "patch_counts"):
            arrays[f"{evidence.kind}_{attr}"] = getattr(evidence, attr)
    arrays["vjepa_direction_raw_errors"] = (vjepa.direction_raw_errors if vjepa.direction_raw_errors is not None
                                             else np.empty((0, n), dtype=np.float32))
    arrays["vjepa_direction_valid_mask"] = (vjepa.direction_valid_mask if vjepa.direction_valid_mask is not None
                                             else np.empty((0, n), dtype=bool))
    arrays["vjepa_direction_counts"] = (vjepa.direction_counts if vjepa.direction_counts is not None
                                         else np.empty((0, n), dtype=np.int64))
    statuses = [vjepa.metadata["status"], ijepa.metadata["status"]]
    status = ("error" if "error" in statuses else "insufficient_evidence"
              if not vjepa.valid_mask.any() else "partial_evidence" if not ijepa.valid_mask.any() else "ok")
    metadata = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "status": status,
        "video": {"path": str(video), "total_frames": sampled.total_frames, "fps": sampled.fps,
                  "fps_fallback_used": False, "timeline_extent_source": "container_reported_frame_count"},
        "jepa_usage": {"vjepa": "true_masked_encoder_predictor_vs_hierarchical_target",
                       "ijepa": "true_masked_context_predictor_vs_layer_normalized_ema_target",
                       "proxy_fallback": False},
        "profile": {"name": "isolated_true_jepa_v1", "requested": asdict(config),
                    "vjepa": vjepa.metadata, "ijepa": ijepa.metadata,
                    "vjepa_preprocessing": "BGR_to_RGB_then_legacy_short_side_resize_center_crop_384_ImageNet_norm",
                    "ijepa_preprocessing": "legacy_BGR_to_RGB_resize_256_bicubic_center_crop_224_ImageNet_norm",
                    "v_target_encoded_once_per_direction": True,
                    "v_context_reuse": "only_same_exact_mask_within_direction",
                    "raw_error_fusion_across_models": "none_different_loss_units",
                    "full_video_interpolation": False,
                    "determinism": "seeded_masks_not_a_guarantee_of_bitwise_cuda_kernel_determinism"},
        "sampling": sampled.metadata,
        "score_normalization": {"vjepa": _normalization_metadata(vjepa),
                                "ijepa": _normalization_metadata(ijepa)},
        "coverage": {"vjepa": _coverage(vjepa), "ijepa": _coverage(ijepa),
                     "direct_vjepa_frames": int(p_mask.sum()), "direct_ijepa_frames": int(c_mask.sum()),
                     "source_frames": sampled.total_frames,
                     "vjepa_frame_fraction": float(p_mask.mean()) if p_mask.size else 0.0,
                     "ijepa_frame_fraction": float(c_mask.mean()) if c_mask.size else 0.0,
                     "both_signals_observed_frames": int((p_mask & c_mask).sum()),
                     "denominator_source": "reported_frame_count_not_full_decode_verified"},
        "missing_value_semantics": {"npz": "NaN", "json": "null", "valid_mask_required": True,
                                    "zero_error_can_be_valid": True,
                                    "tubelet_to_frame": "shared_score_on_actual_two_members_only_not_interval",
                                    "dropped_odd_frame": "no_vjepa_evidence_even_in_reverse"},
        "signal_columns": ["true_vjepa_absolute_error", "true_ijepa_absolute_error"],
        "relative_signal_columns": ["true_vjepa_legacy_relative", "true_ijepa_legacy_relative"],
        "runtime": runtime or {},
        "artifacts": {"arrays": "signals.npz", "json": "signals.json", "npz_allow_pickle": False},
        "array_schema": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape),
                   "semantics": ("bool observation mask; false is missing, not normal" if value.dtype == bool else
                                 "zero-based source frame ids" if name.endswith("frame_ids") else
                                 "unscaled mean predictor-target error; NaN where unobserved" if "raw" in name else
                                 "predicted token observation counts including overlaps/directions/passes" if "counts" in name else
                                 "legacy per-video relative score; NaN where unobserved" if "relative" in name else
                                 "see schema and profile")}
            for name, value in arrays.items()
        },
        "signals": {name: arrays[name] for name in (
            "frame_ids", "timestamps_sec", "physics_raw", "physics_relative", "physics_valid_mask",
            "corruption_raw", "corruption_relative", "corruption_valid_mask")},
        "vjepa": {"raw_errors": vjepa.raw_errors, "relative_scores": vjepa.relative_scores,
                  "valid_mask": vjepa.valid_mask, "tubelet_frame_ids": tubelet_ids,
                  "direction_labels": vjepa.metadata.get("directions", []),
                  "direction_raw_errors": arrays["vjepa_direction_raw_errors"],
                  "direction_valid_mask": arrays["vjepa_direction_valid_mask"],
                  "direction_counts": arrays["vjepa_direction_counts"]},
        "ijepa": {"raw_errors": ijepa.raw_errors, "relative_scores": ijepa.relative_scores,
                  "valid_mask": ijepa.valid_mask, "keyframe_ids": key_ids},
    }
    metadata = _json_safe(metadata)
    # A scalar Unicode string, NOT a pickled object array; NPZ alone is self-describing.
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    metadata["array_schema"]["metadata_json"] = {
        "dtype": "unicode", "shape": [], "semantics": "strict JSON metadata including schemas and frame scores"}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    return arrays, metadata


def extract(video, config: Config, backend_factory=real_backend, capture_factory=None):
    started = time.perf_counter()
    sampled = sample_video(video, config, capture_factory=capture_factory)
    timings = {"sampling_seconds": time.perf_counter() - started}
    models = {}
    results = {}
    for kind, frames, ids in (("vjepa", sampled.frames, sampled.frame_ids),
                              ("ijepa", sampled.keyframes, sampled.keyframe_ids)):
        stage_start = time.perf_counter()
        insufficient = len(frames) < 4 if kind == "vjepa" else not frames
        if insufficient:
            results[kind] = (run_vjepa(frames, ids, None, config) if kind == "vjepa" else
                             empty_evidence("ijepa", 0, "no_successfully_decoded_keyframes"))
            models[kind] = {"loaded": False, "reason": results[kind].metadata["reason"]}
        else:
            backend = None
            try:
                with backend_factory(kind, config) as backend:
                    results[kind] = (run_vjepa(frames, ids, backend, config) if kind == "vjepa" else
                                     run_ijepa(frames, ids, backend, config))
                models[kind] = {"loaded": True, **getattr(backend, "stats", {})}
            except Exception as exc:
                # No encoder-distance or simulated signal fallback on OOM/load/shape errors.
                n = len(frames) // 2 if kind == "vjepa" else len(frames)
                results[kind] = empty_evidence(kind, n, "true_model_inference_failed", status="error")
                results[kind].metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
                models[kind] = {"loaded": backend is not None, **getattr(backend, "stats", {})}
        timings[f"{kind}_seconds_including_load_cleanup"] = time.perf_counter() - stage_start
    timings["extraction_seconds_excluding_serialization"] = time.perf_counter() - started
    runtime = {"timings": timings, "models": models, "memory": _peak_rss(),
               "python_version": sys.version.split()[0], "numpy_version": np.__version__,
               "gpu_benchmark_performed_by_tests": False}
    return build_artifacts(video, sampled, results["vjepa"], results["ijepa"], config, runtime)


def write_outputs(output, arrays, metadata):
    """Explicit output only; exclusive creation prevents overwriting historical runs."""
    output = Path(output)
    if any((output / name).exists() for name in ("signals.json", "signals.npz")):
        raise FileExistsError("signals.json/signals.npz already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "signals.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with (output / "signals.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="WSL-accessible source video path")
    parser.add_argument("--output", required=True, help="new explicit output directory; never overwrite existing signals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=32, help="V-JEPA requested samples, at most 64")
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--bidirectional", action="store_true", help="encode and predict in both time directions")
    parser.add_argument("--context-tubelets", type=int, default=6)
    parser.add_argument("--target-tubelets", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-passes", type=int, default=1, help="rotating complementary patch masks; more passes cost more predictor forwards")
    parser.add_argument("--ijepa-batch-size", type=int, default=2)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    except ValueError as exc:
        parser.error(str(exc))
    try:
        # Reject accidental overwrites BEFORE expensive true-model loading.
        if any((Path(args.output) / name).exists() for name in ("signals.json", "signals.npz")):
            raise FileExistsError("choose a new output directory; existing signals are never overwritten")
        arrays, metadata = extract(args.video, config)
        write_outputs(args.output, arrays, metadata)
    except (ValueError, OSError) as exc:
        print(f"optimized JEPA extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": metadata["status"], "output": str(Path(args.output).resolve()),
                      "coverage": metadata["coverage"]}, allow_nan=False), flush=True)
    return 1 if metadata["status"] == "error" else 0 if metadata["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
