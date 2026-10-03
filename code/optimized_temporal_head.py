"""Small-data, fixed-budget temporal head with NumPy-only deployment.

Public API:
    train_temporal([video_TD, ...], [binary_labels_T, ...], [real_fps, ...],
                   seed, *, steps=400) -> JSON-serializable dict
    predict_temporal(bundle, video_TD, real_fps) -> float32 probabilities[T]

No video identifiers, paths, evaluation labels, downloads or file IO are used.
Training lazily imports PyTorch and runs on CPU with two threads. Invoke it in
an existing PyTorch environment (e.g. WSL); prediction imports only NumPy and
Python's standard library. Hyperparameters and the final step are fixed, not
selected using any validation/test labels.

Times are frame_index / real_fps, not video_duration / frame_count. The 10 Hz
regular grid includes t=0 and appends the exact last-frame time when necessary;
the final interval can therefore be shorter than 0.1 s. Features are linearly
interpolated, labels use nearest-frame half-up rounding, and probabilities are
linearly interpolated back to every original frame. Training and inference use
exactly the same NumPy feature transform. Empty training videos and nonfinite
inputs are rejected; constant features and single-class labels are supported.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

SCHEMA = "optimized-temporal-head-v1"
TARGET_HZ = 10.0
CHANNELS = 32
DILATIONS = (1, 2, 4)
DROPOUT = (0.15, 0.20, 0.20)
CPU_THREADS = 2
BATCH_SIZE = 8
POSITIVE_WEIGHT = 1.25
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
STD_FLOOR = 1e-6
GRID_RULE = "regular_10hz_plus_exact_endpoint"


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _fps(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("fps must be a finite positive real number; no guessed FPS")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("fps must be a finite positive real number; no guessed FPS")
    return result


def _array(value: Any, name: str, ndim: int, *, allow_bool: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    kinds = "biuf" if allow_bool else "iuf"
    if result.ndim != ndim or result.dtype.kind not in kinds:
        raise ValueError(f"{name} must be a real numeric {ndim}D array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite float32-representable values")
    return result


def _features(value: Any, *, dimension: int | None = None, empty: bool = False) -> np.ndarray:
    result = _array(value, "values [T,D]", 2)
    if result.shape[1] < 1 or (result.shape[0] < 1 and not empty):
        raise ValueError("values must have T >= 1 and D >= 1 (only prediction permits T=0)")
    if dimension is not None and result.shape[1] != dimension:
        raise ValueError(f"feature dimension mismatch: expected {dimension}, got {result.shape[1]}")
    return result


def _training_inputs(values: Any, labels: Any, fps: Any) -> tuple[list, list, list]:
    for value, name in ((values, "values"), (labels, "labels"), (fps, "fps")):
        if not isinstance(value, (list, tuple)) or not len(value):
            raise ValueError(f"training {name} must be a nonempty list of videos")
    if not (len(values) == len(labels) == len(fps)):
        raise ValueError("values, labels and fps must have the same video count")
    videos, targets, rates = [], [], []
    dimension = None
    for index, (video, target, rate) in enumerate(zip(values, labels, fps)):
        x = _features(video, dimension=dimension)
        dimension = x.shape[1]
        # Validate binary values before float32 conversion, not after rounding.
        y = _array(target, "labels [T]", 1, allow_bool=True)
        raw_y = np.asarray(target)
        if len(y) != len(x) or not np.all((raw_y == 0) | (raw_y == 1)):
            raise ValueError(f"labels for video index {index} must be binary [T] matching values")
        videos.append(x)
        targets.append(y)
        rates.append(_fps(rate))
    return videos, targets, rates


def _time_grid(frame_count: int, fps: float) -> np.ndarray:
    """A true 10 Hz grid, with the exact source endpoint (never a stretched grid)."""
    frame_count = _integer(frame_count, "frame_count")
    fps = _fps(fps)
    duration = (frame_count - 1) / fps
    if not np.isfinite(duration) or not np.isfinite(duration * TARGET_HZ):
        raise ValueError("frame count / fps produces a nonfinite duration")
    grid = np.arange(int(np.floor(duration * TARGET_HZ)) + 1, dtype=np.float64) / TARGET_HZ
    tolerance = 8 * np.finfo(np.float64).eps * max(1.0, duration)
    if duration - grid[-1] > tolerance:
        grid = np.append(grid, duration)
    else:
        grid[-1] = duration
    return grid


def _resample_features(values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    grid = _time_grid(len(values), fps)
    frame_axis = np.clip(grid * fps, 0.0, float(len(values) - 1))
    frame_axis[-1] = len(values) - 1  # Exact inclusion even for fractional FPS.
    left = np.floor(frame_axis).astype(np.int64)
    right = np.minimum(left + 1, len(values) - 1)
    fraction = (frame_axis - left)[:, None]
    # Float64 interpolation avoids overflow for opposite-sign finite float32s.
    sampled = values[left].astype(np.float64) * (1.0 - fraction)
    sampled += values[right].astype(np.float64) * fraction
    return sampled.astype(np.float32), grid


def _resample_labels(labels: np.ndarray, grid: np.ndarray, fps: float) -> np.ndarray:
    indices = np.floor(grid * fps + 0.5)
    indices = np.clip(indices, 0, len(labels) - 1).astype(np.int64)
    indices[-1] = len(labels) - 1
    return labels[indices].astype(np.float32, copy=True)


def _fit_normalization(videos: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Population moments of ONLY the supplied, resampled training frames."""
    count = sum(len(video) for video in videos)
    total = sum((video.sum(axis=0, dtype=np.float64) for video in videos),
                np.zeros(videos[0].shape[1], dtype=np.float64))
    mean64 = total / count
    squares = sum((((video.astype(np.float64) - mean64) ** 2).sum(axis=0)
                   for video in videos), np.zeros_like(mean64))
    std64 = np.sqrt(squares / count)
    # A constant/tiny-variance feature is not amplified into artificial evidence.
    std64 = np.where(std64 < STD_FLOOR, 1.0, std64)
    return mean64.astype(np.float32), std64.astype(np.float32)


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = ((values.astype(np.float64) - mean) / std).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("normalization produced nonfinite float32 values")
    return result


def _pad_batch(videos: list[np.ndarray], targets: list[np.ndarray], indices: Any
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = max(len(videos[int(index)]) for index in indices)
    x = np.zeros((len(indices), videos[0].shape[1], length), dtype=np.float32)
    y = np.zeros((len(indices), length), dtype=np.float32)
    mask = np.zeros_like(y, dtype=bool)
    for row, index in enumerate(indices):
        index = int(index)
        n = len(videos[index])
        x[row, :, :n] = videos[index].T
        y[row, :n] = targets[index]
        mask[row, :n] = True
    return x, y, mask


def _build_torch_model(torch: Any, feature_dim: int) -> Any:
    """Lazy CPU-only construction; no Torch symbol is needed by deployment."""
    class TemporalHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimensions = (feature_dim, CHANNELS, CHANNELS, CHANNELS)
            self.convs = torch.nn.ModuleList([
                torch.nn.Conv1d(dimensions[i], CHANNELS, 3, padding=dilation,
                                dilation=dilation, device="cpu", dtype=torch.float32)
                for i, dilation in enumerate(DILATIONS)
            ] + [torch.nn.Conv1d(CHANNELS, 1, 1, device="cpu", dtype=torch.float32)])
            self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p) for p in DROPOUT])

        def forward(self, x: Any, mask: Any = None) -> Any:
            valid = None if mask is None else mask[:, None, :].to(dtype=x.dtype)
            if valid is not None:
                x = x * valid
            for conv, dropout in zip(self.convs[:-1], self.dropouts):
                x = dropout(torch.relu(conv(x)))
                # Mask every hidden layer: a loss mask alone does not prevent
                # padded activations/bias from leaking back into valid frames.
                if valid is not None:
                    x = x * valid
            logits = self.convs[-1](x).squeeze(1)
            return logits if mask is None else logits * mask.to(dtype=logits.dtype)

    return TemporalHead()


def _masked_video_loss(torch: Any, logits: Any, targets: Any, mask: Any) -> Any:
    positive_weight = torch.tensor(POSITIVE_WEIGHT, device="cpu", dtype=torch.float32)
    frame_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight, reduction="none")
    masked = torch.where(mask, frame_loss, torch.zeros_like(frame_loss))
    per_video = masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return per_video.mean()  # Each sampled video, not each padded frame, has equal weight.


def train_temporal(values: list[np.ndarray], labels: list[np.ndarray], fps: list[float],
                   seed: int, *, steps: int = 400) -> dict[str, Any]:
    """Fit on these training videos only; export the FINAL fixed-budget CPU model.

    All features must be finite [T,D] with common D, labels binary [T], and FPS
    finite/positive. steps >= 1; 0 <= seed < 2**63. Constant features, no-positive
    labels and all-positive labels are valid; absent/empty training data is not.
    Uniform video sampling + masked per-video means avoids long-video dominance
    in the loss. Mean/std are frame-weighted over resampled training videos.

    No validation arguments or early stopping exist. Torch's CPU RNG, thread
    count and deterministic-algorithm flags are restored on return. These are
    process-global settings: parallel fold callers should use separate processes.
    """
    seed = _integer(seed, "seed", minimum=0)
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    steps = _integer(steps, "steps")
    videos, targets, rates = _training_inputs(values, labels, fps)
    sampled, sampled_targets = [], []
    for video, target, rate in zip(videos, targets, rates):
        features, grid = _resample_features(video, rate)
        sampled.append(features)
        sampled_targets.append(_resample_labels(target, grid, rate))
    mean, std = _fit_normalization(sampled)
    normalized = [_normalize(video, mean, std) for video in sampled]
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("train_temporal requires CPU PyTorch; use the existing WSL "
                           "Python environment. predict_temporal is NumPy-only.") from exc

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    batch_size = min(BATCH_SIZE, len(videos))
    rng = np.random.default_rng(seed)
    try:
        torch.set_num_threads(CPU_THREADS)
        torch.use_deterministic_algorithms(True)
        # Explicitly save/seed ONLY the CPU generator: no CUDA API is called.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = _build_torch_model(torch, videos[0].shape[1])
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)
            for step in range(steps):
                indices = rng.choice(len(videos), size=batch_size, replace=False)
                x, y, mask = _pad_batch(normalized, sampled_targets, indices)
                tx, ty, tm = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)
                optimizer.zero_grad(set_to_none=True)
                loss = _masked_video_loss(torch, model(tx, tm), ty, tm)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite training loss at fixed step {step + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP,
                                               error_if_nonfinite=True)
                optimizer.step()
            model.eval()
            convs = []
            for index, conv in enumerate(model.convs):
                weight = conv.weight.detach().numpy().copy()
                bias = conv.bias.detach().numpy().copy()
                if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                    raise FloatingPointError("nonfinite exported convolution parameters")
                dilation = DILATIONS[index] if index < len(DILATIONS) else 1
                convs.append({"weight": weight.tolist(), "bias": bias.tolist(),
                              "dilation": dilation, "padding": dilation if index < 3 else 0,
                              "activation": "relu" if index < 3 else "sigmoid"})
            final_loss = float(loss.detach())
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.set_num_threads(old_threads)

    return {
        "schema": SCHEMA,
        "feature_dim": int(videos[0].shape[1]),
        "mean": mean.tolist(), "std": std.tolist(), "convs": convs,
        "architecture": {"channels": CHANNELS, "kernel_size": 3,
                         "dilations": list(DILATIONS), "dropout": list(DROPOUT),
                         "output_kernel_size": 1, "receptive_field_samples": 15,
                         "padding": "zero_same", "mask_hidden_padding": True},
        "normalization": {"fit": "training_videos_only", "population_std": True,
                          "statistics_axis": "all_resampled_training_frames",
                          "std_floor": STD_FLOOR, "small_std_scale": 1.0},
        "fps_profile": {"target_hz": TARGET_HZ, "grid": GRID_RULE,
                        "features": "linear", "labels": "nearest_frame_half_up",
                        "output": "linear_to_original_frame_axis",
                        "source_fps": rates,
                        "source_frame_counts": [len(video) for video in videos],
                        "resampled_frame_counts": [len(video) for video in sampled]},
        "training": {"seed": seed, "steps": steps, "optimizer": "AdamW",
                     "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                     "gradient_clip_norm": GRADIENT_CLIP, "cpu_threads": CPU_THREADS,
                     "batch_size": batch_size, "video_count": len(videos),
                     "sampling": "uniform_videos_without_replacement_per_step",
                     "loss": "equal_video_masked_bce_with_logits",
                     "frame_positive_weight": POSITIVE_WEIGHT,
                     "temporal_variation_weight": 0.0,
                     "selection": "none_final_fixed_step", "final_loss": final_loss,
                     "device": "cpu", "torch_version": str(torch.__version__)},
    }


def _bundle_parameters(bundle: Any) -> tuple[int, np.ndarray, np.ndarray, list]:
    if not isinstance(bundle, Mapping) or bundle.get("schema") != SCHEMA:
        raise ValueError(f"bundle schema must be {SCHEMA}")
    dimension = _integer(bundle.get("feature_dim"), "bundle feature_dim")
    mean = _array(bundle.get("mean"), "bundle mean", 1)
    std = _array(bundle.get("std"), "bundle std", 1)
    if mean.shape != (dimension,) or std.shape != (dimension,) or np.any(std <= 0):
        raise ValueError("bundle mean/std must be [D] with finite positive std")
    profile = bundle.get("fps_profile")
    if not isinstance(profile, Mapping) or profile.get("grid") != GRID_RULE:
        raise ValueError("bundle fps_profile must use the supported endpoint-inclusive grid")
    if _fps(profile.get("target_hz")) != TARGET_HZ or profile.get("features") != "linear":
        raise ValueError("bundle fps_profile must specify linear features at 10 Hz")
    layers = bundle.get("convs")
    if not isinstance(layers, (list, tuple)) or len(layers) != 4:
        raise ValueError("bundle must contain three hidden convolutions and one output convolution")
    parsed = []
    in_channels = dimension
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError("each bundle convolution must be a mapping")
        hidden = index < len(DILATIONS)
        out_channels, kernel = (CHANNELS, 3) if hidden else (1, 1)
        dilation = DILATIONS[index] if hidden else 1
        padding = dilation if hidden else 0
        activation = "relu" if hidden else "sigmoid"
        if (_integer(layer.get("dilation"), "convolution dilation") != dilation
                or _integer(layer.get("padding"), "convolution padding", minimum=0) != padding
                or layer.get("activation") != activation):
            raise ValueError("bundle convolution dilation/padding/activation is incompatible")
        weight = _array(layer.get("weight"), "convolution weight", 3)
        bias = _array(layer.get("bias"), "convolution bias", 1)
        if weight.shape != (out_channels, in_channels, kernel) or bias.shape != (out_channels,):
            raise ValueError("bundle convolution weight/bias shape is incompatible")
        parsed.append((weight, bias, dilation))
        in_channels = out_channels
    return dimension, mean, std, parsed


def _conv1d_same(values: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                 dilation: int) -> np.ndarray:
    """Torch-style cross-correlation [out,in,k], explicit zero SAME padding."""
    kernel = weight.shape[2]
    padding = dilation * (kernel - 1) // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="constant")
    output = np.zeros((len(values), weight.shape[0]), dtype=np.float32)
    for tap in range(kernel):
        offset = tap * dilation
        output += padded[offset:offset + len(values)] @ weight[:, :, tap].T
    output += bias
    return output


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    exponential = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1.0 / (1.0 + exponential),
                    exponential / (1.0 + exponential)).astype(np.float32)


def predict_temporal(bundle: Mapping[str, Any], values: np.ndarray, fps: float) -> np.ndarray:
    """NumPy-only inference, returning finite float32 probabilities on source frames.

    Empty [0,D] inference returns an empty float32 array after schema/FPS checks.
    Dropout is absent at deployment; there is no refit or test-time normalization.
    """
    dimension, mean, std, convs = _bundle_parameters(bundle)
    video = _features(values, dimension=dimension, empty=True)
    rate = _fps(fps)
    if not len(video):
        return np.empty(0, dtype=np.float32)
    sampled, grid = _resample_features(video, rate)
    hidden = _normalize(sampled, mean, std)
    with np.errstate(over="raise", invalid="raise"):
        for index, (weight, bias, dilation) in enumerate(convs):
            hidden = _conv1d_same(hidden, weight, bias, dilation)
            if index < len(convs) - 1:
                np.maximum(hidden, 0.0, out=hidden)
        probabilities = _sigmoid(hidden[:, 0])
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("nonfinite temporal inference probabilities")
    original_axis = np.arange(len(video), dtype=np.float64) / rate
    return np.interp(original_axis, grid, probabilities).astype(np.float32)
