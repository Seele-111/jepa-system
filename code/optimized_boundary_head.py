#!/usr/bin/env python3
"""Offline, non-causal boundary refinement; portable inference needs only NumPy.

Features are the caller's unchanged [T,D] rows: no labels, FPS channels, video
identities, calibration, candidate generation or OOF selection are added here.
The caller must fit on training videos only and select boundary_seconds from
0/.2/.4/.6 using inner OOF, never held-out labels. Boundary probabilities are
uncalibrated. Refinement reads future frames and is NOT a causal deployment.

Bundle fields are exactly start_model/end_model, each using optimized_locator's
export_model schema. Empty training exports probability-0 constants; n_features
is 0 only when no video supplied a dimension (a dimension-free constant).
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any

import numpy as np

from optimized_locator import export_model, portable_predict, spans

TOLERANCE_SECONDS = 0.06
STAY_PENALTY = 0.05


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _fps(value: Any) -> float:
    result = _real(value, "fps")
    if result <= 0:
        raise ValueError("fps must be positive; no guessed FPS")
    return result


def _array(value: Any, name: str, ndim: int, *, allow_bool: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if result.ndim != ndim or result.dtype.kind not in ("biuf" if allow_bool else "iuf"):
        raise ValueError(f"{name} must be a real numeric {ndim}D array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _features(value: Any) -> np.ndarray:
    result = _array(value, "values [T,D]", 2)
    if result.shape[1] < 1:
        raise ValueError("values must have D >= 1 (T=0 is allowed)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("values must be finite after float32 conversion")
    return result


def _boundary_targets(labels: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive span endpoints, with clipped binary +/- floor(.06*fps) bands."""
    start = np.zeros(len(labels), dtype=np.uint8)
    end = np.zeros(len(labels), dtype=np.uint8)
    radius = int(np.floor(TOLERANCE_SECONDS * fps))
    for s, e in spans(labels):
        start[max(0, s - radius):min(len(labels), s + radius + 1)] = 1
        end[max(0, e - radius):min(len(labels), e + radius + 1)] = 1
    return start, end


def _balanced_weights(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Equal-video 1/T weights times inverse *weighted* binary class frequency."""
    mass = np.bincount(target, weights=base, minlength=2)
    return base * (base.sum() / (2.0 * mass[target]))


def _fit_head(values: np.ndarray, target: np.ndarray, base: np.ndarray, seed: int) -> dict:
    classes = np.unique(target)
    if len(classes) < 2:
        # export_model reads only these two attributes for a single-class model.
        constant = SimpleNamespace(n_features_in_=values.shape[1],
                                   classes_=classes if len(classes) else np.array([0]))
        return export_model(constant)
    # Training-only dependency. Importing this module or predicting never imports sklearn.
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(n_estimators=128, max_depth=7, min_samples_leaf=5,
                                 max_features=0.5, n_jobs=2, random_state=seed,
                                 class_weight=None)
    model.fit(values, target, sample_weight=_balanced_weights(target, base))
    return export_model(model)


def train_boundary_heads(values: list[np.ndarray], labels: list[np.ndarray],
                         fps: list[float], seed: int) -> dict:
    """Fit two fixed-budget CPU ET heads and return a JSON-serializable dict.

    Every nonempty training video has base weight 1; balancing then gives each
    present class half of the total weight, independently for each head. Empty
    videos contribute no frames/weight but still validate their dimension/FPS.
    Empty or single-class heads use export_model's constant representation.
    """
    seed = _integer(seed, "seed")
    if seed > 2**32 - 1:
        raise ValueError("seed must be <= 2**32 - 1")
    if not all(isinstance(v, (list, tuple)) for v in (values, labels, fps)):
        raise ValueError("values, labels and fps must be lists of videos")
    if not (len(values) == len(labels) == len(fps)):
        raise ValueError("values, labels and fps must have the same video count")
    videos, starts, ends, weights = [], [], [], []
    dimension = None
    for video, target, rate in zip(values, labels, fps):
        x = _features(video)
        if dimension is not None and x.shape[1] != dimension:
            raise ValueError("training feature dimension mismatch")
        dimension = x.shape[1]
        y = _array(target, "labels [T]", 1, allow_bool=True)
        if len(y) != len(x) or not np.all((y == 0) | (y == 1)):
            raise ValueError("labels must be binary [T] matching values")
        start, end = _boundary_targets(y, _fps(rate))
        videos.append(x)
        starts.append(start)
        ends.append(end)
        weights.append(np.full(len(x), 1.0 / len(x) if len(x) else 0.0))
    x = np.concatenate(videos) if videos else np.empty((0, 0), dtype=np.float32)
    start = np.concatenate(starts) if starts else np.empty(0, dtype=np.uint8)
    end = np.concatenate(ends) if ends else np.empty(0, dtype=np.uint8)
    base = np.concatenate(weights) if weights else np.empty(0, dtype=np.float64)
    return {"start_model": _fit_head(x, start, base, seed),
            "end_model": _fit_head(x, end, base, seed)}


def _probabilities(value: Any, name: str) -> np.ndarray:
    result = _array(value, name, 1)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must be in [0,1]")
    return result.astype(np.float64, copy=False)


def predict_boundary_heads(bundle: dict, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return uncalibrated float32 start/end [T] probabilities using NumPy only.

    JSON round trips are supported. A dimension-free empty-training constant
    accepts any D >= 1; every other model requires the training dimension.
    """
    x = _features(values)
    if not isinstance(bundle, Mapping):
        raise ValueError("bundle must contain start_model and end_model")
    predictions, dimensions = [], []
    for key in ("start_model", "end_model"):
        model = bundle.get(key)
        if not isinstance(model, Mapping) or model.get("kind") not in ("constant", "forest"):
            raise ValueError(f"invalid {key}: expected an exported constant or forest")
        dimension = _integer(model.get("n_features"), f"{key}.n_features")
        if dimension == 0 and model["kind"] != "constant":
            raise ValueError("only empty-training constants may be dimension-free")
        if model["kind"] == "constant":
            probability = _real(model.get("probability"), f"{key}.probability")
            if not 0 <= probability <= 1:
                raise ValueError("constant probability must be in [0,1]")
        inputs = x if dimension else np.empty((len(x), 0), dtype=np.float32)
        probability = portable_predict(model, inputs)
        _probabilities(probability, key)
        predictions.append(probability)
        dimensions.append(dimension)
    if dimensions[0] != dimensions[1]:
        raise ValueError("start/end model feature dimensions must match")
    return predictions[0], predictions[1]


def _intervals(value: Any, length: int) -> list[tuple[int, int]]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("intervals must be a sequence of inclusive (start,end) pairs")
    if isinstance(value, np.ndarray) and value.ndim != 2:
        raise ValueError("interval array must have shape [N,2]")
    result = []
    for pair in value:
        if (not isinstance(pair, (list, tuple, np.ndarray))
                or (isinstance(pair, np.ndarray) and pair.ndim != 1) or len(pair) != 2):
            raise ValueError("each interval must be an inclusive (start,end) pair")
        s, e = (_integer(v, "interval boundary") for v in pair)
        if not 0 <= s <= e < length:
            raise ValueError("interval must satisfy 0 <= start <= end < T")
        result.append((s, e))
    return result


def _peak(probabilities: np.ndarray, center: int, lower: int, upper: int, radius: int) -> int:
    indices = np.arange(lower, upper + 1)
    distance = np.abs(indices - center)
    scores = probabilities[indices] - STAY_PENALTY * distance / radius
    tied = np.flatnonzero(scores == scores.max())
    # Nearest-to-original on exact ties; lower index breaks equidistant ties.
    best = tied[np.argmin(distance[tied])]
    return int(indices[best])


def refine_intervals(intervals: list[tuple[int, int]], start_prob: np.ndarray,
                     end_prob: np.ndarray, fps: float, config: dict) -> list[tuple[int, int]]:
    """Refine existing inclusive candidates offline; never force/add an event.

    Only config['boundary_seconds'] is used (default 0). Zero, including a
    rounded-zero radius, is a strict no-op after input validation: original
    order and overlaps remain unchanged. Otherwise r=round(seconds*fps), and
    each endpoint maximizes p[t] - .05*abs(t-original)/r in its clipped window.
    Start <= original end; end >= updated start. End's window stays centered on
    the ORIGINAL end. Only overlapping results are merged, not adjacent ones
    or gaps; locator gap/minimum-duration settings are deliberately not used.
    """
    start = _probabilities(start_prob, "start_prob [T]")
    end = _probabilities(end_prob, "end_prob [T]")
    if len(start) != len(end):
        raise ValueError("start_prob and end_prob must have the same T")
    rate = _fps(fps)
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    seconds = _real(config.get("boundary_seconds", 0.0), "boundary_seconds")
    if seconds < 0:
        raise ValueError("boundary_seconds must be nonnegative")
    candidates = _intervals(intervals, len(start))
    if seconds == 0 or not candidates:
        return candidates
    window = seconds * rate
    if not np.isfinite(window):
        raise ValueError("boundary_seconds * fps must be finite")
    radius = int(round(window))
    if radius == 0:
        return candidates
    refined = []
    for s, e in candidates:
        new_s = _peak(start, s, max(0, s - radius), min(e, s + radius), radius)
        new_e = _peak(end, e, max(new_s, e - radius), min(len(end) - 1, e + radius), radius)
        refined.append((new_s, new_e))
    merged = []
    for s, e in sorted(refined):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
