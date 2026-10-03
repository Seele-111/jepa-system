"""Bounded interval-completeness / IoU learning on frozen frame probabilities.

Inference needs only NumPy. No baseline classifier, q75 verifier, video gate,
identity, labels, dataset access, threshold selection, or persistence lives here.
The caller supplies label-free, already-transformed frame evidence, fits using
training contents only, and selects one of THRESHOLDS using strict inner OOF.
Targets/weights and the lazy sklearn fit are training-only APIs.

All intervals use inclusive endpoints. interval_features returns (features,
names), with float32 [N, 41 + 9*D] features and stable named columns.
Empty contexts have mean zero and explicit valid fractions. Equal thirds use
np.array_split; empty thirds fall back to the inner mean, with zero support.
Short-window support is observed length / requested FPS-relative window length.
Export returns JSON-safe data, not a file; portable inference validates it even
for empty input. No function silently drops candidates to meet the fixed cap.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

MAX_CANDIDATES = 4096
THRESHOLDS = (0.25, 0.35, 0.45, 0.55)
QUALITY_HEAD_KIND = "interval-quality-et-v1"
_MAX_TREES = 192
_MAX_DEPTH = 8
_MAX_TREE_NODES = 2 ** (_MAX_DEPTH + 1) - 1

_P_NAMES = ("p_mean", "p_std", "p_min", "p_max", "p_q25", "p_q75")
_THIRD_NAMES = tuple(name for i in range(3) for name in
                     (f"p_third_{i}_mean", f"third_{i}_validfraction"))
_CONTEXT_NAMES = tuple(f"{name}_{seconds}s" for seconds in ("0.2", "0.5")
                       for name in ("p_before_mean", "before_validfraction",
                                    "p_after_mean", "after_validfraction",
                                    "p_start_inner_mean", "start_inner_validfraction",
                                    "p_end_inner_mean", "end_inner_validfraction",
                                    "p_start_contrast", "p_end_contrast"))
_GLOBAL_NAMES = ("p_start_slope_0.12s_per_second", "start_slope_validfraction",
                 "p_end_slope_0.12s_per_second", "end_slope_validfraction",
                 "duration_seconds", "duration_video_fraction",
                 "whole_video_p_mean", "whole_video_p_std", "video_probability")
_EVIDENCE_STATS = ("mean", "std", "before_0.2s_mean", "after_0.2s_mean",
                   "inner_minus_before_0.2s", "inner_minus_after_0.2s",
                   "third_0_mean", "third_1_mean", "third_2_mean")

__all__ = ["MAX_CANDIDATES", "THRESHOLDS", "QUALITY_HEAD_KIND", "proposal_bank",
           "interval_feature_names", "interval_features", "quality_targets",
           "quality_weights", "fit_quality_head", "export_quality_head",
           "portable_predict_quality", "quality_configurations", "select_intervals"]


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is out of bounds")
    return result


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _fps(value: Any) -> float:
    result = _real(value, "fps")
    if result <= 0:
        raise ValueError("fps must be positive; no guessed FPS")
    return result


def _numeric(value: Any, name: str, ndim: int, *, allow_bool: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    kinds = "biuf" if allow_bool else "iuf"
    if result.ndim != ndim or result.dtype.kind not in kinds:
        raise ValueError(f"{name} must be a real numeric {ndim}D array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if not allow_bool and isinstance(value, (list, tuple)):
        if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
            raise ValueError(f"{name} must not contain booleans")
    return result


def _float32(value: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = value.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must remain finite in float32")
    return result


def _unit_vector(value: Any, name: str) -> np.ndarray:
    result = _numeric(value, name, 1).astype(np.float64)
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _matrix(value: Any, name: str, *, allow_zero_columns: bool = False) -> np.ndarray:
    result = _numeric(value, name, 2)
    if not allow_zero_columns and result.shape[1] == 0:
        raise ValueError(f"{name} needs at least one feature")
    return _float32(result, name)


def _intervals(value: Any, length: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("intervals must be integer [N,2] pairs") from exc
    if result.shape == (0,) and result.dtype.kind in "iuf":
        return np.empty((0, 2), dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 2 or result.dtype.kind not in "iu":
        raise ValueError("intervals must be integer [N,2] pairs")
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
            raise ValueError("intervals must not contain boolean endpoints")
    if len(result) > MAX_CANDIDATES:
        raise ValueError(f"unsupported candidate count > {MAX_CANDIDATES}")
    if np.any(result < 0) or np.any(result > np.iinfo(np.int64).max):
        raise ValueError("interval endpoints are out of bounds")
    result = result.astype(np.int64, copy=False)
    if np.any(result[:, 0] > result[:, 1]):
        raise ValueError("interval starts must not exceed inclusive ends")
    if length is not None and np.any(result[:, 1] >= length):
        raise ValueError("intervals exceed the frame count")
    return result


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    (np.flatnonzero(edges == -1) - 1).tolist()))


def _width(seconds: float, fps: float) -> int:
    return max(1, round(seconds * fps))


def proposal_bank(probabilities: Any, fps: float) -> list[tuple[int, int]]:
    """Label-free union of grid spans, full sliding short windows and moved runs.

    The grid includes both endpoints before applying the 64-point cap. Capped
    grids use np.linspace(..., dtype=int64). Short windows start at every frame
    where their full requested length fits. Each run endpoint moves independently
    by -round(.12*fps), 0, +round(.12*fps), is clipped, then checked for s <= e.
    Every source uses the same inclusive span mean >= .20 and max >= .35 rule.
    """
    p = _unit_vector(probabilities, "probabilities")
    fps = _fps(fps)
    total = len(p)
    if total == 0 or p.max() < 0.35:
        return []
    grid = sorted(set(range(0, total, _width(0.12, fps))) | {0, total - 1})
    if len(grid) > 64:
        grid = np.linspace(0, total - 1, 64, dtype=np.int64).tolist()
    candidates: set[tuple[int, int]] = set()

    def add(start: int, end: int) -> None:
        if start > end or (start, end) in candidates:
            return
        span = p[start:end + 1]
        if span.mean() >= 0.20 and span.max() >= 0.35:
            candidates.add((start, end))
            if len(candidates) > MAX_CANDIDATES:
                raise ValueError(f"unsupported candidate count > {MAX_CANDIDATES}; "
                                 "no subset selection is permitted")

    for i, start in enumerate(grid):
        for end in grid[i:]:
            add(start, end)
    for seconds in (0.04, 0.12, 0.24):
        width = _width(seconds, fps)
        for start in range(max(0, total - width + 1)):
            add(start, start + width - 1)
    shift = round(0.12 * fps)
    for cutoff in (0.25, 0.40, 0.55, 0.70):
        for start, end in _runs(p >= cutoff):
            for ds in (-shift, 0, shift):
                for de in (-shift, 0, shift):
                    add(min(total - 1, max(0, start + ds)),
                        min(total - 1, max(0, end + de)))
    return sorted(candidates)


def interval_feature_names(evidence_dim: int) -> tuple[str, ...]:
    """Stable stat-major names; zero evidence columns are also supported."""
    dimension = _integer(evidence_dim, "evidence_dim")
    return (_P_NAMES + _THIRD_NAMES + _CONTEXT_NAMES + _GLOBAL_NAMES +
            tuple(f"evidence_{j}/{stat}" for stat in _EVIDENCE_STATS
                  for j in range(dimension)))


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)), axis=0)


def _prefix_mean(prefix: np.ndarray, offset: Any, starts: np.ndarray,
                 stops: np.ndarray) -> np.ndarray:
    counts = stops - starts
    sums = prefix[stops] - prefix[starts]
    divisor = counts if sums.ndim == 1 else counts[:, None]
    np.divide(sums, divisor, out=sums, where=divisor > 0)
    sums += offset
    sums[counts == 0] = 0.0
    return sums


def _prefix_std(prefix: np.ndarray, squared: np.ndarray, starts: np.ndarray,
                stops: np.ndarray) -> np.ndarray:
    # All these windows are nonempty. Centering at the video mean improves
    # variance accuracy without accepting negative variances from roundoff.
    divisor = stops - starts
    if prefix.ndim == 2:
        divisor = divisor[:, None]
    centered_mean = (prefix[stops] - prefix[starts]) / divisor
    variance = (squared[stops] - squared[starts]) / divisor - centered_mean ** 2
    np.maximum(variance, 0.0, out=variance)
    return np.sqrt(variance, out=variance)


def _slopes(prefix: np.ndarray, indexed: np.ndarray, endpoints: np.ndarray,
            radius: int, total: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    clipped_radius = min(radius, total)
    starts = np.maximum(0, endpoints - clipped_radius)
    stops = np.minimum(total, endpoints + clipped_radius + 1)
    counts = (stops - starts).astype(np.float64)
    midpoint = starts + (counts - 1) / 2
    numerator = indexed[stops] - indexed[starts] - midpoint * (prefix[stops] - prefix[starts])
    denominator = counts * (counts ** 2 - 1) / 12
    slopes = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=counts > 1)
    with np.errstate(over="ignore", invalid="ignore"):
        slopes *= fps
    return slopes, counts / float(2 * radius + 1)


def interval_features(probabilities: Any, video_probability: float, fps: float,
                      intervals: Any, frame_evidence: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return (float32 [N,41+9D] features, stable names), without identity/labels.

    Contexts are strictly outside each candidate. Contrasts subtract that
    context mean from the corresponding inner endpoint-window mean. Evidence
    differences subtract .2s context means from the full candidate mean.
    Evidence shares p context supports and third supports (no missing values
    are accepted). Third support is length / max(1, ceil(candidate_length/3)).
    Local start/end slopes use centered +/- max(1, round(.12*fps)) windows and
    OLS units per second; a singleton slope is zero with its observed support.
    Moments and context means are batched with centered float64 prefix sums;
    only scalar p min/max/quantiles inspect individual spans. No normalized
    absolute position is used. Empty input is still fully validated.
    """
    p = _unit_vector(probabilities, "probabilities")
    vp = _real(video_probability, "video_probability")
    if not 0 <= vp <= 1:
        raise ValueError("video_probability must be in [0, 1]")
    fps = _fps(fps)
    evidence = _matrix(frame_evidence, "frame_evidence", allow_zero_columns=True)
    if evidence.shape[0] != len(p):
        raise ValueError("frame_evidence must have the same T as probabilities")
    pairs = _intervals(intervals, len(p))
    names = interval_feature_names(evidence.shape[1])
    result = np.empty((len(pairs), len(names)), dtype=np.float32)
    if not len(pairs):
        return result, names
    evidence = evidence.astype(np.float64)
    whole_mean, whole_std = float(p.mean()), float(p.std())
    p_centered = p - whole_mean
    p_prefix, p_squared = _prefix(p_centered), _prefix(p_centered ** 2)
    p_indexed = _prefix(p_centered * np.arange(len(p), dtype=np.float64))
    evidence_offset = evidence.mean(axis=0)
    evidence_centered = evidence - evidence_offset
    evidence_prefix = _prefix(evidence_centered)
    evidence_squared = _prefix(evidence_centered ** 2)
    starts, stops = pairs[:, 0], pairs[:, 1] + 1
    lengths = stops - starts
    inner_p_mean = _prefix_mean(p_prefix, whole_mean, starts, stops)
    column = 0

    def put(values: np.ndarray) -> None:
        nonlocal column
        block = values[:, None] if values.ndim == 1 else values
        width = block.shape[1]
        result[:, column:column + width] = _float32(block, "interval features")
        column += width

    put(inner_p_mean)
    put(_prefix_std(p_prefix, p_squared, starts, stops))
    scalar_stats = np.empty((len(pairs), 4), dtype=np.float64)
    for row, (start, stop) in enumerate(zip(starts, stops)):
        span = p[start:stop]
        scalar_stats[row] = (span.min(), span.max(), *np.quantile(span, [0.25, 0.75]))
    for j in range(4):
        put(scalar_stats[:, j])
    thirds: list[tuple[np.ndarray, np.ndarray]] = []
    third_start = starts.copy()
    third_denominator = (lengths + 2) // 3
    for j in range(3):
        third_lengths = lengths // 3 + (j < lengths % 3)
        third_stop = third_start + third_lengths
        thirds.append((third_start, third_stop))
        means = _prefix_mean(p_prefix, whole_mean, third_start, third_stop)
        means[third_lengths == 0] = inner_p_mean[third_lengths == 0]
        put(means)
        put(third_lengths / third_denominator)
        third_start = third_stop
    evidence_contexts: list[tuple[np.ndarray, np.ndarray]] = []
    for seconds in (0.2, 0.5):
        width = _width(seconds, fps)
        clipped_width = min(width, len(p))
        before_start, after_stop = np.maximum(0, starts - clipped_width), np.minimum(len(p), stops + clipped_width)
        before = _prefix_mean(p_prefix, whole_mean, before_start, starts)
        after = _prefix_mean(p_prefix, whole_mean, stops, after_stop)
        start_stop, end_start = np.minimum(stops, starts + clipped_width), np.maximum(starts, stops - clipped_width)
        start_mean = _prefix_mean(p_prefix, whole_mean, starts, start_stop)
        end_mean = _prefix_mean(p_prefix, whole_mean, end_start, stops)
        put(before)
        put((starts - before_start) / float(width))
        put(after)
        put((after_stop - stops) / float(width))
        put(start_mean)
        put((start_stop - starts) / float(width))
        put(end_mean)
        put((stops - end_start) / float(width))
        put(start_mean - before)
        put(end_mean - after)
        if seconds == 0.2:
            evidence_contexts = [(before_start, starts), (stops, after_stop)]
    radius = _width(0.12, fps)
    for endpoints in (starts, stops - 1):
        slope, support = _slopes(p_prefix, p_indexed, endpoints, radius, len(p), fps)
        put(slope)
        put(support)
    with np.errstate(over="ignore", invalid="ignore"):
        put(lengths.astype(np.float64) / fps)
    put(lengths / len(p))
    for value in (whole_mean, whole_std, vp):
        put(np.full(len(pairs), value, dtype=np.float64))
    inner_mean = _prefix_mean(evidence_prefix, evidence_offset, starts, stops)
    put(inner_mean)
    put(_prefix_std(evidence_prefix, evidence_squared, starts, stops))
    context_means = [_prefix_mean(evidence_prefix, evidence_offset, a, b) for a, b in evidence_contexts]
    for context in context_means:
        put(context)
    for context in context_means:
        put(inner_mean - context)
    for a, b in thirds:
        means = _prefix_mean(evidence_prefix, evidence_offset, a, b)
        means[a == b] = inner_mean[a == b]
        put(means)
    if column != len(names):
        raise ValueError("interval feature names and columns differ")
    return result, names


def quality_targets(intervals: Any, labels: Any) -> np.ndarray:
    """Training-only maximum IoU with contiguous positive GT; endpoints inclusive."""
    truth = _numeric(labels, "labels", 1, allow_bool=True)
    if np.any((truth != 0) & (truth != 1)):
        raise ValueError("labels must be binary")
    pairs = _intervals(intervals, len(truth))
    targets = np.zeros(len(pairs), dtype=np.float32)
    events = _runs(truth == 1)
    for i, (start_raw, end_raw) in enumerate(pairs):
        start, end = int(start_raw), int(end_raw)
        best = 0.0
        for gs, ge in events:
            intersection = max(0, min(end, ge) - max(start, gs) + 1)
            union = end - start + 1 + ge - gs + 1 - intersection
            best = max(best, intersection / union)
        targets[i] = best
    return targets


def quality_weights(targets: Any, is_normal: bool, content_weight: float) -> np.ndarray:
    """Each nonempty positive-content IoU stratum gets one content mass.

    Normal contents get 2*content_weight total, regardless of candidate count.
    Non-normal strata are [0,.1), [.1,.5), [.5,1]; empty strata get no mass.
    """
    target = _unit_vector(targets, "targets")
    if not isinstance(is_normal, (bool, np.bool_)):
        raise ValueError("is_normal must be boolean")
    mass = _real(content_weight, "content_weight")
    if mass < 0:
        raise ValueError("content_weight must be nonnegative")
    if is_normal and np.any(target != 0):
        raise ValueError("normal content must have zero IoU targets")
    weights = np.zeros(len(target), dtype=np.float64)
    if len(target):
        if is_normal:
            weights[:] = mass * (2.0 / len(target))
        else:
            for mask in (target < 0.1, (target >= 0.1) & (target < 0.5), target >= 0.5):
                count = int(mask.sum())
                if count:
                    weights[mask] = mass / count
    result = _float32(weights, "quality weights")
    if np.any((weights > 0) & (result == 0)):
        raise ValueError("content_weight is too small for positive float32 weights")
    return result


def fit_quality_head(feature_rows: Any, target_rows: Any, weight_rows: Any, seed: int) -> Any:
    """One fixed ET regressor; aligned [N,F], [N], [N] training rows only.

    Weights are normalized to mean one without overflowing their sum. No grid,
    split construction, dataset read, classifier retraining, or disk write occurs.
    """
    features = _matrix(feature_rows, "feature_rows")
    targets = _unit_vector(target_rows, "target_rows")
    weights = _numeric(weight_rows, "weight_rows", 1).astype(np.float64)
    seed = _integer(seed, "seed", maximum=2 ** 32 - 1)
    if not len(features) or len(targets) != len(features) or len(weights) != len(features):
        raise ValueError("training rows must be nonempty and aligned")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("training weights must be nonnegative with positive total mass")
    scaled = weights / weights.max()
    normalized = scaled / scaled.mean()
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(n_estimators=192, max_depth=8, min_samples_leaf=12,
                                max_features=0.5, n_jobs=2, random_state=seed)
    model.fit(features, targets, sample_weight=normalized)
    return model


def _int_nodes(value: Any, name: str) -> np.ndarray:
    result = _numeric(value, name, 1)
    if result.dtype.kind not in "iu" or np.any(result > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain int64-compatible integers")
    return result.astype(np.int64, copy=False)


def _validated_head(state: Any) -> tuple[int, list[dict[str, np.ndarray]]]:
    if not isinstance(state, Mapping) or set(state) != {"kind", "n_features", "trees"}:
        raise ValueError("unknown quality head schema")
    if state["kind"] != QUALITY_HEAD_KIND:
        raise ValueError("unknown quality head kind")
    dimension = _integer(state["n_features"], "n_features", minimum=1)
    raw_trees = state["trees"]
    if not isinstance(raw_trees, (list, tuple)) or not 1 <= len(raw_trees) <= _MAX_TREES:
        raise ValueError("quality head requires 1..192 trees")
    trees = []
    for raw in raw_trees:
        if not isinstance(raw, Mapping) or set(raw) != {"left", "right", "feature", "threshold", "value"}:
            raise ValueError("unknown quality tree schema")
        tree = {key: _int_nodes(raw[key], key) for key in ("left", "right", "feature")}
        tree.update({key: _numeric(raw[key], key, 1).astype(np.float64)
                     for key in ("threshold", "value")})
        count = len(tree["left"])
        if not 1 <= count <= _MAX_TREE_NODES or any(len(a) != count for a in tree.values()):
            raise ValueError("quality tree arrays have unsupported node counts")
        left, right, feature = tree["left"], tree["right"], tree["feature"]
        leaf = (left == -1) & (right == -1)
        if (np.any(feature[leaf] != -2) or np.any(tree["threshold"][leaf] != -2) or
                np.any(left[~leaf] < 0) or np.any(right[~leaf] < 0) or
                np.any(left[~leaf] >= count) or np.any(right[~leaf] >= count) or
                np.any(left[~leaf] == right[~leaf]) or np.any(feature[~leaf] < 0) or
                np.any(feature[~leaf] >= dimension) or
                np.any((tree["value"] < 0) | (tree["value"] > 1))):
            raise ValueError("invalid quality tree nodes")
        seen: set[int] = set()
        pending = [(0, 0)]
        while pending:
            node, depth = pending.pop()
            if node in seen or depth > _MAX_DEPTH:
                raise ValueError("cyclic/shared or over-depth quality tree")
            seen.add(node)
            if not leaf[node]:
                pending.extend(((int(left[node]), depth + 1), (int(right[node]), depth + 1)))
        if len(seen) != count:
            raise ValueError("quality tree has unreachable nodes")
        tree["leaf"] = leaf
        trees.append(tree)
    return dimension, trees


def export_quality_head(model: Any) -> dict:
    """Export a fitted, single-output regression forest to JSON-safe tree arrays."""
    dimension = _integer(getattr(model, "n_features_in_", None), "n_features", minimum=1)
    _integer(getattr(model, "n_outputs_", None), "n_outputs", minimum=1, maximum=1)
    estimators = getattr(model, "estimators_", None)
    if hasattr(model, "classes_") or not isinstance(estimators, (list, tuple)) or not 1 <= len(estimators) <= _MAX_TREES:
        raise ValueError("expected a fitted single-output quality regressor")
    trees = []
    for estimator in estimators:
        tree = getattr(estimator, "tree_", None)
        if tree is None or getattr(tree, "n_features", None) != dimension:
            raise ValueError("regression tree feature dimension differs")
        value = _numeric(tree.value, "tree values", 3)
        if value.shape[1:] != (1, 1):
            raise ValueError("only single-output regression trees are supported")
        trees.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                      "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                      "value": value[:, 0, 0].tolist()})
    state = {"kind": QUALITY_HEAD_KIND, "n_features": dimension, "trees": trees}
    _validated_head(state)
    return state


def portable_predict_quality(model: Any, feature_rows: Any) -> np.ndarray:
    """NumPy-only forest mean, using sklearn's float32 input and <= split rule."""
    dimension, trees = _validated_head(model)
    features = _matrix(feature_rows, "feature_rows")
    if features.shape[1] != dimension:
        raise ValueError("quality feature dimension differs")
    total = np.zeros(len(features), dtype=np.float64)
    for tree in trees:
        node = np.zeros(len(features), dtype=np.int64)
        for _ in range(_MAX_DEPTH + 1):
            active = np.flatnonzero(~tree["leaf"][node])
            if not len(active):
                break
            current = node[active]
            node[active] = np.where(features[active, tree["feature"][current]] <= tree["threshold"][current],
                                    tree["left"][current], tree["right"][current])
        total += tree["value"][node]
    return _float32(total / len(trees), "quality predictions")


def quality_configurations() -> tuple[dict[str, float], ...]:
    """Fixed shortlist only; selection is the caller's strict inner-OOF job."""
    return tuple({"threshold": threshold} for threshold in THRESHOLDS)


def select_intervals(intervals: Any, scores: Any, threshold: float) -> list[tuple[int, int]]:
    """Greedy descending quality, deterministic ties, rejecting inclusive overlap.

    There is no maximum event count and no whole-video veto. The primitive
    accepts finite [0,1] thresholds; the experiment shortlist is THRESHOLDS only.
    """
    pairs = _intervals(intervals)
    quality = _unit_vector(scores, "scores")
    threshold = _real(threshold, "threshold")
    if not 0 <= threshold <= 1 or len(quality) != len(pairs):
        raise ValueError("invalid threshold or unaligned scores")
    order = sorted(range(len(pairs)), key=lambda i: (-quality[i], int(pairs[i, 0]), int(pairs[i, 1])))
    accepted: list[tuple[int, int]] = []
    for i in order:
        if quality[i] < threshold:
            break
        start, end = map(int, pairs[i])
        if not any(start <= old_end and old_start <= end for old_start, old_end in accepted):
            accepted.append((start, end))
    return sorted(accepted)
