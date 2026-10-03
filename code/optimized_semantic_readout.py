"""Frozen semantic-content readout; no extraction, runner, decoder or fusion.

Production semantic inputs are frozen last-layer V-JEPA2.1 spatial means
[T,1408], with paired-frame support/direct masks. Only supported TRAIN frames
enter equal-content covariance PCA. The original corrected_motion_rf base,
FPS augmentation, RF/ET losses, frame weighting and video classifier are reused.
Unsupported semantic projections are zero BEFORE augmentation; explicit raw
sem/direct, sem/support flags follow it. No labels or identities enter a view.

Inference and PCA need NumPy only. sklearn is imported lazily by the fitter.
Default PCA width is 16; no parameter grid or selection logic is provided.
Input channel width/order is bound to the serialized PCA (small synthetic
channel widths are also supported for tests). All persistence is caller-owned.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import numpy as np

from optimized_compact_features_v4 import fit_rgb_pca
from optimized_feature_view import feature_view
from optimized_locator import augment_signals, export_model, portable_predict, video_features

KIND = "semantic-content-readout-v1"
BASE_RECIPE = "corrected_motion_rf"
PCA_COMPONENTS = 16
SEMANTIC_DIM = 1408
FLAG_NAMES = ("sem/direct", "sem/support")
_PCA_ROLE = "training_content_only_equal_content_covariance"

__all__ = ["KIND", "BASE_RECIPE", "PCA_COMPONENTS", "SEMANTIC_DIM", "FLAG_NAMES",
           "fit_semantic_pca", "feature_view_semantic", "fit_semantic_member",
           "export_semantic_member", "predict_semantic_member"]


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is out of bounds")
    return result


def _numeric(value: Any, name: str, ndim: int, *, allow_bool: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if result.ndim != ndim or result.dtype.kind not in ("biuf" if allow_bool else "iuf"):
        raise ValueError(f"{name} must be a real numeric {ndim}D array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if not allow_bool and isinstance(value, (list, tuple)):
        if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
            raise ValueError(f"{name} must not contain booleans")
    return result


def _float32(value: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = value.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must remain finite in float32")
    return result


def _names(value: Any, width: int, name: str) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an ordered list of names")
    try:
        result = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an ordered list of names") from exc
    if (len(result) != width or any(not isinstance(v, str) or not v for v in result)
            or len(set(result)) != width):
        raise ValueError(f"{name} must have unique nonempty names matching its width")
    return result


def _indices(records: Any, indices: Any, name: str) -> list[int]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    try:
        result = [_integer(i, name, maximum=len(records) - 1) for i in indices]
    except TypeError as exc:
        raise ValueError(f"{name} must contain record indices") from exc
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and contain no duplicates")
    return result


def _field(record: Any, key: str) -> Any:
    if not isinstance(record, Mapping) or key not in record:
        raise ValueError(f"missing record field: {key}")
    return record[key]


def _sha(record: Mapping) -> str:
    value = _field(record, "sha256")
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("sha256 must be a nonempty content key")
    return value


def _mask(record: Mapping, key: str, length: int) -> np.ndarray:
    try:
        result = np.asarray(_field(record, key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a boolean [T] mask") from exc
    if result.shape != (length,) or result.dtype.kind != "b":
        raise ValueError(f"{key} must be a boolean [T] mask")
    return result


def _semantic(record: Mapping) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    values = _float32(_numeric(_field(record, "semantic"), "semantic", 2), "semantic")
    if not len(values) or values.shape[1] < 1:
        raise ValueError("semantic must be nonempty [T,D]")
    width = values.shape[1]
    default_names = [f"sem/latent_{j:04d}" for j in range(width)]
    names = _names(record.get("semantic_names", default_names), width, "semantic_names")
    support = _mask(record, "semantic_support", len(values))
    direct = _mask(record, "semantic_direct", len(values))
    if np.any(direct & ~support):
        raise ValueError("semantic_direct must be a subset of semantic_support")
    return values, names, support, direct


def _pca(pca: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    required = {"kind", "mean", "components", "semantic_names", "n_components", "fit_role"}
    if not isinstance(pca, Mapping) or set(pca) != required:
        raise ValueError("unknown semantic PCA schema")
    if pca["kind"] != KIND or pca["fit_role"] != _PCA_ROLE:
        raise ValueError("unknown semantic PCA kind/fit role")
    mean = _float32(_numeric(pca["mean"], "PCA mean", 1), "PCA mean")
    components = _float32(_numeric(pca["components"], "PCA components", 2), "PCA components")
    count = _integer(pca["n_components"], "n_components", minimum=1, maximum=len(mean))
    if components.shape != (count, len(mean)):
        raise ValueError("semantic PCA dimensions differ")
    names = _names(pca["semantic_names"], len(mean), "PCA semantic_names")
    return mean, components, names


def fit_semantic_pca(records: Any, train: Any, k: int = 16) -> dict:
    """Train-only supported-frame covariance: each content has total mass one.

    Aliases split a content's mass, each video's supported frames split its
    video mass. Cropped, local rgb copies let the original covariance routine
    perform precisely this calculation without mutating any record field.
    Neither labels nor held-out records nor unsupported semantic values enter
    the fit. A training video without supported frames fails explicitly.
    """
    train = _indices(records, train, "train")
    k = _integer(k, "k", minimum=1)
    copies: dict[int, dict] = {}
    names = None
    for i in train:
        values, current, support, _ = _semantic(records[i])
        if not np.any(support):
            raise ValueError("semantic PCA training video has no supported frames")
        if names is None:
            names = current
        elif current != names:
            raise ValueError("semantic training name/order differs")
        if k > values.shape[1]:
            raise ValueError("k exceeds semantic channel width")
        copies[i] = {"rgb": values[support], "sha256": _sha(records[i])}
    learned = fit_rgb_pca(copies, train, components=k)
    pca = {**learned, "kind": KIND, "semantic_names": names, "n_components": k}
    _pca(pca)
    return pca


def _fps(record: Mapping) -> float:
    value = _field(record, "fps")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("fps must be a finite positive scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("fps must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError("fps must be finite and positive")
    return result


def _view(record: Mapping, pca: Mapping) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    fps = _fps(record)
    # Preserve the original base's values, channel order and raw video summaries.
    for field in ("motion", "corrected"):
        _field(record, field)
        _field(record, field + "_names")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        base_frame, base_video, base_names = feature_view(BASE_RECIPE, record, None)
    base_frame = _float32(_numeric(base_frame, "base frame", 2), "base frame")
    base_video = _float32(_numeric(base_video, "base video", 1), "base video")
    base_names = _names(base_names, base_frame.shape[1], "base frame names")
    values, names, support, direct = _semantic(record)
    mean, components, bound_names = _pca(pca)
    if names != bound_names or values.shape[1] != len(mean):
        raise ValueError("semantic channel name/order differs from PCA")
    if len(values) != len(base_frame):
        raise ValueError("semantic and base frame counts differ")
    projected = np.zeros((len(values), len(components)), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected[support] = (values[support] - mean) @ components.T
        semantic_frame, semantic_names = augment_signals(
            projected, [f"sem/pca_{j:04d}" for j in range(len(components))], fps)
        flags = np.column_stack((direct, support)).astype(np.float32)
        frame = np.concatenate((base_frame, semantic_frame, flags), axis=1)
        video = np.concatenate((base_video, video_features(semantic_frame), video_features(flags)))
    frame = _float32(frame, "semantic frame view")
    video = _float32(video, "semantic video view")
    frame_names = base_names + semantic_names + list(FLAG_NAMES)
    _names(frame_names, frame.shape[1], "frame names")
    video_names = []
    for field in ("motion", "corrected"):
        raw_names = _names(record[field + "_names"], np.asarray(record[field]).shape[1], field + "_names")
        video_names.extend(f"base/{field}/{name}/{stat}" for stat in ("mean", "std", "p10", "p90")
                           for name in raw_names)
    for group in (semantic_names, FLAG_NAMES):
        video_names.extend(f"{name}/video_{stat}" for stat in ("mean", "std", "p10", "p90") for name in group)
    _names(video_names, len(video), "video names")
    return frame, video, frame_names, video_names


def feature_view_semantic(record: Mapping, pca: Mapping) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Original base + supported PCA augmentation + direct/support raw flags.

    Unsupported latent values cannot influence any feature. Context and global
    statistics still use the unchanged augment_signals convention on the masked
    projected series, not a new support-aware augmentation or loss. The explicit
    flags (and their raw video statistics) distinguish padding from observation.
    """
    frame, video, names, _ = _view(record, pca)
    return frame, video, names


def _labels(record: Mapping, length: int) -> np.ndarray:
    labels = _numeric(_field(record, "labels"), "training labels", 1, allow_bool=True)
    if len(labels) != length or np.any((labels != 0) & (labels != 1)):
        raise ValueError("training labels must be aligned binary [T]")
    return labels


def _positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.ndim != 1 or not len(classes) or not np.isin(classes, [0, 1]).all() or len(set(classes)) != len(classes):
        raise ValueError("only binary 0/1 classifier classes are supported")
    probabilities = _numeric(model.predict_proba(values), "classifier probabilities", 2)
    if probabilities.shape != (len(values), len(classes)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("invalid classifier probabilities")
    positive = np.flatnonzero(classes == 1)
    return probabilities[:, positive[0]] if len(positive) else np.zeros(len(values), dtype=np.float64)


def fit_semantic_member(algorithm: str, records: Any, train: Any, predict: Any,
                        seed: int, full_fit: bool = False) -> tuple[dict, dict, dict]:
    """Fit one fixed RF/ET readout and the unchanged video ET using TRAIN labels.

    Ordinary runs require disjoint SHA groups, not just disjoint indices. A
    full fit must be explicit and predict exactly its training index set.
    Frame weights deliberately remain original per-video 1/T (not PCA's alias
    weights); positive balancing and mean-one normalization are unchanged.
    """
    if algorithm not in ("rf", "et") or not isinstance(algorithm, str):
        raise ValueError("algorithm must be rf or et")
    if not isinstance(full_fit, (bool, np.bool_)):
        raise ValueError("full_fit must be explicit boolean")
    seed = _integer(seed, "seed", maximum=2 ** 32 - 1 - 1100)
    train, predict = _indices(records, train, "train"), _indices(records, predict, "predict")
    fit_content = {_sha(records[i]) for i in train}
    predict_content = {_sha(records[i]) for i in predict}
    if full_fit:
        if set(train) != set(predict):
            raise ValueError("full fit must predict exactly its training index set")
    elif fit_content & predict_content:
        raise ValueError("SHA group content leakage between train and predict")
    pca = fit_semantic_pca(records, train, k=PCA_COMPONENTS)
    ids = sorted(set(train) | set(predict))
    views = {i: _view(records[i], pca) for i in ids}
    frame_names, video_names = views[ids[0]][2:]
    if any(v[2] != frame_names or v[3] != video_names for v in views.values()):
        raise ValueError("frame/video feature name/order differs between records")
    labels = {i: _labels(records[i], len(views[i][0])) for i in train}
    x = np.concatenate([views[i][0] for i in train])
    y = np.concatenate([labels[i] for i in train])
    weights = np.concatenate([np.full(len(labels[i]), 1 / len(labels[i]), np.float64) for i in train])
    positive, negative = weights[y > 0].sum(), weights[y == 0].sum()
    if negative <= 0:
        raise ValueError("original positive balancing requires negative frame mass")
    weights = np.where(y > 0, weights * negative / max(1e-8, positive), weights)
    weights *= len(weights) / weights.sum()
    video_x = np.stack([views[i][1] for i in train])
    video_y = np.asarray([bool(np.any(labels[i])) for i in train], dtype=np.int64)
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    if algorithm == "rf":
        model = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=10,
                                       max_features=0.5, n_jobs=4, random_state=seed)
    else:
        model = ExtraTreesClassifier(n_estimators=192, max_depth=9, min_samples_leaf=8,
                                     max_features=0.6, n_jobs=4, random_state=seed)
    model.fit(x, y, sample_weight=weights)
    video_model = ExtraTreesClassifier(n_estimators=128, max_depth=4, min_samples_leaf=2,
                                       max_features=0.75, class_weight="balanced", n_jobs=4,
                                       random_state=seed + 1100)
    video_model.fit(video_x, video_y)
    fp = {i: _positive_probability(model, views[i][0]).astype(np.float32) for i in predict}
    vp = {i: float(_positive_probability(video_model, views[i][1][None, :])[0]) for i in predict}
    state = {"kind": KIND, "algorithm": algorithm, "recipe": f"semantic_corrected_motion_{algorithm}",
             "transform": {"feature_view": KIND, "base_recipe": BASE_RECIPE, "pca": pca,
                           "frame_feature_names": list(frame_names), "video_feature_names": list(video_names)},
             "frame_model": model, "video_model": video_model,
             "fit_content_sha256": sorted(fit_content), "full_fit": bool(full_fit)}
    return fp, vp, state


def _transform(state: Any) -> tuple[str, Mapping, list[str], list[str]]:
    if not isinstance(state, Mapping) or state.get("kind") != KIND or state.get("algorithm") not in ("rf", "et"):
        raise ValueError("unknown semantic member kind/algorithm")
    algorithm = state["algorithm"]
    if state.get("recipe") != f"semantic_corrected_motion_{algorithm}":
        raise ValueError("semantic member recipe differs")
    transform = state.get("transform")
    fields = {"feature_view", "base_recipe", "pca", "frame_feature_names", "video_feature_names"}
    if not isinstance(transform, Mapping) or set(transform) != fields:
        raise ValueError("unknown semantic transform schema")
    if transform["feature_view"] != KIND or transform["base_recipe"] != BASE_RECIPE:
        raise ValueError("semantic transform kind/base recipe differs")
    _pca(transform["pca"])
    names = []
    for key in ("frame_feature_names", "video_feature_names"):
        raw = transform[key]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("missing semantic feature names")
        names.append(_names(raw, len(raw), key))
    if names[0][-2:] != list(FLAG_NAMES):
        raise ValueError("semantic direct/support frame flag order differs")
    return algorithm, transform, names[0], names[1]


def _validate_forest(model: Any, width: int, trees_max: int, depth_max: int) -> None:
    """Validate bounded trees before delegating to the original portable walker."""
    if not isinstance(model, Mapping):
        raise ValueError("portable semantic model must be a mapping")
    dimension = _integer(model.get("n_features"), "n_features", minimum=1)
    if dimension != width:
        raise ValueError("portable semantic model feature dimension differs")
    if model.get("kind") == "constant":
        if set(model) != {"kind", "n_features", "probability"}:
            raise ValueError("unknown constant classifier schema")
        value = model["probability"]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid constant probability")
        return
    if set(model) != {"kind", "n_features", "trees"} or model["kind"] != "forest":
        raise ValueError("only portable forest/constant classifiers are supported")
    trees = model["trees"]
    if not isinstance(trees, (list, tuple)) or not 1 <= len(trees) <= trees_max:
        raise ValueError("unsupported semantic forest tree count")
    for tree in trees:
        if not isinstance(tree, Mapping) or set(tree) != {"left", "right", "feature", "threshold", "value"}:
            raise ValueError("unknown semantic tree schema")
        arrays = {key: _numeric(tree[key], key, 1) for key in tree}
        count = len(arrays["left"])
        if not 1 <= count <= 2 ** (depth_max + 1) - 1 or any(len(a) != count for a in arrays.values()):
            raise ValueError("unsupported semantic tree node count")
        for key in ("left", "right", "feature"):
            if arrays[key].dtype.kind not in "iu" or np.any(arrays[key] > np.iinfo(np.int32).max):
                raise ValueError("semantic tree indices must be int32-compatible")
        left, right, feature = arrays["left"], arrays["right"], arrays["feature"]
        leaf = (left == -1) & (right == -1)
        if (np.any(feature[leaf] != -2) or np.any(arrays["threshold"][leaf] != -2) or
                np.any(left[~leaf] < 0) or np.any(right[~leaf] < 0) or
                np.any(left[~leaf] >= count) or np.any(right[~leaf] >= count) or
                np.any(left[~leaf] == right[~leaf]) or np.any(feature[~leaf] < 0) or
                np.any(feature[~leaf] >= width) or np.any((arrays["value"] < 0) | (arrays["value"] > 1))):
            raise ValueError("invalid semantic classifier tree nodes")
        seen, pending = set(), [(0, 0)]
        while pending:
            node, depth = pending.pop()
            if node in seen or depth > depth_max:
                raise ValueError("cyclic/shared/over-depth semantic tree")
            seen.add(node)
            if not leaf[node]:
                pending.extend(((int(left[node]), depth + 1), (int(right[node]), depth + 1)))
        if len(seen) != count:
            raise ValueError("semantic tree has unreachable nodes")


def export_semantic_member(state: Mapping) -> dict:
    """Return a JSON-safe portable member; no files or default runtime are changed."""
    algorithm, transform, frame_names, video_names = _transform(state)
    forests = []
    for key, names, trees, depth in (("frame_model", frame_names, 160 if algorithm == "rf" else 192, 8 if algorithm == "rf" else 9),
                                    ("video_model", video_names, 128, 4)):
        model = state.get(key)
        if getattr(model, "n_features_in_", None) != len(names) or not hasattr(model, "classes_"):
            raise ValueError("expected fitted semantic classifier matching names")
        try:
            exported = export_model(model)
        except (AttributeError, TypeError, IndexError) as exc:
            raise ValueError("invalid fitted semantic classifier") from exc
        _validate_forest(exported, len(names), trees, depth)
        forests.append(exported)
    content = state.get("fit_content_sha256")
    if not isinstance(content, (list, tuple)) or not content or any(not isinstance(v, str) or not v for v in content):
        raise ValueError("missing fit content provenance")
    if list(content) != sorted(set(content)) or not isinstance(state.get("full_fit"), (bool, np.bool_)):
        raise ValueError("invalid fit content/full-fit provenance")
    return {"kind": KIND, "algorithm": algorithm, "recipe": state["recipe"], "weight": 1.0,
            "transform": deepcopy(dict(transform)), "frame_model": forests[0], "video_model": forests[1],
            "boundary_models": None, "fit_content_sha256": list(content), "full_fit": bool(state["full_fit"])}


def predict_semantic_member(member: Mapping, record: Mapping) -> tuple[np.ndarray, float]:
    """Recompute the same masked view and enforce PCA + frame/video name order."""
    algorithm, transform, frame_names, video_names = _transform(member)
    frame, video, names, vnames = _view(record, transform["pca"])
    if names != frame_names or vnames != video_names:
        raise ValueError("semantic inference frame/video name/order differs")
    frame_model, video_model = member.get("frame_model"), member.get("video_model")
    _validate_forest(frame_model, len(frame_names), 160 if algorithm == "rf" else 192, 8 if algorithm == "rf" else 9)
    _validate_forest(video_model, len(video_names), 128, 4)
    fp = portable_predict(frame_model, frame)
    vp = float(portable_predict(video_model, video[None, :])[0])
    return fp, vp
