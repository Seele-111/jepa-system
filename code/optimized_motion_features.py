"""Lightweight CPU video descriptors; these are features, not error scores.

Usage::

    video = extract_video_features(path)
    context = context_features(video["signals"], video["feature_names"],
                               fps=video["fps"],
                               validity=video["validity"]["feature_valid"])

Every decoded frame has one row. Missing temporal measurements are zero-filled
with explicit validity flags/masks, not confused with observed zero motion.
Intensity uses the fixed 1/255 scale; flow uses analysis-frame diagonals/second.
No per-video min/max scaling, labels, filename-derived features, or error rules
are used. Median flow is a translation-only camera-motion approximation: zoom,
rotation, parallax, and moving foreground can also produce legitimate residuals.

Context windows are centered (offline/non-causal). Always pass the decoded FPS
for real seconds; the two-argument convenience call explicitly records its
24-FPS assumption in metadata. Unknown extraction FPS also uses this fallback,
but marks the whole video's validity false. Nominal timestamps assume CFR.
"""

import math
import os

import cv2
import numpy as np


DEFAULT_FPS = 24.0
_FRAME_FEATURES = ("luma_mean", "luma_std", "laplacian_variance", "blur_score")
_PAIR_FEATURES = ("luma_change", "frame_diff_median", "frame_diff_p90")
_SECOND_FEATURES = ("luma_second_change", "frame_second_diff_p90")
_FLOW_FEATURES = (
    "flow_median", "flow_p90", "flow_p99",
    "camera_dx", "camera_dy", "camera_magnitude",
    "residual_median", "residual_p90", "residual_p99",
    "residual_active_fraction", "residual_outlier_fraction",
)
_ACCEL_FEATURES = ("camera_acceleration", "residual_acceleration")
_FLAG_FEATURES = ("pair_valid", "second_order_valid", "motion_valid", "acceleration_valid")
FEATURE_NAMES = _FRAME_FEATURES + _PAIR_FEATURES + _SECOND_FEATURES + _FLOW_FEATURES + _ACCEL_FEATURES + _FLAG_FEATURES
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_GROUPS = {
    "pair_valid": _PAIR_FEATURES,
    "second_order_valid": _SECOND_FEATURES,
    "motion_valid": _FLOW_FEATURES,
    "acceleration_valid": _ACCEL_FEATURES,
}
_FLOW_PARAMS = dict(pyr_scale=0.5, levels=2, winsize=15, iterations=2,
                    poly_n=5, poly_sigma=1.2, flags=0)
# Fixed floors prevent tiny codec/flow noise acquiring huge relative z-scores.
_SCALE_FLOORS = {name: 1.0 / 255.0 for name in _FRAME_FEATURES + _PAIR_FEATURES + _SECOND_FEATURES}
_SCALE_FLOORS.update({name: 0.01 for name in _FLOW_FEATURES})
_SCALE_FLOORS.update(laplacian_variance=1e-4, blur_score=0.02,
                     camera_acceleration=0.25, residual_acceleration=0.25)
FEATURE_UNITS = {name: "normalized_intensity" for name in _FRAME_FEATURES + _PAIR_FEATURES + _SECOND_FEATURES}
FEATURE_UNITS.update({name: "frame_diagonals/second" for name in _FLOW_FEATURES})
FEATURE_UNITS.update(laplacian_variance="normalized_intensity_squared", blur_score="unitless",
                     residual_active_fraction="fraction", residual_outlier_fraction="fraction",
                     camera_acceleration="frame_diagonals/second_squared",
                     residual_acceleration="frame_diagonals/second_squared")
FEATURE_UNITS.update({name: "boolean" for name in _FLAG_FEATURES})


def _positive_float(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _property(cap, prop):
    try:
        value = float(cap.get(prop))
    except (cv2.error, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _set_group(row, mask, names, values):
    indices = [_INDEX[name] for name in names]
    row[indices] = values
    mask[indices] = True


def _flow_statistics(previous, current, rate, active_threshold):
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, **_FLOW_PARAMS)
    if flow is None or flow.shape != current.shape + (2,) or not np.isfinite(flow).all():
        raise ValueError("invalid_flow")
    # Exclude a narrow border where the flow solver has less image support.
    border = 4 if min(current.shape) >= 16 else 0
    if border:
        flow = flow[border:-border, border:-border]
    vectors = flow.reshape(-1, 2)
    camera = np.median(vectors, axis=0) * rate
    magnitude = np.hypot(vectors[:, 0], vectors[:, 1]) * rate
    residual_vectors = vectors * rate - camera
    residual = np.hypot(residual_vectors[:, 0], residual_vectors[:, 1])
    flow_quantiles = np.quantile(magnitude, [0.5, 0.9, 0.99])
    residual_quantiles = np.quantile(residual, [0.5, 0.9, 0.99])
    mad = float(np.median(np.abs(residual - residual_quantiles[0])))
    cutoff = max(active_threshold, residual_quantiles[0] + 3.0 * 1.4826 * mad)
    values = [*flow_quantiles, *camera, float(np.hypot(*camera)), *residual_quantiles,
              float(np.mean(residual > active_threshold)), float(np.mean(residual > cutoff))]
    return values, camera, float(residual_quantiles[0])


def _finish(rows, masks, fps, fps_valid, metadata, issues):
    signals = np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    feature_valid = np.asarray(masks, dtype=bool).reshape(signals.shape)
    if not fps_valid:
        # Retain explicitly fallback-scaled raw values, but never present their
        # seconds-based units as observed/valid to a downstream temporal head.
        indices = [_INDEX[name] for name in _FLOW_FEATURES + _ACCEL_FEATURES]
        feature_valid[:, indices] = False
        signals[:, [_INDEX["motion_valid"], _INDEX["acceleration_valid"]]] = 0.0
    count = len(signals)
    reported = metadata["reported_frame_count"]
    matches = None if reported is None else count == reported
    if matches is False:
        issues.append("frame_count_mismatch")
    metadata.update(decoded_frame_count=count, duration_seconds=count / fps,
                    issues=list(dict.fromkeys(issues)))
    complete = None if reported is None else bool(matches and metadata["read_status"] == "eof")
    validity = {
        "valid": bool(count and fps_valid and not issues),
        "status": issues[0] if issues else "ok",
        "has_frames": bool(count), "fps_valid": bool(fps_valid),
        "frame_count_matches": matches, "complete_decode": complete,
        "frame_valid": feature_valid[:, _INDEX["luma_mean"]].copy(),
        "feature_valid": feature_valid,
    }
    for name in _FLAG_FEATURES:
        validity[name] = signals[:, _INDEX[name]].astype(bool)
    return {"signals": signals, "feature_names": list(FEATURE_NAMES), "fps": float(fps),
            "feature_units": dict(FEATURE_UNITS), "validity": validity, "metadata": metadata,
            "frame_times_seconds": np.arange(count, dtype=np.float64) / fps}


def extract_video_features(path, *, long_edge=96, fallback_fps=DEFAULT_FPS,
                           residual_active_threshold=0.05):
    """Read all frames of a local video and return a stable [T, 26] schema.

    ``long_edge`` preserves aspect ratio and never upscales. Flow statistics are
    absolute normalized speeds, not within-video ranks. Residual fractions use
    a fixed speed threshold (default .05 diagonals/s); the outlier fraction also
    requires exceeding median + 3 * scaled MAD. These do not classify dynamics
    as errors. Blur is the fixed proxy 1/(1 + Laplacian variance/.01).

    Missing/empty/unopenable inputs return [0, D] with an invalid status. Partial
    decode returns all successfully processed rows with explicit failure/count
    metadata. CAP_PROP_FRAME_COUNT is checked, never used as a reading limit.
    Unknown counts cannot prove complete decoding and yield None for that check.
    Nonpositive/nonfinite FPS is never silently trusted: fallback/source/validity
    are recorded; FPS-dependent motion/acceleration masks and flags are false
    when fallback FPS is used. Flow acceleration needs two valid flow estimates.
    """
    if isinstance(long_edge, bool) or not isinstance(long_edge, (int, np.integer)) or long_edge < 8:
        raise ValueError("long_edge must be an integer >= 8")
    fps = _positive_float(fallback_fps, "fallback_fps")
    active_threshold = _positive_float(residual_active_threshold, "residual_active_threshold")
    path = os.fsdecode(os.fspath(path))
    rows, masks, issues = [], [], []
    fps_valid = False
    metadata = {
        "schema_version": 1, "reported_fps": None, "fps_source": "fallback",
        "reported_frame_count": None, "read_status": "not_started",
        "source_size": None, "analysis_size": None, "long_edge": int(long_edge),
        "flow_failure_count": 0, "frame_size_change_count": 0,
        "flow_parameters": dict(_FLOW_PARAMS), "flow_border_pixels": 4,
        "residual_active_threshold": active_threshold, "blur_variance_reference": 0.01,
        "timestamps_are_nominal": True, "camera_model": "median_translation",
    }
    if not os.path.isfile(path):
        return _finish(rows, masks, fps, fps_valid, metadata, ["not_a_file"])
    try:
        if os.path.getsize(path) == 0:
            return _finish(rows, masks, fps, fps_valid, metadata, ["empty_file"])
    except OSError:
        return _finish(rows, masks, fps, fps_valid, metadata, ["file_stat_failed"])

    cap = None
    previous = previous_float = before_previous = previous_camera = previous_residual = None
    previous_size = None
    try:
        try:
            cap = cv2.VideoCapture(path)
            opened = cap.isOpened()
        except cv2.error:
            opened = False
        if not opened:
            return _finish(rows, masks, fps, fps_valid, metadata, ["open_failed"])
        reported_fps = _property(cap, cv2.CAP_PROP_FPS)
        metadata["reported_fps"] = reported_fps
        if reported_fps is not None and reported_fps > 0:
            fps, fps_valid = reported_fps, True
            metadata["fps_source"] = "reported"
        else:
            issues.append("invalid_fps")
        reported_count = _property(cap, cv2.CAP_PROP_FRAME_COUNT)
        if reported_count is not None and reported_count >= 1 and reported_count.is_integer():
            metadata["reported_frame_count"] = int(reported_count)

        while True:
            try:
                ok, frame = cap.read()
            except cv2.error:
                metadata["read_status"] = "read_error"
                issues.append("read_error")
                break
            if not ok:
                metadata["read_status"] = "eof"
                break
            if (frame is None or not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim not in (2, 3) or min(frame.shape[:2]) < 1
                    or (frame.ndim == 3 and frame.shape[2] not in (1, 3, 4))):
                metadata["read_status"] = "invalid_frame"
                issues.append("invalid_frame")
                break
            size = (frame.shape[1], frame.shape[0])
            if metadata["analysis_size"] is None:
                scale = min(1.0, long_edge / max(size))
                metadata["source_size"] = size
                metadata["analysis_size"] = tuple(max(1, int(round(v * scale))) for v in size)
            if previous_size is not None and size != previous_size:
                metadata["frame_size_change_count"] += 1
                issues.append("frame_size_changed")
                previous = previous_float = before_previous = previous_camera = previous_residual = None
            try:
                if frame.ndim == 2 or frame.shape[2] == 1:
                    gray = frame.reshape(frame.shape[:2])
                else:
                    conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
                    gray = cv2.cvtColor(frame, conversion)
                if size != metadata["analysis_size"]:
                    gray = cv2.resize(gray, metadata["analysis_size"], interpolation=cv2.INTER_AREA)
                gray = np.ascontiguousarray(gray)
                current_float = gray.astype(np.float32) / 255.0
                laplacian = cv2.Laplacian(current_float, cv2.CV_32F, ksize=3)
            except cv2.error:
                metadata["read_status"] = "frame_processing_error"
                issues.append("frame_processing_error")
                break
            row = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            mask = np.zeros(len(FEATURE_NAMES), dtype=bool)
            mask[[_INDEX[name] for name in _FLAG_FEATURES]] = True
            lap_variance = float(laplacian.var())
            _set_group(row, mask, _FRAME_FEATURES,
                       [float(current_float.mean()), float(current_float.std()),
                        lap_variance, 1.0 / (1.0 + lap_variance / 0.01)])
            camera = residual_median = None
            if previous is not None:
                difference = current_float - previous_float
                _set_group(row, mask, _PAIR_FEATURES,
                           [float(difference.mean()), *np.quantile(np.abs(difference), [0.5, 0.9])])
                row[_INDEX["pair_valid"]] = 1.0
                if before_previous is not None:
                    second_difference = current_float - 2.0 * previous_float + before_previous
                    _set_group(row, mask, _SECOND_FEATURES,
                               [float(second_difference.mean()), float(np.quantile(np.abs(second_difference), 0.9))])
                    row[_INDEX["second_order_valid"]] = 1.0
                try:
                    if min(gray.shape) < 8:
                        raise ValueError("motion_size_too_small")
                    rate = fps / math.hypot(*metadata["analysis_size"])
                    values, camera, residual_median = _flow_statistics(previous, gray, rate, active_threshold)
                    if not np.isfinite(values).all():
                        raise ValueError("invalid_flow")
                    _set_group(row, mask, _FLOW_FEATURES, values)
                    row[_INDEX["motion_valid"]] = 1.0
                    if previous_camera is not None:
                        acceleration = float(np.hypot(*(camera - previous_camera)) * fps)
                        residual_acceleration = (residual_median - previous_residual) * fps
                        if np.isfinite([acceleration, residual_acceleration]).all():
                            _set_group(row, mask, _ACCEL_FEATURES, [acceleration, residual_acceleration])
                            row[_INDEX["acceleration_valid"]] = 1.0
                except (cv2.error, ValueError, FloatingPointError):
                    camera = residual_median = None
                    metadata["flow_failure_count"] += 1
                    issues.append("flow_failed")
            rows.append(row)
            masks.append(mask)
            before_previous, previous_float = previous_float, current_float
            previous, previous_size = gray, size
            previous_camera, previous_residual = camera, residual_median
        if not rows:
            issues.append("no_decodable_frames")
    finally:
        if cap is not None:
            cap.release()
    return _finish(rows, masks, fps, fps_valid, metadata, issues)


def _tie_ranks(values, valid):
    """Normalized average ranks; all-equal/singleton observations map to .5."""
    result = np.zeros(len(values), dtype=np.float32)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return result
    order = np.argsort(values[indices], kind="stable")
    ordered_indices = indices[order]
    ordered = values[ordered_indices]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(ordered)]
    ranks = ((starts + ends - 1) * 0.5 / (len(ordered) - 1)
             if len(ordered) > 1 else np.asarray([0.5]))
    result[ordered_indices] = np.repeat(ranks, ends - starts)
    return result


def _rolling_relative(values, valid, frames, floor):
    # Chunk the sliding windows to bound temporary memory even on long videos.
    count = len(values)
    centers = np.zeros(count, dtype=np.float32)
    zscores = np.zeros(count, dtype=np.float32)
    center_valid = np.zeros(count, dtype=bool)
    if not count:
        return centers, zscores, center_valid
    radius = frames // 2
    # Padding with the larger radius has identical observed support, so clamp
    # at video length to avoid huge buffers for very short clips/high FPS.
    radius = min(radius, count - 1)
    width = 2 * radius + 1
    observed = np.where(valid, values.astype(np.float64), np.nan)
    padded = np.pad(observed, (radius, radius), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    chunk_size = max(1, 262144 // width)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        block = windows[start:stop]
        usable = np.isfinite(block).any(axis=1)
        if not usable.any():
            continue
        selected = block[usable]
        medians = np.nanmedian(selected, axis=1)
        scales = np.maximum(1.4826 * np.nanmedian(np.abs(selected - medians[:, None]), axis=1), floor)
        target = np.arange(start, stop)[usable]
        centers[target] = medians
        center_valid[target] = True
        zscores[target] = np.where(valid[target], np.clip((values[target] - medians) / scales, -20.0, 20.0), 0.0)
    return centers, zscores, center_valid


def context_features(signals, feature_names, fps=None, *, window_seconds=(0.5, 1.5), validity=None):
    """Append global robust z/midrank and rolling median/robust z descriptors.

    Returns a dict with ``signals``, ``feature_names``, ``fps``, ``validity`` and
    ``metadata``. The original finite float32 columns remain unchanged at the
    beginning. NaN/inf inputs are zero-filled and invalid. Supplied validity is a
    [T,D] boolean mask; known temporal flags are also honored, never overridden.
    Flags themselves get no rank/z columns. Constant signals rank .5, z = 0.
    MAD scale floors are fixed in feature units and relative z is clipped to
    +/-20; absolute inputs are never clipped or normalized per video.
    """
    source = np.asarray(signals)
    names = list(feature_names)
    if source.ndim != 2 or source.dtype.kind not in "biuf":
        raise ValueError("signals must be a real numeric [T, D] array")
    if (len(names) != source.shape[1] or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("feature_names must be unique nonempty strings matching D")
    effective_fps = DEFAULT_FPS if fps is None else _positive_float(fps, "fps")
    seconds = tuple(_positive_float(v, "window_seconds") for v in window_seconds)
    if not seconds or len(set(seconds)) != len(seconds):
        raise ValueError("window_seconds must be nonempty and unique")
    with np.errstate(over="ignore", invalid="ignore"):
        base = source.astype(np.float32, copy=True)
    base_valid = np.isfinite(base)
    if validity is not None:
        supplied = np.asarray(validity, dtype=bool)
        if supplied.shape != base.shape:
            raise ValueError("validity must have shape [T, D]")
        base_valid &= supplied
    name_indices = {name: i for i, name in enumerate(names)}
    for flag, group in _GROUPS.items():
        if flag in name_indices:
            flag_index = name_indices[flag]
            observed = base_valid[:, flag_index] & (base[:, flag_index] > 0.5)
            for name in group:
                if name in name_indices:
                    base_valid[:, name_indices[name]] &= observed
    base[~np.isfinite(base)] = 0.0
    blocks, masks, output_names = [base], [base_valid], names.copy()
    frame_windows = []
    for duration in seconds:
        target_frames = duration * effective_fps
        if not math.isfinite(target_frames):
            raise ValueError("window_seconds * fps must be finite")
        frames = max(1, int(round(target_frames)))
        frame_windows.append(frames if frames % 2 else frames + 1)
    for index, name in enumerate(names):
        if name in _FLAG_FEATURES:
            continue
        values, valid = base[:, index], base_valid[:, index]
        floor = _SCALE_FLOORS.get(name, 1e-3)
        zscore = np.zeros(len(base), dtype=np.float32)
        if valid.any():
            observed = values[valid].astype(np.float64)
            median = np.median(observed)
            scale = max(1.4826 * np.median(np.abs(observed - median)), floor)
            zscore[valid] = np.clip((observed - median) / scale, -20.0, 20.0)
        blocks.extend([zscore[:, None], _tie_ranks(values, valid)[:, None]])
        masks.extend([valid[:, None], valid[:, None]])
        output_names.extend([name + "__robust_z", name + "__midrank"])
        for seconds_value, frames in zip(seconds, frame_windows):
            center, relative, center_valid = _rolling_relative(values, valid, frames, floor)
            blocks.extend([center[:, None], relative[:, None]])
            masks.extend([center_valid[:, None], (center_valid & valid)[:, None]])
            suffix = format(seconds_value, ".12g") + "s"
            output_names.extend([name + "__rolling_median_" + suffix, name + "__rolling_z_" + suffix])
    if len(set(output_names)) != len(output_names):
        raise ValueError("context feature names collide; rename input features or windows")
    return {
        "signals": np.concatenate(blocks, axis=1).astype(np.float32, copy=False),
        "feature_names": output_names, "fps": float(effective_fps),
        "validity": {"feature_valid": np.concatenate(masks, axis=1), "timing_valid": fps is not None},
        "metadata": {"schema_version": 1, "absolute_feature_count": len(names),
                     "fps_assumed": fps is None, "window_seconds": list(seconds),
                     "window_frames": frame_windows, "window_mode": "centered",
                     "rank_method": "average_ties_zero_to_one", "robust_z_clip": 20.0},
    }
