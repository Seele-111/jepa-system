"""Independent CPU local-motion descriptors, NOT anomaly scores.

Rows correspond to successful decoder reads; all missing quantities are zero
with false feature_valid masks. Speeds use analysis-frame diagonals/second;
acceleration is an Eulerian (fixed image grid) residual-velocity difference,
not tracked-object physical acceleration. Coordinates are normalized to [0,1].
No whole-video ranking, filenames, labels, learned models, or GPU are used.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

VERSION = "local-motion-v1"
DEFAULT_FPS = 24.0
FLOW_PARAMETERS = dict(pyr_scale=0.5, levels=2, winsize=15, iterations=2,
                       poly_n=5, poly_sigma=1.2, flags=0)
DEFAULT_PROFILE = {
    "name": "local-residual-motion-cpu-v1", "long_edge": 128,
    "tile_grid": [3, 3], "border_pixels": 4,
    "active_speed_threshold": 0.05, "texture_gradient_threshold": 0.015,
    "camera_model": "deterministic_huber_affine_with_median_translation_fallback",
    "camera_sample_stride": 4, "camera_irls_iterations": 5,
    "camera_min_samples": 32, "camera_affine_min_tiles": 6,
    "camera_translation_min_tiles": 4, "camera_min_inlier_fraction": 0.60,
    "camera_inlier_floor_pixels": 0.40, "camera_inlier_cap_pixels": 1.50,
    "normalization": "fixed_units_no_video_ranks",
    "acceleration": "Eulerian_residual_vector_difference_times_fps",
    "direction_bins": 8, "flow_parameters": FLOW_PARAMETERS,
    "cpu_threads": 1, "opencl": False,
}
_GLOBAL = (
    "residual_p50", "residual_p90", "residual_p99", "active_fraction",
    "speed_mass", "active_centroid_x", "active_centroid_y", "active_spatial_rms",
    "direction_x", "direction_y", "direction_consistency",
) + tuple(f"direction_bin_{i}" for i in range(8)) + (
    "camera_vx", "camera_vy", "camera_strain_xx", "camera_strain_xy",
    "camera_strain_yx", "camera_strain_yy", "camera_inlier_fraction",
    "camera_fit_error_p90", "observed_fraction",
    "acceleration_p50", "acceleration_p90", "acceleration_mean_x",
    "acceleration_mean_y",
)
_TILE = ("observed_fraction", "residual_p50", "residual_p90", "active_fraction",
         "speed_mass", "active_vx", "active_vy", "direction_consistency",
         "acceleration_p90", "acceleration_mean_x", "acceleration_mean_y")
_FLAGS = ("frame_valid", "pair_valid", "motion_valid", "camera_affine_valid",
          "camera_fallback_used", "camera_compensation_valid", "acceleration_valid",
          "texture_observable", "source_size_changed", "flow_failed",
          "camera_model_changed")
FEATURE_NAMES = _GLOBAL + tuple(f"tile_{y}{x}_{n}" for y in range(3)
                                for x in range(3) for n in _TILE) + _FLAGS
_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}
FEATURE_UNITS = {}
for _name in FEATURE_NAMES:
    if _name in _FLAGS:
        _unit = "boolean"
    elif "acceleration" in _name:
        _unit = "frame_diagonals/second_squared"
    elif "strain" in _name:
        _unit = "1/second"
    elif any(v in _name for v in ("residual_p", "speed_mass", "active_vx", "active_vy")) or _name in ("camera_vx", "camera_vy", "camera_fit_error_p90"):
        _unit = "frame_diagonals/second"
    elif "centroid" in _name:
        _unit = "normalized_image_coordinate"
    elif _name == "active_spatial_rms":
        _unit = "frame_diagonals"
    else:
        _unit = "unitless" if _name in ("direction_x", "direction_y") else "fraction"
    FEATURE_UNITS[_name] = _unit


def _positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _property(cap, prop):
    try:
        value = float(cap.get(prop))
    except (cv2.error, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _set(row, mask, name, value, valid=True):
    i = _INDEX[name]
    if valid and math.isfinite(float(value)) and abs(float(value)) <= np.finfo(np.float32).max:
        row[i], mask[i] = value, True


def _texture(gray):
    image = gray.astype(np.float32) / 255.0
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(dx, dy) >= DEFAULT_PROFILE["texture_gradient_threshold"]


def _geometry(shape):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    coords = np.stack((2.0 * xx / (w - 1) - 1.0,
                       2.0 * yy / (h - 1) - 1.0, np.ones(shape)), axis=-1)
    interior = (xx >= 4) & (xx < w - 4) & (yy >= 4) & (yy < h - 4)
    tile = np.minimum(yy * 3 // h, 2) * 3 + np.minimum(xx * 3 // w, 2)
    sample = (xx % 4 == 0) & (yy % 4 == 0)
    return coords, interior, tile, sample


def _camera(flow, support, geometry):
    """Deterministic IRLS; no RANSAC RNG, cv2 RNG seed, or global video state."""
    coords, interior, tile, sample = geometry
    chosen = support & sample
    a, b = coords[chosen], flow[chosen].astype(np.float64)
    all_vectors = flow[support]
    coeff = np.zeros((3, 2), dtype=np.float64)
    if not len(all_vectors):
        return coeff, "unobservable", False, 0.0, 0.0
    coeff[2] = np.median(all_vectors, axis=0)
    if len(b) < DEFAULT_PROFILE["camera_min_samples"]:
        return coeff, "median_translation", False, 0.0, 0.0

    def quality(candidate):
        error = np.linalg.norm(b - a @ candidate, axis=1)
        med = float(np.median(error))
        mad = float(np.median(np.abs(error - med)))
        threshold = min(1.5, max(0.4, med + 3.0 * 1.4826 * mad))
        inside = error <= threshold
        coverage = len(np.unique(tile[chosen][inside]))
        return float(inside.mean()), coverage, float(np.quantile(error, 0.9))

    try:
        if np.linalg.matrix_rank(a) == 3 and np.linalg.cond(a) <= 50:
            fit = coeff.copy()
            for _ in range(5):
                error = np.linalg.norm(b - a @ fit, axis=1)
                cutoff = max(0.15, float(np.median(error)) * 1.5)
                weights = np.minimum(1.0, cutoff / np.maximum(error, 1e-9))
                root = np.sqrt(weights)
                fit = np.linalg.lstsq(a * root[:, None], b * root[:, None], rcond=None)[0]
            fraction, coverage, error = quality(fit)
            if np.isfinite(fit).all() and fraction >= 0.60 and coverage >= 6:
                return fit, "affine", True, fraction, error
    except np.linalg.LinAlgError:
        pass
    fraction, coverage, error = quality(coeff)
    reliable = fraction >= 0.60 and coverage >= 4
    return coeff, "median_translation", reliable, fraction, error


def _summarize(row, mask, velocity, support, geometry, threshold, previous, fps):
    coords, interior, tiles, sample = geometry
    speed = np.linalg.norm(velocity, axis=2)
    moving = support & (speed >= threshold)
    _set(row, mask, "observed_fraction", support.sum() / interior.sum())
    for name, value in zip(("residual_p50", "residual_p90", "residual_p99"),
                           np.quantile(speed[support], (0.5, 0.9, 0.99))):
        _set(row, mask, name, value)
    _set(row, mask, "active_fraction", moving.sum() / support.sum())
    _set(row, mask, "speed_mass", speed[moving].sum() / support.sum())
    if moving.any():
        weights = speed[moving].astype(np.float64)
        positions = (coords[moving, :2] + 1.0) / 2.0
        centroid = np.average(positions, axis=0, weights=weights)
        # x/y distances restored to pixels then divided by analysis diagonal.
        h, w = speed.shape
        distance = np.sum(((positions - centroid) * [w - 1, h - 1]) ** 2, axis=1)
        _set(row, mask, "active_centroid_x", centroid[0])
        _set(row, mask, "active_centroid_y", centroid[1])
        _set(row, mask, "active_spatial_rms", np.sqrt(np.average(distance, weights=weights)) / math.hypot(w, h))
        direction = velocity[moving].sum(axis=0) / weights.sum()
        _set(row, mask, "direction_x", direction[0])
        _set(row, mask, "direction_y", direction[1])
        _set(row, mask, "direction_consistency", min(1.0, np.linalg.norm(direction)))
        angles = np.mod(np.arctan2(velocity[moving, 1], velocity[moving, 0]), 2 * np.pi)
        bins = np.minimum((angles * 8 / (2 * np.pi)).astype(int), 7)
        histogram = np.bincount(bins, weights=weights, minlength=8) / weights.sum()
        for i, value in enumerate(histogram):
            _set(row, mask, f"direction_bin_{i}", value)
    accel = accel_support = None
    if previous is not None:
        old_velocity, old_support = previous
        accel_support = support & old_support
        if accel_support.any():
            accel = (velocity.astype(np.float64) - old_velocity) * fps
            magnitude = np.linalg.norm(accel, axis=2)
            for name, value in zip(("acceleration_p50", "acceleration_p90"),
                                   np.quantile(magnitude[accel_support], (0.5, 0.9))):
                _set(row, mask, name, value)
            mean = accel[accel_support].mean(axis=0)
            _set(row, mask, "acceleration_mean_x", mean[0])
            _set(row, mask, "acceleration_mean_y", mean[1])
            _set(row, mask, "acceleration_valid", 1)
    for t in range(9):
        prefix = f"tile_{t // 3}{t % 3}_"
        area = interior & (tiles == t)
        supported = support & area
        active = moving & area
        _set(row, mask, prefix + "observed_fraction", supported.sum() / area.sum())
        if supported.sum() >= 8:
            p50, p90 = np.quantile(speed[supported], (0.5, 0.9))
            _set(row, mask, prefix + "residual_p50", p50)
            _set(row, mask, prefix + "residual_p90", p90)
            _set(row, mask, prefix + "active_fraction", active.sum() / supported.sum())
            _set(row, mask, prefix + "speed_mass", speed[active].sum() / supported.sum())
            if active.any():
                mean = velocity[active].mean(axis=0)
                consistency = np.linalg.norm(velocity[active].sum(axis=0)) / speed[active].sum()
                _set(row, mask, prefix + "active_vx", mean[0])
                _set(row, mask, prefix + "active_vy", mean[1])
                _set(row, mask, prefix + "direction_consistency", min(1.0, consistency))
        if accel is not None:
            observed = accel_support & area
            if observed.sum() >= 8:
                _set(row, mask, prefix + "acceleration_p90", np.quantile(magnitude[observed], 0.9))
                mean = accel[observed].mean(axis=0)
                _set(row, mask, prefix + "acceleration_mean_x", mean[0])
                _set(row, mask, prefix + "acceleration_mean_y", mean[1])


def _finish(rows, masks, fps, fps_valid, metadata, issues):
    signals = np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    feature_valid = np.asarray(masks, dtype=bool).reshape(signals.shape)
    count = len(signals)
    expected = metadata["reported_frame_count"]
    matches = None if expected is None else count == expected
    if matches is False:
        issues.append("frame_count_mismatch")
    if not count:
        issues.append("no_decoded_frames")
    issues = list(dict.fromkeys(issues))
    metadata.update(decoded_frame_count=count, usable_frame_count=int(signals[:, _INDEX["frame_valid"]].sum()),
                    duration_seconds=count / fps, issues=issues)
    validity = {
        "valid": bool(count and fps_valid and not issues),
        "status": issues[0] if issues else "ok", "has_frames": bool(count),
        "fps_valid": fps_valid, "frame_count_matches": matches,
        "complete_decode": None if expected is None else bool(matches and metadata["read_status"] == "eof"),
        "feature_valid": feature_valid,
    }
    for name in _FLAGS:
        validity[name] = signals[:, _INDEX[name]].astype(bool)
    return {"signals": signals, "feature_names": list(FEATURE_NAMES), "fps": float(fps),
            "feature_units": dict(FEATURE_UNITS), "validity": validity, "metadata": metadata,
            "frame_times_seconds": np.arange(count, dtype=np.float64) / fps}


def extract_video_features(path, *, long_edge=128, fallback_fps=DEFAULT_FPS,
                           residual_active_threshold=0.05):
    """Extract causal, fixed-unit [T,D] descriptors from all successful reads.

    Low-texture flow is unobservable, not evidence of zero motion. Direction and
    centroid are masked when there are no active pixels. Fallback camera models
    have explicit flags; unreliable compensation masks residual descriptors.
    Acceleration requires adjacent valid estimates of the SAME camera model.
    Bad frames/size changes reset temporal history. Invalid FPS yields a finite
    fallback FPS but masks every motion quantity. CFR timestamps are nominal.
    The CPU/OpenCL settings are scoped and restored; call serially in a process.
    """
    if isinstance(long_edge, bool) or not isinstance(long_edge, (int, np.integer)) or long_edge < 32:
        raise ValueError("long_edge must be an integer >= 32")
    fps = _positive(fallback_fps, "fallback_fps")
    threshold = _positive(residual_active_threshold, "residual_active_threshold")
    path = os.fsdecode(os.fspath(path))
    rows, masks, issues = [], [], []
    fps_valid = False
    profile = dict(DEFAULT_PROFILE, long_edge=int(long_edge), active_speed_threshold=threshold)
    metadata = {
        "schema_version": VERSION, "profile": profile, "reported_fps": None,
        "fps_source": "fallback", "reported_frame_count": None, "read_status": "not_started",
        "source_size": None, "analysis_size": None, "source_sizes": [], "analysis_sizes": [],
        "frame_size_change_count": 0, "bad_frame_count": 0, "flow_failure_count": 0,
        "camera_affine_count": 0, "camera_fallback_count": 0, "camera_unobservable_count": 0,
        "texture_unobservable_count": 0, "timestamps_are_nominal": True,
        "direction_convention": "image x right, y down; bin i=[i*pi/4,(i+1)*pi/4)",
        "centroid_weighting": "active residual speed; texture-supported interior pixels",
        "speed_mass_definition": "sum(active residual speeds)/supported pixel count",
        "spatial_rms_definition": "speed-weighted RMS radius/frame diagonal",
        "acceleration_convention": "Eulerian grid; not object tracking or an anomaly rule",
    }
    if not os.path.isfile(path):
        return _finish(rows, masks, fps, fps_valid, metadata, ["not_a_file"])
    if os.path.getsize(path) == 0:
        return _finish(rows, masks, fps, fps_valid, metadata, ["empty_file"])
    cap = None
    previous = previous_size = previous_motion = previous_model = geometry = None
    old_threads, old_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
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
            metadata["fps_source"] = "decoder"
        else:
            issues.append("invalid_fps")
        reported_count = _property(cap, cv2.CAP_PROP_FRAME_COUNT)
        if reported_count is not None and reported_count >= 0 and reported_count.is_integer():
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
            row = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            mask = np.zeros(len(FEATURE_NAMES), dtype=bool)
            for name in _FLAGS:
                _set(row, mask, name, 0)
            rows.append(row)
            masks.append(mask)
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                    frame.ndim not in (2, 3) or min(frame.shape[:2]) < 1 or
                    (frame.ndim == 3 and frame.shape[2] not in (1, 3, 4))):
                metadata["bad_frame_count"] += 1
                issues.append("bad_frame")
                previous = previous_motion = previous_model = previous_size = None
                continue
            size = (int(frame.shape[1]), int(frame.shape[0]))
            changed = previous_size is not None and size != previous_size
            _set(row, mask, "source_size_changed", int(changed))
            if changed:
                metadata["frame_size_change_count"] += 1
                issues.append("source_size_changed")
                previous = previous_motion = previous_model = None
            scale = min(1.0, long_edge / max(size))
            analysis = (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))
            try:
                if frame.ndim == 2:
                    gray = frame
                elif frame.shape[2] == 1:
                    gray = frame[:, :, 0]
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, analysis, interpolation=cv2.INTER_AREA) if size != analysis else gray.copy()
            except cv2.error:
                metadata["bad_frame_count"] += 1
                issues.append("bad_frame")
                previous = previous_motion = previous_model = previous_size = None
                continue
            _set(row, mask, "frame_valid", 1)
            for key, value in (("source_sizes", list(size)), ("analysis_sizes", list(analysis))):
                if value not in metadata[key]:
                    metadata[key].append(value)
            if metadata["source_size"] is None:
                metadata["source_size"], metadata["analysis_size"] = list(size), list(analysis)
            if min(analysis) < 16:
                issues.append("analysis_too_small")
                previous = previous_motion = previous_model = None
                previous_size = size
                continue
            if geometry is None or changed or geometry[0].shape[:2] != gray.shape:
                geometry = _geometry(gray.shape)
            this_motion = this_model = None
            if previous is not None:
                _set(row, mask, "pair_valid", 1)
                if fps_valid:
                    try:
                        flow = cv2.calcOpticalFlowFarneback(previous, gray, None, **FLOW_PARAMETERS)
                        if flow is None or flow.shape != gray.shape + (2,) or not np.isfinite(flow).all():
                            raise ValueError("invalid_flow")
                        support = _texture(previous) & _texture(gray) & geometry[1]
                        observable = support.sum() >= 32
                        _set(row, mask, "texture_observable", observable)
                        _set(row, mask, "observed_fraction", support.sum() / geometry[1].sum())
                        for t in range(9):
                            area = geometry[1] & (geometry[2] == t)
                            _set(row, mask, f"tile_{t//3}{t%3}_observed_fraction", (support & area).sum() / area.sum())
                        if not observable:
                            metadata["texture_unobservable_count"] += 1
                        else:
                            coeff, model, reliable, inliers, error = _camera(flow, support, geometry)
                            this_model = model
                            _set(row, mask, "camera_affine_valid", model == "affine" and reliable)
                            _set(row, mask, "camera_fallback_used", model == "median_translation")
                            _set(row, mask, "camera_compensation_valid", reliable)
                            metadata["camera_affine_count" if model == "affine" else "camera_fallback_count"] += 1
                            _set(row, mask, "camera_inlier_fraction", inliers)
                            _set(row, mask, "camera_fit_error_p90", error * fps / math.hypot(*analysis))
                            if reliable:
                                rate = fps / math.hypot(*analysis)
                                velocity = (flow.astype(np.float64) - geometry[0] @ coeff) * rate
                                _set(row, mask, "camera_vx", coeff[2, 0] * rate)
                                _set(row, mask, "camera_vy", coeff[2, 1] * rate)
                                for name, v in zip(("camera_strain_xx", "camera_strain_xy", "camera_strain_yx", "camera_strain_yy"),
                                                   (coeff[0, 0] * 2 / (analysis[0]-1), coeff[1, 0] * 2 / (analysis[1]-1),
                                                    coeff[0, 1] * 2 / (analysis[0]-1), coeff[1, 1] * 2 / (analysis[1]-1))):
                                    _set(row, mask, name, v * fps, model == "affine")
                                _set(row, mask, "motion_valid", 1)
                                switched = previous_model is not None and previous_model != model
                                _set(row, mask, "camera_model_changed", switched)
                                _summarize(row, mask, velocity, support, geometry, threshold,
                                           previous_motion if not switched else None, fps)
                                this_motion = (velocity, support)
                            else:
                                metadata["camera_unobservable_count"] += 1
                    except (cv2.error, ValueError, np.linalg.LinAlgError, FloatingPointError):
                        # Never leave a partially computed row looking like valid motion.
                        row[:] = 0
                        mask[:] = False
                        for name in _FLAGS:
                            _set(row, mask, name, 0)
                        _set(row, mask, "frame_valid", 1)
                        _set(row, mask, "pair_valid", 1)
                        _set(row, mask, "flow_failed", 1)
                        metadata["flow_failure_count"] += 1
                        issues.append("flow_failed")
                        this_motion = this_model = None
            previous, previous_size = gray, size
            previous_motion, previous_model = this_motion, this_model
    finally:
        if cap is not None:
            cap.release()
        cv2.ocl.setUseOpenCL(old_opencl)
        cv2.setNumThreads(old_threads)
    return _finish(rows, masks, fps, fps_valid, metadata, issues)
