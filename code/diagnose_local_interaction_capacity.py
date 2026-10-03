#!/usr/bin/env python3
"""Read-only capacity diagnosis for local spatiotemporal interaction evidence.

This is deliberately not a model trainer. It consumes the frozen development
labels only to measure whether a small, label-free local interaction view has
information that is not already present in the existing global JEPA error and
ordinary spatial summaries.

The protocol is frozen in the output receipt:
* V-JEPA tubelet anchors are the primary temporal grid; no dense interpolation
  is used for the diagnostic scores.
* A heatmap is reduced to a fixed 3x3 spatial distribution by valid-patch
  average, then normalized. No label or per-video threshold enters this view.
* The only relation families are V temporal persistence/change, V/I spatial
  co-location, camera-compensated local residual motion, and top-k local
  strength coupled to those relations. The strength is a fixed robust local
  z-score, not a video-level threshold.
* The output directory must be new; historical evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_MANIFEST = ROOT / "output/algorithm-opt-2026-10-02-v2/dataset_manifest.json"
DEFAULT_DATASET_NPZ = ROOT / "output/algorithm-opt-2026-10-02/dataset.npz"
DEFAULT_RAW_ROOT = ROOT / "output/algorithm-opt-2026-10-02/corrected_jepa_features_v2"
DEFAULT_LOCAL_ROOT = ROOT / "output/algorithm-opt-2026-10-02-v2/local_motion"
DEFAULT_LEDGER_ROOT = ROOT / "output/baseline-targeted-semantic-readout/event-change-ledger"

FEATURES = (
    "v_persistence_cos",
    "v_top3_persistence_iou",
    "v_spatial_change",
    "i_persistence_cos",
    "i_top3_persistence_iou",
    "i_spatial_change",
    "vi_colocation_cos",
    "vi_top3_colocation_iou",
    "motion_colocation_cos",
    "motion_colocation_gain",
    "joint_persistent_colocation",
    "v_top3_strength_z",
    "v_strength_change",
    "v_strength_motion_gain",
    "v_strength_vi_colocation",
)
FAMILIES = {
    "temporal_local": (
        "v_persistence_cos", "v_top3_persistence_iou", "v_spatial_change",
        "i_persistence_cos", "i_top3_persistence_iou", "i_spatial_change",
    ),
    "cross_modal_colocation": (
        "vi_colocation_cos", "vi_top3_colocation_iou",
    ),
    "motion_colocation": (
        "motion_colocation_cos", "motion_colocation_gain",
    ),
    "joint_relation": ("joint_persistent_colocation",),
    "local_strength_interaction": (
        "v_top3_strength_z", "v_strength_change",
        "v_strength_motion_gain", "v_strength_vi_colocation",
    ),
}

# These are part of the pre-registered diagnostic, not a tuning grid.
CAPACITY_RULE = {
    "minimum_direct_v_anchor_coverage": 0.80,
    "minimum_two_sided_anchor_auc": 0.60,
    "minimum_low_evidence_stage_auc": 0.60,
    "maximum_absolute_correlation_with_existing_global_or_spatial": 0.90,
    "minimum_qualifying_relation_families": 2,
    "minimum_qualifying_features": 2,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 3:
        return None
    x, y = a[good] - a[good].mean(), b[good] - b[good].mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return 0.0 if den <= 1e-12 else float(np.dot(x, y) / den)


def auc_high(pos: np.ndarray, neg: np.ndarray) -> float | None:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return None
    total = 0.0
    for x in pos:
        total += float((x > neg).sum()) + 0.5 * float((x == neg).sum())
    return float(total / (len(pos) * len(neg)))


def normalized_distribution(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0).astype(np.float64)
    total = float(values.sum())
    return np.zeros_like(values) if total <= 1e-12 else values / total


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return 0.0 if den <= 1e-12 else float(np.dot(a, b) / den)


def top_iou(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not a.any() or not b.any():
        return 0.0
    ka = set(np.argsort(a)[-min(k, len(a)):].tolist())
    kb = set(np.argsort(b)[-min(k, len(b)):].tolist())
    return float(len(ka & kb) / max(1, len(ka | kb)))


def heat_distribution(heat: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """Fixed 3x3 distribution plus a robust top-3 local-strength score."""
    heat, mask = np.asarray(heat, dtype=np.float64), np.asarray(mask, dtype=bool)
    require(heat.ndim == 2 and mask.shape == heat.shape, "invalid heatmap schema")
    valid = mask & np.isfinite(heat)
    values = np.where(valid, np.maximum(heat, 0.0), 0.0)
    h, w = heat.shape
    cells, supported = np.zeros(9, dtype=np.float64), np.zeros(9, dtype=np.float64)
    for r in range(3):
        r0, r1 = (r * h) // 3, ((r + 1) * h) // 3
        for c in range(3):
            c0, c1 = (c * w) // 3, ((c + 1) * w) // 3
            j = 3 * r + c
            block_valid = valid[r0:r1, c0:c1]
            supported[j] = float(block_valid.sum())
            if supported[j] > 0:
                cells[j] = float(values[r0:r1, c0:c1].sum() / supported[j])
    distribution = normalized_distribution(cells)
    total = float(values[valid].sum()) if valid.any() else 0.0
    if valid.any():
        supported_values = values[valid]
        patch_mean = float(supported_values.mean())
        weights = normalized_distribution(supported_values)
        entropy = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum() /
                        math.log(max(2, len(weights))))
        median = float(np.median(supported_values))
        q75 = float(np.percentile(supported_values, 75))
        scale = max((q75 - median) / 0.67448975, float(np.std(supported_values)) * 0.1 + 1e-5)
        order = np.sort(supported_values)[::-1]
        top3_mean = float(order[:min(3, len(order))].mean())
        top3_strength_z = max(0.0, (top3_mean - median) / scale)
    else:
        patch_mean, entropy, top3_strength_z = 0.0, 0.0, 0.0
    return distribution, float(valid.mean()), patch_mean, entropy, float(top3_strength_z)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels).astype(bool)
    change = np.diff(np.r_[False, labels, False].astype(np.int8))
    return [(int(s), int(e - 1)) for s, e in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))]


def load_local_motion(path: Path, frames: int, fps: float, feature_names: list[str]):
    with np.load(path, allow_pickle=False) as a:
        signals = np.asarray(a["signals"], dtype=np.float64)
        valid = np.asarray(a["feature_valid"], dtype=bool)
        ids = np.asarray(a["frame_ids"])
        cache_fps = float(np.asarray(a["fps"]).reshape(()))
        names = [str(x) for x in np.asarray(a["feature_names"]).tolist()]
    require(signals.shape == valid.shape == (frames, len(feature_names)), "local feature shape mismatch")
    require(names == feature_names and np.array_equal(ids, np.arange(frames)), "local schema mismatch")
    require(math.isfinite(cache_fps) and abs(cache_fps - fps) <= 1e-6, "local FPS mismatch")
    require(np.isfinite(signals).all(), "local signals contain nonfinite values")
    return signals, valid, {name: i for i, name in enumerate(feature_names)}


def local_motion_map(signals, valid, name_to_index, frame):
    values, support = [], []
    for r in range(3):
        for c in range(3):
            prefix = f"tile_{r}{c}_"
            vi, oi = name_to_index[prefix + "residual_p90"], name_to_index[prefix + "observed_fraction"]
            ok = bool(valid[frame, vi] and valid[frame, oi] and np.isfinite(signals[frame, vi]))
            values.append(float(max(signals[frame, vi], 0.0)) if ok else 0.0)
            support.append(ok)
    raw, support = np.asarray(values, dtype=np.float64), np.asarray(support, dtype=bool)
    return normalized_distribution(raw), support, float(raw[support].mean()) if support.any() else 0.0, raw


def extract_video_features(raw_path, local_path, frames, fps, local_names):
    with np.load(raw_path, allow_pickle=False) as a:
        frame_ids, timestamps = np.asarray(a["frame_ids"]), np.asarray(a["timestamps_sec"], dtype=np.float64)
        tubelets, keyframes = np.asarray(a["tubelet_frame_ids"]), np.asarray(a["keyframe_ids"])
        v_heat, v_mask, v_obs = np.asarray(a["vjepa_raw_heatmaps"]), np.asarray(a["vjepa_patch_valid_mask"], bool), np.asarray(a["vjepa_valid_mask"], bool)
        i_heat, i_mask, i_obs = np.asarray(a["ijepa_raw_heatmaps"]), np.asarray(a["ijepa_patch_valid_mask"], bool), np.asarray(a["ijepa_valid_mask"], bool)
        v_error, i_error = np.asarray(a["vjepa_raw_errors"], dtype=np.float64), np.asarray(a["ijepa_raw_errors"], dtype=np.float64)
    require(np.array_equal(frame_ids, np.arange(frames)) and np.allclose(timestamps, np.arange(frames) / fps, rtol=0, atol=1e-6), "raw frame/FPS schema mismatch")
    require(v_heat.ndim == 3 and v_mask.shape == v_heat.shape and len(v_heat) == len(tubelets) == len(v_obs), "V raw spatial schema mismatch")
    require(i_heat.ndim == 3 and i_mask.shape == i_heat.shape and len(i_heat) == len(keyframes) == len(i_obs), "I raw spatial schema mismatch")
    require(v_obs.all() and i_obs.all() and np.isfinite(v_error).all() and np.isfinite(i_error).all(), "raw anchor coverage/error schema mismatch")
    v_centers, i_centers = tubelets.astype(np.float64).mean(axis=1), keyframes.astype(np.float64).reshape(-1)
    require(np.all(np.diff(v_centers) > 0) and np.all(np.diff(i_centers) > 0), "anchor order mismatch")
    v_data = [heat_distribution(h, m) for h, m in zip(v_heat, v_mask)]
    i_data = [heat_distribution(h, m) for h, m in zip(i_heat, i_mask)]
    v_dist, i_dist = np.stack([x[0] for x in v_data]), np.stack([x[0] for x in i_data])
    v_entropy = np.asarray([x[3] for x in v_data], dtype=np.float64)
    v_strength = np.asarray([x[4] for x in v_data], dtype=np.float64)
    v_top10 = []
    for heat, mask in zip(v_heat, v_mask):
        valid = np.asarray(mask, bool) & np.isfinite(heat)
        vals = np.maximum(np.asarray(heat, dtype=np.float64)[valid], 0.0)
        if not len(vals) or vals.sum() <= 1e-12:
            v_top10.append(0.0)
        else:
            order = np.argsort(vals)[::-1]
            v_top10.append(float(vals[order[:max(1, int(math.ceil(.10 * len(vals))))]].sum() / vals.sum()))
    v_persist, v_iou, v_valid = np.zeros(len(v_dist)), np.zeros(len(v_dist)), np.zeros(len(v_dist), bool)
    for k in range(1, len(v_dist)):
        v_persist[k], v_iou[k], v_valid[k] = cosine(v_dist[k], v_dist[k - 1]), top_iou(v_dist[k], v_dist[k - 1]), True
    i_persist, i_iou, i_valid = np.zeros(len(i_dist)), np.zeros(len(i_dist)), np.zeros(len(i_dist), bool)
    for k in range(1, len(i_dist)):
        i_persist[k], i_iou[k], i_valid[k] = cosine(i_dist[k], i_dist[k - 1]), top_iou(i_dist[k], i_dist[k - 1]), True
    signals, motion_valid, name_to_index = load_local_motion(local_path, frames, fps, local_names)
    for r in range(3):
        for c in range(3):
            for suffix in ("residual_p90", "observed_fraction"):
                require(f"tile_{r}{c}_{suffix}" in name_to_index, "local tile schema incomplete")
    values = {name: np.zeros(len(v_dist), dtype=np.float64) for name in FEATURES}
    feature_valid = {name: np.ones(len(v_dist), dtype=bool) for name in FEATURES}
    for name in ("v_persistence_cos", "v_top3_persistence_iou", "v_spatial_change"):
        feature_valid[name] = v_valid.copy()
    for k, center in enumerate(v_centers):
        j = int(np.argmin(np.abs(i_centers - center)))
        values["v_persistence_cos"][k], values["v_top3_persistence_iou"][k] = v_persist[k], v_iou[k]
        values["v_spatial_change"][k] = 1.0 - v_persist[k]
        values["i_persistence_cos"][k], values["i_top3_persistence_iou"][k] = i_persist[j], i_iou[j]
        values["i_spatial_change"][k] = 1.0 - i_persist[j]
        for name in ("i_persistence_cos", "i_top3_persistence_iou", "i_spatial_change"):
            feature_valid[name][k] = bool(i_valid[j])
        vi_cos = cosine(v_dist[k], i_dist[j])
        values["vi_colocation_cos"][k], values["vi_top3_colocation_iou"][k] = vi_cos, top_iou(v_dist[k], i_dist[j])
        frame = int(np.clip(round(center), 0, frames - 1))
        motion_dist, motion_support, motion_mean, motion_raw = local_motion_map(signals, motion_valid, name_to_index, frame)
        values["motion_colocation_cos"][k] = cosine(v_dist[k], motion_dist)
        values["motion_colocation_gain"][k] = float(np.dot(v_dist[k][motion_support], motion_raw[motion_support]) / motion_mean) if motion_support.any() and motion_mean > 1e-12 else 0.0
        values["joint_persistent_colocation"][k] = v_persist[k] * vi_cos
        values["v_top3_strength_z"][k] = v_strength[k]
        values["v_strength_change"][k] = v_strength[k] * (1.0 - v_persist[k])
        values["v_strength_motion_gain"][k] = v_strength[k] * values["motion_colocation_gain"][k]
        values["v_strength_vi_colocation"][k] = v_strength[k] * vi_cos
    for name in FEATURES:
        require(np.isfinite(values[name]).all(), f"nonfinite local feature: {name}")
    return {"v_centers": v_centers, "i_centers": i_centers, "v_tubelets": tubelets.astype(int), "v_error": v_error,
            "features": values, "feature_valid": feature_valid, "v_top10": np.asarray(v_top10), "v_entropy": v_entropy, "v_strength": v_strength}


def parse_event_anchors(record, start, end):
    tubelets = np.asarray(record["v_tubelets"])
    return [int(k) for k, pair in enumerate(tubelets) if np.any((pair >= start) & (pair <= end))]


def valid_values(record, name, indices):
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return np.empty(0, dtype=np.float64)
    values, valid = record["features"][name][idx], record["feature_valid"][name][idx]
    return values[valid & np.isfinite(values)]


def safe_max(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if not len(values) else float(values.max())


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if not len(values) else float(values.mean())


def summarize_feature(name, records, events, negative_values, normal_values, category=None):
    selected = [e for e in events if category is None or e["category"] == category]
    pos_anchor, peaks, mins = [], [], []
    covered = 0
    for e in selected:
        vals = valid_values(records[e["index"]], name, e["v_anchor_indices"])
        if len(vals):
            covered += 1; pos_anchor.extend(vals.tolist()); peaks.append(float(vals.max())); mins.append(float(vals.min()))
    pos_anchor, peaks, mins = np.asarray(pos_anchor), np.asarray(peaks), np.asarray(mins)
    neg, normal = negative_values[name], normal_values[name]
    high_auc = auc_high(pos_anchor, neg)
    low_auc = None if high_auc is None else 1.0 - high_auc
    two_auc = None if high_auc is None else max(high_auc, low_auc)
    q05 = float(np.quantile(normal, .05)) if len(normal) else None
    q95 = float(np.quantile(normal, .95)) if len(normal) else None
    normal_video_peaks = []
    for r in records:
        if r["is_normal"]:
            vals = valid_values(r, name, range(len(r["v_centers"])))
            if len(vals): normal_video_peaks.append(float(vals.max()))
    normal_median = float(np.median(normal_video_peaks)) if normal_video_peaks else None
    corr_global, corr_top10, corr_entropy = [], [], []
    for r in records:
        idx = np.flatnonzero(r["feature_valid"][name]); vals = r["features"][name][idx]
        corr_global.append(finite_corr(vals, r["v_error"][idx])); corr_top10.append(finite_corr(vals, r["v_top10"][idx])); corr_entropy.append(finite_corr(vals, r["v_entropy"][idx]))
    all_corr = [x for x in corr_global + corr_top10 + corr_entropy if x is not None]
    return {
        "feature": name, "category": category or "all", "event_count": len(selected),
        "direct_v_anchor_coverage": float(covered / len(selected)) if selected else None,
        "positive_anchor_count": int(len(pos_anchor)), "negative_anchor_count": int(len(neg)), "normal_anchor_count": int(len(normal)),
        "auc_high": high_auc, "auc_low": low_auc, "two_sided_anchor_auc": two_auc,
        "normal_anchor_q05": q05, "normal_anchor_q95": q95,
        "event_peak_recall_at_normal_q95": float(np.mean(peaks >= q95)) if len(peaks) and q95 is not None else None,
        "event_min_recall_at_normal_q05": float(np.mean(mins <= q05)) if len(mins) and q05 is not None else None,
        "normal_video_peak_median": normal_median,
        "event_peak_beats_normal_video_median": float(np.mean(peaks >= normal_median)) if len(peaks) and normal_median is not None else None,
        "max_absolute_correlation_with_existing_global_or_spatial": max((abs(x) for x in all_corr), default=None),
        "correlation_with_v_raw_error_max_abs": max((abs(x) for x in corr_global if x is not None), default=None),
        "correlation_with_existing_top10_max_abs": max((abs(x) for x in corr_top10 if x is not None), default=None),
        "correlation_with_existing_entropy_max_abs": max((abs(x) for x in corr_entropy if x is not None), default=None),
    }


def feature_passes(summary, low_summary):
    return bool(
        summary.get("direct_v_anchor_coverage") is not None and summary["direct_v_anchor_coverage"] >= CAPACITY_RULE["minimum_direct_v_anchor_coverage"]
        and summary.get("two_sided_anchor_auc") is not None and summary["two_sided_anchor_auc"] >= CAPACITY_RULE["minimum_two_sided_anchor_auc"]
        and low_summary.get("two_sided_anchor_auc") is not None and low_summary["two_sided_anchor_auc"] >= CAPACITY_RULE["minimum_low_evidence_stage_auc"]
        and summary.get("max_absolute_correlation_with_existing_global_or_spatial") is not None and summary["max_absolute_correlation_with_existing_global_or_spatial"] <= CAPACITY_RULE["maximum_absolute_correlation_with_existing_global_or_spatial"]
        and max(summary.get("event_peak_recall_at_normal_q95") or 0.0, summary.get("event_min_recall_at_normal_q05") or 0.0) >= 0.20
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--dataset-npz", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve(); output = args.output if args.output.is_absolute() else root / args.output; output = output.resolve()
    require(not output.exists(), "output already exists; choose a new directory"); output.mkdir(parents=True, exist_ok=False)
    dataset_manifest_path = args.dataset_manifest if args.dataset_manifest.is_absolute() else root / args.dataset_manifest
    dataset_npz_path = args.dataset_npz if args.dataset_npz.is_absolute() else root / args.dataset_npz
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    local_root = args.local_root if args.local_root.is_absolute() else root / args.local_root
    ledger_root = args.ledger_root if args.ledger_root.is_absolute() else root / args.ledger_root
    raw_manifest_path, local_manifest_path = raw_root / "manifest.json", local_root / "manifest.json"
    event_csv, video_csv = ledger_root / "events.csv", ledger_root / "videos.csv"
    for path in (dataset_manifest_path, dataset_npz_path, raw_manifest_path, local_manifest_path, event_csv, video_csv): require(path.is_file(), f"missing input: {path}")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8")); raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8")); local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    rows = dataset_manifest["rows"]; require(len(rows) == 77, f"unexpected development row count: {len(rows)}")
    raw_entries = {e.get("name", e.get("video_name")): e for e in raw_manifest["videos"]}
    local_entries = {}
    for entry in local_manifest["videos"]:
        row_id = int(entry["row_id"])
        require(0 <= row_id < len(rows), "local motion row id outside dataset")
        require(entry.get("source_sha256") == rows[row_id]["sha256"], "local motion source SHA differs from dataset")
        name = rows[row_id]["name"]
        require(name not in local_entries, "duplicate local motion row")
        local_entries[name] = entry
    require(set(raw_entries) == {r["name"] for r in rows} and set(local_entries) == {r["name"] for r in rows}, "feature cache names differ from dataset")
    local_names = [str(x) for x in json.loads((local_root / "feature_names.json").read_text(encoding="utf-8"))]
    inventory_paths = [dataset_manifest_path, dataset_npz_path, raw_manifest_path, local_manifest_path, event_csv, video_csv, Path(__file__).resolve()]
    inventory = {"schema_version": "local-interaction-input-inventory-v1", "files": [{"path": str(p), "sha256": digest(p)} for p in inventory_paths], "feature_files": []}
    for name, e in sorted(raw_entries.items()):
        feature_p = raw_root / e["file"]
        raw_p = raw_root / e["raw_directory"] / "signals.npz"
        feature_h, raw_h = digest(feature_p), digest(raw_p)
        require(not e.get("feature_sha256") or feature_h == e["feature_sha256"], f"raw feature hash mismatch: {name}")
        inventory["feature_files"].append({"role": "raw_spatial_feature", "name": name, "path": str(feature_p), "sha256": feature_h})
        inventory["feature_files"].append({"role": "raw_spatial_heatmaps", "name": name, "path": str(raw_p), "sha256": raw_h})
    for name, e in sorted(local_entries.items()):
        p = local_root / e["file"]; h = digest(p); require(not e.get("feature_sha256") or h == e["feature_sha256"], f"local feature hash mismatch: {name}"); inventory["feature_files"].append({"role": "local_motion", "name": name, "path": str(p), "sha256": h})
    inventory["inventory_sha256"] = value_hash(inventory); write_json(output / "input_inventory.json", inventory)
    event_rows, video_rows = load_csv(event_csv), load_csv(video_csv); require(len(event_rows) == 85 and len(video_rows) == 77, "unexpected ledger row counts")
    event_lookup = {(int(e["index"]), int(e["event"])): e for e in event_rows}; require(len(event_lookup) == 85, "duplicate event ledger key")
    with np.load(dataset_npz_path, allow_pickle=False) as data: labels = [np.asarray(data[f"labels_{i}"]).astype(bool) for i in range(len(rows))]
    records = []
    for i, row in enumerate(rows):
        require(len(labels[i]) == int(row["frames"]), f"label/frame mismatch: {row['name']}"); spans = parse_spans(labels[i]); require(len(spans) == int(row["event_count"]), f"event count mismatch: {row['name']}")
        raw_e, local_e = raw_entries[row["name"]], local_entries[row["name"]]
        raw_p, local_p = raw_root / raw_e["raw_directory"] / "signals.npz", local_root / local_e["file"]
        record = extract_video_features(raw_p, local_p, int(row["frames"]), float(row["fps"]), local_names)
        record.update({"index": i, "name": row["name"], "sha256": row["sha256"], "labels": labels[i], "spans": spans, "is_normal": not spans}); records.append(record)
    events = []
    for r in records:
        for event_id, (start, end) in enumerate(r["spans"]):
            ledger = event_lookup[(r["index"], event_id)]; matched = ledger["default_control_matched_iou_0_5"].lower() == "true"
            events.append({"index": r["index"], "name": r["name"], "event": event_id, "start_frame": start, "end_frame": end, "duration_frames": end - start + 1, "category": "matched_iou_0_5" if matched else ledger["default_control_diagnostic_stage"], "stage": ledger["default_control_diagnostic_stage"], "baseline_matched_iou_0_5": matched, "v_anchor_indices": parse_event_anchors(r, start, end)})
    require(len(events) == 85, "event construction count mismatch")
    negative_indices = {r["index"]: [k for k in range(len(r["v_centers"])) if not any(k in e["v_anchor_indices"] for e in events if e["index"] == r["index"])] for r in records}
    negative_values = {name: np.asarray([float(r["features"][name][k]) for r in records for k in negative_indices[r["index"]] if r["feature_valid"][name][k]]) for name in FEATURES}
    normal_values = {name: np.asarray([float(r["features"][name][k]) for r in records if r["is_normal"] for k in range(len(r["v_centers"])) if r["feature_valid"][name][k]]) for name in FEATURES}
    categories = sorted({e["category"] for e in events}); summaries = []
    for name in FEATURES:
        summaries.append(summarize_feature(name, records, events, negative_values, normal_values))
        for category in categories: summaries.append(summarize_feature(name, records, events, negative_values, normal_values, category))
    by_key = {(s["feature"], s["category"]): s for s in summaries}; low_category = "frame_evidence_below_high_seed"; qualifying = []
    for name in FEATURES:
        if feature_passes(by_key[(name, "all")], by_key.get((name, low_category), {"two_sided_anchor_auc": None})): qualifying.append(name)
    qualifying_families = sorted({family for family, members in FAMILIES.items() if any(x in qualifying for x in members)}); direct_coverage = float(np.mean([bool(e["v_anchor_indices"]) for e in events]))
    capacity_supported = bool(direct_coverage >= CAPACITY_RULE["minimum_direct_v_anchor_coverage"] and len(qualifying) >= CAPACITY_RULE["minimum_qualifying_features"] and len(qualifying_families) >= CAPACITY_RULE["minimum_qualifying_relation_families"])
    npz_arrays = {}
    for r in records:
        i = r["index"]; npz_arrays[f"v_centers_{i}"] = r["v_centers"].astype(np.float32); npz_arrays[f"v_error_{i}"] = r["v_error"].astype(np.float32)
        for name in FEATURES: npz_arrays[f"feature_{name}_{i}"] = r["features"][name].astype(np.float32); npz_arrays[f"valid_{name}_{i}"] = r["feature_valid"][name].astype(np.uint8)
    np.savez_compressed(output / "anchor_features.npz", **npz_arrays)
    event_out = []
    for e in events:
        r = records[e["index"]]; row = dict(e)
        for name in FEATURES:
            vals = valid_values(r, name, e["v_anchor_indices"]); row[name + "_peak"], row[name + "_mean"] = safe_max(vals), safe_mean(vals)
        row["v_raw_error_peak"] = safe_max(r["v_error"][e["v_anchor_indices"]]) if e["v_anchor_indices"] else None; event_out.append(row)
    write_json(output / "event_evidence.json", event_out); write_json(output / "feature_summaries.json", summaries)
    video_out = []
    for r in records:
        row = {"index": r["index"], "name": r["name"], "is_normal": r["is_normal"], "event_count": len(r["spans"]), "v_anchor_count": len(r["v_centers"])}
        for name in FEATURES:
            vals = valid_values(r, name, range(len(r["v_centers"]))); row[name + "_video_peak"], row[name + "_video_mean"] = safe_max(vals), safe_mean(vals)
        video_out.append(row)
    write_json(output / "video_evidence.json", video_out)
    protocol = {"schema_version": "local-interaction-capacity-protocol-v2", "role": "read_only_capacity_diagnostic_not_training_or_selection", "created_at": "2026-10-03", "dataset": {"rows": len(rows), "events": len(events), "normal_videos": sum(r["is_normal"] for r in records), "content_groups": len({r["sha256"] for r in rows})}, "anchor_policy": {"primary": "V-JEPA tubelet centers", "event_membership": "tubelet intersects inclusive GT interval", "no_dense_interpolation": True, "no_label_in_feature_transform": True}, "feature_families": FAMILIES, "features": list(FEATURES), "capacity_rule": CAPACITY_RULE, "input_inventory_sha256": inventory["inventory_sha256"], "limitations": ["Repeated development set, not an independent test.", "AUC and fixed normal-tail comparisons use labels for diagnosis only; no estimator is fitted.", "A local relation signal is not evidence that a trained downstream head will improve OOF metrics.", "Exact SHA grouping does not establish near-duplicate or source isolation."]}; protocol["protocol_sha256"] = value_hash(protocol); write_json(output / "protocol.json", protocol)
    report = {"schema_version": "local-interaction-capacity-report-v2", "status": "complete", "role": "read_only_capacity_diagnostic_not_training_or_selection", "protocol_sha256": protocol["protocol_sha256"], "input_inventory_sha256": inventory["inventory_sha256"], "event_category_counts": {c: sum(e["category"] == c for e in events) for c in categories}, "direct_v_anchor_coverage": direct_coverage, "qualifying_features": qualifying, "qualifying_relation_families": qualifying_families, "capacity_supported": capacity_supported, "training_started": False, "default_model_modified": False, "feature_summaries": summaries}; report["report_sha256"] = value_hash(report); write_json(output / "report.json", report)
    decision = {"schema_version": "local-interaction-capacity-decision-v2", "status": "capacity_supported_continue_to_single_oof_comparison" if capacity_supported else "capacity_not_supported_stop_before_training", "capacity_supported": capacity_supported, "training_started": False, "default_promoted": False, "default_model_modified": False, "qualifying_features": qualifying, "qualifying_relation_families": qualifying_families, "report_sha256": report["report_sha256"], "protocol_sha256": protocol["protocol_sha256"]}; decision["decision_sha256"] = value_hash(decision); write_json(output / "decision.json", decision)
    print(json.dumps({"status": "complete", "output": str(output), "capacity_supported": capacity_supported, "qualifying_features": qualifying, "qualifying_families": qualifying_families, "direct_v_anchor_coverage": direct_coverage}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
