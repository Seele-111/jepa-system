#!/usr/bin/env python3
"""Amended read-only native-patch local capacity screen; never trains a model.
Freeze source, inputs and protocol before calculating corrected statistics.
The development set and run2/run3 were seen: NOT fresh-data preregistration.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/algorithm-opt-2026-10-02/dataset.npz"
DATA_MANIFEST = ROOT / "output/algorithm-opt-2026-10-02-v2/dataset_manifest.json"
RAW_ROOT = ROOT / "output/algorithm-opt-2026-10-02/corrected_jepa_features_v2"
LOCAL_ROOT = ROOT / "output/algorithm-opt-2026-10-02-v2/local_motion"
LEDGER = ROOT / "output/baseline-targeted-semantic-readout/event-change-ledger"
SOURCES = Path("C:/Users/admin/Desktop/测试")
PRIOR = ROOT / "output/baseline-targeted-semantic-readout/final_delivery_manifest.json"
EARLY = ROOT / "output/baseline-targeted-local-interaction-run3-20261003/input_inventory.json"
FEATURE_FAMILIES = {
    "v_patch_change": "temporal_local",
    "v_strength_change": "temporal_local",
    "v_strength_persistence": "temporal_local",
    "vi_topk_overlap": "cross_modal",
    "vi_strength_colocation": "cross_modal",
    "motion_topk_gain": "coarse_motion",
    "motion_persistent_gain": "joint_relation",
    "vi_motion_joint": "joint_relation",
}
FEATURES = tuple(FEATURE_FAMILIES)
REPRESENTATIVES = {"temporal_local": "v_strength_change", "cross_modal": "vi_strength_colocation", "coarse_motion": "motion_topk_gain"}
LOW = "frame_evidence_below_high_seed"
GATED = "video_attenuation_drops_event_peak_below_seed"
BOUNDARY = "surviving_overlap_needs_boundary_or_structure"
RULE = {
    "minimum_event_valid_coverage_all_and_target": 0.8,
    "minimum_event_anchor_valid_fraction_all_and_target": 0.8,
    "minimum_normal_anchor_valid_fraction": 0.8,
    "minimum_overall_high_direction_group_auc": 0.55,
    "minimum_target_high_direction_group_auc": 0.60,
    "minimum_target_group_bootstrap_ci_lower_strict": 0.50,
    "minimum_target_within_video_auc": 0.55,
    "minimum_target_within_video_available_fraction": 0.50,
    "minimum_target_strict_normal_q95_event_recall": 0.20,
    "minimum_qualifying_features": 2,
    "minimum_qualifying_families": 2,
    "target_categories": [LOW, GATED],
}
PARAMETERS = {"top_fraction": 0.05, "minimum_common_patches": 32,
              "minimum_projected_area_coverage": 0.50, "normal_tail_quantile": 0.95,
              "bootstrap_replicates": 2000, "bootstrap_seed": 20261003,
              "boundary_neighbor_count": 2, "gate_one_sided_familywise_alpha": 0.05,
              "gate_representative_target_tests": 6}
PROTECTED = {
    "models/optimized_locator_v1.json": "093c14ec9b2ffc91dea776f694a8278bbd2f4d66a1881fe10d8399fc24301dbd",
    "models/optimized_motion_locator_v1.json": "77e5d7eeb63653c6ddc0ea33da8e455c5b737d46f87c1cfde96c43a1381d5670",
    "code/optimized_detector.py": "899e3b5128e687a9be7e27702db91487ef3319154b713fdf594331cc2a0c9ea7",
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def spans(labels):
    changes = np.diff(np.r_[False, np.asarray(labels, bool), False].astype(np.int8))
    return [(int(s), int(e - 1)) for s, e in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]

def crop_geometry(height, width, model):
    """Source normalized pixel-edge crop, including actual integer rounding."""
    require(min(height, width) > 0, "invalid source geometry")
    if model == "v":
        short, crop = int(256 / 224 * 384), 384
        scale = short / min(height, width)
        nh, nw = int(height * scale), int(width * scale)
        top, left = (nh - crop) // 2, (nw - crop) // 2
    elif model == "i":
        crop, short = 224, 256
        if height <= width:
            nh, nw = short, int(short * width / height)
        else:
            nh, nw = int(short * height / width), short
        top, left = int(round((nh - crop) / 2)), int(round((nw - crop) / 2))
    else:
        raise ValueError("unknown crop model")
    return {"resized_height": nh, "resized_width": nw, "crop_top": top, "crop_left": left,
            "crop_size": crop, "box_xyxy": [left / nw, top / nh, (left + crop) / nw, (top + crop) / nh]}

def patch_rectangles(box, grid):
    x0, y0, x1, y1 = box
    x, y = np.linspace(x0, x1, grid + 1), np.linspace(y0, y1, grid + 1)
    return np.asarray([[x[c], y[r], x[c + 1], y[r + 1]] for r in range(grid) for c in range(grid)])

def motion_rectangles(height, width):
    # floor(3*x/w) tiles, actual analysis rounding and four-pixel flow border.
    scale = min(1.0, 128 / max(height, width))
    ah, aw = max(1, round(height * scale)), max(1, round(width * scale))
    rects = []
    for r in range(3):
        for c in range(3):
            left, right = max(4, math.ceil(c * aw / 3)), min(aw - 4, math.ceil((c + 1) * aw / 3))
            top, bottom = max(4, math.ceil(r * ah / 3)), min(ah - 4, math.ceil((r + 1) * ah / 3))
            rects.append([left / aw, top / ah, max(left, right) / aw, max(top, bottom) / ah])
    return np.asarray(rects), [ah, aw]

def overlap_weights(target, source):
    lower = np.maximum(target[:, None, :2], source[None, :, :2])
    upper = np.minimum(target[:, None, 2:], source[None, :, 2:])
    areas = np.prod(np.maximum(upper - lower, 0), axis=2)
    target_areas = np.prod(target[:, 2:] - target[:, :2], axis=1)
    require((target_areas > 0).all(), "empty target patch")
    return areas / target_areas[:, None]

def project(values, valid, weights):
    values, valid = np.asarray(values, float).reshape(-1), np.asarray(valid, bool).reshape(-1)
    valid = valid & np.isfinite(values)
    area = weights[:, valid].sum(axis=1)
    output = np.zeros(len(weights))
    nonzero = area > 1e-12
    if valid.any():
        output[nonzero] = (weights[:, valid] @ values[valid])[nonzero] / area[nonzero]
    return output, area >= PARAMETERS["minimum_projected_area_coverage"], area

def positive_strength(values):
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-8)
    z = np.maximum(np.asarray(values, float) - median, 0) / scale
    k = max(1, math.ceil(len(z) * PARAMETERS["top_fraction"]))
    return z, float(np.sort(z)[-k:].mean())

def local_relation(a, b, mask):
    """Normalize both operands on identical actually observed support."""
    mask = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < PARAMETERS["minimum_common_patches"]:
        return None
    a, b = np.asarray(a, float)[mask], np.asarray(b, float)[mask]
    za, sa = positive_strength(a)
    zb, sb = positive_strength(b)
    n = max(1, math.ceil(len(a) * PARAMETERS["top_fraction"]))
    # Constant maps provide genuine zero, not arbitrary argsort peaks.
    if min(sa, sb) <= 1e-12:
        return {"cos": 0.0, "iou": 0.0, "a_strength": sa, "b_strength": sb, "common_count": len(a)}
    ta, tb = np.zeros(len(a), bool), np.zeros(len(b), bool)
    ta[np.argsort(-za, kind="stable")[:n]] = True
    tb[np.argsort(-zb, kind="stable")[:n]] = True
    norm = float(np.linalg.norm(za) * np.linalg.norm(zb))
    return {"cos": float(np.dot(za, zb) / norm) if norm > 1e-12 else 0.0,
            "iou": float((ta & tb).sum() / (ta | tb).sum()),
            "a_strength": sa, "b_strength": sb, "common_count": len(a)}

def temporal_features(now, previous, valid_now, valid_previous):
    rel = local_relation(now, previous, valid_now & valid_previous)
    if rel is None:
        return None
    if min(rel["a_strength"], rel["b_strength"]) <= 1e-12:
        return dict.fromkeys(FEATURES[:3], 0.0)
    return {"v_patch_change": 1 - rel["cos"],
            "v_strength_change": rel["a_strength"] * (1 - rel["cos"]),
            "v_strength_persistence": min(rel["a_strength"], rel["b_strength"]) * rel["iou"]}

def cross_features(v, i, mask):
    rel = local_relation(v, i, mask)
    if rel is None:
        return None
    return {"vi_topk_overlap": rel["iou"],
            "vi_strength_colocation": math.sqrt(rel["a_strength"] * rel["b_strength"]) * rel["cos"]}

def motion_gain(v, motion, mask):
    mask = mask & np.isfinite(v) & np.isfinite(motion)
    if mask.sum() < PARAMETERS["minimum_common_patches"]:
        return None
    v, motion = np.asarray(v)[mask], np.asarray(motion)[mask]
    z, strength = positive_strength(v)
    if strength <= 1e-12 or float(motion.mean()) <= 1e-12:
        return 0.0
    k = max(1, math.ceil(len(v) * PARAMETERS["top_fraction"]))
    return float(motion[np.argsort(-z, kind="stable")[:k]].mean() / motion.mean())

def extract(record, names):
    with np.load(record["raw_path"], allow_pickle=False) as raw:
        t, keyframes = np.asarray(raw["tubelet_frame_ids"], int), np.asarray(raw["keyframe_ids"], int)
        v, i = np.asarray(raw["vjepa_raw_heatmaps"], float), np.asarray(raw["ijepa_raw_heatmaps"], float)
        vm = np.asarray(raw["vjepa_patch_valid_mask"], bool) & (raw["vjepa_patch_counts"] > 0)
        im = np.asarray(raw["ijepa_patch_valid_mask"], bool) & (raw["ijepa_patch_counts"] > 0)
        vm &= np.asarray(raw["vjepa_valid_mask"], bool)[:, None, None] & np.isfinite(v)
        im &= np.asarray(raw["ijepa_valid_mask"], bool)[:, None, None] & np.isfinite(i)
        error = np.asarray(raw["vjepa_raw_errors"], float)
        require(np.array_equal(raw["frame_ids"], np.arange(record["frames"])), "raw timeline mismatch")
        require(np.allclose(raw["timestamps_sec"], np.arange(record["frames"]) / record["fps"], atol=1e-6, rtol=0), "raw FPS mismatch")
    require(v.shape == vm.shape == (len(t), 24, 24), "V spatial schema mismatch")
    require(i.shape == im.shape == (len(keyframes), 14, 14), "I spatial schema mismatch")
    require(t.shape == (len(v), 2) and np.all(np.diff(t.mean(axis=1)) > 0), "tubelet schema mismatch")
    require(np.all(np.diff(keyframes) > 0), "keyframe order mismatch")
    with np.load(record["local_path"], allow_pickle=False) as local:
        signals, valid = np.asarray(local["signals"], float), np.asarray(local["feature_valid"], bool)
        require(signals.shape == valid.shape == (record["frames"], len(names)), "motion shape mismatch")
        require(list(local["feature_names"]) == names and np.array_equal(local["frame_ids"], np.arange(record["frames"])), "motion schema mismatch")
        require(abs(float(local["fps"]) - record["fps"]) < 1e-6, "motion FPS mismatch")
    idx = {name: n for n, name in enumerate(names)}
    rect_v = patch_rectangles(record["geometry"]["v"]["box_xyxy"], 24)
    rect_i = patch_rectangles(record["geometry"]["i"]["box_xyxy"], 14)
    rect_motion, analysis_shape = motion_rectangles(record["height"], record["width"])
    require(analysis_shape == record["geometry"]["motion_analysis_shape"], "motion geometry mismatch")
    wi, wm = overlap_weights(rect_v, rect_i), overlap_weights(rect_v, rect_motion)
    vv, ii = v.reshape(len(v), -1), i.reshape(len(i), -1)
    vvalid, ivalid = vm.reshape(len(v), -1), im.reshape(len(i), -1)
    values, supported = np.zeros((len(v), len(FEATURES))), np.zeros((len(v), len(FEATURES)), bool)
    support_info = []
    def put(k, result):
        if result is not None:
            for name, value in result.items():
                n = FEATURES.index(name)
                require(math.isfinite(value), "nonfinite feature")
                values[k, n], supported[k, n] = value, True
    centers = t.mean(axis=1)
    for k, center in enumerate(centers):
        temporal = temporal_features(vv[k], vv[k - 1], vvalid[k], vvalid[k - 1]) if k else None
        put(k, temporal)
        j = int(np.argmin(np.abs(keyframes - center)))
        ip, ipvalid, _ = project(ii[j], ivalid[j], wi)
        joint = cross_features(vv[k], ip, vvalid[k] & ipvalid)
        put(k, joint)
        frame = int(np.clip(round(center), 0, record["frames"] - 1))
        motion, motion_ok = np.zeros(9), np.zeros(9, bool)
        for tile in range(9):
            p = f"tile_{tile // 3}{tile % 3}_"
            mi, oi = idx[p + "residual_p90"], idx[p + "observed_fraction"]
            motion_ok[tile] = valid[frame, mi] and valid[frame, oi] and signals[frame, oi] > 0 and np.isfinite(signals[frame, mi])
            if motion_ok[tile]:
                motion[tile] = max(signals[frame, mi], 0)
        mp, mpvalid, _ = project(motion, motion_ok, wm)
        gain = motion_gain(vv[k], mp, vvalid[k] & mpvalid)
        if gain is not None:
            put(k, {"motion_topk_gain": gain})
            persistence_mask = vvalid[k] & vvalid[k - 1] & mpvalid if k else np.zeros_like(vvalid[k])
            rel = local_relation(vv[k], vv[k - 1], persistence_mask) if k else None
            persistent_gain = motion_gain(vv[k], mp, persistence_mask) if k else None
            if rel is not None and persistent_gain is not None:
                put(k, {"motion_persistent_gain": persistent_gain * rel["iou"]})
            triple_mask = vvalid[k] & ipvalid & mpvalid
            triple = cross_features(vv[k], ip, triple_mask)
            triple_gain = motion_gain(vv[k], mp, triple_mask)
            if triple is not None and triple_gain is not None:
                put(k, {"vi_motion_joint": triple_gain * triple["vi_topk_overlap"]})
        support_info.append({"anchor": k, "nearest_i_anchor": j,
                             "i_offset_seconds": float(abs(keyframes[j] - center) / record["fps"]),
                             "v_observed_patches": int(vvalid[k].sum()),
                             "temporal_common_patches": int((vvalid[k] & vvalid[k - 1]).sum()) if k else 0,
                             "vi_common_patches": int((vvalid[k] & ipvalid).sum()),
                             "motion_common_patches": int((vvalid[k] & mpvalid).sum()),
                             "motion_supported_tiles": int(motion_ok.sum())})
    return {"tubelets": t, "centers": centers, "values": values, "valid": supported,
            "v_error": error, "support_info": support_info}

def weighted_auc(pos, neg, pos_weight=None, neg_weight=None):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    pw = np.ones(len(pos)) if pos_weight is None else np.asarray(pos_weight, float)
    nw = np.ones(len(neg)) if neg_weight is None else np.asarray(neg_weight, float)
    require(np.isfinite(pos).all() and np.isfinite(neg).all() and (pw >= 0).all() and (nw >= 0).all(), "invalid AUC input")
    require(pw.sum() > 0 and nw.sum() > 0, "empty AUC mass")
    order = np.argsort(neg, kind="stable")
    neg, nw = neg[order], nw[order]
    mass = np.r_[0.0, np.cumsum(nw)]
    lower, upper = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float(np.dot(pw, mass[lower] + .5 * (mass[upper] - mass[lower])) / (pw.sum() * nw.sum()))

def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[min(len(values) - 1, np.searchsorted(np.cumsum(weights), q * weights.sum(), "left"))])

def group_auc_summary(positive_groups, normal_groups):
    pkeys, nkeys = sorted(positive_groups), sorted(normal_groups)
    if not pkeys or not nkeys:
        return {"auc": None, "ci95": None, "positive_content_groups": len(pkeys), "normal_content_groups": len(nkeys), "bonferroni_lower": None}
    matrix = np.asarray([[weighted_auc(positive_groups[p][0], normal_groups[n][0],
                                      positive_groups[p][1], normal_groups[n][1])
                          for n in nkeys] for p in pkeys])
    rng = np.random.default_rng(PARAMETERS["bootstrap_seed"])
    count = PARAMETERS["bootstrap_replicates"]
    wp = rng.multinomial(len(pkeys), np.full(len(pkeys), 1 / len(pkeys)), count) / len(pkeys)
    wn = rng.multinomial(len(nkeys), np.full(len(nkeys), 1 / len(nkeys)), count) / len(nkeys)
    boot = np.einsum("bi,ij,bj->b", wp, matrix, wn, optimize=True)
    return {"auc": float(matrix.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "positive_content_groups": len(pkeys), "normal_content_groups": len(nkeys),
            "bonferroni_lower": float(np.quantile(boot, PARAMETERS["gate_one_sided_familywise_alpha"] / PARAMETERS["gate_representative_target_tests"]))}

def pooled_groups(items):
    grouped = defaultdict(list)
    for key, scores in items:
        if len(scores):
            grouped[key].append(np.asarray(scores, float))
    output = {}
    for key, arrays in grouped.items():
        values = np.concatenate(arrays)
        weights = np.concatenate([np.full(len(a), 1 / (len(a) * len(arrays))) for a in arrays])
        output[key] = (values, weights)
    return output

def feature_values(r, feature, ids):
    ids = np.asarray(ids, int)
    return r["values"][ids, feature][r["valid"][ids, feature]] if len(ids) else np.empty(0)

def group_mean(pairs):
    grouped = defaultdict(list)
    for group, value in pairs:
        if value is not None:
            grouped[group].append(value)
    return float(np.mean([np.mean(x) for x in grouped.values()])) if grouped else None

def summarize(records, events, feature, category):
    chosen = [e for e in events if category == "all" or e["category"] == category]
    positives, normal, coverage, within, tail, ties = [], [], [], [], [], []
    event_valid_fraction, normal_valid_fraction = [], []
    for r in records:
        if r["is_normal"]:
            normal.append((r["sha256"], feature_values(r, feature, range(len(r["centers"])))))
            normal_valid_fraction.append((r["sha256"], float(r["valid"][:, feature].mean())))
    n_groups = pooled_groups(normal)
    if n_groups:
        nv = np.concatenate([v for v, w in n_groups.values()])
        nw = np.concatenate([w / len(n_groups) for v, w in n_groups.values()])
        q95 = weighted_quantile(nv, nw, PARAMETERS["normal_tail_quantile"])
        normal_tail_mass = float(nw[nv > q95].sum() / nw.sum())
        normal_tie_mass = float(nw[nv == q95].sum() / nw.sum())
    else:
        q95, normal_tail_mass, normal_tie_mass = None, None, None
    for e in chosen:
        r = records[e["index"]]
        vals = feature_values(r, feature, e["pure_anchor_indices"])
        positives.append((r["sha256"], vals))
        coverage.append((r["sha256"], float(len(vals) > 0)))
        event_valid_fraction.append((r["sha256"], len(vals) / len(e["pure_anchor_indices"]) if e["pure_anchor_indices"] else 0.0))
        if len(vals):
            bg = feature_values(r, feature, r["background_indices"])
            within.append((r["sha256"], weighted_auc(vals, bg)))
            if q95 is not None:
                tail.append((r["sha256"], float(np.max(vals) > q95)))
                ties.append((r["sha256"], float(np.max(vals) == q95)))
    result = group_auc_summary(pooled_groups(positives), n_groups)
    result.update({"feature": FEATURES[feature], "family": FEATURE_FAMILIES[FEATURES[feature]], "category": category,
                   "event_count": len(chosen), "valid_event_count": sum(len(s) > 0 for _, s in positives),
                   "event_valid_coverage_content_balanced": group_mean(coverage),
                   "event_anchor_valid_fraction_content_balanced": group_mean(event_valid_fraction),
                   "normal_anchor_valid_fraction_content_balanced": group_mean(normal_valid_fraction),
                   "within_video_auc_content_balanced": group_mean(within),
                   "within_video_available_events": sum(v is not None for _, v in within),
                   "within_video_available_fraction": sum(v is not None for _, v in within) / len(chosen) if chosen else None,
                   "strict_normal_q95_event_recall_content_balanced": group_mean(tail),
                   "event_peak_equals_q95_content_balanced": group_mean(ties), "normal_q95": q95,
                   "normal_strict_tail_mass": normal_tail_mass, "normal_tie_mass_at_q95": normal_tie_mass,
                   "normal_video_any_strict_tail_fraction": group_mean([(key, float(np.max(v) > q95)) for key, v in normal if len(v)]) if q95 is not None else None,
                   "direction": "high_fixed_before_repaired_scores",
                   "ci_role": "pointwise_group_bootstrap_diagnostic_not_multiple_testing_or_generalization_proof"})
    return result

def qualifies(overall, target):
    coverage = RULE["minimum_event_valid_coverage_all_and_target"]
    return bool(overall["event_valid_coverage_content_balanced"] is not None
                and overall["event_valid_coverage_content_balanced"] >= coverage
                and target["event_valid_coverage_content_balanced"] is not None
                and target["event_valid_coverage_content_balanced"] >= coverage
                and overall["auc"] is not None and overall["auc"] >= RULE["minimum_overall_high_direction_group_auc"]
                and target["auc"] is not None and target["auc"] >= RULE["minimum_target_high_direction_group_auc"]
                and target["bonferroni_lower"] is not None and target["bonferroni_lower"] > RULE["minimum_target_group_bootstrap_ci_lower_strict"]
                and overall["event_anchor_valid_fraction_content_balanced"] >= RULE["minimum_event_anchor_valid_fraction_all_and_target"]
                and target["event_anchor_valid_fraction_content_balanced"] >= RULE["minimum_event_anchor_valid_fraction_all_and_target"]
                and target["normal_anchor_valid_fraction_content_balanced"] >= RULE["minimum_normal_anchor_valid_fraction"]
                and target["within_video_auc_content_balanced"] is not None
                and target["within_video_auc_content_balanced"] >= RULE["minimum_target_within_video_auc"]
                and target["within_video_available_fraction"] >= RULE["minimum_target_within_video_available_fraction"]
                and target["strict_normal_q95_event_recall_content_balanced"] is not None
                and target["strict_normal_q95_event_recall_content_balanced"] >= RULE["minimum_target_strict_normal_q95_event_recall"])

def boundary_summary(records, events, feature):
    result = []
    for e in events:
        if e["category"] != BOUNDARY:
            continue
        r, inside = records[e["index"]], e["pure_anchor_indices"]
        before = [k for k in r["background_indices"] if r["tubelets"][k].max() < e["start_frame"]]
        after = [k for k in r["background_indices"] if r["tubelets"][k].min() > e["end_frame"]]
        count = PARAMETERS["boundary_neighbor_count"]
        vi, vo = feature_values(r, feature, inside[:count]), feature_values(r, feature, before[-count:])
        ve, va = feature_values(r, feature, inside[-count:]), feature_values(r, feature, after[:count])
        start = float(vi.mean() - vo.mean()) if len(vi) and len(vo) else None
        end = float(ve.mean() - va.mean()) if len(ve) and len(va) else None
        result.append({"index": e["index"], "event": e["event"], "start_contrast": start, "end_contrast": end,
                       "both_observed": start is not None and end is not None,
                       "both_correct_high_direction": start is not None and end is not None and start > 0 and end > 0})
    available = [r for r in result if r["both_observed"]]
    return {"feature": FEATURES[feature], "events": len(result), "both_observed": len(available),
            "both_correct_fraction_of_observed": sum(r["both_correct_high_direction"] for r in available) / len(available) if available else None,
            "role": "GT_boundary_contrast_diagnostic_not_a_boundary_detector", "rows": result}


def capacity_decision(summaries):
    lookup = {(s["feature"], s["category"]): s for s in summaries}
    qualified, targets = [], {}
    for category in RULE["target_categories"]:
        accepted = []
        for family, name in REPRESENTATIVES.items():
            if qualifies(lookup[(name, "all")], lookup[(name, category)]):
                accepted.append(family)
                qualified.append({"feature": name, "family": family, "target": category})
        targets[category] = {"qualifying_base_families": accepted,
                             "passes": len(accepted) >= RULE["minimum_qualifying_families"]}
    return {"capacity_screen_supported": any(t["passes"] for t in targets.values()),
            "qualifying": qualified, "target_decisions": targets}

def inventory_entry(path, role):
    return {"path": str(Path(path).resolve()), "role": role, "sha256": digest(path)}

def freeze(output, source_root):
    require(not output.exists(), "output must be new; no historical overwrite")
    for relative, expected in PROTECTED.items():
        require(digest(ROOT / relative) == expected, f"protected artifact changed: {relative}")
    rows = read_json(DATA_MANIFEST)["rows"]
    raw_entries = {e.get("name", e.get("video_name")): e for e in read_json(RAW_ROOT / "manifest.json")["videos"]}
    local_entries = {int(e["row_id"]): e for e in read_json(LOCAL_ROOT / "manifest.json")["videos"]}
    require(len(rows) == 77 and len(raw_entries) == 77 and len(local_entries) == 77, "inventory count mismatch")
    source_paths = [Path(__file__), ROOT / "code/test_local_interaction_capacity_repaired.py",
                    ROOT / "code/vjepa_predictor.py", ROOT / "code/ijepa_predictor.py",
                    ROOT / "code/optimized_local_motion.py", ROOT / "code/optimized_jepa_extractor.py"]
    files = [inventory_entry(p, "source") for p in source_paths]
    for p in [DATA, DATA_MANIFEST, RAW_ROOT / "manifest.json", LOCAL_ROOT / "manifest.json",
              LOCAL_ROOT / "feature_names.json", LEDGER / "events.csv", LEDGER / "videos.csv", PRIOR, EARLY]:
        files.append(inventory_entry(p, "input_metadata"))
    early_pins = {e["path"]: e["sha256"] for e in read_json(EARLY)["feature_files"]}
    prior_pins = read_json(PRIOR)["artifact_sha256"]
    import cv2
    geometry = []
    for index, row in enumerate(rows):
        e, m = raw_entries[row["name"]], local_entries[index]
        require(m["source_sha256"] == row["sha256"], "motion source mismatch")
        raw_path = RAW_ROOT / e["raw_directory"] / "signals.npz"
        meta_path = raw_path.with_suffix(".json")
        local_path, feature_path = LOCAL_ROOT / m["file"], RAW_ROOT / e["file"]
        source = source_root / row["name"]
        for path, role in [(raw_path, "raw_spatial"), (meta_path, "raw_metadata"), (local_path, "coarse_local_motion"),
                           (feature_path, "prior_dense_cache_integrity_only"), (source, "original_video")]:
            pin = inventory_entry(path, role)
            if role == "original_video":
                require(pin["sha256"] == row["sha256"], "original video source SHA mismatch")
            if str(path) in early_pins:
                require(pin["sha256"] == early_pins[str(path)], "cache changed since early diagnosis")
            if str(path) in prior_pins:
                require(pin["sha256"] == prior_pins[str(path)], "old delivery pin mismatch")
            files.append(pin)
        require(digest(feature_path) == e["feature_sha256"], "dense cache manifest pin mismatch")
        require(digest(local_path) == m["feature_sha256"], "motion cache manifest pin mismatch")
        metadata = read_json(meta_path)
        require(metadata["profile"]["vjepa_preprocessing"] == "BGR_to_RGB_then_legacy_short_side_resize_center_crop_384_ImageNet_norm", "V preprocessing differs")
        require(metadata["profile"]["ijepa_preprocessing"] == "legacy_BGR_to_RGB_resize_256_bicubic_center_crop_224_ImageNet_norm", "I preprocessing differs")
        cap = cv2.VideoCapture(str(source))
        try:
            ok, first = cap.read()
            require(ok, "video header decode failed")
            height, width = first.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frames"]) - 1)
            ok, last = cap.read()
            require(ok and last.shape[:2] == (height, width), "source endpoint geometry changed")
        finally:
            cap.release()
        vgeo, igeo = crop_geometry(height, width, "v"), crop_geometry(height, width, "i")
        rects, shape = motion_rectangles(height, width)
        box = vgeo["box_xyxy"]
        geometry.append({"index": index, "name": row["name"], "sha256": row["sha256"],
                         "height": height, "width": width, "v": vgeo, "i": igeo,
                         "motion_analysis_shape": shape, "motion_rectangles": rects.tolist(),
                         "v_source_area_fraction": (box[2] - box[0]) * (box[3] - box[1]),
                         "source_geometry_checked": "first_and_last_decoded_frames_not_full_frame_shape_scan"})
    protocol = {"schema_version": "amended-local-interaction-capacity-v3", "created_utc": datetime.now(timezone.utc).isoformat(),
                "role": "read_only_capacity_screen_not_training_not_product_validation",
                "amendment_reason": ["crop/full-frame coordinates differed", "unobserved cells were treated as zero evidence",
                                     "first-anchor interaction validity was wrong", "pooled/two-sided AUC and maximum-video correlation were unsafe stopping rules"],
                "prior_diagnoses_seen": ["run2", "run3"], "not_original_or_fresh_data_preregistration": True,
                "features": list(FEATURES), "feature_families": FEATURE_FAMILIES, "gate_representatives": REPRESENTATIVES,
                "parameters": PARAMETERS, "training_capacity_rule": RULE, "protected": PROTECTED,
                "feature_policy": {"labels_used_in_feature_transform": False, "direction": "higher fixed for all features and stages",
                                   "V_grid": [24, 24], "I_grid": [14, 14], "motion_grid": [3, 3],
                                   "spatial_registration": "source normalized pixel-edge boxes, observed-area projection onto native V patches",
                                   "motion_limit": "projection of nine tile p90 summaries; NOT recovered pixel motion",
                                   "temporal_limit": "nearest cached I anchor, record offset; no interpolation or object tracking",
                                   "missing": "invalid mask, never zero evidence; initial temporal/joint anchors invalid",
                                   "positives": "both decoded tubelet members inside the same event; mixed anchors excluded",
                                   "negative_background": "both members outside every event",
                                   "weighting": "equal SHA group, equal event within group, equal observed anchor within event; duplicate annotation aliases retained",
                                   "correlation": "not used as a redundancy rejection rule",
                                   "confidence": "pointwise group bootstrap CI shown; fixed six representative/target tests use .05/6 one-sided lower",
                                   "gate": "at least two distinct BASE families on the SAME target; joint products never count as independent families",
                                   "tail": "strict > normal anchor q95, disclose real anchor and video tail mass; NOT 5% video FPR control"},
                "scope": {"training_only_after_screen_and_audit": True, "one_OOF_comparison_only": True,
                          "change_defaults": False, "new_dependencies": False, "download_weights": False},
                "sources_checked": ["local legacy V preprocessing", "installed I preprocessing and torchvision functional source",
                                    "https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional.py",
                                    "https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.roc_auc_score.html",
                                    "https://openaccess.thecvf.com/content_ECCV_2018/papers/Tianwei_Lin_BSN_Boundary_Sensitive_ECCV_2018_paper.pdf"],
                "limitations": ["77 videos repeatedly used for development, not a blind test",
                                "SHA groups do not guarantee near-duplicate/source/scene isolation",
                                "native prediction-error maps are not object detections or anomaly likelihoods",
                                "sparse masks and centered crop may omit the erroneous object",
                                "these fixed features can fail even if a different local representation would work",
                                "amended data-dependent workflow and bootstrap intervals are not independent confirmatory evidence"],
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__}}
    output.mkdir(parents=True)
    (output / "source_snapshot").mkdir()
    for path in source_paths:
        (output / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    write_json(output / "input_inventory.json", {"files": files})
    write_json(output / "geometry_inventory.json", geometry)
    write_json(output / "protocol.json", protocol)
    write_json(output / "freeze_receipt.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": digest(output / "protocol.json"), "input_inventory_sha256": digest(output / "input_inventory.json"),
               "geometry_inventory_sha256": digest(output / "geometry_inventory.json"), "repaired_statistics_not_yet_calculated": True})
    print(json.dumps({"status": "frozen_before_repaired_scores", "output": str(output)}, ensure_ascii=False))

def run(output):
    require(not (output / "report.json").exists(), "refuse to overwrite completed report")
    receipt = read_json(output / "freeze_receipt.json")
    for name in ["protocol", "input_inventory", "geometry_inventory"]:
        require(digest(output / f"{name}.json") == receipt[f"{name}_sha256"], "freeze receipt mismatch")
    protocol = read_json(output / "protocol.json")
    require(protocol["features"] == list(FEATURES) and protocol["parameters"] == PARAMETERS
            and protocol["training_capacity_rule"] == RULE and protocol["gate_representatives"] == REPRESENTATIVES, "code/protocol mismatch")
    for entry in read_json(output / "input_inventory.json")["files"]:
        require(digest(entry["path"]) == entry["sha256"], f"frozen input changed: {entry['role']}")
    rows, geometries = read_json(DATA_MANIFEST)["rows"], read_json(output / "geometry_inventory.json")
    raw_entries = {e.get("name", e.get("video_name")): e for e in read_json(RAW_ROOT / "manifest.json")["videos"]}
    local_entries = {int(e["row_id"]): e for e in read_json(LOCAL_ROOT / "manifest.json")["videos"]}
    names = read_json(LOCAL_ROOT / "feature_names.json")
    ledger = {(int(e["index"]), int(e["event"])): e for e in load_csv(LEDGER / "events.csv")}
    require(len(ledger) == 85, "ledger count mismatch")
    records, events = [], []
    with np.load(DATA, allow_pickle=False) as data:
        for index, row in enumerate(rows):
            r = {**row, "index": index, "height": geometries[index]["height"], "width": geometries[index]["width"],
                 "geometry": geometries[index], "raw_path": str(RAW_ROOT / raw_entries[row["name"]]["raw_directory"] / "signals.npz"),
                 "local_path": str(LOCAL_ROOT / local_entries[index]["file"])}
            labels = np.asarray(data[f"labels_{index}"], bool)
            require(len(labels) == row["frames"], "label/frame count mismatch")
            r.update(extract(r, names))
            r["spans"], r["is_normal"] = spans(labels), not labels.any()
            require(len(r["spans"]) == row["event_count"], "event count mismatch")
            r["background_indices"] = np.flatnonzero(~labels[r["tubelets"]].any(axis=1)).tolist()
            for event_id, (start, end) in enumerate(r["spans"]):
                old = ledger[(index, event_id)]
                require(int(old["start_frame"]) == start and int(old["end_frame"]) == end, "ledger label alignment mismatch")
                inside = (r["tubelets"] >= start) & (r["tubelets"] <= end)
                matched = old["default_control_matched_iou_0_5"].lower() == "true"
                events.append({"index": index, "event": event_id, "name": row["name"], "sha256": row["sha256"],
                               "start_frame": start, "end_frame": end, "duration_seconds": (end - start + 1) / row["fps"],
                               "category": "matched_iou_0_5" if matched else old["default_control_diagnostic_stage"],
                               "pure_anchor_indices": np.flatnonzero(inside.all(axis=1)).tolist(),
                               "any_member_anchor_indices": np.flatnonzero(inside.any(axis=1)).tolist(),
                               "nearest_sample_to_start_seconds": float(np.min(abs(r["tubelets"] - start)) / row["fps"]),
                               "nearest_sample_to_end_seconds": float(np.min(abs(r["tubelets"] - end)) / row["fps"]),
                               "median_anchor_spacing_seconds": float(np.median(np.diff(r["centers"])) / row["fps"])})
            records.append(r)
    require(len(events) == 85 and sum(r["is_normal"] for r in records) == 14, "dataset category counts changed")
    categories = ["all", LOW, GATED, BOUNDARY, "matched_iou_0_5"]
    summaries = [summarize(records, events, feature, category) for feature in range(len(FEATURES)) for category in categories]
    capacity = capacity_decision(summaries)
    arrays, video_rows = {}, []
    for r in records:
        index = r["index"]
        arrays[f"values_{index}"], arrays[f"valid_{index}"] = r["values"], r["valid"].astype(np.uint8)
        arrays[f"tubelets_{index}"], arrays[f"centers_{index}"] = r["tubelets"], r["centers"]
        arrays[f"v_error_{index}"], arrays[f"background_{index}"] = r["v_error"], np.asarray(r["background_indices"], int)
        video_rows.append({"index": index, "name": r["name"], "sha256": r["sha256"], "is_normal": bool(r["is_normal"]),
                           "support_info": r["support_info"], "valid_anchor_counts": dict(zip(FEATURES, r["valid"].sum(axis=0).astype(int).tolist()))})
    np.savez_compressed(output / "anchor_features.npz", **arrays)
    write_json(output / "events.json", events)
    write_json(output / "video_support.json", video_rows)
    write_json(output / "feature_summaries.json", summaries)
    write_json(output / "boundary_diagnostic.json", [boundary_summary(records, events, n) for n in range(len(FEATURES))])
    geo = read_json(output / "geometry_inventory.json")
    areas = [g["v_source_area_fraction"] for g in geo]
    report = {"status": "complete_read_only_screen", "schema_version": "amended-local-interaction-report-v3",
              "protocol_sha256": receipt["protocol_sha256"], "rows": len(records), "content_groups": len({r["sha256"] for r in records}),
              "events": len(events), "category_counts": {c: sum(e["category"] == c for e in events) for c in categories[1:]},
              "coverage": {"events_with_any_direct_V_member": sum(bool(e["any_member_anchor_indices"]) for e in events),
                           "events_with_pure_V_tubelet": sum(bool(e["pure_anchor_indices"]) for e in events),
                           "total_V_anchors": sum(len(r["centers"]) for r in records),
                           "source_area_seen_by_V_min_median_max": [float(np.min(areas)), float(np.median(areas)), float(np.max(areas))]},
              **capacity, "training_started": False, "default_model_modified": False,
              "performance_improvement_proven": False, "feature_summaries": summaries,
              "conclusion_scope": "Only these fixed error-map relations and sparse cache; not proof local evidence or JEPA cannot help."}
    write_json(output / "report.json", report)
    names_out = ["protocol.json", "input_inventory.json", "geometry_inventory.json", "freeze_receipt.json", "anchor_features.npz",
                 "events.json", "video_support.json", "feature_summaries.json", "boundary_diagnostic.json", "report.json"]
    write_json(output / "decision.json", {"status": "eligible_for_single_controlled_OOF_after_audit" if capacity["capacity_screen_supported"] else "stop_before_training_insufficient_fixed_feature_evidence",
               **capacity, "performance_improvement_proven": False, "training_started": False, "default_promoted": False,
               "artifact_sha256": {name: digest(output / name) for name in names_out},
               "source_and_inputs_verified_again_before_run": True})
    print(json.dumps({"status": "complete", "output": str(output), **capacity}, ensure_ascii=False))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCES)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        freeze(args.output.resolve(), args.source_root.resolve())
    else:
        run(args.output.resolve())

if __name__ == "__main__":
    main()
