#!/usr/bin/env python3
"""Independent, non-training audit of a frozen amended local-capacity screen.

Only the requested NEW audit JSON is written. No producer module is imported,
no feature values are generated, and all original evidence is read-only.
Statistics are recalculated from stored float64 values; observation masks are
checked independently against pinned caches and source-coordinate rectangles.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOW = "frame_evidence_below_high_seed"
GATED = "video_attenuation_drops_event_peak_below_seed"
BOUNDARY = "surviving_overlap_needs_boundary_or_structure"
CATEGORIES = ("all", LOW, GATED, BOUNDARY, "matched_iou_0_5")
FEATURES = (
    "v_patch_change", "v_strength_change", "v_strength_persistence",
    "vi_topk_overlap", "vi_strength_colocation", "motion_topk_gain",
    "motion_persistent_gain", "vi_motion_joint",
)
FAMILIES = dict(zip(FEATURES, (
    "temporal_local", "temporal_local", "temporal_local", "cross_modal",
    "cross_modal", "coarse_motion", "joint_relation", "joint_relation",
)))
REPRESENTATIVES = {
    "temporal_local": "v_strength_change", "cross_modal": "vi_strength_colocation",
    "coarse_motion": "motion_topk_gain",
}
ARTIFACTS = (
    "protocol.json", "input_inventory.json", "geometry_inventory.json",
    "freeze_receipt.json", "anchor_features.npz", "events.json",
    "video_support.json", "feature_summaries.json", "boundary_diagnostic.json",
    "report.json", "decision.json",
)
PROTECTED = {
    "models/optimized_locator_v1.json": "093c14ec9b2ffc91dea776f694a8278bbd2f4d66a1881fe10d8399fc24301dbd",
    "models/optimized_motion_locator_v1.json": "77e5d7eeb63653c6ddc0ea33da8e455c5b737d46f87c1cfde96c43a1381d5670",
    "code/optimized_detector.py": "899e3b5128e687a9be7e27702db91487ef3319154b713fdf594331cc2a0c9ea7",
}
HISTORICAL_PINS = {
    "code/diagnose_local_interaction_capacity.py": "d67cf56605099e4353276451e672654e420b0e45c1fdb21d1bb5ae8a09a7ed5c",
    "output/baseline-targeted-local-interaction-run2-20261003/anchor_features.npz": "d0eec9548ec6e1bfa7780c414dc9a2d0490ccf99c065a983481a31a28a4db128",
    "output/baseline-targeted-local-interaction-run2-20261003/decision.json": "464854440f1f4af7f14e9fa29317949248d350e42841cd45b2b68cbbccd9e240",
    "output/baseline-targeted-local-interaction-run2-20261003/event_evidence.json": "8b24517a4cbbc6c8a6c9c6df5bbe0bb5ca616fd434ef30465d41fed085b7b90e",
    "output/baseline-targeted-local-interaction-run2-20261003/feature_summaries.json": "a0475078cc6d33765c439df0473c7243fd74def5c3207ac3dea7779cbf4be7a2",
    "output/baseline-targeted-local-interaction-run2-20261003/input_inventory.json": "0624ddba8f79513f27bc545a62d472760d9ed8f67040ef662b0f8d1ef9a12982",
    "output/baseline-targeted-local-interaction-run2-20261003/protocol.json": "ba64868b4cf92dc7251e165659741007de51e6db43b958f2b3a5269842ec4ccc",
    "output/baseline-targeted-local-interaction-run2-20261003/report.json": "d77063cac5250c8c6bfc085565e3379b95ac5b1cb43997843e21ea5194c5873a",
    "output/baseline-targeted-local-interaction-run2-20261003/video_evidence.json": "98b96aca05f627acaf3dc6a279d9b0036ab5dd4a98e7587fdbfc81ec29d88d2e",
    "output/baseline-targeted-local-interaction-run3-20261003/anchor_features.npz": "084d6c647a21a7271930aec215ae66b7e3556f73157d3fcb6682356507891f17",
    "output/baseline-targeted-local-interaction-run3-20261003/decision.json": "2dbf3ba060cfce4bc6ec4d2dcc57521fe2820e58d9e8cebe17795e236bb9a340",
    "output/baseline-targeted-local-interaction-run3-20261003/event_evidence.json": "888ad63493ec39515202020a9b66141a04df26e4564a3195f50470c40f2a1967",
    "output/baseline-targeted-local-interaction-run3-20261003/feature_summaries.json": "682bcec08069b19c234c56a6480b3213a915e17f99ebc8ea7b407fcb85c2ef20",
    "output/baseline-targeted-local-interaction-run3-20261003/input_inventory.json": "f4839a83ec50d44bbb366108c83321cff28a7b6e1e62df3715ab2f8019a2e65d",
    "output/baseline-targeted-local-interaction-run3-20261003/protocol.json": "eb9ece67643038f2d8f494b33bd2cd4e2c5b677848140222ec20ffb89687d5cb",
    "output/baseline-targeted-local-interaction-run3-20261003/report.json": "0d9613f12408390eceac58f14b87145bb3e47ef49564a1845d6c8984b99750e0",
    "output/baseline-targeted-local-interaction-run3-20261003/video_evidence.json": "ae46f4152cd5b23efd7fce976fa4f610578dca30732a7e778ea0147d70becef3",
}


class AuditError(ValueError):
    """Malformed, missing, or inconsistent evidence: fail closed."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise AuditError(f"nonfinite JSON literal: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=invalid)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.max_numeric_difference = 0.0
        self.max_numeric_difference_path = None

    def check(self, condition, path, expected=None, actual=None):
        self.count += 1
        if not condition and len(self.failures) < 200:
            self.failures.append({"path": path, "expected": expected, "actual": actual})

    def same(self, actual, expected, path, atol=2e-12):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), path + "/type", "dict", type(actual).__name__)
            if not isinstance(actual, dict):
                return
            self.check(set(actual) == set(expected), path + "/keys",
                       sorted(expected), sorted(actual))
            for key in expected.keys() & actual.keys():
                self.same(actual[key], expected[key], path + "/" + str(key), atol)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)), path + "/type", "list", type(actual).__name__)
            if isinstance(actual, (list, tuple)):
                self.check(len(actual) == len(expected), path + "/length", len(expected), len(actual))
                for i, (left, right) in enumerate(zip(actual, expected)):
                    self.same(left, right, path + "/" + str(i), atol)
        elif isinstance(expected, (float, np.floating)):
            ok = isinstance(actual, (int, float)) and not isinstance(actual, bool)
            if ok:
                difference = abs(float(actual) - float(expected))
                if difference > self.max_numeric_difference:
                    self.max_numeric_difference = difference
                    self.max_numeric_difference_path = path
                ok = math.isfinite(float(actual)) and difference <= atol
            self.check(ok, path, float(expected), actual)
        else:
            # bool and integer fields must not quietly compare True == 1.
            ok = actual == expected and type(actual) is type(expected)
            self.check(ok, path, expected, actual)


class Evidence:
    """Records every consumed file hash and rehashes it before publication."""
    def __init__(self):
        self.initial = {}
        self.pins_checked = 0

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), f"missing evidence: {path}")
        actual = sha256(path)
        if expected is not None:
            require(isinstance(expected, str) and len(expected) == 64 and
                    all(c in "0123456789abcdef" for c in expected), f"bad SHA256 pin: {path}")
            require(actual == expected, f"SHA256 mismatch: {path}; expected={expected}; actual={actual}")
            self.pins_checked += 1
        old = self.initial.setdefault(str(path), actual)
        require(old == actual, f"evidence changed during audit: {path}")
        return actual

    def json(self, path, expected=None):
        self.pin(path, expected)
        return strict_json(path)

    def verify_unchanged(self):
        for path, expected in self.initial.items():
            require(sha256(path) == expected, f"post-audit drift: {path}")


def literal_constants(source):
    """Inspect frozen declarations with AST only; never execute producer code."""
    wanted = {"FEATURE_FAMILIES", "FEATURES", "REPRESENTATIVES", "LOW", "GATED",
              "BOUNDARY", "RULE", "PARAMETERS", "PROTECTED"}
    constants = {}

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            seq = [value(x) for x in node.elts]
            return tuple(seq) if isinstance(node, ast.Tuple) else seq
        if isinstance(node, ast.Dict):
            return {value(k): value(v) for k, v in zip(node.keys, node.values)}
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                node.func.id == "tuple" and len(node.args) == 1 and not node.keywords):
            return tuple(value(node.args[0]))
        raise AuditError("unsupported frozen declaration; no expression execution")

    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                constants[name] = value(node.value)
    require(wanted <= constants.keys(), "missing frozen source declarations")
    return constants


def weighted_auc(positive, negative, positive_weights=None, negative_weights=None):
    """Direct pairwise U statistic, independent of producer's sorted-rank code."""
    p, n = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    require(p.ndim == n.ndim == 1, "AUC scores must be vectors")
    if not p.size or not n.size:
        return None
    pw = np.ones(p.size) if positive_weights is None else np.asarray(positive_weights, np.float64)
    nw = np.ones(n.size) if negative_weights is None else np.asarray(negative_weights, np.float64)
    require(pw.shape == p.shape and nw.shape == n.shape, "AUC weight shape mismatch")
    require(np.isfinite(p).all() and np.isfinite(n).all() and
            np.isfinite(pw).all() and np.isfinite(nw).all(), "nonfinite AUC input")
    require((pw >= 0).all() and (nw >= 0).all() and pw.sum() > 0 and nw.sum() > 0,
            "invalid AUC weights")
    comparisons = (p[:, None] > n[None, :]).astype(np.float64)
    comparisons += 0.5 * (p[:, None] == n[None, :])
    return float((pw / pw.sum()) @ comparisons @ (nw / nw.sum()))


def equal_block_groups(items):
    grouped = defaultdict(list)
    for key, scores in items:
        x = np.asarray(scores, np.float64)
        require(x.ndim == 1 and np.isfinite(x).all(), "bad observed scores")
        if x.size:
            grouped[key].append(x)
    balanced = {}
    for key, blocks in grouped.items():
        # Each observed event (or normal alias) has equal total probability.
        x = np.hstack(blocks)
        w = np.hstack([np.repeat(1.0 / len(blocks) / b.size, b.size) for b in blocks])
        require(abs(w.sum() - 1.0) < 1e-12, "unbalanced group mass")
        balanced[key] = (x, w)
    return balanced


def group_average(pairs):
    grouped = defaultdict(list)
    for key, value in pairs:
        if value is not None:
            require(math.isfinite(float(value)), "nonfinite group statistic")
            grouped[key].append(float(value))
    return float(np.mean([sum(values) / len(values) for values in grouped.values()])) if grouped else None


def weighted_quantile(values, weights, quantile):
    require(0 <= quantile <= 1, "bad quantile")
    x, w = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    require(x.ndim == 1 and x.shape == w.shape and x.size and np.isfinite(x).all() and
            np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0, "bad quantile input")
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cumulative = 0.0
    cutoff = quantile * float(w.sum())
    for score, mass in zip(x, w):
        cumulative += float(mass)
        if cumulative >= cutoff:
            return float(score)
    return float(x[-1])


def linear_quantile(values, quantiles):
    """Explicit NumPy default 'linear' order-statistic interpolation."""
    x = np.sort(np.asarray(values, np.float64))
    q = np.asarray(quantiles, np.float64)
    require(x.ndim == 1 and x.size and np.isfinite(x).all(), "empty/nonfinite bootstrap")
    at = (x.size - 1) * q
    low, high = np.floor(at).astype(int), np.ceil(at).astype(int)
    return x[low] + (x[high] - x[low]) * (at - low)


def group_auc_summary(positive, normal, parameters):
    pk, nk = sorted(positive), sorted(normal)
    result = {"auc": None, "ci95": None, "positive_content_groups": len(pk),
              "normal_content_groups": len(nk), "bonferroni_lower": None}
    if not pk or not nk:
        return result
    matrix = np.empty((len(pk), len(nk)), np.float64)
    for i, pkey in enumerate(pk):
        for j, nkey in enumerate(nk):
            p, pw = positive[pkey]
            n, nw = normal[nkey]
            matrix[i, j] = weighted_auc(p, n, pw, nw)
    rng = np.random.Generator(np.random.PCG64(int(parameters["bootstrap_seed"])))
    count = int(parameters["bootstrap_replicates"])
    # Match the frozen random draw contract; evaluate each two-sample draw by
    # matrix multiplication rather than producer's optimized einsum.
    pc = rng.multinomial(len(pk), np.ones(len(pk)) / len(pk), size=count)
    nc = rng.multinomial(len(nk), np.ones(len(nk)) / len(nk), size=count)
    bootstrap = np.array([float(left @ matrix @ right) / (len(pk) * len(nk))
                          for left, right in zip(pc, nc)])
    result.update(auc=float(matrix.mean()), ci95=linear_quantile(bootstrap, [.025, .975]).tolist(),
                  bonferroni_lower=float(linear_quantile(bootstrap, parameters["gate_one_sided_familywise_alpha"] /
                                                        parameters["gate_representative_target_tests"])))
    return result


def observed_values(record, feature, indices):
    indices = np.asarray(list(indices), dtype=np.int64)
    if not indices.size:
        return np.empty(0, np.float64)
    require(indices.ndim == 1 and (indices >= 0).all() and
            (indices < len(record["centers"])).all(), "feature index outside anchor grid")
    scores = record["values"][indices, feature]
    valid = record["valid"][indices, feature]
    require(np.isfinite(scores[valid]).all(), "nonfinite observed feature")
    return scores[valid]


def recompute_summary(records, events, feature, category, parameters):
    selected = [e for e in events if category == "all" or e["category"] == category]
    normal_items = [(r["sha256"], observed_values(r, feature, range(len(r["centers"]))))
                    for r in records if r["is_normal"]]
    normal_coverage = [(r["sha256"], float(r["valid"][:, feature].mean()))
                       for r in records if r["is_normal"]]
    ng = equal_block_groups(normal_items)
    q95, tail_mass, tie_mass = None, None, None
    if ng:
        x = np.hstack([value for value, weight in ng.values()])
        w = np.hstack([weight / len(ng) for value, weight in ng.values()])
        q95 = weighted_quantile(x, w, float(parameters["normal_tail_quantile"]))
        tail_mass = float(w[x > q95].sum() / w.sum())
        tie_mass = float(w[x == q95].sum() / w.sum())
    positive_items, coverage, anchor_coverage = [], [], []
    within, tail, ties = [], [], []
    for e in selected:
        r = records[e["index"]]
        ids = e["pure_anchor_indices"]
        scores = observed_values(r, feature, ids)
        positive_items.append((r["sha256"], scores))
        coverage.append((r["sha256"], float(bool(scores.size))))
        anchor_coverage.append((r["sha256"], scores.size / len(ids) if ids else 0.0))
        if scores.size:
            background = observed_values(r, feature, r["background_indices"])
            within.append((r["sha256"], weighted_auc(scores, background)))
            if q95 is not None:
                tail.append((r["sha256"], float(scores.max() > q95)))
                ties.append((r["sha256"], float(scores.max() == q95)))
    name = FEATURES[feature]
    result = group_auc_summary(equal_block_groups(positive_items), ng, parameters)
    result.update(
        feature=name, family=FAMILIES[name], category=category, event_count=len(selected),
        valid_event_count=sum(bool(scores.size) for key, scores in positive_items),
        event_valid_coverage_content_balanced=group_average(coverage),
        event_anchor_valid_fraction_content_balanced=group_average(anchor_coverage),
        normal_anchor_valid_fraction_content_balanced=group_average(normal_coverage),
        within_video_auc_content_balanced=group_average(within),
        within_video_available_events=sum(value is not None for key, value in within),
        within_video_available_fraction=(sum(value is not None for key, value in within) /
                                         len(selected)) if selected else None,
        strict_normal_q95_event_recall_content_balanced=group_average(tail),
        event_peak_equals_q95_content_balanced=group_average(ties), normal_q95=q95,
        normal_strict_tail_mass=tail_mass, normal_tie_mass_at_q95=tie_mass,
        normal_video_any_strict_tail_fraction=group_average(
            [(key, float(scores.max() > q95)) for key, scores in normal_items if scores.size]
        ) if q95 is not None else None,
        direction="high_fixed_before_repaired_scores",
        ci_role="pointwise_group_bootstrap_diagnostic_not_multiple_testing_or_generalization_proof",
    )
    return result


def gate_requirements(overall, target, rules):
    comparisons = (
        ("all_event_coverage", overall, "event_valid_coverage_content_balanced", "minimum_event_valid_coverage_all_and_target", False),
        ("target_event_coverage", target, "event_valid_coverage_content_balanced", "minimum_event_valid_coverage_all_and_target", False),
        ("all_anchor_coverage", overall, "event_anchor_valid_fraction_content_balanced", "minimum_event_anchor_valid_fraction_all_and_target", False),
        ("target_anchor_coverage", target, "event_anchor_valid_fraction_content_balanced", "minimum_event_anchor_valid_fraction_all_and_target", False),
        ("normal_anchor_coverage", target, "normal_anchor_valid_fraction_content_balanced", "minimum_normal_anchor_valid_fraction", False),
        ("overall_fixed_direction_auc", overall, "auc", "minimum_overall_high_direction_group_auc", False),
        ("target_fixed_direction_auc", target, "auc", "minimum_target_high_direction_group_auc", False),
        ("target_bonferroni_lower", target, "bonferroni_lower", "minimum_target_group_bootstrap_ci_lower_strict", True),
        ("target_within_video_auc", target, "within_video_auc_content_balanced", "minimum_target_within_video_auc", False),
        ("target_within_video_available", target, "within_video_available_fraction", "minimum_target_within_video_available_fraction", False),
        ("target_strict_tail_recall", target, "strict_normal_q95_event_recall_content_balanced", "minimum_target_strict_normal_q95_event_recall", False),
    )
    checks = {}
    for label, source, field, threshold, strict in comparisons:
        value, cutoff = source.get(field), float(rules[threshold])
        checks[label] = bool(value is not None and math.isfinite(float(value)) and
                             (float(value) > cutoff if strict else float(value) >= cutoff))
    return checks


def recompute_gate(summaries, protocol):
    lookup = {(s["feature"], s["category"]): s for s in summaries}
    require(len(lookup) == len(summaries), "duplicate feature/category summary")
    rules = protocol["training_capacity_rule"]
    representatives = protocol["gate_representatives"]
    qualifying, decisions, details = [], {}, []
    for target in rules["target_categories"]:
        accepted = []
        for family, feature in representatives.items():
            conditions = gate_requirements(lookup[feature, "all"], lookup[feature, target], rules)
            passes = all(conditions.values())
            details.append({"feature": feature, "family": family, "target": target,
                            "passes": passes, "conditions": conditions})
            if passes:
                accepted.append(family)
                qualifying.append({"feature": feature, "family": family, "target": target})
        # Count BOTH distinct representatives and families within this target.
        target_passes = (len(accepted) >= int(rules["minimum_qualifying_families"]) and
                         len(accepted) >= int(rules["minimum_qualifying_features"]))
        decisions[target] = {"qualifying_base_families": accepted, "passes": target_passes}
    result = {"capacity_screen_supported": any(d["passes"] for d in decisions.values()),
              "qualifying": qualifying, "target_decisions": decisions}
    return result, details


def recompute_boundary(records, events, feature, parameters):
    rows = []
    count = int(parameters["boundary_neighbor_count"])
    for e in events:
        if e["category"] != BOUNDARY:
            continue
        r = records[e["index"]]
        before = [k for k in r["background_indices"] if r["tubelets"][k].max() < e["start_frame"]]
        after = [k for k in r["background_indices"] if r["tubelets"][k].min() > e["end_frame"]]
        inside = e["pure_anchor_indices"]
        pairs = ((inside[:count], before[-count:]), (inside[-count:], after[:count]))
        contrasts = []
        for positive, negative in pairs:
            p, n = observed_values(r, feature, positive), observed_values(r, feature, negative)
            contrasts.append(float(p.mean() - n.mean()) if p.size and n.size else None)
        start, end = contrasts
        observed = start is not None and end is not None
        rows.append({"index": e["index"], "event": e["event"], "start_contrast": start,
                     "end_contrast": end, "both_observed": observed,
                     "both_correct_high_direction": bool(observed and start > 0 and end > 0)})
    available = [row for row in rows if row["both_observed"]]
    return {"feature": FEATURES[feature], "events": len(rows), "both_observed": len(available),
            "both_correct_fraction_of_observed": (sum(row["both_correct_high_direction"] for row in available) /
                                                   len(available)) if available else None,
            "role": "GT_boundary_contrast_diagnostic_not_a_boundary_detector", "rows": rows}


def expected_crop(height, width, kind):
    require(type(height) is int and type(width) is int and min(height, width) > 0,
            "invalid geometry dimensions")
    if kind == "v":
        # Preserve the pinned legacy implementation's float-multiply then int;
        # do NOT silently substitute ideal short-side 438 for effective 437.
        scale = int(256 / 224 * 384) / min(height, width)
        rh, rw = int(height * scale), int(width * scale)
        size = 384
        top, left = (rh - size) // 2, (rw - size) // 2
    else:
        rh, rw = (256, int(256 * width / height)) if height <= width else (int(256 * height / width), 256)
        size = 224
        top, left = round((rh - size) / 2), round((rw - size) / 2)
    return {"resized_height": rh, "resized_width": rw, "crop_top": top, "crop_left": left,
            "crop_size": size, "box_xyxy": [left / rw, top / rh, (left + size) / rw, (top + size) / rh]}


def grid_rectangles(box, rows, columns):
    x = np.linspace(box[0], box[2], columns + 1)
    y = np.linspace(box[1], box[3], rows + 1)
    return np.array([(x[c], y[r], x[c + 1], y[r + 1])
                     for r in range(rows) for c in range(columns)], np.float64)


def expected_motion_geometry(height, width):
    scale = min(1.0, 128 / max(height, width))
    ah, aw = max(1, round(height * scale)), max(1, round(width * scale))
    rectangles = []
    for r in range(3):
        for c in range(3):
            left, right = max(4, math.ceil(c * aw / 3)), min(aw - 4, math.ceil((c + 1) * aw / 3))
            top, bottom = max(4, math.ceil(r * ah / 3)), min(ah - 4, math.ceil((r + 1) * ah / 3))
            rectangles.append([left / aw, top / ah, max(left, right) / aw, max(top, bottom) / ah])
    return np.array(rectangles, np.float64), [ah, aw]


def observation_area(targets, sources, observed):
    """Only support area, never feature-value projection or extraction."""
    area = np.zeros(len(targets), np.float64)
    for source, supported in zip(sources, observed):
        if not supported:
            continue
        width = np.maximum(0, np.minimum(targets[:, 2], source[2]) - np.maximum(targets[:, 0], source[0]))
        height = np.maximum(0, np.minimum(targets[:, 3], source[3]) - np.maximum(targets[:, 1], source[1]))
        area += width * height
    total = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    require((total > 0).all(), "zero-area target patch")
    return area / total


def label_spans(labels):
    # Scan inclusive spans independently, without producer's padded diff.
    result, start = [], None
    for i, abnormal in enumerate(labels):
        if abnormal and start is None:
            start = i
        elif not abnormal and start is not None:
            result.append((start, i - 1))
            start = None
    if start is not None:
        result.append((start, len(labels) - 1))
    return result



def expected_feature_valid(temporal_mask, cross_mask, motion_mask, initial, minimum):
    temporal = bool(not initial and temporal_mask.sum() >= minimum)
    cross = bool(cross_mask.sum() >= minimum)
    motion = bool(motion_mask.sum() >= minimum)
    persistent = bool(not initial and (temporal_mask & motion_mask).sum() >= minimum)
    triple = bool((cross_mask & motion_mask).sum() >= minimum)
    return np.asarray([temporal] * 3 + [cross] * 2 + [motion, motion and persistent,
                                                    motion and triple], dtype=bool)


def validate_source_mask_contract(source, checks):
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({"extract", "local_relation", "motion_gain", "cross_features", "temporal_features"} <= functions.keys(),
            "missing frozen feature functions")

    def expression(text):
        return ast.dump(ast.parse(text, mode="eval").body, include_attributes=False)

    def assignments(function, variable):
        return [node.value for node in ast.walk(functions[function]) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)]

    for variable, text in (("persistence_mask", "vvalid[k] & vvalid[k - 1] & mpvalid if k else np.zeros_like(vvalid[k])"),
                           ("triple_mask", "vvalid[k] & ipvalid & mpvalid")):
        values = assignments("extract", variable)
        checks.check(len(values) == 1 and ast.dump(values[0], include_attributes=False) == expression(text),
                     "frozen_source/shared_mask/" + variable, text,
                     ast.unparse(values[0]) if values else None)
    calls = [node for node in ast.walk(functions["extract"]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    for function, mask in (("local_relation", "persistence_mask"), ("motion_gain", "persistence_mask"),
                           ("cross_features", "triple_mask"), ("motion_gain", "triple_mask")):
        checks.check(any(node.func.id == function and len(node.args) == 3 and
                         ast.dump(node.args[2], include_attributes=False) == expression(mask) for node in calls),
                     "frozen_source/shared_mask_call/" + function + "/" + mask, True, None)
    # Verify both operands are sliced by the very same passed mask. This checks
    # support/normalization plumbing, not the numerical feature transform.
    for function, operands in (("local_relation", ("a", "b")), ("motion_gain", ("v", "motion"))):
        indexed = set()
        for node in ast.walk(functions[function]):
            if not (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name) and node.slice.id == "mask"):
                continue
            operand = node.value
            if (isinstance(operand, ast.Call) and isinstance(operand.func, ast.Attribute) and
                    operand.func.attr == "asarray" and operand.args):
                operand = operand.args[0]
            if isinstance(operand, ast.Name):
                indexed.add(operand.id)
        checks.check(set(operands) <= indexed, "frozen_source/common_normalization/" + function,
                     list(operands), sorted(indexed))
    return {"three_way_joint_masks_verified_from_frozen_AST": True,
            "both_operands_indexed_on_common_mask": True,
            "joint_source_lines": [functions["extract"].lineno,
                                   max(n.lineno for n in ast.walk(functions["extract"]) if hasattr(n, "lineno"))]}

def validate_protocol(protocol, source, checks):
    checks.same(protocol["schema_version"], "amended-local-interaction-capacity-v3", "protocol/schema")
    checks.same(protocol["features"], list(FEATURES), "protocol/features")
    checks.same(protocol["feature_families"], FAMILIES, "protocol/families")
    checks.same(protocol["gate_representatives"], REPRESENTATIVES, "protocol/representatives")
    checks.same(protocol["protected"], PROTECTED, "protocol/protected")
    checks.same(protocol["not_original_or_fresh_data_preregistration"], True, "protocol/amendment")
    checks.same(protocol["prior_diagnoses_seen"], ["run2", "run3"], "protocol/previous_diagnoses")
    checks.same(protocol["feature_policy"]["direction"], "higher fixed for all features and stages", "protocol/direction")
    checks.same(protocol["feature_policy"]["labels_used_in_feature_transform"], False, "protocol/label_free_transform")
    for key, expected in (("V_grid", [24, 24]), ("I_grid", [14, 14]), ("motion_grid", [3, 3])):
        checks.same(protocol["feature_policy"][key], expected, "protocol/" + key)
    validate_source_mask_contract(source, checks)
    declarations = literal_constants(source)
    for field, name in (("features", "FEATURES"), ("feature_families", "FEATURE_FAMILIES"),
                        ("gate_representatives", "REPRESENTATIVES"), ("parameters", "PARAMETERS"),
                        ("training_capacity_rule", "RULE"), ("protected", "PROTECTED")):
        expected = list(declarations[name]) if name == "FEATURES" else declarations[name]
        checks.same(protocol[field], expected, "frozen_source/declarations/" + name)
    parameters, rules = protocol["parameters"], protocol["training_capacity_rule"]
    expected_parameters = {
        "top_fraction": .05, "minimum_common_patches": 32,
        "minimum_projected_area_coverage": .5, "normal_tail_quantile": .95,
        "bootstrap_replicates": 2000, "bootstrap_seed": 20261003,
        "boundary_neighbor_count": 2, "gate_one_sided_familywise_alpha": .05,
        "gate_representative_target_tests": 6,
    }
    checks.same(parameters, expected_parameters, "protocol/approved_parameters")
    checks.same(rules["target_categories"], [LOW, GATED], "protocol/targets")
    checks.same(rules["minimum_qualifying_features"], 2, "protocol/two_features")
    checks.same(rules["minimum_qualifying_families"], 2, "protocol/two_families")
    for key, value in rules.items():
        if key != "target_categories":
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                    math.isfinite(value) and value > 0, f"bad gate threshold: {key}")
    checks.check(len(REPRESENTATIVES) * len(rules["target_categories"]) ==
                 parameters["gate_representative_target_tests"], "protocol/Bonferroni_test_count", 6,
                 len(REPRESENTATIVES) * len(rules["target_categories"]))
    require(not checks.failures, "unsupported or inconsistent frozen protocol")


def verify_bindings(output, evidence, checks):
    for name in ARTIFACTS:
        evidence.pin(output / name)
    protocol = strict_json(output / "protocol.json")
    receipt = strict_json(output / "freeze_receipt.json")
    decision = strict_json(output / "decision.json")
    for name in ("protocol", "input_inventory", "geometry_inventory"):
        evidence.pin(output / (name + ".json"), receipt[name + "_sha256"])
    checks.same(receipt["repaired_statistics_not_yet_calculated"], True, "freeze/no_scores_yet")
    created = datetime.fromisoformat(protocol["created_utc"])
    frozen = datetime.fromisoformat(receipt["frozen_utc"])
    require(created.tzinfo is not None and frozen.tzinfo is not None, "freeze time must have timezone")
    checks.check(created <= frozen, "freeze/protocol_precedes_receipt", True, created <= frozen)
    for name in ARTIFACTS:
        if name not in ("protocol.json", "input_inventory.json", "geometry_inventory.json", "freeze_receipt.json"):
            checks.check((output / name).stat().st_mtime_ns >= (output / "freeze_receipt.json").stat().st_mtime_ns,
                         "freeze/local_order/" + name, "after freeze", "after" if
                         (output / name).stat().st_mtime_ns >= (output / "freeze_receipt.json").stat().st_mtime_ns else "before")
    checks.check(set(decision["artifact_sha256"]) == set(ARTIFACTS) - {"decision.json"},
                 "decision/artifact_hash_scope", sorted(set(ARTIFACTS) - {"decision.json"}),
                 sorted(decision["artifact_sha256"]))
    for name, pin in decision["artifact_sha256"].items():
        require(name in ARTIFACTS, "unsafe or unknown artifact path in decision")
        evidence.pin(output / name, pin)
    inventory = strict_json(output / "input_inventory.json")["files"]
    pins, roles = {}, Counter()
    for entry in inventory:
        p = str(Path(entry["path"]).resolve())
        require(p not in pins, f"duplicate frozen input path: {p}")
        pins[p] = entry["sha256"]
        roles[entry["role"]] += 1
        evidence.pin(p, entry["sha256"])
        if entry["role"] == "source":
            evidence.pin(output / "source_snapshot" / Path(p).name, entry["sha256"])
    checks.same(dict(roles), {"source": 6, "input_metadata": 9, "raw_spatial": 77,
                            "raw_metadata": 77, "coarse_local_motion": 77,
                            "prior_dense_cache_integrity_only": 77, "original_video": 77}, "pins/role_counts")
    for relative, pin in {**HISTORICAL_PINS, **PROTECTED}.items():
        evidence.pin(ROOT / relative, pin)
    paths = {
        "dataset": ROOT / "output/algorithm-opt-2026-10-02/dataset.npz",
        "dataset_manifest": ROOT / "output/algorithm-opt-2026-10-02-v2/dataset_manifest.json",
        "raw_manifest": ROOT / "output/algorithm-opt-2026-10-02/corrected_jepa_features_v2/manifest.json",
        "local_manifest": ROOT / "output/algorithm-opt-2026-10-02-v2/local_motion/manifest.json",
        "local_names": ROOT / "output/algorithm-opt-2026-10-02-v2/local_motion/feature_names.json",
        "events_ledger": ROOT / "output/baseline-targeted-semantic-readout/event-change-ledger/events.csv",
        "videos_ledger": ROOT / "output/baseline-targeted-semantic-readout/event-change-ledger/videos.csv",
        "prior": ROOT / "output/baseline-targeted-semantic-readout/final_delivery_manifest.json",
        "early": ROOT / "output/baseline-targeted-local-interaction-run3-20261003/input_inventory.json",
        "producer": ROOT / "code/diagnose_local_interaction_capacity_repaired.py",
    }
    for key, path in paths.items():
        require(str(path.resolve()) in pins, f"consumed dependency not frozen: {key}")
    producer = paths["producer"].read_text(encoding="utf-8")
    validate_protocol(protocol, producer, checks)
    # The old delivery and early receipts are themselves pinned; validate every
    # intersecting consumed cache pin, not the unrelated full old fit history.
    earlier = evidence.json(paths["early"])
    previous = evidence.json(paths["prior"])["artifact_sha256"]
    historical_matches = 0
    for entry in earlier["feature_files"]:
        path = str(Path(entry["path"]).resolve())
        require(path in pins, f"prior cache omitted from amendment: {path}")
        checks.same(pins[path], entry["sha256"], "pins/prior_cache/" + path)
        historical_matches += 1
    for path, pin in previous.items():
        key = str(Path(path).resolve())
        if key in pins:
            checks.same(pins[key], pin, "pins/old_delivery_consumed/" + key)
    return protocol, paths, pins, inventory, {
        "frozen_input_files": len(inventory), "source_snapshots": roles["source"],
        "old_cache_pins_matched": historical_matches, "protected_model_runtime_files": len(PROTECTED),
        "original_script_and_run2_run3_files": len(HISTORICAL_PINS),
        "freeze_order": "receipt bindings and local file order verified; not a cryptographic external timestamp",
    }


def validate_cache_support(record, geometry, raw_path, local_path, names, parameters, checks):
    with np.load(raw_path, allow_pickle=False) as cache:
        tubelets = np.asarray(cache["tubelet_frame_ids"])
        keyframes = np.asarray(cache["keyframe_ids"]).reshape(-1)
        vheat, iheat = np.asarray(cache["vjepa_raw_heatmaps"]), np.asarray(cache["ijepa_raw_heatmaps"])
        vm = np.asarray(cache["vjepa_patch_valid_mask"], bool) & (cache["vjepa_patch_counts"] > 0)
        im = np.asarray(cache["ijepa_patch_valid_mask"], bool) & (cache["ijepa_patch_counts"] > 0)
        vm &= np.asarray(cache["vjepa_valid_mask"], bool)[:, None, None] & np.isfinite(vheat)
        im &= np.asarray(cache["ijepa_valid_mask"], bool)[:, None, None] & np.isfinite(iheat)
        verror = np.asarray(cache["vjepa_raw_errors"], np.float64)
        require(np.array_equal(cache["frame_ids"], np.arange(record["frames"])), "raw timeline mismatch")
        require(np.allclose(cache["timestamps_sec"], np.arange(record["frames"]) / record["fps"],
                            atol=1e-6, rtol=0), "raw timestamps mismatch")
    require(tubelets.shape == (len(record["centers"]), 2) and tubelets.dtype.kind in "iu", "raw tubelet schema")
    require(vheat.shape == vm.shape == (len(tubelets), 24, 24), "V patch schema")
    require(iheat.shape == im.shape == (len(keyframes), 14, 14), "I patch schema")
    require(keyframes.size and keyframes.dtype.kind in "iu" and np.all(np.diff(keyframes) > 0), "I anchor order")
    checks.check(np.array_equal(record["tubelets"], tubelets), f"cache/{record['index']}/tubelets", True, False if
                 not np.array_equal(record["tubelets"], tubelets) else True)
    checks.check(np.array_equal(record["v_error"], verror), f"cache/{record['index']}/v_error", True,
                 bool(np.array_equal(record["v_error"], verror)))
    with np.load(local_path, allow_pickle=False) as cache:
        signals, valid = np.asarray(cache["signals"], np.float64), np.asarray(cache["feature_valid"], bool)
        require(signals.shape == valid.shape == (record["frames"], len(names)), "motion signal schema")
        require(list(cache["feature_names"]) == names, "motion feature ordering")
        require(np.array_equal(cache["frame_ids"], np.arange(record["frames"])), "motion timeline mismatch")
        require(abs(float(cache["fps"]) - record["fps"]) <= 1e-6, "motion FPS mismatch")
    target = grid_rectangles(geometry["v"]["box_xyxy"], 24, 24)
    source_i = grid_rectangles(geometry["i"]["box_xyxy"], 14, 14)
    source_m = np.asarray(geometry["motion_rectangles"], np.float64)
    feature_valid = np.zeros(record["valid"].shape, bool)
    support_rows, extra = [], Counter()
    joint_examples = []
    name_index = {name: i for i, name in enumerate(names)}
    required = [f"tile_{r}{c}_{suffix}" for r in range(3) for c in range(3)
                for suffix in ("residual_p90", "observed_fraction")]
    require(all(name in name_index for name in required), "missing motion tile column")
    vm, im = vm.reshape(len(vm), -1), im.reshape(len(im), -1)
    for k, center in enumerate(record["centers"]):
        previous = vm[k - 1] if k else np.zeros(576, bool)
        temporal_mask = vm[k] & previous
        j = int(np.argmin(np.abs(keyframes - center)))
        i_area = observation_area(target, source_i, im[j])
        i_observed = i_area >= parameters["minimum_projected_area_coverage"]
        cross_mask = vm[k] & i_observed
        frame = int(np.clip(round(float(center)), 0, record["frames"] - 1))
        tiles = np.zeros(9, bool)
        for tile in range(9):
            prefix = f"tile_{tile // 3}{tile % 3}_"
            residual, observed = name_index[prefix + "residual_p90"], name_index[prefix + "observed_fraction"]
            tiles[tile] = bool(valid[frame, residual] and valid[frame, observed] and
                               signals[frame, observed] > 0 and math.isfinite(signals[frame, residual]))
        m_area = observation_area(target, source_m, tiles)
        motion_mask = vm[k] & (m_area >= parameters["minimum_projected_area_coverage"])
        minimum = int(parameters["minimum_common_patches"])
        temporal_ok = bool(k and temporal_mask.sum() >= minimum)
        cross_ok, motion_ok = bool(cross_mask.sum() >= minimum), bool(motion_mask.sum() >= minimum)
        feature_valid[k] = expected_feature_valid(temporal_mask, cross_mask, motion_mask, k == 0, minimum)
        support_rows.append({"anchor": k, "nearest_i_anchor": j,
                             "i_offset_seconds": float(abs(keyframes[j] - center) / record["fps"]),
                             "v_observed_patches": int(vm[k].sum()),
                             "temporal_common_patches": int(temporal_mask.sum()),
                             "vi_common_patches": int(cross_mask.sum()),
                             "motion_common_patches": int(motion_mask.sum()),
                             "motion_supported_tiles": int(tiles.sum())})
        extra["anchors"] += 1
        extra["motion_zero_observed_tiles"] += int(not tiles.any())
        extra["motion_below_minimum_common"] += int(not motion_ok)
        extra["first_temporal_invalid"] += int(k == 0 and not record["valid"][k, :3].any())
        extra["first_temporal_joint_invalid"] += int(k == 0 and not record["valid"][k, 6])
        extra["first_non_temporal_vi_motion_joint_valid"] += int(k == 0 and record["valid"][k, 7])
        for n, relation_mask, joint_name in ((6, temporal_mask, "motion_persistent_gain"),
                                              (7, cross_mask, "vi_motion_joint")):
            common = int((relation_mask & motion_mask).sum())
            candidate = motion_ok and (temporal_ok if n == 6 else cross_ok)
            extra[joint_name + "_three_way_common_below_minimum"] += int(candidate and common < minimum)
            extra[joint_name + "_three_way_valid"] += int(feature_valid[k, n])
            if common < minimum:
                checks.check(not record["valid"][k, n],
                             f"support/{record['index']}/{k}/{joint_name}/three_way_insufficient_invalid",
                             True, bool(not record["valid"][k, n]))
            if candidate and len(joint_examples) < 4:
                joint_examples.append({"index": record["index"], "anchor": k, "feature": joint_name,
                                       "three_way_common": common, "expected_valid": bool(feature_valid[k, n]),
                                       "actual_valid": bool(record["valid"][k, n])})
        # Insufficient or completely unobserved support must never be valid,
        # but an observed constant/zero map is genuine evidence, not missing.
        if not tiles.any():
            checks.check(not record["valid"][k, 5:].any(), f"support/{record['index']}/{k}/missing_motion_invalid",
                         True, bool(not record["valid"][k, 5:].any()))
    mismatch = np.argwhere(record["valid"] != feature_valid)
    checks.check(not mismatch.size, f"support/{record['index']}/feature_valid", [], mismatch[:10].tolist())
    checks.check(not record["valid"][0, :3].any() and not record["valid"][0, 6],
                 f"support/{record['index']}/initial_temporal_and_temporal_joint", "invalid", "invalid" if
                 not record["valid"][0, :3].any() and not record["valid"][0, 6] else "valid")
    return support_rows, extra, joint_examples


def load_and_check_records(output, protocol, paths, pins, evidence, checks):
    manifest = evidence.json(paths["dataset_manifest"])
    rows = manifest["rows"]
    raw_root, local_root = paths["raw_manifest"].parent, paths["local_manifest"].parent
    raw_manifest, local_manifest = evidence.json(paths["raw_manifest"]), evidence.json(paths["local_manifest"])
    raw_entries = {e.get("name", e.get("video_name")): e for e in raw_manifest["videos"]}
    local_entries = {int(e["row_id"]): e for e in local_manifest["videos"]}
    require(len(rows) == len(raw_entries) == len(local_entries) == 77, "development cache row counts")
    require(set(raw_entries) == {r["name"] for r in rows} and set(local_entries) == set(range(77)), "cache identities")
    names = evidence.json(paths["local_names"])
    require(len(set(names)) == len(names), "duplicate motion feature names")
    geometry = evidence.json(output / "geometry_inventory.json")
    stored_events = evidence.json(output / "events.json")
    stored_support = evidence.json(output / "video_support.json")
    require(len(geometry) == len(stored_support) == 77, "geometry/support row counts")
    event_ledger_rows = read_csv(paths["events_ledger"])
    video_ledger_rows = read_csv(paths["videos_ledger"])
    ledger = {(int(e["index"]), int(e["event"])): e for e in event_ledger_rows}
    video_ledger = {int(e["index"]): e for e in video_ledger_rows}
    require(len(ledger) == len(event_ledger_rows) == 85 and len(video_ledger) == len(video_ledger_rows) == 77,
            "ledger duplicates or count mismatch")
    originals = {Path(entry["path"]).name: Path(entry["path"]) for entry in
                 strict_json(output / "input_inventory.json")["files"] if entry["role"] == "original_video"}
    require(len(originals) == 77, "source video names not unique")
    records, events, counters, joint_examples = [], [], Counter(), []
    with np.load(output / "anchor_features.npz", allow_pickle=False) as saved, np.load(paths["dataset"], allow_pickle=False) as data:
        required_keys = {f"{base}_{i}" for base in ("values", "valid", "tubelets", "centers", "v_error", "background")
                         for i in range(len(rows))}
        require(set(saved.files) == required_keys, "unexpected/missing saved arrays")
        for index, row in enumerate(rows):
            r = {**row, "index": index}
            values, valid = np.asarray(saved[f"values_{index}"]), np.asarray(saved[f"valid_{index}"])
            t, centers = np.asarray(saved[f"tubelets_{index}"]), np.asarray(saved[f"centers_{index}"])
            verror, background = np.asarray(saved[f"v_error_{index}"]), np.asarray(saved[f"background_{index}"])
            require(values.dtype == centers.dtype == verror.dtype == np.dtype("float64"), "authority arrays must be float64")
            require(values.shape == valid.shape == (len(centers), len(FEATURES)) and len(centers) > 0, "saved feature schema")
            require(valid.dtype == np.dtype("uint8") and np.isin(valid, [0, 1]).all(), "validity encoding")
            require(np.isfinite(values).all() and np.isfinite(centers).all() and np.isfinite(verror).all(), "saved nonfinite value")
            require(t.shape == (len(centers), 2) and t.dtype.kind in "iu" and (t >= 0).all() and
                    (t < row["frames"]).all() and (t[:, 0] <= t[:, 1]).all(), "saved tubelet schema")
            require(np.all(np.diff(centers) > 0) and np.array_equal(centers, t.mean(axis=1)), "saved centers/order")
            require(background.ndim == 1 and background.dtype.kind in "iu", "saved background schema")
            require(verror.shape == centers.shape, "saved raw error shape")
            valid = valid.astype(bool)
            checks.check(np.all(values[~valid] == 0), f"arrays/{index}/invalid_placeholders", "zero with valid=false",
                         "zero" if np.all(values[~valid] == 0) else "nonzero")
            r.update(values=values, valid=valid, tubelets=t, centers=centers, v_error=verror,
                     background_indices=background.tolist())
            labels = np.asarray(data[f"labels_{index}"])
            require(labels.shape == (row["frames"],) and np.isin(labels, [False, True]).all(), "label timeline/schema")
            labels = labels.astype(bool)
            spans = label_spans(labels)
            checks.same(len(spans), int(row["event_count"]), f"events/{index}/manifest_count")
            r.update(is_normal=not bool(labels.any()), spans=spans)
            expected_background = [k for k, members in enumerate(t) if not labels[members].any()]
            checks.same(r["background_indices"], expected_background, f"events/{index}/pure_background")
            for event_id, (start, end) in enumerate(spans):
                old = ledger[index, event_id]
                checks.same(old["name"], row["name"], f"events/{index}/{event_id}/ledger_name")
                checks.same([int(old["start_frame"]), int(old["end_frame"])], [start, end], f"events/{index}/{event_id}/ledger_bounds")
                require(old["default_control_matched_iou_0_5"].lower() in ("true", "false"), "ledger match flag")
                matched = old["default_control_matched_iou_0_5"].lower() == "true"
                pure = [k for k, pair in enumerate(t) if all(start <= frame <= end for frame in pair)]
                any_member = [k for k, pair in enumerate(t) if any(start <= frame <= end for frame in pair)]
                events.append({"index": index, "event": event_id, "name": row["name"], "sha256": row["sha256"],
                               "start_frame": start, "end_frame": end, "duration_seconds": (end - start + 1) / row["fps"],
                               "category": "matched_iou_0_5" if matched else old["default_control_diagnostic_stage"],
                               "pure_anchor_indices": pure, "any_member_anchor_indices": any_member,
                               "nearest_sample_to_start_seconds": float(np.min(np.abs(t - start)) / row["fps"]),
                               "nearest_sample_to_end_seconds": float(np.min(np.abs(t - end)) / row["fps"]),
                               "median_anchor_spacing_seconds": float(np.median(np.diff(centers)) / row["fps"])})
            g = geometry[index]
            for field, expected in (("index", index), ("name", row["name"]), ("sha256", row["sha256"])):
                checks.same(g[field], expected, f"geometry/{index}/{field}")
            raw_entry, local_entry = raw_entries[row["name"]], local_entries[index]
            raw_path = raw_root / raw_entry["raw_directory"] / "signals.npz"
            meta_path, local_path = raw_path.with_suffix(".json"), local_root / local_entry["file"]
            dense_path = raw_root / raw_entry["file"]
            source_path = originals[row["name"]]
            for dep in (raw_path, meta_path, local_path, dense_path, source_path):
                require(str(dep.resolve()) in pins, f"cache/source dependency not pinned: {dep}")
            checks.same(pins[str(source_path.resolve())], row["sha256"], f"source/{index}/SHA")
            checks.same(local_entry["source_sha256"], row["sha256"], f"source/{index}/motion_binding")
            checks.same(pins[str(local_path.resolve())], local_entry["feature_sha256"], f"source/{index}/motion_manifest_pin")
            checks.same(pins[str(dense_path.resolve())], raw_entry["feature_sha256"], f"source/{index}/dense_manifest_pin")
            metadata = evidence.json(meta_path)
            checks.same(metadata["profile"]["vjepa_preprocessing"],
                        "BGR_to_RGB_then_legacy_short_side_resize_center_crop_384_ImageNet_norm", f"geometry/{index}/V_profile")
            checks.same(metadata["profile"]["ijepa_preprocessing"],
                        "legacy_BGR_to_RGB_resize_256_bicubic_center_crop_224_ImageNet_norm", f"geometry/{index}/I_profile")
            checks.same(int(metadata["video"]["total_frames"]), int(row["frames"]), f"source/{index}/raw_frames")
            checks.same(float(metadata["video"]["fps"]), float(row["fps"]), f"source/{index}/raw_fps")
            # No new visual inspection/decoding: use the pinned original video
            # SHA and local cache's decoder geometry, as requested by the owner.
            lm = local_entry["metadata"]
            checks.same(lm["source_size"], [g["width"], g["height"]], f"geometry/{index}/pinned_source_shape")
            checks.same(lm["frame_size_change_count"], 0, f"geometry/{index}/cache_shape_change_count")
            checks.same(g["v"], expected_crop(g["height"], g["width"], "v"), f"geometry/{index}/V_integer_crop")
            checks.same(g["i"], expected_crop(g["height"], g["width"], "i"), f"geometry/{index}/I_integer_crop")
            rectangles, shape = expected_motion_geometry(g["height"], g["width"])
            checks.same(g["motion_analysis_shape"], shape, f"geometry/{index}/motion_shape")
            checks.same(g["motion_rectangles"], rectangles.tolist(), f"geometry/{index}/motion_border_tiles")
            checks.same(lm["analysis_size"], [shape[1], shape[0]], f"geometry/{index}/cache_analysis_shape")
            box = g["v"]["box_xyxy"]
            checks.same(g["v_source_area_fraction"], (box[2] - box[0]) * (box[3] - box[1]), f"geometry/{index}/V_source_area")
            support, counts, examples = validate_cache_support(r, g, raw_path, local_path, names, protocol["parameters"], checks)
            counters.update(counts)
            joint_examples.extend(examples)
            expected_support = {"index": index, "name": row["name"], "sha256": row["sha256"],
                                "is_normal": r["is_normal"], "support_info": support,
                                "valid_anchor_counts": dict(zip(FEATURES, r["valid"].sum(axis=0).astype(int).tolist()))}
            checks.same(stored_support[index], expected_support, f"video_support/{index}")
            video = video_ledger[index]
            checks.same(video["name"], row["name"], f"video_ledger/{index}/name")
            checks.same(video["source_sha256"], row["sha256"], f"video_ledger/{index}/source")
            records.append(r)
    checks.same(stored_events, events, "events")
    checks.same(len(events), 85, "dataset/event_count")
    checks.same(sum(r["is_normal"] for r in records), 14, "dataset/normal_videos")
    checks.same(len({r["sha256"] for r in records}), 76, "dataset/SHA_content_groups")
    for r in records:
        pure = [k for e in events if e["index"] == r["index"] for k in e["pure_anchor_indices"]]
        checks.check(len(pure) == len(set(pure)) and not set(pure) & set(r["background_indices"]),
                     f"events/{r['index']}/positive_negative_disjoint_and_unique", True,
                     len(pure) == len(set(pure)) and not bool(set(pure) & set(r["background_indices"])))
    return records, events, geometry, {"counts": dict(counters), "joint_support_examples": joint_examples[:8],
                                     "geometry_rows_checked": len(geometry),
                                     "new_video_decodes_or_visual_inspections": 0,
                                     "geometry_scope": "pinned source SHA/cache decoder dimensions, integer crops, areas and observation masks; no visual alignment claim",
                                     "feature_arithmetic_recomputed": False}


def audit_directory(output):
    start = time.perf_counter()
    output = Path(output).resolve()
    evidence, checks = Evidence(), Checks()
    result = {"schema_version": "independent-amended-local-capacity-audit-v1",
              "output": str(output), "started_utc": datetime.now(timezone.utc).isoformat(),
              "training_started_by_auditor": False, "producer_imported": False,
              "feature_values_regenerated": False, "checks": {}, "findings": [],
              "scope": "stored float64 statistics/gate, observation support, geometry metadata and pins; not full feature-generation or semantic/physical-error validation"}
    try:
        require(output.is_dir(), "--output must be an existing frozen repair directory")
        evidence.pin(Path(__file__))
        protocol, paths, pins, inventory, binding = verify_bindings(output, evidence, checks)
        result["checks"]["freeze_source_input_artifact_protection"] = binding
        records, events, geometry, support = load_and_check_records(output, protocol, paths, pins, evidence, checks)
        result["checks"]["valid_initial_commonmask_geometry"] = support
        result["checks"]["valid_initial_commonmask_geometry"]["three_way_joint_masks_verified_from_frozen_AST"] = True
        stored = evidence.json(output / "feature_summaries.json")
        summaries = [recompute_summary(records, events, n, category, protocol["parameters"])
                     for n in range(len(FEATURES)) for category in CATEGORIES]
        summary_checks = Checks()
        summary_checks.same(stored, summaries, "summaries")
        checks.count += summary_checks.count
        checks.failures.extend(summary_checks.failures)
        summary_delta = summary_checks.max_numeric_difference
        if summary_delta > checks.max_numeric_difference:
            checks.max_numeric_difference = summary_delta
            checks.max_numeric_difference_path = summary_checks.max_numeric_difference_path
        boundary = [recompute_boundary(records, events, n, protocol["parameters"]) for n in range(len(FEATURES))]
        checks.same(evidence.json(output / "boundary_diagnostic.json"), boundary, "boundary_diagnostic")
        capacity, gate_details = recompute_gate(summaries, protocol)
        report, decision = evidence.json(output / "report.json"), evidence.json(output / "decision.json")
        for key, expected in capacity.items():
            checks.same(report[key], expected, "report/" + key)
            checks.same(decision[key], expected, "decision/" + key)
        checks.same(report["feature_summaries"], summaries, "report/feature_summaries")
        for field, expected in (("status", "complete_read_only_screen"), ("schema_version", "amended-local-interaction-report-v3"),
                                ("protocol_sha256", sha256(output / "protocol.json")), ("rows", len(records)),
                                ("content_groups", len({r['sha256'] for r in records})), ("events", len(events)),
                                ("training_started", False), ("default_model_modified", False), ("performance_improvement_proven", False)):
            checks.same(report[field], expected, "report/" + field)
        category_counts = {c: sum(e["category"] == c for e in events) for c in CATEGORIES[1:]}
        checks.same(report["category_counts"], category_counts, "report/category_counts")
        areas = np.asarray([g["v_source_area_fraction"] for g in geometry])
        coverage = {"events_with_any_direct_V_member": sum(bool(e["any_member_anchor_indices"]) for e in events),
                    "events_with_pure_V_tubelet": sum(bool(e["pure_anchor_indices"]) for e in events),
                    "total_V_anchors": sum(len(r["centers"]) for r in records),
                    "source_area_seen_by_V_min_median_max": [float(areas.min()), float(np.median(areas)), float(areas.max())]}
        checks.same(report["coverage"], coverage, "report/coverage")
        for field, expected in (("status", "eligible_for_single_controlled_OOF_after_audit" if capacity["capacity_screen_supported"] else
                                 "stop_before_training_insufficient_fixed_feature_evidence"),
                                ("training_started", False), ("default_promoted", False),
                                ("performance_improvement_proven", False), ("source_and_inputs_verified_again_before_run", True)):
            checks.same(decision[field], expected, "decision/" + field)
        result["recomputed_capacity"] = capacity
        result["gate_condition_details"] = gate_details
        result["recomputed_feature_summaries"] = summaries
        result["checks"]["statistics"] = {
            "summary_rows": len(summaries), "max_summary_absolute_difference": summary_delta,
            "directions": "high fixed; no reversal/max(AUC,1-AUC)",
            "positive_weighting": "equal SHA content, equal nonempty event within content, equal observed anchor within event",
            "normal_weighting": "equal normal SHA content, equal aliases within content, equal observed anchor within alias",
            "within_video_background": "separate event AUC then content-balanced mean; never pooled with normal videos",
            "bootstrap": {"replicates": protocol["parameters"]["bootstrap_replicates"],
                          "seed": protocol["parameters"]["bootstrap_seed"], "rng": "PCG64",
                          "unit": "separate positive/normal SHA-content multinomial draws",
                          "pointwise_ci": [.025, .975], "one_sided_gate_quantile": .05 / 6,
                          "interpolation": "linear", "independent_pairwise_auc_and_matrix_bootstrap": True},
            "same_target_two_base_families_checked": True,
            "joint_features_excluded_from_gate": True, "category_counts": category_counts,
            "target_content_groups": {c: len({e["sha256"] for e in events if e["category"] == c}) for c in (LOW, GATED)},
        }
        counts = support["counts"]
        if counts.get("first_non_temporal_vi_motion_joint_valid", 0):
            result["findings"].append({"severity": "P3", "code": "INITIAL_JOINT_POLICY_OVERBROAD",
                "gate_affecting": False, "path": str(output / "protocol.json"),
                "detail": "Protocol says initial temporal/joint anchors invalid, but non-temporal vi_motion_joint has valid initial anchors. It does not require a previous anchor; wording should distinguish temporal from non-temporal joint products.",
                "valid_initial_vi_motion_joint_rows": counts["first_non_temporal_vi_motion_joint_valid"]})
        evidence.verify_unchanged()
        result["checks"]["consumed_evidence_posthash_unchanged"] = True
        result["audit_status"] = "failed" if checks.failures else ("passed_with_non_gate_findings" if result["findings"] else "passed")
        result["gate_recalculation_verified"] = not checks.failures
        result["training_allowed_by_screen"] = bool(not checks.failures and capacity["capacity_screen_supported"])
        result["conclusion"] = ("insufficient fixed-feature evidence: stop before training; not proof absence of trainable capacity"
                                if not capacity["capacity_screen_supported"] else
                                "fixed screen passes; independent audit is not performance or production approval")
    except (AuditError, KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        result.update(audit_status="failed", gate_recalculation_verified=False, training_allowed_by_screen=False,
                      fatal_error={"type": type(exc).__name__, "message": str(exc)})
        try:
            evidence.verify_unchanged()
            result["checks"]["consumed_evidence_posthash_unchanged"] = True
        except (AuditError, OSError) as drift:
            result["checks"]["consumed_evidence_posthash_unchanged"] = False
            result["posthash_error"] = str(drift)
    result.update(comparisons=checks.count, failures=checks.failures,
                  max_absolute_numeric_difference=checks.max_numeric_difference,
                  max_absolute_numeric_difference_path=checks.max_numeric_difference_path,
                  consumed_files=len(evidence.initial), pin_comparisons=evidence.pins_checked,
                  auditor_source_sha256=sha256(Path(__file__)),
                  elapsed_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat(),
                  runtime={"python": platform.python_version(), "numpy": np.__version__})
    result["consumed_file_sha256"] = evidence.initial
    return result


def self_test():
    """Focused in-memory checks; no fixtures, directories, or reports written."""
    names = []

    def test(name, condition):
        require(bool(condition), "self-test failed: " + name)
        names.append(name)

    def rejects(name, operation):
        try:
            operation()
        except AuditError:
            names.append(name)
        else:
            raise AuditError("self-test must reject: " + name)

    test("AUC_high_direction", weighted_auc([3], [1]) == 1)
    test("AUC_no_two_sided_flip", weighted_auc([1], [3]) == 0)
    test("AUC_ties_half", weighted_auc([2, 2], [2]) == .5)
    test("AUC_mixed_U_statistic", abs(weighted_auc([1, 1, 3], [0, 1, 2]) - 2 / 3) < 1e-14)
    test("AUC_explicit_weight", weighted_auc([0, 2], [1], [1, 3]) == .75)
    test("AUC_empty_is_unavailable", weighted_auc([], [1]) is None)
    rejects("AUC_nonfinite_rejected", lambda: weighted_auc([np.nan], [1]))
    rejects("AUC_negative_weight_rejected", lambda: weighted_auc([1], [2], [-1]))
    rejects("AUC_nan_weight_rejected", lambda: weighted_auc([1], [2], [np.nan]))
    rejects("AUC_shape_rejected", lambda: weighted_auc([[1]], [2]))
    group = equal_block_groups([("a", [0]), ("a", [3, 3, 3]), ("b", [])])
    test("equal_event_not_equal_anchor", weighted_auc(group["a"][0], [1], group["a"][1]) == .5)
    test("empty_score_groups_not_bootstrapped", set(group) == {"a"})
    test("SHA_mean_not_event_mean", abs(group_average([("a", 0), ("a", 0), ("b", 1)]) - .5) < 1e-14)
    test("weighted_quantile_inverse_CDF", weighted_quantile([1, 2, 3], [.5, .25, .25], .5) == 1)
    test("linear_quantile_explicit", np.array_equal(linear_quantile([0, 10], [.025, .975]), [.25, 9.75]))
    r = {"values": np.ones((3, 8)), "valid": np.ones((3, 8), bool), "centers": np.arange(3)}
    r["values"][0, 0], r["valid"][0, 0] = 999, False
    test("invalid_score_excluded_not_imputed", observed_values(r, 0, [0, 1]).tolist() == [1])
    test("initial_invalid_excluded", observed_values(r, 0, [0]).size == 0)
    test("constant_observed_zero_not_missing", weighted_auc([0], [0]) == .5)
    test("inclusive_label_spans", label_spans([True, True, False, True]) == [(0, 1), (3, 3)])
    targets = grid_rectangles([0, 0, 1, 1], 2, 2)
    test("projection_no_observation_zero_support", np.array_equal(observation_area(targets, targets, [False] * 4), np.zeros(4)))
    test("projection_fully_observed", np.array_equal(observation_area(targets, targets, [True] * 4), np.ones(4)))
    half = observation_area(np.array([[0, 0, 1, 1.]]), np.array([[0, 0, .5, 1]]), [True])
    test("projected_half_area_threshold", half.tolist() == [.5])
    disjoint_a, disjoint_b = np.r_[np.ones(32, bool), np.zeros(32, bool)], np.r_[np.zeros(32, bool), np.ones(32, bool)]
    triple_valid = expected_feature_valid(disjoint_a, disjoint_a, disjoint_b, False, 32)
    test("joint_three_way_missing_invalid", triple_valid[:6].all() and not triple_valid[6:].any())
    all_same = expected_feature_valid(disjoint_a, disjoint_a, disjoint_a, False, 32)
    test("joint_three_way_supported_valid", all_same.all())
    first = expected_feature_valid(np.zeros(64, bool), disjoint_a, disjoint_a, True, 32)
    test("non_temporal_joint_initial_allowed", not first[:3].any() and not first[6] and first[7])
    test("V_legacy_float_integer_geometry", expected_crop(720, 1280, "v")["resized_height"] == 437)
    test("I_integer_center_rounding", expected_crop(720, 1280, "i")["crop_left"] == 116)
    rects, shape = expected_motion_geometry(720, 1280)
    test("motion_border4_integer_tiles", shape == [72, 128] and rects[0, 0] == 4 / 128 and rects[0, 1] == 4 / 72)
    # Tests do not depend on the presence of a real output directory.
    rules = {
        "minimum_event_valid_coverage_all_and_target": .8,
        "minimum_event_anchor_valid_fraction_all_and_target": .8,
        "minimum_normal_anchor_valid_fraction": .8,
        "minimum_overall_high_direction_group_auc": .55,
        "minimum_target_high_direction_group_auc": .6,
        "minimum_target_group_bootstrap_ci_lower_strict": .5,
        "minimum_target_within_video_auc": .55,
        "minimum_target_within_video_available_fraction": .5,
        "minimum_target_strict_normal_q95_event_recall": .2,
        "minimum_qualifying_features": 2, "minimum_qualifying_families": 2,
        "target_categories": [LOW, GATED],
    }
    protocol = {"training_capacity_rule": rules, "gate_representatives": REPRESENTATIVES}
    passing = {"event_valid_coverage_content_balanced": .8, "event_anchor_valid_fraction_content_balanced": .8,
               "normal_anchor_valid_fraction_content_balanced": .8, "auc": .6, "bonferroni_lower": .51,
               "within_video_auc_content_balanced": .55, "within_video_available_fraction": .5,
               "strict_normal_q95_event_recall_content_balanced": .2}
    summary = []
    for feature in REPRESENTATIVES.values():
        for category in ("all", LOW, GATED):
            item = {**passing, "feature": feature, "category": category}
            if category != "all":
                item["auc"] = .4
            summary.append(item)
    for item in summary:
        if (item["feature"], item["category"]) in (("v_strength_change", LOW), ("motion_topk_gain", GATED)):
            item["auc"] = .7
    gate, details = recompute_gate(summary, protocol)
    test("no_cross_target_family_stitch", not gate["capacity_screen_supported"] and len(gate["qualifying"]) == 2)
    for item in summary:
        if (item["feature"], item["category"]) == ("vi_strength_colocation", LOW):
            item["auc"] = .7
    test("two_families_same_target_pass", recompute_gate(summary, protocol)[0]["capacity_screen_supported"])
    test("Bonferroni_lower_strict", not gate_requirements(passing, {**passing, "bonferroni_lower": .5}, rules)["target_bonferroni_lower"])
    test("pointwise_CI_never_substitutes", not gate_requirements(passing, {**passing, "bonferroni_lower": .49, "ci95": [.6, .9]}, rules)["target_bonferroni_lower"])
    test("unavailable_background_fails_gate", not gate_requirements(passing, {**passing, "within_video_auc_content_balanced": None}, rules)["target_within_video_auc"])
    test("joint_features_never_qualify", all(q["feature"] in REPRESENTATIVES.values() for q in recompute_gate(summary, protocol)[0]["qualifying"]))
    parameters = {"bootstrap_seed": 20261003, "bootstrap_replicates": 2000,
                  "gate_one_sided_familywise_alpha": .05, "gate_representative_target_tests": 6,
                  "normal_tail_quantile": .95}
    p, n = equal_block_groups([("p", [3])]), equal_block_groups([("n", [1])])
    interval = group_auc_summary(p, n, parameters)
    test("degenerate_valid_bootstrap", interval["ci95"] == [1., 1.] and interval["bonferroni_lower"] == 1)
    test("bootstrap_deterministic", interval == group_auc_summary(p, n, parameters))
    test("missing_normal_CI_unavailable", group_auc_summary(p, {}, parameters)["ci95"] is None)
    p_record = {"index": 0, "sha256": "p", "is_normal": False, "centers": np.arange(2),
                "values": np.full((2, 8), 3.), "valid": np.ones((2, 8), bool), "background_indices": []}
    p_record["valid"][1] = False
    n_record = {"index": 1, "sha256": "n", "is_normal": True, "centers": np.arange(1),
                "values": np.ones((1, 8)), "valid": np.ones((1, 8), bool), "background_indices": [0]}
    events = [{"index": 0, "category": LOW, "pure_anchor_indices": [i]} for i in (0, 1)]
    partial = recompute_summary([p_record, n_record], events, 0, LOW, parameters)
    test("missing_event_remains_coverage_denominator", partial["event_valid_coverage_content_balanced"] == .5 and partial["valid_event_count"] == 1)
    test("pure_anchor_coverage_separate", partial["event_anchor_valid_fraction_content_balanced"] == .5)
    test("strict_tail_ties_not_hits", partial["normal_strict_tail_mass"] == 0 and partial["normal_tie_mass_at_q95"] == 1)
    probe = Checks()
    probe.same(True, 1, "bool_is_not_integer")
    test("typed_report_fields", len(probe.failures) == 1)
    rejects("producer_AST_never_executes", lambda: literal_constants("LOW=__import__('os').system('not-run')"))
    return {"passed": len(names), "failed": 0, "writes": 0, "checks": names}


def validate_audit_target(output, target):
    output, target = Path(output).resolve(), Path(target).resolve()
    require(output.is_dir(), "--output does not exist")
    require(target.suffix.lower() == ".json", "--audit-output must be a JSON file")
    require(target.parent.is_dir(), "audit-output parent must already exist; no directories are created")
    require(not target.exists(), f"refusing to overwrite existing audit-output: {target}")
    require(target not in {output / name for name in ARTIFACTS}, "audit output cannot replace producer artifact")
    require((output / "source_snapshot") not in target.parents, "audit output cannot enter frozen source_snapshot")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="existing, explicitly frozen and completed repair directory")
    parser.add_argument("--audit-output", type=Path, help="NEW JSON; exclusive creation; no overwrite")
    parser.add_argument("--self-test", action="store_true", help="run focused in-memory tests without any writes")
    args = parser.parse_args(argv)
    if args.self_test:
        try:
            print(json.dumps(self_test(), ensure_ascii=False))
            return 0
        except (AuditError, ValueError, TypeError) as exc:
            print(json.dumps({"self_test": "failed", "error": str(exc)}, ensure_ascii=False))
            return 1
    if args.output is None or args.audit_output is None:
        parser.error("--output and --audit-output are required unless --self-test is used")
    try:
        output = args.output.resolve()
        target = validate_audit_target(output, args.audit_output)
        result = audit_directory(output)
        result["audit_output"] = str(target)
        result["audit_content_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                                                   ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        # Exclusive open is the sole write. No placeholders, mkdir, backups,
        # mutation of producer evidence, or fallback to an existing JSON.
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"audit_status": result["audit_status"], "audit_output": str(target),
                          "gate_recalculation_verified": result["gate_recalculation_verified"],
                          "capacity_screen_supported": result.get("recomputed_capacity", {}).get("capacity_screen_supported"),
                          "max_absolute_numeric_difference": result["max_absolute_numeric_difference"],
                          "failure_count": len(result["failures"]), "findings": result["findings"],
                          "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))
        return 1 if result["audit_status"] == "failed" else 0
    except (AuditError, OSError, ValueError) as exc:
        print(json.dumps({"audit_status": "not_written", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
