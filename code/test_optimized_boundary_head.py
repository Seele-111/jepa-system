#!/usr/bin/env python3
"""Synthetic-only CPU checks; run with python -B -m unittest this module.

No dataset access, GPU libraries, installation, training artifacts or caches.
Actual ET training/parity is checked in the existing WSL sklearn environment.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

import optimized_boundary_head as boundary
from optimized_locator import export_model, interval_iou

HAS_SKLEARN = os.name != "nt" and importlib.util.find_spec("sklearn") is not None


def constant_bundle(dimension=2, probability=0.0):
    return {key: {"kind": "constant", "n_features": dimension, "probability": probability}
            for key in ("start_model", "end_model")}


def stump_bundle():
    tree = {"left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
            "threshold": [1.0, -2.0, -2.0], "value": [0.5, 0.2, 0.8]}
    return {key: {"kind": "forest", "n_features": 2, "trees": [tree.copy()]}
            for key in ("start_model", "end_model")}


class TargetsAndInputTests(unittest.TestCase):
    def test_inclusive_targets_floor_fps_tolerance_union_and_edge_clipping(self):
        labels = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
        for fps, expected_start, expected_end in (
            (10, [0, 4, 7], [1, 4, 9]),
            (16.66, [0, 4, 7], [1, 4, 9]),
            (29.97, [0, 1, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 8, 9]),
            (50, list(range(10)), list(range(10))),
        ):
            with self.subTest(fps=fps):
                start, end = boundary._boundary_targets(labels, fps)
                np.testing.assert_array_equal(np.flatnonzero(start), expected_start)
                np.testing.assert_array_equal(np.flatnonzero(end), expected_end)
                self.assertTrue(set(np.unique(start)) <= {0, 1})
                self.assertTrue(set(np.unique(end)) <= {0, 1})
        start, end = boundary._boundary_targets(np.ones(5), 10)
        np.testing.assert_array_equal(start, [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(end, [0, 0, 0, 0, 1])  # end, not end+1

    def test_empty_and_single_class_exports(self):
        for values, labels, fps, dimension, probability in (
            ([], [], [], 0, 0.0),
            ([np.empty((0, 3))], [np.empty(0)], [30], 3, 0.0),
            ([np.zeros((8, 3))], [np.zeros(8)], [30], 3, 0.0),
            ([np.zeros((1, 3))], [np.ones(1)], [30], 3, 1.0),
            ([np.zeros((2, 3))], [np.ones(2)], [30], 3, 1.0),
        ):
            with self.subTest(dimension=dimension, probability=probability, videos=len(values)):
                bundle = boundary.train_boundary_heads(values, labels, fps, seed=19)
                self.assertEqual(set(bundle), {"start_model", "end_model"})
                for model in bundle.values():
                    self.assertEqual(model, {"kind": "constant", "n_features": dimension,
                                             "probability": probability})
                saved = json.loads(json.dumps(bundle, allow_nan=False))
                for result in boundary.predict_boundary_heads(saved, np.zeros((7, 3))):
                    np.testing.assert_array_equal(result, np.full(7, probability, np.float32))
                for result in boundary.predict_boundary_heads(saved, np.empty((0, 3))):
                    self.assertEqual(result.shape, (0,))
                    self.assertEqual(result.dtype, np.float32)

    def test_training_rejects_nonfinite_nonreal_or_invalid_shapes(self):
        for values in (np.zeros(2), np.zeros((2, 0)), np.zeros((2, 1, 1)),
                       [[np.nan], [0]], [[np.inf], [0]], [[1e300], [0]],
                       [["1"], ["2"]], np.ones((2, 1), dtype=complex),
                       np.ones((2, 1), dtype=bool)):
            with self.subTest(values=repr(values)):
                with self.assertRaises(ValueError):
                    boundary.train_boundary_heads([values], [np.zeros(2)], [10], 1)
        with self.assertRaises(ValueError):
            boundary.train_boundary_heads([np.zeros((2, 1)), np.empty((0, 2))],
                                          [np.zeros(2), []], [10, 10], 1)

    def test_training_rejects_labels_before_float32_rounding(self):
        for labels in ([0, 2], [-1, 1], [0, 1 + 1e-12], [0, np.nan],
                       [0, np.inf], [[0, 1]], [1], [0j, 1j], ["0", "1"]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    boundary.train_boundary_heads([np.zeros((2, 2))], [labels], [30], 1)

    def test_training_rejects_video_counts_fps_and_seed(self):
        for values, labels, fps in (([], [], [30]), ([], [[]], []),
                                   (np.empty((0, 2)), [], [])):
            with self.assertRaises(ValueError):
                boundary.train_boundary_heads(values, labels, fps, 1)
        for fps in (0, -1, np.nan, np.inf, True, "30"):
            with self.subTest(fps=fps):
                with self.assertRaises(ValueError):
                    boundary.train_boundary_heads([np.empty((0, 2))], [[]], [fps], 1)
        for seed in (-1, 2**32, 1.5, True, "7"):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    boundary.train_boundary_heads([], [], [], seed)


class PortablePredictionTests(unittest.TestCase):
    def test_hand_exported_forest_and_json_round_trip(self):
        bundle = json.loads(json.dumps(stump_bundle(), allow_nan=False))
        values = np.array([[0, 5], [1, 6], [2, 7]], dtype=np.float64)
        original = values.copy()
        for result in boundary.predict_boundary_heads(bundle, values):
            np.testing.assert_array_equal(result, np.array([0.2, 0.2, 0.8], np.float32))
            self.assertEqual(result.shape, (3,))
        np.testing.assert_array_equal(values, original)

    def test_prediction_feature_and_bundle_validity(self):
        for values in (np.zeros(2), np.zeros((2, 0)), np.zeros((2, 3)),
                       [[np.nan, 0]], [[np.inf, 0]], [[1e300, 0]], [["0", "1"]]):
            with self.subTest(values=repr(values)):
                with self.assertRaises(ValueError):
                    boundary.predict_boundary_heads(constant_bundle(), values)
        for bundle in (None, {}, {"start_model": {}},
                       {"start_model": {"kind": "unsupported"}, "end_model": {}},
                       constant_bundle(probability=np.nan), constant_bundle(probability=1.01),
                       constant_bundle(dimension=-1), constant_bundle(dimension=True)):
            with self.subTest(bundle=bundle):
                with self.assertRaises(ValueError):
                    boundary.predict_boundary_heads(bundle, np.zeros((2, 2)))
        bundle = constant_bundle()
        bundle["start_model"]["n_features"] = 0
        with self.assertRaises(ValueError):
            boundary.predict_boundary_heads(bundle, np.zeros((2, 2)))
        bundle = stump_bundle()
        bundle["start_model"]["trees"][0]["value"] = [0.5, np.nan, 0.8]
        with self.assertRaises(ValueError):
            boundary.predict_boundary_heads(bundle, np.zeros((2, 2)))

    def test_import_prediction_constants_and_refinement_without_sklearn_or_torch(self):
        source = r'''
import builtins
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split(".")[0] in {"sklearn", "torch"}:
        raise AssertionError("forbidden training/GPU dependency: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
import numpy as np
from optimized_boundary_head import train_boundary_heads, predict_boundary_heads, refine_intervals
for values, labels, fps in (([], [], []), ([np.zeros((1, 2))], [[1]], [30])):
    result = predict_boundary_heads(train_boundary_heads(values, labels, fps, 1), np.zeros((8, 2)))
    assert all(p.shape == (8,) for p in result)
tree = {"left": [1,-1,-1], "right": [2,-1,-1], "feature": [0,-2,-2],
        "threshold": [0.5,-2,-2], "value": [0.5,0.2,0.8]}
bundle = {k: {"kind": "forest", "n_features": 2, "trees": [tree]}
          for k in ("start_model", "end_model")}
a, b = predict_boundary_heads(bundle, np.zeros((8, 2)))
assert refine_intervals([], a, b, 10, {"boundary_seconds": .4}) == []
assert refine_intervals([(2,5)], a, b, 10, {"boundary_seconds": .4}) == [(2,5)]
'''
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-B", "-c", source],
                                cwd=Path(__file__).resolve().parent, env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class RefinementTests(unittest.TestCase):
    def refine(self, intervals, start, end, seconds=.2, fps=10, **extra):
        return boundary.refine_intervals(intervals, start, end, fps,
                                         {"boundary_seconds": seconds, **extra})

    def test_disabled_is_exact_noop_even_for_unsorted_overlaps(self):
        intervals = [(8, 10), (2, 9), (0, 0)]
        start = np.linspace(0, 1, 12)
        for seconds, fps in ((0, 30), (.2, 1)):
            self.assertEqual(self.refine(intervals, start, start[::-1], seconds, fps), intervals)
        self.assertEqual(boundary.refine_intervals(intervals, start, start, 30, {}), intervals)

    def test_no_candidates_stays_empty_even_with_certain_heads(self):
        self.assertEqual(self.refine([], np.ones(12), np.ones(12), .6), [])
        self.assertEqual(self.refine([], np.empty(0), np.empty(0), .6), [])

    def test_local_windows_and_original_end_center(self):
        start, end = np.zeros(20), np.zeros(20)
        start[[0, 7, 9]] = [1, .9, 1]  # 0 and 9 outside start window [3,7]
        end[[11, 15, 17]] = [.95, .8, 1]  # original end window [11,15]
        self.assertEqual(self.refine([(5, 13)], start, end), [(7, 11)])
        self.assertEqual(self.refine([(5, 13)], start, end, .2, 5), [(5, 13)])

    def test_stay_penalty_resists_farther_peak_but_allows_a_clear_improvement(self):
        start, end = np.zeros(20), np.zeros(20)
        start[[3, 4, 7]] = [.5, .51, .54]
        end[[12, 16]] = [.4, .449]
        self.assertEqual(self.refine([(3, 12)], start, end, .4), [(3, 12)])
        start[7], end[16] = .56, .46
        self.assertEqual(self.refine([(3, 12)], start, end, .4), [(7, 16)])

    def test_exact_ties_prefer_staying_then_nearest_then_lower_frame(self):
        start, end = np.zeros(14), np.zeros(14)
        start[[3, 5]], end[[9, 11]] = .05, .05
        self.assertEqual(self.refine([(4, 10)], start, end, .2, 5), [(4, 10)])
        start[[3, 5]], end[[9, 11]] = .9, .9
        self.assertEqual(self.refine([(4, 10)], start, end), [(3, 9)])
        # Same adjusted score at unequal distances: prefer the nearer frame.
        start[:] = 0
        start[5], start[7] = .05 / 4, .15 / 4
        self.assertEqual(self.refine([(4, 10)], start, np.zeros(14), .4), [(4, 10)])

    def test_start_not_beyond_original_end_and_end_not_before_updated_start(self):
        start, end = np.zeros(14), np.zeros(14)
        start[[6, 7, 9]], end[[6, 8]] = [.6, .8, 1], [1, .9]
        self.assertEqual(self.refine([(5, 7)], start, end, .4), [(7, 8)])

    def test_frame_edges_singletons_and_inclusive_iou(self):
        start, end = np.zeros(12), np.zeros(12)
        start[0], end[11] = 1, 1
        self.assertEqual(self.refine([(0, 3), (9, 11)], start, end, .4), [(0, 3), (9, 11)])
        start[:], end[:] = 0, 0
        start[4], end[4] = 1, 1
        result = self.refine([(4, 4)], start, end, .6)
        self.assertEqual(result, [(4, 4)])
        self.assertEqual(interval_iou(result[0], (4, 4)), 1)
        self.assertEqual(interval_iou(result[0], (4, 5)), .5)
        self.assertEqual(interval_iou(result[0], (5, 5)), 0)

    def test_merge_only_real_overlap_without_gap_or_adjacent_bridging(self):
        start, end = np.zeros(20), np.zeros(20)
        start[[2, 7]], end[[7, 12]] = 1, 1
        self.assertEqual(self.refine([(8, 12), (2, 6)], start, end), [(2, 12)])
        for intervals, expected in (([(2, 4), (5, 8)], [(2, 4), (5, 8)]),
                                    ([(2, 4), (6, 8)], [(2, 4), (6, 8)]),
                                    ([(3, 8), (1, 12)], [(1, 12)])):
            with self.subTest(intervals=intervals):
                self.assertEqual(self.refine(intervals, np.zeros(20), np.zeros(20),
                                             gap_seconds=100, min_seconds=100), expected)

    def test_synthetic_short_and_fractional_fps_windows_stay_valid_without_mutation(self):
        rng = np.random.default_rng(42)
        for length in (1, 2, 7, 25):
            start, end = rng.random(length), rng.random(length)
            original_start, original_end = start.copy(), end.copy()
            for fps in (2.5, 10, 29.97, 60):
                s, e = sorted(rng.integers(0, length, size=2).tolist())
                for seconds in (.2, .4, .6):
                    result = self.refine([(s, e)], start, end, seconds, fps)
                    self.assertEqual(len(result), 1)
                    a, b = result[0]
                    radius = round(seconds * fps)
                    self.assertTrue(0 <= a <= e and a <= b < length)
                    self.assertLessEqual(abs(a - s), radius)
                    self.assertLessEqual(abs(b - e), radius)
            np.testing.assert_array_equal(start, original_start)
            np.testing.assert_array_equal(end, original_end)

    def test_refinement_rejects_invalid_probabilities_fps_config_and_intervals(self):
        for bad in ([np.nan, 0], [np.inf, 0], [-.01, 0], [1.01, 0], [[.2, .3]],
                    ["0", "1"], [0j, 1j]):
            for start, end in ((bad, np.zeros(2)), (np.zeros(2), bad)):
                with self.assertRaises(ValueError):
                    self.refine([], start, end)
        with self.assertRaises(ValueError):
            self.refine([], np.zeros(2), np.zeros(3))
        for fps in (0, -1, np.nan, np.inf, True, "30"):
            with self.assertRaises(ValueError):
                self.refine([], np.zeros(4), np.zeros(4), fps=fps)
        for config in (None, {"boundary_seconds": -.1}, {"boundary_seconds": np.inf},
                       {"boundary_seconds": np.nan}, {"boundary_seconds": True},
                       {"boundary_seconds": "0.2"}):
            with self.assertRaises(ValueError):
                boundary.refine_intervals([], np.zeros(4), np.zeros(4), 10, config)
        for intervals in ([(2, 1)], [(-1, 1)], [(0, 4)], [(0.0, 1)], [(True, 1)],
                          [(0,)], [0], [(0, 1, 2)], np.array(1), "0,1"):
            with self.subTest(intervals=intervals):
                with self.assertRaises(ValueError):
                    self.refine(intervals, np.zeros(4), np.zeros(4), 0)


@unittest.skipUnless(HAS_SKLEARN, "ET CPU training/parity requires the existing WSL sklearn environment")
class ExtraTreesCPUTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.ensemble import ExtraTreesClassifier

        rng = np.random.default_rng(105)
        cls.values = [rng.normal(size=(length, 4)) for length in (13, 37, 59, 0)]
        cls.labels = [np.zeros(len(x), np.uint8) for x in cls.values]
        cls.labels[0][1:6] = cls.labels[0][9:11] = 1
        cls.labels[1][7:19] = cls.labels[1][27:35] = 1
        cls.fps, cls.models, cls.fits = [10, 29.97, 60, 30], [], []
        original_fit = ExtraTreesClassifier.fit

        def recorded_fit(model, values, labels, sample_weight=None):
            cls.fits.append((values.copy(), labels.copy(), sample_weight.copy()))
            return original_fit(model, values, labels, sample_weight=sample_weight)

        def recorded_export(model):
            cls.models.append(model)
            return export_model(model)

        cls.before_values = [x.copy() for x in cls.values]
        cls.before_labels = [y.copy() for y in cls.labels]
        with mock.patch.object(ExtraTreesClassifier, "fit", recorded_fit), \
                mock.patch.object(boundary, "export_model", side_effect=recorded_export):
            cls.bundle = boundary.train_boundary_heads(cls.values, cls.labels, cls.fps, seed=71)

    def test_fixed_et_budget_export_schema_and_unchanged_feature_rows(self):
        self.assertEqual(set(self.bundle), {"start_model", "end_model"})
        for model, exported, (features, _, _) in zip(self.models, self.bundle.values(), self.fits):
            expected = {"n_estimators": 128, "max_depth": 7, "min_samples_leaf": 5,
                        "max_features": .5, "n_jobs": 2, "random_state": 71, "class_weight": None}
            self.assertEqual({key: model.get_params()[key] for key in expected}, expected)
            self.assertEqual(set(exported), {"kind", "n_features", "trees"})
            self.assertEqual(exported["kind"], "forest")
            self.assertEqual(exported["n_features"], 4)
            self.assertEqual(len(exported["trees"]), 128)
            self.assertEqual(set(exported["trees"][0]), {"left", "right", "feature", "threshold", "value"})
            np.testing.assert_array_equal(features, np.concatenate(self.values).astype(np.float32))
        for before, after in zip(self.before_values + self.before_labels, self.values + self.labels):
            np.testing.assert_array_equal(before, after)

    def test_inclusive_training_bands_equal_video_and_weighted_class_balance(self):
        base = np.concatenate([np.full(len(x), 1 / len(x)) for x in self.values if len(x)])
        np.testing.assert_allclose([base[:13].sum(), base[13:50].sum(), base[50:].sum()], 1)
        expected_starts, expected_ends = np.zeros(109, np.uint8), np.zeros(109, np.uint8)
        expected_starts[[1, 9]] = 1
        expected_ends[[5, 10]] = 1
        for start, end in ((7, 18), (27, 34)):
            expected_starts[13 + start - 1:13 + start + 2] = 1
            expected_ends[13 + end - 1:13 + end + 2] = 1
        for (_, target, weight), expected in zip(self.fits, (expected_starts, expected_ends)):
            np.testing.assert_array_equal(target, expected)
            masses = np.array([base[target == c].sum() for c in (0, 1)])
            np.testing.assert_allclose(weight, base * 3 / (2 * masses[target]), rtol=1e-14)
            np.testing.assert_allclose([weight[target == c].sum() for c in (0, 1)], [1.5, 1.5])

    def test_uncalibrated_numpy_export_parity_on_training_and_unseen_rows(self):
        bundle = json.loads(json.dumps(self.bundle, allow_nan=False))
        for values in (np.concatenate(self.values), np.random.default_rng(6).normal(size=(27, 4))):
            for result, model in zip(boundary.predict_boundary_heads(bundle, values), self.models):
                np.testing.assert_allclose(result, model.predict_proba(values)[:, 1], atol=1e-7, rtol=0)
                self.assertEqual(result.dtype, np.float32)
        for result in boundary.predict_boundary_heads(bundle, np.empty((0, 4))):
            self.assertEqual(result.shape, (0,))

    def test_seeded_cpu_training_is_deterministic(self):
        repeated = boundary.train_boundary_heads(self.values, self.labels, self.fps, seed=71)
        self.assertEqual(repeated, self.bundle)


if __name__ == "__main__":
    unittest.main()
