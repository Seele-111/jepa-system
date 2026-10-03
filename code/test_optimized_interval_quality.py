#!/usr/bin/env python3
"""Synthetic CPU tests only: python -B -m unittest test_optimized_interval_quality -v.

No dataset, baseline training, installation, model file or cache is touched.
The one real, small ET parity fit is skipped explicitly without sklearn, and
can run in the already-installed WSL environment. Other tests need only NumPy.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import optimized_interval_quality as quality

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def named_features(p, fps, intervals, evidence, video_probability=0.3):
    values, names = quality.interval_features(p, video_probability, fps, intervals, evidence)
    return values, {name: values[:, j] for j, name in enumerate(names)}


def stump_state():
    stump = {"left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
             "threshold": [0.500000005, -2.0, -2.0], "value": [0.5, 0.1, 0.9]}
    constant = {"left": [-1], "right": [-1], "feature": [-2],
                "threshold": [-2.0], "value": [0.4]}
    return {"kind": quality.QUALITY_HEAD_KIND, "n_features": 2, "trees": [stump, constant]}


class ProposalBankTests(unittest.TestCase):
    def test_empty_low_peak_and_inclusive_peak(self):
        self.assertEqual(quality.proposal_bank([], 25), [])
        self.assertEqual(quality.proposal_bank([0.0, 0.349999, 0.2], 25), [])
        self.assertEqual(quality.proposal_bank([0.35], 25), [(0, 0)])
        self.assertEqual(quality.proposal_bank([0.2, 0.35], 1), [(0, 1), (1, 1)])

    def test_mean_cutoff_and_all_small_grid_spans(self):
        self.assertIn((0, 1), quality.proposal_bank([0.0, 0.4], 1))
        self.assertNotIn((0, 1), quality.proposal_bank([0.049, 0.35], 1))
        self.assertEqual(quality.proposal_bank([0.8] * 4, 1),
                         [(s, e) for s in range(4) for e in range(s, 4)])

    def test_grid_includes_endpoints_and_caps_at_64(self):
        total = 66
        actual = set(quality.proposal_bank(np.full(total, 0.8), 0.1))
        grid = np.linspace(0, total - 1, 64, dtype=np.int64).tolist()
        expected = {(s, e) for s in grid for e in grid if s <= e}
        expected.update((i, i) for i in range(total))
        self.assertEqual(actual, expected)
        self.assertIn((0, total - 1), actual)
        self.assertEqual(len(actual), 2082)

    def test_full_sliding_short_windows_at_all_starts(self):
        actual = set(quality.proposal_bank(np.full(23, 0.6), 50))
        for width in (2, 6, 12):
            for start in range(23 - width + 1):
                self.assertIn((start, start + width - 1), actual)
        self.assertIn((1, 2), actual)  # Neither endpoint is on the six-frame grid.
        self.assertIn((5, 16), actual)

    def test_run_endpoints_move_independently_and_clip(self):
        p = np.zeros(21)
        p[7:15] = 0.6
        actual = set(quality.proposal_bank(p, 25))
        for start in (4, 7, 10):
            for end in (11, 14, 17):
                self.assertIn((start, end), actual)
        edge = set(quality.proposal_bank(np.full(10, 0.8), 25))
        for candidate in ((0, 6), (0, 9), (3, 6), (3, 9)):
            self.assertIn(candidate, edge)
        self.assertTrue(all(0 <= s <= e < 10 for s, e in edge))

    def test_all_four_run_cutoffs_are_used(self):
        p = np.full(29, 0.26)
        p[7:23] = 0.41
        p[10:20] = 0.56
        p[13:17] = 0.71
        actual = set(quality.proposal_bank(p, 25))
        for candidate in ((0, 28), (7, 22), (10, 19), (13, 16)):
            self.assertIn(candidate, actual)

    def test_low_and_high_fps_are_relative_and_deterministic(self):
        p = np.array([0.1, 0.8, 0.4, 0.7, 0.1])
        original = p.copy()
        for fps in (0.01, 0.5, 1, 12.5, 25, 240, 10000):
            with self.subTest(fps=fps):
                actual = quality.proposal_bank(p, fps)
                self.assertEqual(actual, sorted(set(actual)))
                self.assertEqual(actual, quality.proposal_bank(p, fps))
                for s, e in actual:
                    self.assertGreaterEqual(p[s:e + 1].mean(), 0.20)
                    self.assertGreaterEqual(p[s:e + 1].max(), 0.35)
        np.testing.assert_array_equal(p, original)

    def test_candidate_overflow_explicit_not_subsampled(self):
        with self.assertRaisesRegex(ValueError, "unsupported candidate count > 4096"):
            quality.proposal_bank(np.full(5000, 0.8), 1)
        # The declared development range can retain every valid grid span.
        self.assertLessEqual(len(quality.proposal_bank(np.full(125, 0.8), 25)), 4096)

    def test_invalid_probability_inputs_fail_closed(self):
        for p in (None, 0.5, [[0.5]], [np.nan], [np.inf], [-0.01], [1.01],
                  [True], [True, 0.5], ["0.5"], [0.5 + 0j], np.array([0.5], dtype=object)):
            with self.subTest(p=repr(p)):
                with self.assertRaises(ValueError):
                    quality.proposal_bank(p, 25)

    def test_invalid_fps_rejected_even_for_empty_or_low_peak(self):
        for fps in (0, -1, np.nan, np.inf, True, "25", None, 25 + 0j):
            for p in ([], [0.0]):
                with self.subTest(fps=repr(fps), p=p):
                    with self.assertRaises(ValueError):
                        quality.proposal_bank(p, fps)


class IntervalFeatureTests(unittest.TestCase):
    def test_named_tuple_api_shape_dtype_and_stable_schema(self):
        p, x = np.full(10, 0.5), np.arange(30, dtype=np.float32).reshape(10, 3)
        features, names = quality.interval_features(p, 0.2, 25, [(0, 0), (2, 6)], x)
        self.assertEqual(features.shape, (2, 41 + 9 * 3))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(names, quality.interval_feature_names(3))
        self.assertEqual(len(names), len(set(names)))
        _, reordered_names = quality.interval_features(p, 0.8, 1, [(2, 6)], x)
        self.assertEqual(names, reordered_names)
        self.assertEqual(names[:6], ("p_mean", "p_std", "p_min", "p_max", "p_q25", "p_q75"))

    def test_empty_candidates_and_empty_video_keep_named_dimension(self):
        for total in (0, 5):
            features, names = quality.interval_features(np.zeros(total), 0.2, 25, [],
                                                        np.zeros((total, 220)))
            self.assertEqual(features.shape, (0, 2021))
            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(names, quality.interval_feature_names(220))
        features, names = quality.interval_features([0.5], 0.2, 25, [(0, 0)], np.empty((1, 0)))
        self.assertEqual(features.shape, (1, 41))
        self.assertEqual(len(names), 41)

    def test_probability_statistics_and_equal_thirds(self):
        p = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.6, 0.3, 0.7])
        _, f = named_features(p, 10, [(1, 6)], p[:, None])
        span = p[1:7]
        expected = {"p_mean": span.mean(), "p_std": span.std(), "p_min": span.min(),
                    "p_max": span.max(), "p_q25": 0.325, "p_q75": 0.75,
                    "duration_seconds": 0.6, "duration_video_fraction": 0.75,
                    "whole_video_p_mean": p.mean(), "whole_video_p_std": p.std(),
                    "video_probability": 0.3}
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(float(f[name][0]), value, places=6)
        for i, value in enumerate((0.5, 0.65, 0.45)):
            self.assertAlmostEqual(float(f[f"p_third_{i}_mean"][0]), value, places=6)
            self.assertEqual(f[f"third_{i}_validfraction"][0], 1)

    def test_context_excludes_candidate_and_has_fps_support(self):
        p = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.6, 0.3, 0.7])
        _, f = named_features(p, 10, [(2, 4)], np.column_stack((p, p * 10)))
        expected = {"p_before_mean_0.2s": 0.15, "p_after_mean_0.2s": 0.45,
                    "before_validfraction_0.2s": 1, "after_validfraction_0.2s": 1,
                    "p_start_inner_mean_0.2s": 0.85, "p_end_inner_mean_0.2s": 0.65,
                    "p_start_contrast_0.2s": 0.7, "p_end_contrast_0.2s": 0.2,
                    "p_before_mean_0.5s": 0.15, "p_after_mean_0.5s": 1.6 / 3,
                    "before_validfraction_0.5s": 0.4, "after_validfraction_0.5s": 0.6,
                    "start_inner_validfraction_0.5s": 0.6, "end_inner_validfraction_0.5s": 0.6,
                    "evidence_1/before_0.2s_mean": 1.5, "evidence_1/after_0.2s_mean": 4.5,
                    "evidence_1/inner_minus_before_0.2s": 5.5,
                    "evidence_1/inner_minus_after_0.2s": 2.5}
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(float(f[name][0]), value, places=6)

    def test_singleton_empty_context_and_short_third_fallbacks(self):
        for fps in (0.01, 0.5, 1, 25, 240):
            _, f = named_features([0.6], fps, [(0, 0)], [[2.0, -3.0]], video_probability=0.0)
            for i, support in enumerate((1, 0, 0)):
                self.assertAlmostEqual(float(f[f"p_third_{i}_mean"][0]), 0.6, places=6)
                self.assertEqual(f[f"third_{i}_validfraction"][0], support)
                self.assertEqual(f[f"evidence_1/third_{i}_mean"][0], -3)
            for seconds in (0.2, 0.5):
                self.assertEqual(f[f"p_before_mean_{seconds}s"][0], 0)
                self.assertEqual(f[f"p_after_mean_{seconds}s"][0], 0)
                self.assertEqual(f[f"before_validfraction_{seconds}s"][0], 0)
                self.assertEqual(f[f"after_validfraction_{seconds}s"][0], 0)
                self.assertAlmostEqual(float(f[f"start_inner_validfraction_{seconds}s"][0]),
                                       1 / max(1, round(seconds * fps)), places=6)
            self.assertEqual(f["p_start_slope_0.12s_per_second"][0], 0)
            self.assertAlmostEqual(float(f["start_slope_validfraction"][0]),
                                   1 / (2 * max(1, round(0.12 * fps)) + 1), places=6)
            self.assertEqual(f["evidence_0/before_0.2s_mean"][0], 0)
            self.assertEqual(f["evidence_0/inner_minus_before_0.2s"][0], 2)
            self.assertEqual(f["evidence_0/std"][0], 0)
            self.assertAlmostEqual(float(f["duration_seconds"][0]), 1 / fps, places=5)

    def test_two_frame_thirds_have_explicit_missing_support(self):
        _, f = named_features([0.2, 0.8], 25, [(0, 1)], [[1.0], [3.0]])
        for i, p_mean, evidence_mean, support in ((0, 0.2, 1, 1), (1, 0.8, 3, 1), (2, 0.5, 2, 0)):
            self.assertAlmostEqual(float(f[f"p_third_{i}_mean"][0]), p_mean, places=6)
            self.assertEqual(f[f"evidence_0/third_{i}_mean"][0], evidence_mean)
            self.assertEqual(f[f"third_{i}_validfraction"][0], support)

    def test_local_slopes_have_per_second_units_and_boundary_support(self):
        p = 0.1 + np.arange(20) * 0.02
        _, f = named_features(p, 20, [(0, 0), (8, 11), (19, 19)], np.zeros((20, 1)))
        for name in ("p_start_slope_0.12s_per_second", "p_end_slope_0.12s_per_second"):
            np.testing.assert_allclose(f[name], 0.4, atol=1e-6)
        np.testing.assert_allclose(f["start_slope_validfraction"], [0.6, 1.0, 0.6], atol=1e-7)

    def test_prefix_evidence_moments_match_direct_window_oracle(self):
        rng = np.random.default_rng(71)
        p = rng.uniform(0, 1, 125)
        evidence = rng.normal(size=(125, 220)).astype(np.float32)
        evidence[:, 0] = 100000.0  # Centered prefix variance must stay zero.
        raw = rng.integers(0, 125, size=(150, 2))
        intervals = np.sort(raw, axis=1)
        _, f = named_features(p, 25, intervals, evidence)
        for i, (s, e) in enumerate(intervals):
            inside = evidence[s:e + 1].astype(np.float64)
            parts = np.array_split(inside, 3)
            before, after = evidence[max(0, s - 5):s], evidence[e + 1:min(125, e + 6)]
            for j in (0, 1, 57, 219):
                expected = {"mean": inside[:, j].mean(), "std": inside[:, j].std(),
                            "before_0.2s_mean": before[:, j].astype(np.float64).mean() if len(before) else 0.0,
                            "after_0.2s_mean": after[:, j].astype(np.float64).mean() if len(after) else 0.0}
                expected.update({f"third_{k}_mean": part[:, j].mean() if len(part) else inside[:, j].mean()
                                 for k, part in enumerate(parts)})
                for stat, target in expected.items():
                    self.assertAlmostEqual(float(f[f"evidence_{j}/{stat}"][i]), float(target), places=6)
            self.assertAlmostEqual(float(f["p_mean"][i]), p[s:e + 1].mean(), places=6)
            self.assertAlmostEqual(float(f["p_std"][i]), p[s:e + 1].std(), places=6)

    def test_no_labels_identity_or_normalized_absolute_position(self):
        self.assertEqual(tuple(inspect.signature(quality.interval_features).parameters),
                         ("probabilities", "video_probability", "fps", "intervals", "frame_evidence"))
        args = (np.full(80, 0.5), 0.01, 10, [(10, 15), (40, 45)], np.full((80, 2), 2.0))
        features, names = quality.interval_features(*args)
        np.testing.assert_array_equal(features[0], features[1])
        for field in ("labels", "filename", "sha", "event_count"):
            self.assertFalse(any(field in name for name in names))
            with self.assertRaises(TypeError):
                quality.interval_features(*args, **{field: None})
        self.assertFalse(any("absolute" in name or "normalized_start" in name or "normalized_end" in name
                             for name in names))

    def test_inputs_are_not_mutated(self):
        p = np.linspace(0, 1, 8)
        x = np.arange(24, dtype=np.float32).reshape(8, 3)
        intervals = np.array([[0, 1], [2, 7]])
        copies = [a.copy() for a in (p, x, intervals)]
        quality.interval_features(p, 0.5, 25, intervals, x)
        for actual, original in zip((p, x, intervals), copies):
            np.testing.assert_array_equal(actual, original)

    def test_invalid_inputs_rejected_even_without_candidates(self):
        for evidence in (None, [], [1, 2], np.zeros((3, 1)), [[np.nan], [0]],
                         [[np.inf], [0]], [[1e300], [0]], [["0"], ["1"]], np.zeros((2, 1, 1))):
            with self.subTest(evidence=repr(evidence)):
                with self.assertRaises(ValueError):
                    quality.interval_features([0.5, 0.6], 0.2, 25, [], evidence)
        for vp in (True, None, "0.2", np.nan, np.inf, -0.1, 1.1):
            with self.subTest(vp=repr(vp)):
                with self.assertRaises(ValueError):
                    quality.interval_features([], vp, 25, [], np.empty((0, 2)))
        for fps in (0, -1, True, np.nan):
            with self.assertRaises(ValueError):
                quality.interval_features([], 0.2, fps, [], np.empty((0, 2)))

    def test_invalid_intervals_and_candidate_cap_fail_closed(self):
        bad = (None, [0, 1], [[0]], [[1, 0]], [[-1, 0]], [[0, 3]], [[0.0, 1.0]],
               [[True, 1]], [[0, False]], [["0", "1"]], [[np.nan, 1]],
               np.array([[0, 2 ** 63]], dtype=np.uint64), np.empty((0,), dtype=bool))
        for intervals in bad:
            with self.subTest(intervals=repr(intervals)):
                with self.assertRaises(ValueError):
                    quality.interval_features([0.4] * 3, 0.2, 25, intervals, np.zeros((3, 1)))
        with self.assertRaisesRegex(ValueError, "candidate count > 4096"):
            quality.interval_features([0.4], 0.2, 25, [(0, 0)] * 4097, [[0.0]])

    def test_unrepresentable_outputs_rejected_not_clipped(self):
        with self.assertRaises(ValueError):
            quality.interval_features([0.6], 0.2, 1e-40, [(0, 0)], [[0.0]])
        with self.assertRaises(ValueError):
            quality.interval_features([0.6, 0.7], 0.2, 1, [(1, 1)], [[-3e38], [3e38]])
        with self.assertRaises(ValueError):
            quality.interval_feature_names(True)


class TargetAndWeightTests(unittest.TestCase):
    def test_known_inclusive_iou_and_max_not_sum_over_events(self):
        labels = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
        intervals = [(1, 2), (0, 2), (2, 5), (6, 9), (0, 9), (3, 4), (5, 5), (2, 2)]
        actual = quality.quality_targets(intervals, labels)
        np.testing.assert_allclose(actual, [1, 2 / 3, 1 / 4, 3 / 4, 3 / 10, 0, 1, 1 / 2], atol=1e-7)
        self.assertEqual(actual.dtype, np.float32)

    def test_normal_empty_and_boolean_ground_truth(self):
        np.testing.assert_array_equal(quality.quality_targets([(0, 3), (1, 1)], [0] * 4), [0, 0])
        actual = quality.quality_targets([(0, 0)], np.array([True, False]))
        np.testing.assert_array_equal(actual, [1])
        self.assertEqual(quality.quality_targets([], []).shape, (0,))
        self.assertEqual(quality.quality_targets([], [1]).dtype, np.float32)

    def test_invalid_ground_truth_and_intervals(self):
        for labels in (None, [[0, 1]], [np.nan], [np.inf], [-1], [2], [0.2], ["1"]):
            with self.subTest(labels=repr(labels)):
                with self.assertRaises(ValueError):
                    quality.quality_targets([], labels)
        for intervals in ([(0, 1)], [(1, 0)], [(0.0, 0.0)], [(True, 0)]):
            with self.assertRaises(ValueError):
                quality.quality_targets(intervals, [1])
        with self.assertRaises(ValueError):
            quality.quality_targets([(0, 0)], [])

    def test_positive_stratum_mass_and_exact_boundaries(self):
        targets = np.array([0, 0.09999, 0.1, 0.49999, 0.5, 1], dtype=np.float64)
        weights = quality.quality_weights(targets, False, 2.7)
        self.assertEqual(weights.dtype, np.float32)
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(weights, 1.35, rtol=1e-6)
        for group in (weights[:2], weights[2:4], weights[4:]):
            self.assertAlmostEqual(float(group.sum()), 2.7, places=6)

    def test_empty_strata_do_not_receive_or_redistribute_mass(self):
        for targets in ([0, 0.01, 0.09], [0.1, 0.2], [0.5]):
            weights = quality.quality_weights(targets, False, 2.0)
            self.assertAlmostEqual(float(weights.sum()), 2.0, places=6)
        self.assertEqual(quality.quality_weights([], False, 2).shape, (0,))
        np.testing.assert_array_equal(quality.quality_weights([0, 0.5], False, 0), [0, 0])

    def test_fixed_runner_sample_budget_preserves_content_mass(self):
        targets = np.repeat([0.01, 0.2, 0.8], 32)
        weights = quality.quality_weights(targets, False, 0.75)
        for group in np.split(weights, 3):
            self.assertEqual(float(group.sum()), 0.75)
        for count in (1, 3, 96):
            normal = quality.quality_weights(np.zeros(count), True, 0.75)
            self.assertAlmostEqual(float(normal.sum()), 1.5, places=6)
            self.assertEqual(normal.dtype, np.float32)
        self.assertEqual(quality.quality_weights([], True, 0.75).shape, (0,))

    def test_invalid_weights_fail_closed(self):
        for targets, normal, mass in (([-0.1], False, 1), ([1.1], False, 1), ([np.nan], False, 1),
                                      ([[0]], False, 1), ([True], False, 1), ([0.5], True, 1),
                                      ([0], 1, 1), ([0], False, -1), ([0], False, np.inf),
                                      ([0], False, True), ([0], False, "1"), ([0], True, 1e308),
                                      ([0], False, 1e-300)):
            with self.subTest(targets=targets, normal=normal, mass=mass):
                with self.assertRaises(ValueError):
                    quality.quality_weights(targets, normal, mass)


class QualityHeadTests(unittest.TestCase):
    def test_fit_uses_only_fixed_parameters_and_mean_one_weights(self):
        features = np.arange(8, dtype=np.float64).reshape(4, 2)
        target = np.array([0, 0.2, 0.5, 1.0])
        weights = np.array([1e308, 5e307, 0, 1e308])
        originals = [a.copy() for a in (features, target, weights)]
        model = mock.Mock()
        constructor = mock.Mock(return_value=model)
        fake_sklearn, fake_ensemble = ModuleType("sklearn"), ModuleType("sklearn.ensemble")
        fake_ensemble.ExtraTreesRegressor = constructor
        with mock.patch.dict(sys.modules, {"sklearn": fake_sklearn, "sklearn.ensemble": fake_ensemble}):
            actual = quality.fit_quality_head(features, target, weights, 123)
        self.assertIs(actual, model)
        constructor.assert_called_once_with(n_estimators=192, max_depth=8, min_samples_leaf=12,
                                            max_features=0.5, n_jobs=2, random_state=123)
        model.fit.assert_called_once()
        args, kwargs = model.fit.call_args
        self.assertEqual(args[0].dtype, np.float32)
        np.testing.assert_array_equal(args[0], features)
        np.testing.assert_array_equal(args[1], target)
        np.testing.assert_allclose(kwargs["sample_weight"], [1.6, 0.8, 0, 1.6], atol=1e-12)
        self.assertAlmostEqual(float(kwargs["sample_weight"].mean()), 1.0, places=12)
        for actual_array, original in zip((features, target, weights), originals):
            np.testing.assert_array_equal(actual_array, original)

    def test_invalid_training_data_rejected_before_sklearn_import(self):
        cases = (([], [], [], 1), (np.empty((0, 2)), [], [], 1), ([[0, 1]], [], [1], 1),
                 ([[0, 1]], [0], [], 1), ([[0, 1]], [0], [0], 1), ([[0, 1]], [0], [-1], 1),
                 ([[0, 1]], [0], [np.inf], 1), ([[np.nan, 1]], [0], [1], 1),
                 ([[1e300, 1]], [0], [1], 1), ([[0, 1]], [np.nan], [1], 1),
                 ([[0, 1]], [1.01], [1], 1), ([[0, 1]], [0], [1], True),
                 ([[0, 1]], [0], [1], -1), ([[0, 1]], [0], [1], 2 ** 32),
                 ([[0, 1]], [0], [1], 1.0), ([[0, 1]], [0], [1], None),
                 (np.empty((1, 0)), [0], [1], 1))
        for features, target, weight, seed in cases:
            with self.subTest(seed=seed, features=repr(features)):
                with self.assertRaises(ValueError):
                    quality.fit_quality_head(features, target, weight, seed)

    def test_import_features_and_portable_prediction_do_not_import_sklearn(self):
        program = '''import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"sklearn", "torch"}:
        raise AssertionError("unexpected training dependency import")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import numpy as np
import optimized_interval_quality as quality
pairs = quality.proposal_bank([0.2, 0.7], 25)
features, names = quality.interval_features([0.2, 0.7], 0.0, 25, pairs, [[1.0], [2.0]])
state = {"kind": quality.QUALITY_HEAD_KIND, "n_features": features.shape[1],
         "trees": [{"left": [-1], "right": [-1], "feature": [-2], "threshold": [-2], "value": [0.5]}]}
scores = quality.portable_predict_quality(state, features)
assert quality.select_intervals(pairs, scores, 0.25)
assert not any(key == "sklearn" or key.startswith("sklearn.") for key in sys.modules)
print("numpy-only inference OK")
'''
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-B", "-c", program],
                             cwd=Path(__file__).resolve().parent, env=environment,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("numpy-only inference OK", run.stdout)

    @unittest.skipUnless(HAS_SKLEARN, "sklearn unavailable: real ET fit/parity SKIPPED; use the existing WSL environment")
    def test_small_sklearn_fit_export_json_and_portable_parity(self):
        # One synthetic fit only; no real training records or artifacts.
        rng = np.random.default_rng(24)
        features = rng.uniform(-1, 1, (240, 6))
        targets = np.clip(0.4 + features[:, 0] * 0.4 + features[:, 1] * 0.2, 0, 1)
        weights = np.linspace(0.1, 2.0, len(features))
        model = quality.fit_quality_head(features, targets, weights, 73)
        expected_params = {"n_estimators": 192, "max_depth": 8, "min_samples_leaf": 12,
                           "max_features": 0.5, "n_jobs": 2, "random_state": 73}
        for key, expected in expected_params.items():
            self.assertEqual(model.get_params()[key], expected)
        state = json.loads(json.dumps(quality.export_quality_head(model), allow_nan=False))
        self.assertEqual(len(state["trees"]), 192)
        self.assertEqual(state["n_features"], 6)
        for tree in state["trees"]:
            self.assertEqual(set(tree), {"left", "right", "feature", "threshold", "value"})
        query = rng.uniform(-1, 1, (120, 6))
        # Exercise comparisons exactly at (and just around) a real float64 split.
        first = state["trees"][0]
        splits = [i for i, feature in enumerate(first["feature"]) if feature >= 0]
        self.assertTrue(splits)
        node = splits[0]
        feature, threshold = first["feature"][node], first["threshold"][node]
        query[:3, feature] = [threshold, np.nextafter(threshold, -np.inf), np.nextafter(threshold, np.inf)]
        for data in (features, query, query.astype(np.float32)):
            actual, expected = quality.portable_predict_quality(state, data), model.predict(data)
            self.assertEqual(actual.dtype, np.float32)
            self.assertLessEqual(float(np.max(np.abs(actual - expected))), 2e-6)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
        self.assertEqual(quality.portable_predict_quality(state, np.empty((0, 6))).shape, (0,))

    def test_export_rejects_unfitted_classifier_or_multioutput_models(self):
        for model in (None, SimpleNamespace(), SimpleNamespace(n_features_in_=2, n_outputs_=2),
                      SimpleNamespace(n_features_in_=2, n_outputs_=1, estimators_=[], classes_=[0, 1]),
                      SimpleNamespace(n_features_in_=2, n_outputs_=1, estimators_=[object()])):
            with self.subTest(model=repr(model)):
                with self.assertRaises(ValueError):
                    quality.export_quality_head(model)

    def test_portable_means_leaves_float32_cast_and_inclusive_split(self):
        features = np.array([[0.0, 99], [0.5, 0], [0.500000004, -99], [0.50000004, 0], [1.0, 0]])
        actual = quality.portable_predict_quality(stump_state(), features)
        np.testing.assert_allclose(actual, [0.25, 0.25, 0.25, 0.65, 0.65], atol=1e-7)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(quality.portable_predict_quality(stump_state(), np.empty((0, 2))).shape, (0,))

    def test_portable_rejects_invalid_features(self):
        for features in (None, [], [0, 1], [[0]], [[0, 1, 2]], [[np.nan, 0]],
                         [[np.inf, 0]], [[1e300, 0]], [["0", "1"]], [[True, False]]):
            with self.subTest(features=repr(features)):
                with self.assertRaises(ValueError):
                    quality.portable_predict_quality(stump_state(), features)

    def test_portable_rejects_invalid_schema_and_node_arrays_even_if_empty(self):
        states = [None, {}, {"kind": "wrong", "n_features": 2, "trees": []}]
        for field, value in (("kind", "unknown"), ("n_features", 0), ("n_features", True),
                             ("n_features", 2.0), ("trees", []), ("trees", {}),
                             ("trees", [stump_state()["trees"][0]] * 193)):
            state = stump_state()
            state[field] = value
            states.append(state)
        extra = stump_state()
        extra["filename"] = "forbidden"
        states.append(extra)
        mutations = (("left", [1.0, -1.0, -1.0]), ("left", [True, -1, -1]),
                     ("left", [0, -1, -1]), ("left", [3, -1, -1]),
                     ("left", [-1, -1, -1]), ("right", [1, -1, -1]),
                     ("feature", [2, -2, -2]), ("feature", [-1, -2, -2]),
                     ("feature", [0, 0, -2]), ("threshold", [np.nan, -2, -2]),
                     ("threshold", [0.5, 0, -2]), ("value", [0.5, np.inf, 0.9]),
                     ("value", [0.5, -0.1, 0.9]), ("value", [0.5, 0.1, 1.1]),
                     ("value", [0.1]), ("value", [[0.1], [0.2], [0.3]]))
        for field, value in mutations:
            state = stump_state()
            state["trees"][0][field] = value
            states.append(state)
        bad_tree = stump_state()
        bad_tree["trees"][0]["label"] = [0, 1, 0]
        states.append(bad_tree)
        for state in states:
            with self.subTest(state=repr(state)[:180]):
                with self.assertRaises(ValueError):
                    quality.portable_predict_quality(state, np.empty((0, 2)))

    def test_tree_orphans_depth_shared_nodes_and_node_cap_fail_closed(self):
        orphan = stump_state()
        for key in orphan["trees"][0]:
            orphan["trees"][0][key].append(orphan["trees"][0][key][-1])
        # Properly connected chain, but depth nine is outside the fixed head.
        count, depth = 19, 9
        chain = {"left": [-1] * count, "right": [-1] * count, "feature": [-2] * count,
                 "threshold": [-2.0] * count, "value": [0.5] * count}
        for i in range(depth):
            node = 2 * i
            chain["left"][node], chain["right"][node] = node + 1, node + 2
            chain["feature"][node], chain["threshold"][node] = 0, 0.5
        over_depth = stump_state()
        over_depth["trees"] = [chain]
        shared = stump_state()
        shared["trees"][0] = {"left": [1, 3, 3, -1, -1], "right": [2, 4, 4, -1, -1],
                               "feature": [0, 0, 0, -2, -2], "threshold": [0.5, 0.5, 0.5, -2, -2],
                               "value": [0.5] * 5}
        over_count = stump_state()
        over_count["trees"] = [{key: [value[0]] * 512 for key, value in chain.items()}]
        for state in (orphan, over_depth, shared, over_count):
            with self.assertRaises(ValueError):
                quality.portable_predict_quality(state, np.empty((0, 2)))

    def test_portable_state_and_features_are_not_mutated(self):
        state = stump_state()
        original = deepcopy(state)
        features = np.array([[0.1, 0.0], [0.9, 1.0]])
        original_features = features.copy()
        quality.portable_predict_quality(state, features)
        self.assertEqual(state, original)
        np.testing.assert_array_equal(features, original_features)


class SelectionTests(unittest.TestCase):
    def test_descending_quality_and_inclusive_overlap(self):
        intervals = [(0, 2), (2, 4), (3, 5), (1, 1), (6, 6)]
        actual = quality.select_intervals(intervals, [0.9, 0.8, 0.7, 0.6, 0.55], 0.55)
        self.assertEqual(actual, [(0, 2), (3, 5), (6, 6)])
        # A high-quality inner singleton rejects the lower-quality enclosing span.
        self.assertEqual(quality.select_intervals([(0, 3), (1, 1)], [0.6, 0.9], 0.55), [(1, 1)])

    def test_ties_resolve_by_start_then_end_independent_of_input_order(self):
        intervals = [(1, 3), (2, 2), (1, 1), (0, 0)]
        rng = np.random.default_rng(17)
        for _ in range(8):
            reordered = [intervals[i] for i in rng.permutation(len(intervals))]
            self.assertEqual(quality.select_intervals(reordered, [0.8] * 4, 0.55),
                             [(0, 0), (1, 1), (2, 2)])

    def test_threshold_inclusive_duplicates_and_empty(self):
        self.assertEqual(quality.select_intervals([(0, 0), (0, 0), (1, 1)], [0.25, 0.25, 0.2499], 0.25),
                         [(0, 0)])
        self.assertEqual(quality.select_intervals([], [], 0.45), [])
        self.assertEqual(quality.select_intervals([(1, 2)], [0.34], 0.35), [])

    def test_no_event_count_limit_or_video_gate(self):
        intervals = [(2 * i, 2 * i) for i in range(128)]
        self.assertEqual(quality.select_intervals(intervals, [0.9] * 128, 0.55), intervals)
        self.assertEqual(tuple(inspect.signature(quality.select_intervals).parameters),
                         ("intervals", "scores", "threshold"))
        with self.assertRaises(TypeError):
            quality.select_intervals([(0, 0)], [0.9], 0.55, video_probability=0.0)

    def test_fixed_threshold_configs_have_no_selection_or_mutable_global_state(self):
        self.assertEqual(quality.THRESHOLDS, (0.25, 0.35, 0.45, 0.55))
        configs = quality.quality_configurations()
        self.assertEqual(configs, tuple({"threshold": value} for value in quality.THRESHOLDS))
        configs[0]["threshold"] = 0.99
        self.assertEqual(quality.quality_configurations()[0], {"threshold": 0.25})
        self.assertFalse(any(name.startswith("choose") for name in vars(quality)))

    def test_invalid_selection_inputs_fail_closed(self):
        for intervals, scores, threshold in (([(0, 0)], [], 0.25), ([], [0.5], 0.25),
                                             ([(0, 0)], [np.nan], 0.25), ([(0, 0)], [np.inf], 0.25),
                                             ([(0, 0)], [-0.1], 0.25), ([(0, 0)], [1.1], 0.25),
                                             ([(1, 0)], [0.5], 0.25), ([(True, 0)], [0.5], 0.25),
                                             ([(0, 0)], [0.5], True), ([], [], np.nan),
                                             ([], [], -0.1), ([], [], 1.1), ([], [], "0.25")):
            with self.subTest(threshold=repr(threshold), intervals=intervals):
                with self.assertRaises(ValueError):
                    quality.select_intervals(intervals, scores, threshold)
        with self.assertRaisesRegex(ValueError, "candidate count > 4096"):
            quality.select_intervals([(0, 0)] * 4097, [0.5] * 4097, 0.25)


if __name__ == "__main__":
    unittest.main()
