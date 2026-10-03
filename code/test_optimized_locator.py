"""Focused CPU regression tests for optimized_locator; no data/model files.

Only NumPy is required for the core tests. sklearn parity tests skip when that
package is absent; run the same file in the existing WSL environment for parity.
Confirmed production regressions intentionally remain strict failures/errors,
not expectedFailure or assertions treating a crash as correct behavior.
Legacy parity executes only the three reviewed pure evaluation functions via
AST extraction, avoiding imports of training modules and their GPU dependencies.
"""

import ast
import importlib.util
import json
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from optimized_locator import (
    SCHEMA,
    augment_signals,
    decode,
    export_model,
    interval_iou,
    load_bundle,
    metrics,
    objective,
    portable_predict,
    rolling_mean,
    spans,
    video_features,
)


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def decoder_config(**overrides):
    config = {"threshold": 0.75, "low_ratio": 1.0, "smooth_seconds": 0.0,
              "gap_seconds": 0.0, "min_seconds": 0.0, "video_threshold": 0.0}
    config.update(overrides)
    return config


def stump(threshold=1.5, left_value=0.2, right_value=0.8):
    return {"left": [1, -1, -1], "right": [2, -1, -1],
            "feature": [0, -2, -2], "threshold": [threshold, -2.0, -2.0],
            "value": [0.5, left_value, right_value]}


def forest(*trees):
    return {"kind": "forest", "n_features": 1, "trees": list(trees)}


def load_legacy_evaluation_functions():
    namespace = {"np": np}
    for filename, wanted in (
        ("train_segment_locator.py", {"contiguous_segments"}),
        ("train_proposal_calibrator.py", {"interval_iou", "evaluate_segment_predictions"}),
    ):
        path = Path(__file__).resolve().with_name(filename)
        parsed = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        if {node.name for node in selected} != wanted:
            raise AssertionError(f"legacy evaluation functions missing in {path}")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace


class SpanAndMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = load_legacy_evaluation_functions()

    def test_spans_empty_singleton_edges_and_nonzero_values(self):
        cases = [([], []), ([0], []), ([1], [(0, 0)]),
                 ([1, 1, 0, 1], [(0, 1), (3, 3)]),
                 ([0, 1, 1, 0, 0, 1, 0], [(1, 2), (5, 5)]),
                 ([-1, 0, 2], [(0, 0), (2, 2)])]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(spans(values), expected)

    def test_spans_match_actual_legacy_contiguous_segments(self):
        rng = np.random.default_rng(801)
        for count in range(40):
            labels = rng.integers(0, 2, count, dtype=np.int64)
            self.assertEqual(spans(labels), self.legacy["contiguous_segments"](labels))

    def test_iou_uses_inclusive_single_frame_and_touching_endpoints(self):
        cases = [((2, 2), (2, 2), 1.0), ((0, 1), (1, 2), 1 / 3),
                 ((0, 0), (1, 1), 0.0), ((0, 4), (1, 3), 3 / 5),
                 ((0, 2), (0, 9), 0.3), ((0, 1), (0, 3), 0.5)]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(interval_iou(a, b), expected)
                self.assertAlmostEqual(interval_iou(b, a), expected)

    def test_iou_exhaustive_small_intervals_match_legacy(self):
        intervals = [(a, b) for a in range(-2, 5) for b in range(a, 5)]
        for a in intervals:
            for b in intervals:
                self.assertEqual(interval_iou(a, b), self.legacy["interval_iou"](a, b))

    def test_greedy_preserves_prediction_order_and_first_gt_on_ties(self):
        labels = [np.asarray([1, 1, 0, 1, 1], dtype=np.int64)]
        bridge_first = metrics([[(0, 4), (0, 1)]], labels)
        exact_first = metrics([[(0, 1), (0, 4)]], labels)
        self.assertEqual([bridge_first["iou_0.3"][k] for k in ("tp", "fp", "fn")], [1, 1, 1])
        self.assertEqual([exact_first["iou_0.3"][k] for k in ("tp", "fp", "fn")], [2, 0, 0])
        for predictions in ([[(0, 4), (0, 1)]], [[(0, 1), (0, 4)]], [[(0, 1), (0, 1)]]):
            report = metrics(predictions, labels)
            for threshold in (0.3, 0.5):
                reference = self.legacy["evaluate_segment_predictions"](predictions, labels, threshold)
                self.assertEqual({k: report[f"iou_{threshold}"][k] for k in ("tp", "fp", "fn")},
                                 {k: reference[k] for k in ("tp", "fp", "fn")})

    def test_randomized_greedy_counts_match_actual_legacy_protocol(self):
        rng = np.random.default_rng(902)
        for _ in range(24):
            labels, predictions = [], []
            for _ in range(4):
                count = int(rng.integers(5, 30))
                labels.append((rng.random(count) > 0.7).astype(np.int64))
                candidates = []
                for _ in range(int(rng.integers(0, 7))):
                    start = int(rng.integers(0, count))
                    candidates.append((start, int(rng.integers(start, count))))
                predictions.append(candidates)
            report = metrics(predictions, labels)
            for threshold in (0.3, 0.5):
                reference = self.legacy["evaluate_segment_predictions"](predictions, labels, threshold)
                for key in ("tp", "fp", "fn"):
                    self.assertEqual(report[f"iou_{threshold}"][key], reference[key])

    def test_frame_union_clipping_normal_gate_statistics_and_objective(self):
        labels = [np.asarray([0, 1, 1, 0, 1, 0]), np.zeros(5), np.asarray([1, 0])]
        predictions = [[(-1, 1), (1, 2), (4, 20)], [(2, 2)], []]
        report = metrics(predictions, labels)
        self.assertEqual(report["frame"], {"tp": 3, "fp": 3, "fn": 1, "f1": 0.6})
        self.assertEqual(report["normal"], {"videos": 1, "false_positive_videos": 1, "false_positive_rate": 1.0})
        self.assertEqual(report["positive_videos_without_candidate"], 1)
        self.assertEqual(report["predicted_segments"], 4)
        expected = 0.5 * (report["iou_0.3"]["f1"] + report["iou_0.5"]["f1"]) - 0.2
        self.assertAlmostEqual(objective(report), expected)

    def test_empty_metrics_and_exact_iou_thresholds(self):
        empty = metrics([], [])
        self.assertEqual(empty["frame"], {"tp": 0, "fp": 0, "fn": 0, "f1": 0.0})
        self.assertEqual(objective(empty), 0.0)
        at_point_three = metrics([[(0, 9)]], [np.asarray([1, 1, 1] + [0] * 7)])
        self.assertEqual(at_point_three["iou_0.3"]["tp"], 1)
        at_point_five = metrics([[(0, 3)]], [np.asarray([1, 1, 0, 0])])
        self.assertEqual(at_point_five["iou_0.5"]["tp"], 1)

    def test_segments_wholly_outside_frame_domain_do_not_paint_frames(self):
        """Production regression: negative NumPy slice end currently creates FP."""
        report = metrics([[(-4, -2), (7, 9)]], [np.zeros(5, dtype=np.int64)])
        self.assertEqual(report["frame"], {"tp": 0, "fp": 0, "fn": 0, "f1": 0.0})


class AugmentationTests(unittest.TestCase):
    def test_rolling_mean_matches_independent_edge_padded_boxes(self):
        for values in (np.asarray([1, 3, 7, 9], np.float32),
                       np.asarray([[1, 8], [3, 2], [7, 4], [9, 6]], np.float32)):
            for requested in (1, 2, 3, 4, 99):
                with self.subTest(shape=values.shape, window=requested):
                    width = min(requested, len(values))
                    padding = [(width // 2, width - 1 - width // 2)] + [(0, 0)] * (values.ndim - 1)
                    padded = np.pad(values, padding, mode="edge")
                    reference = np.asarray([padded[i:i + width].mean(axis=0) for i in range(len(values))])
                    got = rolling_mean(values, requested)
                    np.testing.assert_allclose(got, reference, atol=1e-6)
                    self.assertEqual(got.dtype, np.float32)

    def test_rolling_mean_empty_and_window_one_return_copies(self):
        for values in (np.empty(0, np.float32), np.empty((0, 2), np.float32),
                       np.asarray([3], np.float32), np.asarray([[3, 5]], np.float32)):
            got = rolling_mean(values, 1)
            np.testing.assert_array_equal(got, values)
            self.assertFalse(np.shares_memory(got, values))

    def test_no_context_keeps_absolute_values_and_has_five_times_dimension(self):
        values = np.asarray([[20, 100], [40, 200], [60, 300]], np.float32)
        original = values.copy()
        names = ["motion_absolute", "quality_absolute"]
        got, output_names = augment_signals(values, names, fps=24, context=False)
        self.assertEqual(got.shape, (3, 10))
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(output_names[:2], names)
        self.assertEqual(len(set(output_names)), 10)
        np.testing.assert_array_equal(got[:, :2], original)
        np.testing.assert_array_equal(values, original)
        expected = np.concatenate([values.mean(axis=0), values.std(axis=0),
                                   np.percentile(values, 10, axis=0), np.percentile(values, 90, axis=0)])
        np.testing.assert_allclose(got[:, 2:], np.broadcast_to(expected, (3, 8)), atol=1e-6)

    def test_context_keeps_prefix_and_has_seventeen_times_dimension_stable_names(self):
        values = np.asarray([[2, 100], [4, 200], [8, 400], [16, 800]], np.float32)
        names = ["magnitude", "laplacian"]
        first, first_names = augment_signals(values, names, fps=12)
        second, second_names = augment_signals(values, names, fps=30)
        self.assertEqual(first.shape, (4, 34))
        self.assertEqual(second.shape, first.shape)
        self.assertEqual(first_names, second_names)
        self.assertEqual(len(set(first_names)), first.shape[1])
        np.testing.assert_array_equal(first[:, :2], values)
        np.testing.assert_array_equal(second[:, :2], values)
        self.assertTrue(np.isfinite(first).all())

    def test_derivative_sign_seconds_units_and_zero_first_frame(self):
        values = np.asarray([[100, 2], [102, -1], [104, 1]], np.float32)
        got, names = augment_signals(values, ["motion", "quality"], fps=4)
        expected = np.asarray([[0, 0], [8, -12], [8, 8]], np.float32)
        delta_columns = [names.index(n + "/delta_per_second") for n in ("motion", "quality")]
        abs_columns = [names.index(n + "/abs_delta_per_second") for n in ("motion", "quality")]
        np.testing.assert_array_equal(got[:, delta_columns], expected)
        np.testing.assert_array_equal(got[:, abs_columns], np.abs(expected))

    def test_constant_features_remain_absolute_with_zero_relative_variation(self):
        values = np.full((8, 1), 123.0, np.float32)
        got, names = augment_signals(values, ["absolute"], fps=24)
        np.testing.assert_array_equal(got[:, 0], values[:, 0])
        for i, name in enumerate(names):
            if "/std_" in name or "/contrast_" in name or name.endswith("relative_robust_z") or name.endswith("delta_per_second"):
                np.testing.assert_array_equal(got[:, i], np.zeros(8))
        self.assertTrue(np.isfinite(got).all())

    def test_invalid_augmentation_input_is_rejected(self):
        cases = [(np.empty((0, 2)), ["a", "b"]), (np.ones(3), ["a"]),
                 (np.ones((3, 2)), ["a"]), (np.asarray([[np.nan]]), ["a"]),
                 (np.asarray([[np.inf]]), ["a"])]
        for values, names in cases:
            with self.subTest(shape=values.shape, names=names):
                with self.assertRaises(ValueError):
                    augment_signals(values, names, fps=24)

    def test_video_features_global_order_and_float32(self):
        values = np.asarray([[1, 10], [3, 20], [8, 30]], np.float32)
        expected = np.concatenate([values.mean(axis=0), values.std(axis=0),
                                   np.percentile(values, 10, axis=0), np.percentile(values, 90, axis=0)])
        got = video_features(values)
        self.assertEqual(got.shape, (8,))
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, expected, atol=1e-6)


class DecoderTests(unittest.TestCase):
    def test_empty_and_all_below_threshold(self):
        self.assertEqual(decode([], 24, 1.0, decoder_config()), [])
        self.assertEqual(decode(np.empty((0, 1)), 24, 1.0, decoder_config()), [])
        self.assertEqual(decode([0.1, 0.2, 0.4], 24, 1.0, decoder_config()), [])

    def test_hysteresis_without_strong_seed_stays_empty(self):
        self.assertEqual(decode([0.4, 0.5, 0.4], 10, 1.0,
                                decoder_config(low_ratio=0.5)), [])

    def test_hysteresis_expands_only_seeded_low_components(self):
        values = [0.0, 0.4, 0.8, 0.4, 0.0, 0.4, 0.4, 0.0]
        self.assertEqual(decode(values, 10, 1.0, decoder_config(low_ratio=0.5)), [(1, 3)])

    def test_exact_high_and_low_thresholds_include_endpoints(self):
        self.assertEqual(decode([0.375, 0.75, 0.375], 10, 1.0,
                                decoder_config(low_ratio=0.5)), [(0, 2)])
        self.assertEqual(decode([0.374, 0.75, 0.374], 10, 1.0,
                                decoder_config(low_ratio=0.5)), [(1, 1)])

    def test_low_ratio_one_matches_inclusive_binary_spans(self):
        values = np.asarray([0.8, 0.9, 0.0, 0.75, 0.0, 0.9], np.float32)
        self.assertEqual(decode(values, 10, 1.0, decoder_config()), spans(values >= 0.75))

    def test_smoothing_window_converts_seconds_using_fps(self):
        config = decoder_config(threshold=0.5, smooth_seconds=0.5)
        self.assertEqual(decode([0, 1, 0, 0], 2, 1.0, config), [(1, 1)])
        self.assertEqual(decode([0, 1, 0, 0], 4, 1.0, config), [(1, 2)])

    def test_gap_counts_missing_frames_and_converts_seconds(self):
        values = [0.9, 0.0, 0.9, 0.0, 0.0, 0.9]
        self.assertEqual(decode(values, 10, 1.0, decoder_config(gap_seconds=0)), [(0, 0), (2, 2), (5, 5)])
        self.assertEqual(decode(values, 10, 1.0, decoder_config(gap_seconds=0.1)), [(0, 2), (5, 5)])
        self.assertEqual(decode(values, 10, 1.0, decoder_config(gap_seconds=0.2)), [(0, 5)])
        self.assertEqual(decode(values, 20, 1.0, decoder_config(gap_seconds=0.1)), [(0, 5)])

    def test_gap_merging_precedes_minimum_length_filter(self):
        self.assertEqual(decode([0.9, 0, 0.9], 10, 1.0,
                                decoder_config(gap_seconds=0.1, min_seconds=0.3)), [(0, 2)])

    def test_minimum_length_uses_inclusive_count_and_fps(self):
        values = [0.9, 0.9, 0, 0.9, 0.9, 0.9, 0]
        self.assertEqual(decode(values, 10, 1.0, decoder_config(min_seconds=0.3)), [(3, 5)])
        self.assertEqual(decode(values, 20, 1.0, decoder_config(min_seconds=0.3)), [])
        self.assertEqual(decode([0.9], 24, 1.0, decoder_config(min_seconds=0)), [(0, 0)])

    def test_video_gate_rejects_below_but_accepts_equality_and_disabled_gate(self):
        config = decoder_config(video_threshold=0.7)
        self.assertEqual(decode([0.9, 0.9], 10, 0.69, config), [])
        self.assertEqual(decode([0.9, 0.9], 10, 0.7, config), [(0, 1)])
        self.assertEqual(decode([0.9], 10, 0.0, decoder_config(video_threshold=0)), [(0, 0)])

    def test_nonempty_invalid_fps_is_rejected(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                with self.assertRaises(ValueError):
                    decode([0.9], fps, 1.0, decoder_config())

    def test_nonfinite_frame_probabilities_are_rejected(self):
        for values in ([np.nan], [np.inf], [0.9, -np.inf]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    decode(values, 24, 1.0, decoder_config())

    def test_nonfinite_video_probability_is_rejected_not_gate_bypassed(self):
        """Production regression: NaN video probability currently bypasses gate."""
        with self.assertRaises(ValueError):
            decode([0.9], 24, float("nan"), decoder_config(video_threshold=0.7))

    def test_fractional_fps_gap_and_minimum_seconds(self):
        values = [0.9, 0, 0, 0, 0.9]
        self.assertEqual(decode(values, 29.97, 1.0, decoder_config(gap_seconds=0.1, min_seconds=0.1667)), [(0, 4)])
        self.assertEqual(decode(values, 29.97, 1.0, decoder_config(gap_seconds=0.1, min_seconds=0.2)), [])


class PortableModelAndSchemaTests(unittest.TestCase):
    def test_constant_predictor_probabilities_shape_dtype_and_empty_input(self):
        for probability in (0.0, 0.25, 1.0):
            model = {"kind": "constant", "n_features": 2, "probability": probability}
            result = portable_predict(model, np.zeros((3, 2)))
            self.assertEqual(result.shape, (3,))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, np.full(3, probability, np.float32))
            self.assertEqual(portable_predict(model, np.empty((0, 2))).shape, (0,))

    def test_forest_averages_trees_and_equal_threshold_goes_left(self):
        model = forest(stump(), stump(left_value=0.4, right_value=0.6))
        result = portable_predict(model, np.asarray([[-1], [1.5], [2]], np.float64))
        np.testing.assert_allclose(result, [0.3, 0.3, 0.7], atol=1e-7)
        self.assertEqual(portable_predict(model, np.empty((0, 1))).shape, (0,))

    def test_float64_threshold_is_not_rounded_to_float32(self):
        high = np.nextafter(np.float32(1.0), np.float32(np.inf))
        result = portable_predict(forest(stump(threshold=1.00000006)),
                                  np.asarray([[1.0], [high]], np.float32))
        np.testing.assert_allclose(result, [0.2, 0.8], atol=1e-7)

    def test_export_preserves_thresholds_and_normalizes_counts(self):
        definition = stump(threshold=1.00000006)
        tree = SimpleNamespace(children_left=np.asarray(definition["left"]),
                               children_right=np.asarray(definition["right"]),
                               feature=np.asarray(definition["feature"]),
                               threshold=np.asarray(definition["threshold"], np.float64),
                               value=np.asarray([[[7, 13]], [[4, 0]], [[0, 8]]], np.float64))
        fitted = SimpleNamespace(n_features_in_=1, classes_=np.asarray([0, 1]), estimators_=[SimpleNamespace(tree_=tree)])
        exported = json.loads(json.dumps(export_model(fitted), allow_nan=False))
        self.assertEqual(exported["trees"][0]["threshold"], definition["threshold"])
        np.testing.assert_allclose(exported["trees"][0]["value"], [0.65, 0.0, 1.0])
        high = np.nextafter(np.float32(1), np.float32(np.inf))
        np.testing.assert_array_equal(portable_predict(exported, [[1.0], [high]]), [0.0, 1.0])

    def test_gradient_boosting_log_odds_learning_rate_and_stable_sigmoid(self):
        prior = np.log(0.25 / 0.75)
        model = {"kind": "gradient_boosting", "n_features": 1,
                 "initial_log_odds": prior, "learning_rate": 0.4,
                 "trees": [stump(left_value=-3, right_value=3)]}
        got = portable_predict(model, [[0], [2]])
        expected = 1 / (1 + np.exp(-np.asarray([prior - 1.2, prior + 1.2])))
        np.testing.assert_allclose(got, expected, atol=1e-7)
        extreme = dict(model, trees=[stump(left_value=-1e6, right_value=1e6)])
        result = portable_predict(extreme, [[0], [2]])
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        self.assertEqual(portable_predict(model, np.empty((0, 1))).shape, (0,))

    def test_invalid_feature_schema_and_nonfinite_features_are_rejected(self):
        model = {"kind": "constant", "n_features": 2, "probability": 0.5}
        for values in (np.ones(2), np.ones((2, 1)), np.ones((2, 3)),
                       np.asarray([[0, np.nan]]), np.asarray([[0, np.inf]])):
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    portable_predict(model, values)

    def test_unknown_model_kind_and_missing_required_key_are_rejected(self):
        with self.assertRaises(ValueError):
            portable_predict({"kind": "unsupported", "n_features": 1}, [[0]])
        with self.assertRaises((ValueError, KeyError)):
            portable_predict({"kind": "constant", "probability": 0.5}, [[0]])

    def test_out_of_range_tree_child_is_rejected(self):
        invalid = stump()
        invalid["left"][0] = 99
        with self.assertRaises((ValueError, IndexError)):
            portable_predict(forest(invalid), [[0]])

    def test_empty_forest_is_invalid_schema_not_scalar_nan(self):
        """Production regression: mean([]) currently returns scalar NaN."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(ValueError):
                portable_predict(forest(), np.zeros((2, 1)))

    def test_bundle_invalid_versions_non_objects_and_bad_json_are_rejected(self):
        for value in ({}, {"schema_version": "old"}, {"schema_version": 1}, [], None):
            with self.subTest(value=value):
                with patch.object(Path, "read_text", return_value=json.dumps(value)):
                    with self.assertRaises((ValueError, TypeError, AttributeError)):
                        load_bundle(Path("in-memory-only.json"))
        with patch.object(Path, "read_text", return_value="{not-json"):
            with self.assertRaises(ValueError):
                load_bundle(Path("in-memory-only.json"))

    def test_valid_bundle_version_round_trip_uses_utf8(self):
        bundle = {"schema_version": SCHEMA, "feature_names": ["absolute"],
                  "frame_model": {"kind": "constant", "n_features": 1, "probability": 0.5}}
        with patch.object(Path, "read_text", return_value=json.dumps(bundle)) as reader:
            self.assertEqual(load_bundle(Path("in-memory-only.json")), bundle)
            reader.assert_called_once_with(encoding="utf-8")


@unittest.skipUnless(HAS_SKLEARN, "sklearn not installed; run parity with the existing WSL Python")
class SklearnParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
        cls.RF, cls.ET, cls.GB = RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
        rng = np.random.default_rng(601)
        cls.x = rng.normal(size=(96, 4)).astype(np.float32)
        cls.y = (cls.x[:, 0] + 0.5 * cls.x[:, 1] ** 2 - 0.3 * cls.x[:, 2] > 0.2).astype(np.int64)
        cls.weights = np.linspace(0.25, 2.0, len(cls.x))
        cls.query = np.vstack([cls.x, rng.normal(size=(49, 4)).astype(np.float32)])
        cls.models = {
            "rf": cls.RF(n_estimators=5, max_depth=3, min_samples_leaf=2, n_jobs=1, random_state=41),
            "et": cls.ET(n_estimators=5, max_depth=3, min_samples_leaf=2, n_jobs=1, random_state=42),
            "gb": cls.GB(n_estimators=7, max_depth=2, learning_rate=0.13, random_state=43),
        }
        for model in cls.models.values():
            model.fit(cls.x, cls.y, sample_weight=cls.weights)

    def assert_parity(self, model, values):
        exported = json.loads(json.dumps(export_model(model), allow_nan=False))
        positive = np.flatnonzero(np.asarray(model.classes_) == 1)
        expected = (model.predict_proba(values)[:, positive[0]] if len(positive)
                    else np.zeros(len(values), dtype=np.float64))
        got = portable_predict(exported, values)
        self.assertEqual(got.shape, (len(values),))
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
        return exported

    def test_random_forest_json_probabilities_match_sklearn(self):
        self.assert_parity(self.models["rf"], self.query.astype(np.float64))

    def test_extra_trees_json_probabilities_match_sklearn(self):
        self.assert_parity(self.models["et"], self.query)

    def test_gradient_boosting_json_probabilities_match_sklearn(self):
        self.assert_parity(self.models["gb"], self.query)

    def test_exported_split_arrays_and_adjacent_float32_threshold_probes(self):
        for name, model in self.models.items():
            with self.subTest(model=name):
                exported = export_model(model)
                estimators = np.asarray(model.estimators_, dtype=object).reshape(-1)
                queries = []
                for definition, estimator in zip(exported["trees"], estimators):
                    tree = estimator.tree_
                    np.testing.assert_array_equal(definition["left"], tree.children_left)
                    np.testing.assert_array_equal(definition["right"], tree.children_right)
                    np.testing.assert_array_equal(definition["feature"], tree.feature)
                    np.testing.assert_array_equal(np.asarray(definition["threshold"], np.float64), tree.threshold)
                    for node in np.flatnonzero(tree.children_left >= 0):
                        threshold = np.float32(tree.threshold[node])
                        for value in (np.nextafter(threshold, np.float32(-np.inf)), threshold,
                                      np.nextafter(threshold, np.float32(np.inf))):
                            row = np.zeros(model.n_features_in_, dtype=np.float32)
                            row[tree.feature[node]] = value
                            queries.append(row)
                self.assertTrue(queries)
                self.assert_parity(model, np.asarray(queries, np.float32))

    def assert_one_class_forest_parity(self, classifier, label):
        model = classifier(n_estimators=3, max_depth=2, n_jobs=1, random_state=44)
        model.fit(self.x, np.full(len(self.x), label, dtype=np.int64))
        self.assert_parity(model, self.query)

    def test_random_forest_one_negative_class_matches_sklearn(self):
        """Production regression: exporter indexes nonexistent probability col 1."""
        self.assert_one_class_forest_parity(self.RF, 0)

    def test_random_forest_one_positive_class_matches_sklearn(self):
        self.assert_one_class_forest_parity(self.RF, 1)

    def test_extra_trees_one_negative_class_matches_sklearn(self):
        self.assert_one_class_forest_parity(self.ET, 0)

    def test_extra_trees_one_positive_class_matches_sklearn(self):
        self.assert_one_class_forest_parity(self.ET, 1)

    def test_gradient_boosting_zero_init_matches_sklearn(self):
        """Production regression: valid init='zero' makes init_ a string."""
        model = self.GB(init="zero", n_estimators=5, max_depth=2, learning_rate=0.13, random_state=45)
        model.fit(self.x, self.y, sample_weight=self.weights)
        self.assert_parity(model, self.query)


if __name__ == "__main__":
    unittest.main()
