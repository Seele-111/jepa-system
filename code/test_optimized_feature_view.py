"""Shared feature-view contract tests on tiny synthetic train/eval records.

No dataset access, disk artifacts, GPU, RF/TCN fitting, or dependency installs.
Only PCA is fitted, on synthetic TRAIN records. sklearn checks skip if absent;
use the existing WSL venv to execute them. Production regressions are strict
failures, not expectedFailure, so the owner can fix the production modules.
"""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from optimized_feature_view import feature_view, temporal_features


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
CORRECTED_NAMES = [prefix + "_" + suffix for prefix in ("v", "i") for suffix in (
    "absolute_mean", "patch_std", "patch_p90", "patch_p99", "patch_fraction",
    "direct_observation", "interpolation_range_supported")]
RANK_NAMES = ["diff_rank", "flow_rank", "mean_rank", "max_rank"]


def synthetic_record():
    rng = np.random.default_rng(221)
    jepa = rng.uniform(0.1, 2, (6, 5)).astype(np.float32)
    jepa[0, 0] = 0
    return {
        "fps": 12., "jepa": jepa,
        "jepa_names": ["legacy_v", "legacy_i", "legacy_composite", "legacy_aux0", "legacy_aux1"],
        "motion": rng.uniform(1, 10, (6, 2)).astype(np.float32),
        "motion_names": ["absolute_diff", "absolute_flow"],
        "rank_motion": rng.uniform(0, 1, (6, 4)).astype(np.float32),
        "corrected": rng.uniform(0, 20, (6, 14)).astype(np.float32),
        "corrected_names": CORRECTED_NAMES[:],
        "rgb": rng.normal(size=(6, 4)).astype(np.float32),
        "labels": np.asarray([0, 0, 1, 1, 0, 0]),
        "name": "synthetic", "source_sha256": "not-a-real-file",
    }


def synthetic_pca_state():
    return {"mean": [1., 2., 3., 4.],
            "components": [[1., 0., 0., 0.], [0., 0.6, 0., 0.8], [0., 0., 1., 0.]]}


def video_stats(values):
    """Independent statistical oracle, not a call to the production helper."""
    return np.concatenate([np.mean(values, axis=0), np.std(values, axis=0),
                           np.quantile(values, 0.1, axis=0), np.quantile(values, 0.9, axis=0)])


def augmented_names(raw_names, temporal):
    names = list(raw_names) + [n + "/delta_per_second" for n in raw_names]
    if temporal:
        return names + [n + "/relative_robust_z" for n in raw_names]
    names += [n + "/abs_delta_per_second" for n in raw_names]
    for seconds in ("0.12", "0.35", "0.75"):
        for stat in ("mean", "std", "contrast"):
            names += [n + "/" + stat + "_" + seconds + "s" for n in raw_names]
    names += [n + "/relative_robust_z" for n in raw_names]
    for stat in ("mean", "std", "p10", "p90"):
        names += [n + "/global_" + stat for n in raw_names]
    return names


def source_block(source, record, pca, temporal):
    if source == "jepa":
        raw, names = record["jepa"], record["jepa_names"]
        base = raw[:, :3]
        if temporal:
            return base, augmented_names(names[:3], True), video_stats(base)
        return raw, names + [f"jepa_global_{i}" for i in range(12)] + ["jepa_positive_error_support"], video_stats(base)
    if source == "rank":
        raw, names = record["rank_motion"], RANK_NAMES
    elif source == "rgb":
        raw = (record["rgb"] - np.asarray(pca["mean"], dtype=np.float32)) @ np.asarray(pca["components"], dtype=np.float32).T
        names = [f"r3d_pca_{i}" for i in range(raw.shape[1])]
    else:
        raw, names = record[source], record[source + "_names"]
    return raw, augmented_names(names, temporal), video_stats(raw)


def load_training_make_view():
    """Execute the actual PCA wrapper only, without importing its training driver.

    The other module bodies (dataset loaders, experiment driver, etc.) are NOT
    executed. This is a named source read, not a data/dataset scan.
    """
    path = Path(__file__).with_name("run_algorithm_optimization.py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_view"]
    if len(selected) != 1:
        raise AssertionError("expected one actual training make_view function")
    namespace = {"np": np, "feature_view": feature_view}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["make_view"]


class FeatureViewTests(unittest.TestCase):
    def assert_rejected_cases(self, cases):
        """Collect the full guard matrix without masking production bugs as skips."""
        unexpected = []
        with np.errstate(all="ignore"):
            for label, recipe, record, state in cases:
                try:
                    feature_view(recipe, record, state)
                except ValueError:
                    continue
                except Exception as exc:
                    unexpected.append(label + ": wrong exception " + type(exc).__name__)
                else:
                    unexpected.append(label + ": accepted invalid input")
        self.assertEqual(unexpected, [], "feature_view validation regressions: " + "; ".join(unexpected))

    def assert_recipe_schema(self, cases, temporal):
        record, pca = synthetic_record(), synthetic_pca_state()
        for recipe, sources, frame_dim, video_dim in cases:
            with self.subTest(recipe=recipe):
                frames, video, names = feature_view(recipe, record, pca)
                self.assertEqual(frames.shape, (6, frame_dim))
                self.assertEqual(video.shape, (video_dim,))
                self.assertEqual(frames.dtype, np.float32)
                self.assertEqual(video.dtype, np.float32)
                self.assertTrue(np.isfinite(frames).all() and np.isfinite(video).all())
                expected_names, expected_stats, offset = [], [], 0
                for source in sources:
                    raw, block_names, stats = source_block(source, record, pca, temporal)
                    np.testing.assert_allclose(frames[:, offset:offset+raw.shape[1]], raw, rtol=1e-6, atol=1e-6)
                    if source == "jepa" and not temporal:
                        np.testing.assert_allclose(frames[:, offset+5:offset+17], np.broadcast_to(stats, (6, 12)), rtol=1e-6)
                        np.testing.assert_array_equal(frames[:, offset+17], (raw[:, 0] > 0).astype(np.float32))
                    expected_names.extend(block_names)
                    expected_stats.append(stats)
                    offset += len(block_names)
                self.assertEqual(names, expected_names)
                self.assertEqual(len(names), frame_dim)
                self.assertEqual(len(set(names)), frame_dim)
                np.testing.assert_allclose(video, np.concatenate(expected_stats), rtol=1e-6, atol=1e-6)
                if temporal:
                    self.assertFalse(any("global" in name for name in names))

    def test_rf_recipes_have_explicit_dimensions_ordered_names_and_absolute_raw_blocks(self):
        self.assert_recipe_schema([
            ("motion_rf", ("motion",), 34, 8),
            ("motion_rank_rf", ("rank",), 68, 16),
            ("rgb_rf", ("rgb",), 51, 12),
            ("jepa_rf", ("jepa",), 18, 12),
            ("hybrid_rf", ("jepa", "motion"), 52, 20),
            ("corrected_rf", ("corrected",), 238, 56),
            ("corrected_hybrid_rf", ("motion", "corrected"), 272, 64),
            ("rgb_hybrid_rf", ("jepa", "motion", "rgb"), 103, 32),
            ("rgb_corrected_motion_rf", ("motion", "corrected", "rgb"), 323, 76),
        ], temporal=False)

    def test_tcn_recipes_have_three_channels_per_raw_feature_without_global_identity(self):
        self.assert_recipe_schema([
            ("motion_tcn", ("motion",), 6, 8),
            ("motion_rank_tcn", ("rank",), 12, 16),
            ("rgb_tcn", ("rgb",), 9, 12),
            ("jepa_tcn", ("jepa",), 9, 12),
            ("hybrid_tcn", ("jepa", "motion"), 15, 20),
            ("corrected_tcn", ("corrected",), 42, 56),
            ("corrected_hybrid_tcn", ("motion", "corrected"), 48, 64),
            ("rgb_hybrid_tcn", ("jepa", "motion", "rgb"), 24, 32),
            ("rgb_corrected_motion_tcn", ("motion", "corrected", "rgb"), 57, 76),
        ], temporal=True)

    def test_temporal_delta_uses_fps_while_absolute_and_robust_channels_do_not(self):
        raw = np.asarray([[2, 1], [4, 3], [1, 8], [7, 5]], dtype=np.float32)
        raw_before = raw.copy()
        first, names = temporal_features(raw, ["x", "y"], 4.)
        faster, other_names = temporal_features(raw, ["x", "y"], 8.)
        np.testing.assert_array_equal(first[:, :2], raw)
        np.testing.assert_array_equal(first[:, 2:4], [[0, 0], [8, 8], [-12, 20], [24, -12]])
        np.testing.assert_array_equal(faster[:, 2:4], 2 * first[:, 2:4])
        np.testing.assert_array_equal(faster[:, 4:], first[:, 4:])
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((np.abs(first[:, 4:]) <= 8).all())
        self.assertEqual(names, ["x", "y", "x/delta_per_second", "y/delta_per_second",
                                 "x/relative_robust_z", "y/relative_robust_z"])
        self.assertEqual(names, other_names)
        np.testing.assert_array_equal(raw, raw_before)

    def test_pca_applies_supplied_center_and_matrix_without_normalizing_or_refitting(self):
        record = synthetic_record()
        # Deliberately not an orthonormal matrix: the view must apply exactly the
        # supplied transform, not infer a different transform from eval data.
        state = {"mean": [1., -2., 3., 0.5], "components": [[2., 0., 0., 1.], [0., 3., -1., 0.]]}
        expected = np.column_stack([
            2 * (record["rgb"][:, 0] - 1) + record["rgb"][:, 3] - 0.5,
            3 * (record["rgb"][:, 1] + 2) - record["rgb"][:, 2] + 3,
        ])
        for recipe in ("rgb_rf", "rgb_tcn"):
            with self.subTest(recipe=recipe):
                frames, video, names = feature_view(recipe, record, state)
                np.testing.assert_allclose(frames[:, :2], expected, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(video, video_stats(expected), rtol=1e-6, atol=1e-6)
                self.assertEqual(names[:2], ["r3d_pca_0", "r3d_pca_1"])

    @unittest.skipUnless(HAS_SKLEARN, "sklearn unavailable; rerun in existing WSL venv")
    def test_json_portable_pca_matches_sklearn_transform_on_disjoint_eval_inputs(self):
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(501)
        train = rng.normal(size=(40, 4)) * np.asarray([1., 2., 3., 4.])
        evaluation = rng.normal(size=(6, 4)) + 7.  # Never used in fit.
        fitted = PCA(n_components=3, svd_solver="full", whiten=False).fit(train)
        state = json.loads(json.dumps({"mean": fitted.mean_.tolist(), "components": fitted.components_.tolist()}))
        state_before = deepcopy(state)
        expected = fitted.transform(evaluation)
        record = synthetic_record()
        record["rgb"] = evaluation
        with patch.object(PCA, "fit", side_effect=AssertionError("cannot fit on eval")), \
                patch.object(PCA, "fit_transform", side_effect=AssertionError("cannot fit_transform on eval")):
            for recipe in ("rgb_rf", "rgb_tcn"):
                frames, _, _ = feature_view(recipe, record, state)
                np.testing.assert_allclose(frames[:, :3], expected, rtol=5e-6, atol=5e-6)
        self.assertEqual(state, state_before)

    @unittest.skipUnless(HAS_SKLEARN, "sklearn unavailable; rerun in existing WSL venv")
    def test_actual_training_wrapper_fits_pca_only_on_train_and_eval_perturbation_cannot_change_it(self):
        from sklearn.decomposition import PCA
        make_view = load_training_make_view()
        records = [synthetic_record() for _ in range(3)]
        records[1]["rgb"] = records[1]["rgb"] + 2.
        records[2]["rgb"] = records[2]["rgb"] + 1000.  # Sentinel eval distribution.
        train_pool = np.concatenate([records[0]["rgb"], records[1]["rgb"]])
        seen_fit_inputs = []
        original_fit = PCA.fit

        def traced_fit(instance, values, *args, **kwargs):
            seen_fit_inputs.append(np.asarray(values).copy())
            return original_fit(instance, values, *args, **kwargs)

        with patch.object(PCA, "fit", new=traced_fit):
            first_frames, first_video, first_state = make_view("rgb_rf", records, [0, 1], 31)
            altered = deepcopy(records)
            altered[2]["rgb"] += 5000.
            second_frames, second_video, second_state = make_view("rgb_rf", altered, [0, 1], 31)
        self.assertEqual(len(seen_fit_inputs), 2)
        for fitted_input in seen_fit_inputs:
            np.testing.assert_array_equal(fitted_input, train_pool)
        np.testing.assert_allclose(first_state["pca"]["mean"], train_pool.mean(axis=0), rtol=1e-6, atol=1e-6)
        self.assertEqual(first_state, second_state)
        for i in (0, 1):
            np.testing.assert_array_equal(first_frames[i], second_frames[i])
            np.testing.assert_array_equal(first_video[i], second_video[i])
        self.assertFalse(np.array_equal(first_frames[2], second_frames[2]))

    def test_invalid_fps_is_rejected_on_every_feature_branch(self):
        cases = []
        for recipe in ("jepa_rf", "motion_rf", "jepa_tcn", "motion_tcn", "rgb_tcn", "corrected_tcn"):
            for fps in (0., -1., np.nan, np.inf):
                record = synthetic_record()
                record["fps"] = fps
                cases.append((recipe + "/fps=" + str(fps), recipe, record, synthetic_pca_state()))
        self.assert_rejected_cases(cases)

    def test_unaligned_selected_modalities_are_rejected_in_rf_and_tcn(self):
        cases = []
        for recipe, field in (("hybrid_rf", "motion"), ("hybrid_tcn", "motion"),
                              ("corrected_hybrid_rf", "corrected"), ("corrected_hybrid_tcn", "corrected"),
                              ("rgb_hybrid_rf", "rgb"), ("rgb_corrected_motion_tcn", "rgb")):
            record = synthetic_record()
            record[field] = record[field][:-1]
            cases.append((recipe + "/short=" + field, recipe, record, synthetic_pca_state()))
        self.assert_rejected_cases(cases)

    def test_nonfinite_selected_features_are_rejected_including_legacy_jepa_rf(self):
        cases = []
        for recipe, field in (("jepa_rf", "jepa"), ("jepa_tcn", "jepa"),
                              ("motion_rf", "motion"), ("motion_tcn", "motion"),
                              ("corrected_rf", "corrected"), ("corrected_tcn", "corrected"),
                              ("rgb_rf", "rgb"), ("rgb_tcn", "rgb")):
            for value in (np.nan, np.inf):
                record = synthetic_record()
                record[field][0, 0] = value
                cases.append((recipe + "/value=" + str(value), recipe, record, synthetic_pca_state()))
        self.assert_rejected_cases(cases)

    def test_raw_name_width_mismatches_cannot_create_a_false_feature_schema(self):
        cases = []
        for recipe, names_key in (("jepa_rf", "jepa_names"), ("jepa_tcn", "jepa_names"),
                                  ("motion_rf", "motion_names"), ("corrected_tcn", "corrected_names")):
            record = synthetic_record()
            # For J-TCN the first 3 names themselves must be present.
            record[names_key] = record[names_key][:2] if recipe == "jepa_tcn" else record[names_key][:-1]
            cases.append((recipe + "/short_names", recipe, record, synthetic_pca_state()))
        self.assert_rejected_cases(cases)

    def test_missing_malformed_or_zero_dimensional_pca_is_rejected(self):
        states = [
            ("missing", None),
            ("wrong_mean_width", {"mean": [0, 0, 0], "components": np.ones((2, 4))}),
            ("wrong_component_width", {"mean": np.zeros(4), "components": np.ones((2, 5))}),
            ("nonfinite_matrix", {"mean": np.zeros(4), "components": np.full((2, 4), np.nan)}),
            ("zero_components", {"mean": np.zeros(4), "components": np.empty((0, 4))}),
        ]
        cases = [(recipe + "/" + label, recipe, synthetic_record(), state)
                 for recipe in ("rgb_rf", "rgb_tcn") for label, state in states]
        self.assert_rejected_cases(cases)

    def test_feature_view_is_label_free_deterministic_and_preserves_inputs_and_pca(self):
        record, state = synthetic_record(), synthetic_pca_state()
        original, original_state = deepcopy(record), deepcopy(state)
        first = feature_view("rgb_corrected_motion_tcn", record, state)
        changed_labels = deepcopy(record)
        changed_labels["labels"] = 1 - record["labels"]
        changed_labels["name"] = "different-identity-not-a-feature"
        second = feature_view("rgb_corrected_motion_tcn", changed_labels, state)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        for key, before in original.items():
            if isinstance(before, np.ndarray):
                np.testing.assert_array_equal(record[key], before)
            else:
                self.assertEqual(record[key], before)
        self.assertEqual(state, original_state)


if __name__ == "__main__":
    unittest.main()
