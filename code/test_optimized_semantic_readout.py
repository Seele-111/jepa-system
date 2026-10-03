#!/usr/bin/env python3
"""Synthetic-only CPU tests, with explicit sklearn skip on Windows if absent.

Run python -B -m unittest test_optimized_semantic_readout -v. No checkpoints,
extraction, data access, training runner, installation, file artifacts or caches.
Only the parity test fits two tiny readouts (RF/ET plus their fixed video ET).
Covariance tests use small channel counts; a separate view checks [T,1408].
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import optimized_semantic_readout as semantic
from optimized_feature_view import feature_view
from optimized_locator import augment_signals, video_features

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def record(seed=0, frames=12, width=24):
    rng = np.random.default_rng(seed + 310)
    support = np.ones(frames, dtype=bool)
    if frames > 2:
        support[-1] = False
    direct = (np.arange(frames) % 3 == 0) & support
    labels = np.zeros(frames, dtype=np.uint8)
    if seed % 2:
        labels[frames // 3:2 * frames // 3] = 1
    return {"fps": 25.0, "semantic": rng.normal(size=(frames, width)).astype(np.float32),
            "semantic_names": [f"sem/latent_{j:04d}" for j in range(width)],
            "semantic_support": support, "semantic_direct": direct,
            "motion": rng.uniform(0.1, 2, (frames, 26)).astype(np.float32),
            "motion_names": [f"motion_{j:02d}" for j in range(26)],
            "corrected": rng.uniform(0.1, 2, (frames, 14)).astype(np.float32),
            "corrected_names": [f"corrected_{j:02d}" for j in range(14)],
            "rgb": rng.normal(size=(frames, 4)).astype(np.float32),
            "labels": labels, "sha256": f"synthetic-content-{seed}", "name": f"synthetic-{seed}"}


def identity_pca(width=24, k=16):
    return {"kind": semantic.KIND, "fit_role": "training_content_only_equal_content_covariance",
            "mean": np.zeros(width, dtype=np.float32).tolist(),
            "components": np.eye(width, dtype=np.float32)[:k].tolist(), "n_components": k,
            "semantic_names": [f"sem/latent_{j:04d}" for j in range(width)]}


def assert_records_equal(test, actual, original):
    test.assertEqual(len(actual), len(original))
    for a, b in zip(actual, original):
        test.assertEqual(set(a), set(b))
        for key in a:
            if isinstance(a[key], np.ndarray):
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            else:
                test.assertEqual(a[key], b[key], key)


class NoLabels(dict):
    def __getitem__(self, key):
        if key == "labels":
            raise AssertionError("held-out label read")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "labels":
            raise AssertionError("held-out label read")
        return super().get(key, default)


class NoReads(dict):
    def __getitem__(self, key):
        raise AssertionError("unselected record read")

    def get(self, key, default=None):
        raise AssertionError("unselected record read")


class CaptureClassifier:
    def __init__(self, **params):
        self.params = params
        self.classes_ = np.array([0, 1])
        self.fit = mock.Mock(side_effect=self._fit)

    def _fit(self, x, y, **kwargs):
        self.x, self.y = x.copy(), y.copy()
        self.weights = kwargs.get("sample_weight")
        if self.weights is not None:
            self.weights = self.weights.copy()
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        return np.tile([0.65, 0.35], (len(x), 1))


class CapturedSklearn:
    def __init__(self):
        self.models = []
        self.module, self.ensemble = ModuleType("sklearn"), ModuleType("sklearn.ensemble")
        self.ensemble.RandomForestClassifier = self.build
        self.ensemble.ExtraTreesClassifier = self.build

    def build(self, **kwargs):
        model = CaptureClassifier(**kwargs)
        self.models.append(model)
        return model

    def patch(self):
        return mock.patch.dict(sys.modules, {"sklearn": self.module, "sklearn.ensemble": self.ensemble})


def constant_state(r):
    pca = identity_pca(r["semantic"].shape[1])
    frame, video, names, vnames = semantic._view(r, pca)
    return {"kind": semantic.KIND, "algorithm": "rf", "recipe": "semantic_corrected_motion_rf",
            "transform": {"feature_view": semantic.KIND, "base_recipe": semantic.BASE_RECIPE,
                          "pca": pca, "frame_feature_names": names, "video_feature_names": vnames},
            "frame_model": SimpleNamespace(n_features_in_=frame.shape[1], classes_=np.array([0])),
            "video_model": SimpleNamespace(n_features_in_=len(video), classes_=np.array([1])),
            "fit_content_sha256": ["synthetic-content-train"], "full_fit": False}


class SemanticPcaTests(unittest.TestCase):
    def test_heldout_and_all_labels_cannot_affect_pca(self):
        records = [record(i) for i in range(4)]
        a = semantic.fit_semantic_pca(records, [0, 1], k=16)
        changed = deepcopy(records)
        for r in changed:
            r["labels"] = "must not be read"
            r["rgb"] = None
        changed[2] = NoReads(changed[2])
        changed[3]["semantic"][:] = np.nan
        changed[3]["semantic_names"] = None
        b = semantic.fit_semantic_pca(changed, [0, 1], k=16)
        self.assertEqual(a, b)
        self.assertEqual(a["kind"], semantic.KIND)
        self.assertEqual(a["semantic_names"], records[0]["semantic_names"])
        self.assertEqual(np.asarray(a["components"]).shape, (16, 24))

    def test_aliases_split_content_mass_not_frame_or_video_pool_mass(self):
        records = [record(0), record(1, frames=23)]
        original = semantic.fit_semantic_pca(records, [0, 1], k=16)
        alias = deepcopy(records[0])
        for key in ("semantic", "semantic_support", "semantic_direct"):
            alias[key] = np.repeat(alias[key], 2, axis=0)
        alias["name"] = "alternate-filename"
        records.append(alias)
        duplicate = semantic.fit_semantic_pca(records, [0, 1, 2], k=16)
        np.testing.assert_allclose(duplicate["mean"], original["mean"], atol=2e-6, rtol=0)
        np.testing.assert_allclose(duplicate["components"], original["components"], atol=2e-6, rtol=0)

    def test_supported_frame_normalization_has_known_equal_content_mean(self):
        records = [record(0, frames=3), record(1, frames=7)]
        records[0]["semantic"][0] = 0
        records[0]["semantic"][1] = 2
        records[0]["semantic"][2] = 100000
        records[1]["semantic"][:] = 10
        pca = semantic.fit_semantic_pca(records, [0, 1], k=16)
        np.testing.assert_allclose(pca["mean"], 5.5, atol=1e-6, rtol=0)
        # Pooling all supported frames would be (0+2+6*10)/8 = 7.75.
        self.assertNotEqual(pca["mean"][0], 7.75)

    def test_unsupported_values_and_direct_masks_do_not_affect_pca(self):
        records = [record(0), record(1)]
        expected = semantic.fit_semantic_pca(records, [0, 1])
        changed = deepcopy(records)
        for r in changed:
            r["semantic"][~r["semantic_support"]] = 1e20
            r["semantic_direct"] = r["semantic_support"].copy()
        self.assertEqual(semantic.fit_semantic_pca(changed, [0, 1]), expected)

    def test_records_and_existing_rgb_fields_are_not_mutated(self):
        records = [record(0), record(1)]
        original = deepcopy(records)
        semantic.fit_semantic_pca(records, [1, 0])
        assert_records_equal(self, records, original)

    def test_default_names_are_bound_and_components_have_repeatable_sign(self):
        records = [record(0), record(1)]
        for r in records:
            del r["semantic_names"]
        state = semantic.fit_semantic_pca(records, [0, 1])
        self.assertEqual(state["semantic_names"], identity_pca()["semantic_names"])
        components = np.asarray(state["components"])
        np.testing.assert_allclose(components @ components.T, np.eye(16), atol=2e-6)
        for row in components:
            self.assertGreaterEqual(row[np.argmax(np.abs(row))], 0)

    def test_training_name_order_mismatch_and_no_supported_frames_fail(self):
        records = [record(0), record(1)]
        records[1]["semantic_names"].reverse()
        with self.assertRaisesRegex(ValueError, "name/order"):
            semantic.fit_semantic_pca(records, [0, 1])
        records[1] = record(1)
        records[1]["semantic_support"][:] = False
        records[1]["semantic_direct"][:] = False
        with self.assertRaisesRegex(ValueError, "no supported frames"):
            semantic.fit_semantic_pca(records, [0, 1])

    def test_invalid_partition_k_and_content_keys_fail_closed(self):
        records = [record(0), record(1)]
        for train in ([], [0, 0], [-1], [2], [True], [0.0], None):
            with self.subTest(train=train):
                with self.assertRaises(ValueError):
                    semantic.fit_semantic_pca(records, train)
        for k in (0, -1, 25, True, 16.0, None):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    semantic.fit_semantic_pca(records, [0], k)
        for sha in (None, "", " padded ", 12):
            changed = deepcopy(records)
            changed[0]["sha256"] = sha
            with self.assertRaises(ValueError):
                semantic.fit_semantic_pca(changed, [0])


class SemanticViewTests(unittest.TestCase):
    def test_preserves_exact_base_then_original_augmented_semantic_then_flags(self):
        r, pca = record(), identity_pca()
        base_frame, base_video, base_names = feature_view("corrected_motion_rf", r, None)
        projected = np.zeros((len(r["semantic"]), 16), dtype=np.float32)
        support = r["semantic_support"]
        projected[support] = r["semantic"][support, :16]
        sem_frame, sem_names = augment_signals(projected, [f"sem/pca_{j:04d}" for j in range(16)], r["fps"])
        flags = np.column_stack((r["semantic_direct"], support)).astype(np.float32)
        frame, video, names = semantic.feature_view_semantic(r, pca)
        np.testing.assert_array_equal(frame[:, :base_frame.shape[1]], base_frame)
        np.testing.assert_array_equal(frame[:, base_frame.shape[1]:-2], sem_frame)
        np.testing.assert_array_equal(frame[:, -2:], flags)
        np.testing.assert_array_equal(video, np.concatenate((base_video, video_features(sem_frame), video_features(flags))))
        self.assertEqual(names, base_names + sem_names + ["sem/direct", "sem/support"])
        self.assertEqual(frame.shape, (12, 954))
        self.assertEqual(video.shape, (1256,))
        self.assertEqual(frame.dtype, np.float32)
        self.assertEqual(video.dtype, np.float32)

    def test_true_1408_channel_contract_uses_bound_16_components(self):
        r = record(frames=7, width=1408)
        frame, video, names = semantic.feature_view_semantic(r, identity_pca(1408))
        self.assertEqual(frame.shape, (7, 954))
        self.assertEqual(video.shape, (1256,))
        self.assertEqual(names[-2:], ["sem/direct", "sem/support"])
        self.assertTrue(np.isfinite(frame).all())

    def test_fps_names_and_pca_order_are_stable(self):
        r = record()
        expected_names = None
        for fps in (0.1, 1, 12.5, 25, 240):
            r["fps"] = fps
            frame, _, names = semantic.feature_view_semantic(r, identity_pca())
            if expected_names is None:
                expected_names = names
            self.assertEqual(names, expected_names)
            column = names.index("sem/pca_0000/delta_per_second")
            projected = np.where(r["semantic_support"], r["semantic"][:, 0], 0)
            expected = np.r_[0.0, np.diff(projected)] * fps
            np.testing.assert_allclose(frame[:, column], expected, atol=1e-6, rtol=1e-6)

    def test_unsupported_tail_values_cannot_influence_any_feature(self):
        r = record()
        original = semantic.feature_view_semantic(r, identity_pca())
        changed = deepcopy(r)
        changed["semantic"][~changed["semantic_support"]] = -3e38
        actual = semantic.feature_view_semantic(changed, identity_pca())
        np.testing.assert_array_equal(actual[0], original[0])
        np.testing.assert_array_equal(actual[1], original[1])
        self.assertEqual(actual[2], original[2])

    def test_direct_support_are_explicit_raw_flags_and_video_statistics(self):
        r = record()
        frame, video, names = semantic.feature_view_semantic(r, identity_pca())
        np.testing.assert_array_equal(frame[:, -2], r["semantic_direct"])
        np.testing.assert_array_equal(frame[:, -1], r["semantic_support"])
        self.assertEqual(names[-2:], ["sem/direct", "sem/support"])
        flags = frame[:, -2:]
        expected = np.concatenate((flags.mean(axis=0), flags.std(axis=0),
                                   np.percentile(flags, 10, axis=0), np.percentile(flags, 90, axis=0)))
        np.testing.assert_allclose(video[-8:], expected, atol=1e-7)

    def test_all_unsupported_inference_has_zero_semantic_features(self):
        r = record()
        r["semantic_support"][:] = False
        r["semantic_direct"][:] = False
        frame, video, _ = semantic.feature_view_semantic(r, identity_pca())
        base_frame, base_video, _ = feature_view("corrected_motion_rf", r, None)
        np.testing.assert_array_equal(frame[:, :base_frame.shape[1]], base_frame)
        np.testing.assert_array_equal(frame[:, base_frame.shape[1]:], 0)
        np.testing.assert_array_equal(video[:len(base_video)], base_video)
        np.testing.assert_array_equal(video[len(base_video):], 0)

    def test_labels_filename_sha_rgb_and_other_fields_do_not_enter_view(self):
        r, pca = record(), identity_pca()
        expected = semantic.feature_view_semantic(r, pca)
        changed = deepcopy(r)
        for key in ("name", "filename", "sha256", "event_count", "rgb", "rgb_names", "labels"):
            changed[key] = None
        actual = semantic.feature_view_semantic(NoLabels(changed), pca)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])

    def test_view_and_pca_are_not_mutated(self):
        r, pca = record(), identity_pca()
        original_record, original_pca = deepcopy(r), deepcopy(pca)
        semantic.feature_view_semantic(r, pca)
        assert_records_equal(self, [r], [original_record])
        self.assertEqual(pca, original_pca)

    def test_missing_nonboolean_misaligned_or_inconsistent_masks_fail(self):
        cases = []
        for key in ("semantic_support", "semantic_direct"):
            missing = record()
            del missing[key]
            cases.append(missing)
            for value in (None, np.ones(12), np.ones((12, 1), dtype=bool), np.ones(11, dtype=bool)):
                changed = record()
                changed[key] = value
                cases.append(changed)
        inconsistent = record()
        inconsistent["semantic_direct"][-1] = True
        cases.append(inconsistent)
        for changed in cases:
            with self.assertRaises(ValueError):
                semantic.feature_view_semantic(changed, identity_pca())
            with self.assertRaises(ValueError):
                semantic.fit_semantic_pca([changed], [0])

    def test_invalid_semantic_fps_names_and_pca_fail_closed(self):
        for value in (None, np.zeros(12), np.zeros((0, 24)), np.zeros((12, 0)),
                      np.full((12, 24), np.nan), np.full((12, 24), np.inf),
                      np.full((12, 24), 1e300), np.zeros((11, 24)), np.full((12, 24), "0")):
            changed = record()
            changed["semantic"] = value
            with self.assertRaises(ValueError):
                semantic.feature_view_semantic(changed, identity_pca())
        for value in (0, -1, True, np.nan, np.inf, "25", None):
            changed = record()
            changed["fps"] = value
            with self.assertRaises(ValueError):
                semantic.feature_view_semantic(changed, identity_pca())
        for names in (None, ["duplicate"] * 24, ["too-short"], list(reversed(record()["semantic_names"]))):
            changed = record()
            changed["semantic_names"] = names
            with self.assertRaises(ValueError):
                semantic.feature_view_semantic(changed, identity_pca())
        for key, value in (("kind", "other"), ("fit_role", "heldout"), ("n_components", True),
                           ("mean", [0]), ("components", [[0] * 24]), ("mean", [np.nan] * 24),
                           ("semantic_names", list(reversed(record()["semantic_names"])))):
            pca = identity_pca()
            pca[key] = value
            with self.assertRaises(ValueError):
                semantic.feature_view_semantic(record(), pca)


class SemanticMemberFitTests(unittest.TestCase):
    def test_exact_original_estimator_parameters_labels_and_frame_video_weights(self):
        records = [record(0, frames=9), record(1, frames=15)]
        records.append(deepcopy(records[0]))
        records.append(NoLabels(record(3, frames=11)))
        records.append(NoReads())
        train, predict = [0, 1, 2], [3]
        for algorithm in ("rf", "et"):
            captured = CapturedSklearn()
            with captured.patch():
                fp, vp, state = semantic.fit_semantic_member(algorithm, records, train, predict, 42)
            frame_model, video_model = captured.models
            expected_frame = {"n_estimators": 160 if algorithm == "rf" else 192,
                              "max_depth": 8 if algorithm == "rf" else 9,
                              "min_samples_leaf": 10 if algorithm == "rf" else 8,
                              "max_features": 0.5 if algorithm == "rf" else 0.6,
                              "n_jobs": 4, "random_state": 42}
            self.assertEqual(frame_model.params, expected_frame)
            self.assertEqual(video_model.params, {"n_estimators": 128, "max_depth": 4,
                             "min_samples_leaf": 2, "max_features": 0.75, "class_weight": "balanced",
                             "n_jobs": 4, "random_state": 1142})
            y = np.concatenate([records[i]["labels"] for i in train])
            w = np.concatenate([np.full(len(records[i]["labels"]), 1 / len(records[i]["labels"]), np.float64)
                                for i in train])
            positive, negative = w[y > 0].sum(), w[y == 0].sum()
            w = np.where(y > 0, w * negative / max(1e-8, positive), w)
            w *= len(w) / w.sum()
            np.testing.assert_array_equal(frame_model.y, y)
            np.testing.assert_array_equal(frame_model.weights, w)
            self.assertAlmostEqual(float(frame_model.weights.mean()), 1, places=12)
            self.assertAlmostEqual(float(frame_model.weights[y > 0].sum()), len(w) / 2, places=12)
            np.testing.assert_array_equal(video_model.y, [0, 1, 0])
            self.assertEqual(video_model.fit.call_args.kwargs, {})
            views = [semantic.feature_view_semantic(records[i], state["transform"]["pca"]) for i in train]
            np.testing.assert_array_equal(frame_model.x, np.concatenate([v[0] for v in views]))
            np.testing.assert_array_equal(video_model.x, np.stack([v[1] for v in views]))
            self.assertEqual(set(fp), {3})
            self.assertEqual(set(vp), {3})
            self.assertEqual(fp[3].dtype, np.float32)
            self.assertEqual(state["fit_content_sha256"], sorted({records[i]["sha256"] for i in train}))
            self.assertEqual(state["transform"]["feature_view"], "semantic-content-readout-v1")
            self.assertEqual(state["transform"]["base_recipe"], "corrected_motion_rf")
            self.assertEqual(state["transform"]["frame_feature_names"], views[0][2])
            self.assertEqual(state["transform"]["pca"]["n_components"], 16)
            self.assertFalse(state["full_fit"])

    def test_sha_group_holdout_and_index_leakage_rejected_before_pca_or_fit(self):
        records = [record(0), record(1), record(2)]
        records[2]["sha256"] = records[0]["sha256"]
        with mock.patch.object(semantic, "fit_semantic_pca", side_effect=AssertionError("unexpected PCA")) as pca:
            for train, predict, full_fit in (([0], [0], False), ([0, 1], [2], False),
                                             ([0], [2], True), ([0, 1], [0], True)):
                with self.assertRaises(ValueError):
                    semantic.fit_semantic_member("rf", records, train, predict, 1, full_fit)
            pca.assert_not_called()

    def test_full_fit_is_explicit_and_predicts_exact_training_set(self):
        records = [record(0), record(1)]
        captured = CapturedSklearn()
        with captured.patch():
            fp, vp, state = semantic.fit_semantic_member("et", records, [0, 1], [1, 0], 8, full_fit=True)
        self.assertEqual(set(fp), {0, 1})
        self.assertEqual(set(vp), {0, 1})
        self.assertTrue(state["full_fit"])
        self.assertEqual(state["fit_content_sha256"], sorted(r["sha256"] for r in records))

    def test_predict_labels_and_unselected_records_are_not_read_or_mutated(self):
        records = [record(0), record(1), record(2)]
        originals = deepcopy(records)
        captured = CapturedSklearn()
        with captured.patch():
            semantic.fit_semantic_member("rf", records, [0, 1], [2], 9)
        assert_records_equal(self, records, originals)
        guarded = [records[0], records[1], NoLabels(records[2]), NoReads()]
        captured2 = CapturedSklearn()
        with captured2.patch():
            fp, vp, state = semantic.fit_semantic_member("rf", guarded, [0, 1], [2], 9)
        np.testing.assert_array_equal(captured.models[0].x, captured2.models[0].x)
        np.testing.assert_array_equal(captured.models[0].y, captured2.models[0].y)
        self.assertEqual(state["fit_content_sha256"], sorted(r["sha256"] for r in records[:2]))
        self.assertEqual(set(fp), {2})
        self.assertEqual(set(vp), {2})

    def test_base_feature_order_mismatch_is_rejected_before_sklearn(self):
        records = [record(0), record(1), NoLabels(record(2))]
        records[2]["motion_names"] = list(reversed(records[2]["motion_names"]))
        with self.assertRaisesRegex(ValueError, "name/order differs"):
            semantic.fit_semantic_member("rf", records, [0, 1], [2], 1)

    def test_invalid_algorithms_partitions_seed_fullfit_and_labels_fail(self):
        records = [record(0), record(1), record(2)]
        for algorithm in ("gb", "tcn", "RF", "et_new", None):
            with self.assertRaises(ValueError):
                semantic.fit_semantic_member(algorithm, records, [0, 1], [2], 1)
        for train, predict in (([], [2]), ([0, 0], [2]), ([0], []), ([0], [2, 2]),
                               ([True], [2]), ([0], [3]), ([0.0], [2]), ([0], [-1])):
            with self.assertRaises(ValueError):
                semantic.fit_semantic_member("rf", records, train, predict, 1)
        for seed in (True, -1, 1.0, None, 2 ** 32 - 1100):
            with self.assertRaises(ValueError):
                semantic.fit_semantic_member("rf", records, [0, 1], [2], seed)
        for full_fit in (1, "true", None):
            with self.assertRaises(ValueError):
                semantic.fit_semantic_member("rf", records, [0, 1], [2], 1, full_fit)
        for labels in (None, [0], np.full(12, np.nan), np.full(12, 2), np.ones(12)):
            changed = deepcopy(records)
            changed[0]["labels"] = labels
            # np.ones makes all training frames positive and exercises the
            # original weight normalization's undefined no-negative case.
            with self.assertRaises(ValueError):
                semantic.fit_semantic_member("rf", changed, [0], [2], 1)

    @unittest.skipUnless(HAS_SKLEARN, "sklearn unavailable: real RF/ET portable parity SKIPPED; run existing WSL environment")
    def test_small_rf_et_fits_json_export_and_portable_parity(self):
        records = [record(i, frames=16) for i in range(4)]
        records.append(NoLabels(record(4, frames=13)))
        records.append(NoReads())
        for algorithm in ("rf", "et"):
            fp, vp, state = semantic.fit_semantic_member(algorithm, records, [0, 1, 2, 3], [4], 72)
            expected = {"n_estimators": 160 if algorithm == "rf" else 192,
                        "max_depth": 8 if algorithm == "rf" else 9,
                        "min_samples_leaf": 10 if algorithm == "rf" else 8,
                        "max_features": 0.5 if algorithm == "rf" else 0.6,
                        "n_jobs": 4, "random_state": 72}
            for key, value in expected.items():
                self.assertEqual(state["frame_model"].get_params()[key], value)
            expected_video = {"n_estimators": 128, "max_depth": 4, "min_samples_leaf": 2,
                              "max_features": 0.75, "class_weight": "balanced", "n_jobs": 4,
                              "random_state": 1172}
            for key, value in expected_video.items():
                self.assertEqual(state["video_model"].get_params()[key], value)
            member = json.loads(json.dumps(semantic.export_semantic_member(state), allow_nan=False))
            max_frame, max_video = 0.0, 0.0
            for i in range(5):
                frame, video, _ = semantic.feature_view_semantic(records[i], state["transform"]["pca"])
                actual_frame, actual_video = semantic.predict_semantic_member(member, records[i])
                expected_frame = state["frame_model"].predict_proba(frame)[:, 1]
                expected_video_p = state["video_model"].predict_proba(video[None, :])[0, 1]
                max_frame = max(max_frame, float(np.max(np.abs(actual_frame - expected_frame))))
                max_video = max(max_video, abs(actual_video - expected_video_p))
                np.testing.assert_allclose(actual_frame, expected_frame, atol=2e-6, rtol=0)
                self.assertAlmostEqual(actual_video, expected_video_p, delta=2e-6)
                self.assertEqual(actual_frame.dtype, np.float32)
            reloaded_frame, reloaded_video = semantic.predict_semantic_member(member, records[4])
            np.testing.assert_allclose(reloaded_frame, fp[4], atol=2e-6, rtol=0)
            self.assertAlmostEqual(reloaded_video, vp[4], delta=2e-6)
            self.assertEqual(member["fit_content_sha256"], sorted(r["sha256"] for r in records[:4]))
            print(f"[semantic parity] {algorithm}: max_abs_frame={max_frame:.9g}, max_abs_video={max_video:.9g}", flush=True)


class SemanticPortableTests(unittest.TestCase):
    def test_constant_export_prediction_provenance_and_deepcopy(self):
        r = record()
        state = constant_state(r)
        original_pca = deepcopy(state["transform"]["pca"])
        member = json.loads(json.dumps(semantic.export_semantic_member(state), allow_nan=False))
        self.assertEqual(member["kind"], "semantic-content-readout-v1")
        self.assertEqual(member["boundary_models"], None)
        self.assertEqual(member["weight"], 1.0)
        fp, vp = semantic.predict_semantic_member(member, NoLabels(r))
        np.testing.assert_array_equal(fp, np.zeros(len(r["semantic"]), dtype=np.float32))
        self.assertEqual(vp, 1.0)
        member["transform"]["pca"]["mean"][0] = 100
        self.assertEqual(state["transform"]["pca"], original_pca)

    def test_portable_forest_uses_original_inclusive_split(self):
        r = record()
        member = semantic.export_semantic_member(constant_state(r))
        width = member["frame_model"]["n_features"]
        member["frame_model"] = {"kind": "forest", "n_features": width,
             "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [width - 1, -2, -2],
                        "threshold": [0.5, -2, -2], "value": [0.5, 0.1, 0.9]}]}
        fp, _ = semantic.predict_semantic_member(member, r)
        np.testing.assert_allclose(fp, np.where(r["semantic_support"], 0.9, 0.1), atol=1e-7)

    def test_predict_checks_frame_video_and_pca_name_order(self):
        r = record()
        original = semantic.export_semantic_member(constant_state(r))
        for field in ("frame_feature_names", "video_feature_names"):
            changed = deepcopy(original)
            changed["transform"][field][0:2] = reversed(changed["transform"][field][0:2])
            with self.assertRaisesRegex(ValueError, "name/order"):
                semantic.predict_semantic_member(changed, r)
        changed = deepcopy(r)
        changed["semantic_names"].reverse()
        with self.assertRaisesRegex(ValueError, "name/order"):
            semantic.predict_semantic_member(original, changed)
        changed = deepcopy(original)
        changed["transform"]["pca"]["mean"] = [0.0]
        with self.assertRaises(ValueError):
            semantic.predict_semantic_member(changed, r)

    def test_invalid_member_transform_models_and_provenance_fail_closed(self):
        r = record()
        state, member = constant_state(r), semantic.export_semantic_member(constant_state(r))
        for key, value in (("kind", "other"), ("algorithm", "tcn"), ("recipe", "wrong"),
                           ("transform", {}), ("frame_model", None), ("video_model", None)):
            changed = deepcopy(member)
            changed[key] = value
            with self.assertRaises(ValueError):
                semantic.predict_semantic_member(changed, r)
        for field, value in (("feature_view", "other"), ("base_recipe", "rgb_corrected_motion_rf"),
                             ("frame_feature_names", []), ("video_feature_names", [])):
            changed = deepcopy(member)
            changed["transform"][field] = value
            with self.assertRaises(ValueError):
                semantic.predict_semantic_member(changed, r)
        for key, value in (("frame_model", object()), ("fit_content_sha256", []),
                           ("fit_content_sha256", ["duplicate", "duplicate"]), ("full_fit", 1)):
            changed = deepcopy(state)
            changed[key] = value
            with self.assertRaises(ValueError):
                semantic.export_semantic_member(changed)

    def test_invalid_portable_constants_and_cyclic_trees_fail_before_walking(self):
        r = record()
        original = semantic.export_semantic_member(constant_state(r))
        for value in (np.nan, np.inf, -0.1, 1.1, True):
            changed = deepcopy(original)
            changed["frame_model"]["probability"] = value
            with self.assertRaises(ValueError):
                semantic.predict_semantic_member(changed, r)
        for value in (True, 954.0, 1):
            changed = deepcopy(original)
            changed["frame_model"]["n_features"] = value
            with self.assertRaises(ValueError):
                semantic.predict_semantic_member(changed, r)
        width = original["frame_model"]["n_features"]
        cyclic = deepcopy(original)
        cyclic["frame_model"] = {"kind": "forest", "n_features": width,
             "trees": [{"left": [0, -1], "right": [1, -1], "feature": [0, -2],
                        "threshold": [0.5, -2], "value": [0.5, 0.1]}]}
        with self.assertRaisesRegex(ValueError, "cyclic"):
            semantic.predict_semantic_member(cyclic, r)

    def test_import_pca_view_export_and_predict_are_numpy_only(self):
        program = '''import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"sklearn", "torch"}:
        raise AssertionError("training/extraction dependency imported")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import numpy as np
from types import SimpleNamespace
import optimized_semantic_readout as semantic
r = {"fps": 25.0, "semantic": np.arange(48, dtype=np.float32).reshape(3,16),
     "semantic_support": np.array([True, True, False]), "semantic_direct": np.array([True, True, False]),
     "motion": np.full((3,2),0.4,np.float32), "motion_names": ["motion0","motion1"],
     "corrected": np.full((3,14),0.5,np.float32), "corrected_names": ["corrected"+str(i) for i in range(14)],
     "sha256": "synthetic-train"}
pca = semantic.fit_semantic_pca([r], [0])
f, v, names, vnames = semantic._view(r,pca)
state = {"kind": semantic.KIND, "algorithm": "rf", "recipe": "semantic_corrected_motion_rf",
         "transform": {"feature_view": semantic.KIND, "base_recipe": semantic.BASE_RECIPE,
                       "pca": pca, "frame_feature_names": names, "video_feature_names": vnames},
         "frame_model": SimpleNamespace(n_features_in_=f.shape[1],classes_=np.array([0])),
         "video_model": SimpleNamespace(n_features_in_=len(v),classes_=np.array([1])),
         "fit_content_sha256": ["synthetic-train"], "full_fit": True}
member = semantic.export_semantic_member(state)
fp, vp = semantic.predict_semantic_member(member,r)
assert np.all(fp == 0) and vp == 1
assert not any(k == "sklearn" or k.startswith("sklearn.") for k in sys.modules)
print("numpy-only semantic PCA and inference OK")
'''
        run = subprocess.run([sys.executable, "-B", "-c", program],
                             cwd=Path(__file__).resolve().parent,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("numpy-only semantic PCA and inference OK", run.stdout)


if __name__ == "__main__":
    unittest.main()
