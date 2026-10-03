"""Synthetic-only temporal head tests; no dataset, installs, file writes or GPU.

Windows (NumPy checks; Torch tests are explicitly skipped):
    python -B -m unittest discover -s E:\\jepa-system\\code \
        -p test_optimized_temporal_head.py -v
WSL (existing CPU PyTorch environment, CUDA hidden):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 /home/zzy/vjepa2-main/vjepa-env/bin/python -B \
        -m unittest discover -s /mnt/e/jepa-system/code \
        -p test_optimized_temporal_head.py -v
"""
from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from optimized_temporal_head import (
    CHANNELS, DILATIONS, DROPOUT, GRID_RULE, POSITIVE_WEIGHT, SCHEMA,
    _build_torch_model, _conv1d_same, _fit_normalization, _masked_video_loss,
    _normalize, _pad_batch, _resample_features, _resample_labels, _sigmoid,
    _time_grid, predict_temporal, train_temporal,
)

# Do not import/use a Windows Torch installation: the supported training check
# runs explicitly in the existing WSL CPU environment, without CUDA inspection.
if os.name == "nt":
    TORCH = None
    TORCH_SKIP = "Windows: Torch CPU training/parity tests must run in the existing WSL environment"
else:
    try:
        import torch as TORCH
    except ImportError:
        TORCH = None
    TORCH_SKIP = "CPU PyTorch unavailable; run these tests in the existing WSL environment"


def synthetic_bundle(dimension: int = 7, seed: int = 41, *, zero: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    layers = []
    inputs = dimension
    for index in range(4):
        outputs, kernel = (CHANNELS, 3) if index < 3 else (1, 1)
        dilation = DILATIONS[index] if index < 3 else 1
        weight = rng.normal(0, np.sqrt(1 / (inputs * kernel)),
                            (outputs, inputs, kernel)).astype(np.float32)
        bias = rng.normal(0, 0.04, outputs).astype(np.float32)
        if zero:
            weight.fill(0)
            bias.fill(0)
        layers.append({"weight": weight.tolist(), "bias": bias.tolist(),
                       "dilation": dilation, "padding": dilation if index < 3 else 0,
                       "activation": "relu" if index < 3 else "sigmoid"})
        inputs = outputs
    return {"schema": SCHEMA, "feature_dim": dimension,
            "mean": np.zeros(dimension, np.float32).tolist(),
            "std": np.ones(dimension, np.float32).tolist(), "convs": layers,
            "fps_profile": {"target_hz": 10.0, "grid": GRID_RULE, "features": "linear"}}


def synthetic_training() -> tuple[list, list, list]:
    rng = np.random.default_rng(92)
    videos, labels = [], []
    rates = [10.0, 24.0, 30000 / 1001, 7.5]
    for length, rate in zip((1, 27, 53, 79), rates):
        time = np.arange(length) / rate
        y = ((time >= 0.25) & (time < 0.9)).astype(np.float32)
        x = rng.normal(0, 0.15, (length, 7)).astype(np.float32)
        x[:, 0] += 2 * y
        x[:, 1] += time.astype(np.float32)
        x[:, -1] = 3.0  # A train-only constant channel must have unit scale.
        videos.append(x)
        labels.append(y)
    return videos, labels, rates


class TimingAndTransformTests(unittest.TestCase):
    def test_real_fps_grid_includes_both_endpoints_without_stretching(self):
        for length in (1, 2, 4, 17, 123):
            for rate in (5.0, 8.0, 10.0, 16.0, 24.0, 30000 / 1001, 60.0):
                with self.subTest(length=length, fps=rate):
                    grid = _time_grid(length, rate)
                    self.assertEqual(grid[0], 0.0)
                    self.assertEqual(grid[-1], (length - 1) / rate)
                    np.testing.assert_allclose(grid[:-1], np.arange(len(grid) - 1) / 10,
                                               rtol=0, atol=1e-12)
                    self.assertTrue(np.all(np.diff(grid) > 0))
                    self.assertTrue(np.all(np.diff(grid) <= 0.1 + 1e-12))

    def test_grid_has_no_duplicate_aligned_endpoint_and_keeps_short_video(self):
        np.testing.assert_array_equal(_time_grid(4, 30), [0, 0.1])
        np.testing.assert_array_equal(_time_grid(1, 24), [0])
        np.testing.assert_array_equal(_time_grid(2, 24), [0, 1 / 24])
        np.testing.assert_array_equal(_time_grid(11, 10), np.arange(11) / 10)

    def test_linear_feature_resampling_and_single_frame(self):
        for rate in (8, 24, 30000 / 1001):
            time = np.arange(43) / rate
            x = np.column_stack((2 * time + 1, -3 * time, np.ones(43))).astype(np.float32)
            sampled, grid = _resample_features(x, rate)
            expected = np.column_stack((2 * grid + 1, -3 * grid, np.ones(len(grid))))
            np.testing.assert_allclose(sampled, expected, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(sampled[[0, -1]], x[[0, -1]])
            self.assertEqual(sampled.dtype, np.float32)
        one = np.array([[7, -2]], np.float32)
        np.testing.assert_array_equal(_resample_features(one, 29.97)[0], one)

    def test_label_resampling_is_nearest_frame_half_up(self):
        labels = np.array([0, 1, 0, 1, 0, 1], np.float32)
        sampled = _resample_labels(labels, _time_grid(6, 5), 5)
        np.testing.assert_array_equal(sampled, labels[[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
        np.testing.assert_array_equal(_resample_labels(labels, _time_grid(6, 24), 24)[[0, -1]],
                                      labels[[0, -1]])

    def test_statistics_use_only_passed_training_frames_and_handle_no_variance(self):
        videos = [np.array([[0, 5], [2, 5]], np.float32),
                  np.array([[4, 5], [6, 5], [8, 5]], np.float32)]
        mean, std = _fit_normalization(videos)
        joined = np.concatenate(videos).astype(np.float64)
        np.testing.assert_array_equal(mean, joined.mean(axis=0).astype(np.float32))
        self.assertAlmostEqual(float(std[0]), float(joined[:, 0].std()), places=6)
        self.assertEqual(std[1], 1.0)
        normalized = _normalize(videos[0], mean, std)
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_array_equal(normalized[:, 1], 0)
        # Opposite-sign extreme finite float32s should not overflow subtraction.
        extreme = [np.array([[-3e38], [3e38]], np.float32)]
        emean, estd = _fit_normalization(extreme)
        self.assertTrue(np.isfinite(_normalize(extreme[0], emean, estd)).all())

    def test_invalid_timing_is_rejected_without_fps_guess(self):
        for rate in (0, -1, float("nan"), float("inf"), True, "24", None):
            with self.subTest(fps=rate), self.assertRaises(ValueError):
                _time_grid(5, rate)
        for length in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                _time_grid(length, 24)


class NumpyInferenceTests(unittest.TestCase):
    def test_explicit_zero_padding_dilation_and_cross_correlation(self):
        values = np.arange(1, 8, dtype=np.float32)[:, None]
        weight = np.array([[[1, 2, 3]]], np.float32)
        bias = np.array([0.5], np.float32)
        output = _conv1d_same(values, weight, bias, 2)[:, 0]
        expected = []
        for time in range(7):
            total = 0.5
            for tap, coefficient in enumerate((1, 2, 3)):
                index = time + (tap - 1) * 2
                if 0 <= index < 7:
                    total += coefficient * values[index, 0]
            expected.append(total)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(_conv1d_same(values[:1], weight, bias, 4), [[2.5]])

    def test_stable_sigmoid_has_no_overflow_and_float32_output(self):
        with np.errstate(over="raise", invalid="raise"):
            probabilities = _sigmoid(np.array([-1e30, -90, -5, 0, 5, 90, 1e30], np.float32))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(probabilities.dtype, np.float32)
        self.assertEqual(probabilities[3], 0.5)
        self.assertEqual(probabilities[0], 0)
        self.assertEqual(probabilities[-1], 1)
        self.assertTrue(np.all(np.diff(probabilities) >= 0))

    def test_output_matches_original_variable_length_and_fps(self):
        bundle = synthetic_bundle()
        rng = np.random.default_rng(14)
        for length in (1, 2, 13, 67):
            for rate in (5.0, 10.0, 24.0, 30000 / 1001):
                with self.subTest(length=length, fps=rate):
                    result = predict_temporal(bundle, rng.normal(size=(length, 7)), rate)
                    self.assertEqual(result.shape, (length,))
                    self.assertEqual(result.dtype, np.float32)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_no_evidence_zero_features_and_empty_inference(self):
        bundle = synthetic_bundle(zero=True)
        for length in (1, 2, 39):
            np.testing.assert_array_equal(predict_temporal(bundle, np.zeros((length, 7)), 24),
                                          np.full(length, 0.5, np.float32))
        result = predict_temporal(bundle, np.empty((0, 7), np.float32), 24)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.float32)

    def test_json_roundtrip_and_prediction_never_refits_or_mutates_bundle(self):
        bundle = synthetic_bundle()
        serialized = json.dumps(bundle, allow_nan=False, sort_keys=True)
        reloaded = json.loads(serialized)
        x = np.random.default_rng(1).normal(size=(51, 7)).astype(np.float32)
        np.testing.assert_array_equal(predict_temporal(bundle, x, 29.97),
                                      predict_temporal(reloaded, x, 29.97))
        predict_temporal(bundle, x + 100, 5)
        self.assertEqual(serialized, json.dumps(bundle, allow_nan=False, sort_keys=True))

    def test_import_and_inference_do_not_require_or_import_torch(self):
        source = r'''
import builtins, sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("unexpected torch dependency")
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
import numpy as np
from optimized_temporal_head import predict_temporal, SCHEMA, GRID_RULE
layers = []
inputs = 2
for i, dilation in enumerate((1, 2, 4, 1)):
    outputs, kernel = (32, 3) if i < 3 else (1, 1)
    layers.append({"weight": np.zeros((outputs, inputs, kernel)).tolist(),
                   "bias": np.zeros(outputs).tolist(), "dilation": dilation,
                   "padding": dilation if i < 3 else 0,
                   "activation": "relu" if i < 3 else "sigmoid"})
    inputs = outputs
bundle = {"schema": SCHEMA, "feature_dim": 2, "mean": [0, 0], "std": [1, 1],
          "convs": layers, "fps_profile": {"target_hz": 10., "grid": GRID_RULE,
                                           "features": "linear"}}
result = predict_temporal(bundle, np.zeros((7, 2)), 24)
assert result.dtype == np.float32 and result.shape == (7,)
assert "torch" not in sys.modules
print("NUMPY_ONLY_OK")
'''
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", source],
                                cwd=Path(__file__).resolve().parent, env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NUMPY_ONLY_OK", result.stdout)


class SchemaTests(unittest.TestCase):
    def test_public_signature_has_no_evaluation_or_identity_inputs(self):
        signature = inspect.signature(train_temporal)
        self.assertEqual(list(signature.parameters), ["values", "labels", "fps", "seed", "steps"])
        self.assertEqual(signature.parameters["steps"].default, 400)
        self.assertEqual(signature.parameters["steps"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_inference_rejects_bad_features_or_fps_even_if_empty(self):
        bundle = synthetic_bundle()
        invalid = [None, [], [1, 2], np.zeros((3, 8)), np.zeros((3, 7, 1)),
                   np.ones((3, 7), bool), np.full((3, 7), "1"),
                   np.full((3, 7), np.nan), np.full((3, 7), np.inf),
                   np.ones((3, 7), complex), {"video_name": "not_features"},
                   [[1], [2, 3]]]
        for values in invalid:
            with self.subTest(type=type(values).__name__), self.assertRaises(ValueError):
                predict_temporal(bundle, values, 24)
        for rate in (0, -1, np.nan, np.inf, True, "24", None):
            with self.assertRaises(ValueError):
                predict_temporal(bundle, np.empty((0, 7)), rate)

    def test_inference_rejects_incompatible_or_nonfinite_bundle(self):
        changes = [lambda b: b.update(schema="unknown"),
                   lambda b: b.update(feature_dim=True),
                   lambda b: b.update(mean=[0]),
                   lambda b: b["std"].__setitem__(0, 0),
                   lambda b: b["std"].__setitem__(0, float("nan")),
                   lambda b: b["fps_profile"].update(target_hz=12),
                   lambda b: b["fps_profile"].update(features="nearest"),
                   lambda b: b["fps_profile"].update(grid="stretch"),
                   lambda b: b["convs"].pop(),
                   lambda b: b["convs"][0].update(dilation=2),
                   lambda b: b["convs"][0].update(padding=0),
                   lambda b: b["convs"][0].update(activation="gelu"),
                   lambda b: b["convs"][0].update(weight=[[[0]]]),
                   lambda b: b["convs"][0]["bias"].__setitem__(0, float("inf"))]
        for index, change in enumerate(changes):
            bundle = synthetic_bundle()
            change(bundle)
            with self.subTest(change=index), self.assertRaises(ValueError):
                predict_temporal(bundle, np.zeros((2, 7)), 24)
        for bundle in (None, [], {}, {"schema": SCHEMA}):
            with self.assertRaises(ValueError):
                predict_temporal(bundle, np.zeros((2, 7)), 24)

    def test_training_rejects_invalid_schema_before_importing_torch(self):
        x, y, rates = [np.zeros((3, 2))], [np.array([0, 1, 0])], [24.0]
        cases = [([], [], []), (x, [], rates), (x, y, []), (x, y * 2, rates),
                 (np.zeros((1, 3, 2)), y, rates),
                 ([np.empty((0, 2))], [np.empty(0)], rates),
                 ([np.zeros((3, 0))], y, rates),
                 (x, [np.array([0, 1])], rates),
                 (x, [np.array([0, 2, 0])], rates),
                 (x, [np.array([0, np.nan, 0])], rates),
                 (x, [np.array([0, 1 + 1e-9, 0])], rates),
                 (x, [np.zeros((3, 1))], rates),
                 (x + [np.zeros((3, 4))], y * 2, rates * 2),
                 ([np.full((3, 2), np.inf)], y, rates),
                 ([{"id": 1, "features": x[0]}], y, rates), (x, y, [0])]
        for index, (values, labels, fps) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                train_temporal(values, labels, fps, seed=1, steps=1)
        for seed in (-1, 1.5, True, 2**63):
            with self.assertRaises(ValueError):
                train_temporal(x, y, rates, seed, steps=1)
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                train_temporal(x, y, rates, 1, steps=steps)


@unittest.skipIf(TORCH is None, TORCH_SKIP)
class TorchCPUTrainingAndParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = TORCH.get_num_threads()
        TORCH.set_num_threads(2)
        cls.videos, cls.labels, cls.rates = synthetic_training()
        cls.bundle = train_temporal(cls.videos, cls.labels, cls.rates, 17, steps=12)

    @classmethod
    def tearDownClass(cls):
        TORCH.set_num_threads(cls.old_threads)

    def torch_reference(self, bundle, values, fps):
        if not len(values):
            return np.empty(0, np.float32)
        sampled, grid = _resample_features(values.astype(np.float32), fps)
        normalized = _normalize(sampled, np.asarray(bundle["mean"], np.float32),
                                np.asarray(bundle["std"], np.float32))
        x = TORCH.from_numpy(normalized.T.copy())[None]
        with TORCH.no_grad():
            for index, layer in enumerate(bundle["convs"]):
                weight = TORCH.tensor(layer["weight"], dtype=TORCH.float32, device="cpu")
                bias = TORCH.tensor(layer["bias"], dtype=TORCH.float32, device="cpu")
                x = TORCH.nn.functional.conv1d(x, weight, bias, padding=layer["padding"],
                                               dilation=layer["dilation"])
                if index < 3:
                    x = TORCH.relu(x)
            probabilities = TORCH.sigmoid(x)[0, 0].numpy()
        return np.interp(np.arange(len(values)) / fps, grid, probabilities).astype(np.float32)

    def test_tiny_training_export_is_json_serializable_and_fixed_budget(self):
        bundle = json.loads(json.dumps(self.bundle, allow_nan=False))
        self.assertEqual(bundle["training"]["steps"], 12)
        self.assertEqual(bundle["training"]["seed"], 17)
        self.assertEqual(bundle["training"]["cpu_threads"], 2)
        self.assertEqual(bundle["training"]["device"], "cpu")
        self.assertEqual(bundle["training"]["selection"], "none_final_fixed_step")
        self.assertEqual(bundle["training"]["frame_positive_weight"], 1.25)
        self.assertEqual(bundle["architecture"]["channels"], 32)
        self.assertEqual(bundle["architecture"]["dropout"], list(DROPOUT))
        self.assertEqual(bundle["fps_profile"]["source_fps"], self.rates)
        self.assertEqual(bundle["fps_profile"]["source_frame_counts"], [1, 27, 53, 79])
        self.assertTrue(np.isfinite(bundle["training"]["final_loss"]))
        for video, rate in zip(self.videos, self.rates):
            self.assertEqual(predict_temporal(bundle, video, rate).shape, (len(video),))

    def test_numpy_vs_trained_torch_inference_error_below_2e_6(self):
        rng = np.random.default_rng(6)
        worst = 0.0
        for length in (0, 1, 2, 17, 63, 201):
            for fps in (7.5, 10.0, 24.0, 30000 / 1001, 60.0):
                x = rng.normal(size=(length, 7)).astype(np.float32)
                x[:, -1] = 3
                actual = predict_temporal(self.bundle, x, fps)
                reference = self.torch_reference(self.bundle, x, fps)
                error = float(np.max(np.abs(actual - reference))) if length else 0.0
                worst = max(worst, error)
                self.assertLess(error, 2e-6, (length, fps, error))
        print(f"\nTRAINED_NUMPY_TORCH_MAX_ABS_ERROR={worst:.9g}", flush=True)

    def test_numpy_vs_torch_wide_features_and_dilated_boundaries(self):
        rng = np.random.default_rng(42)
        worst = 0.0
        for dimension in (1, 7, 64, 512):
            bundle = synthetic_bundle(dimension)
            for length in (1, 2, 29, 137):
                x = rng.normal(size=(length, dimension)).astype(np.float32)
                actual = predict_temporal(bundle, x, 30000 / 1001)
                reference = self.torch_reference(bundle, x, 30000 / 1001)
                error = float(np.max(np.abs(actual - reference)))
                worst = max(worst, error)
                self.assertLess(error, 2e-6, (dimension, length, error))
        print(f"\nWIDE_NUMPY_TORCH_MAX_ABS_ERROR={worst:.9g}", flush=True)

    def test_preprocessing_matches_independent_torch_interpolation_and_normalization(self):
        for video, rate in zip(self.videos, self.rates):
            sampled, grid = _resample_features(video, rate)
            source = TORCH.from_numpy(video.astype(np.float64))
            positions = TORCH.from_numpy(grid.copy()) * rate
            positions = positions.clamp(0, len(video) - 1)
            positions[-1] = len(video) - 1
            left = TORCH.floor(positions).to(TORCH.int64)
            right = (left + 1).clamp_max(len(video) - 1)
            fraction = (positions - left)[:, None]
            reference = (source[left] * (1 - fraction) + source[right] * fraction).float()
            np.testing.assert_array_equal(sampled, reference.numpy())
            mean = np.asarray(self.bundle["mean"], np.float32)
            std = np.asarray(self.bundle["std"], np.float32)
            torch_normalized = ((reference.double() - TORCH.from_numpy(mean))
                                / TORCH.from_numpy(std)).float()
            np.testing.assert_array_equal(_normalize(sampled, mean, std), torch_normalized.numpy())

    def test_variable_length_hidden_mask_prevents_padding_bias_leakage(self):
        model = _build_torch_model(TORCH, 7)
        with TORCH.no_grad():
            for conv, layer in zip(model.convs, self.bundle["convs"]):
                conv.weight.copy_(TORCH.tensor(layer["weight"], device="cpu"))
                conv.bias.fill_(0.6)  # Deliberately expose hidden-layer padding bias.
        model.eval()
        rng = np.random.default_rng(20)
        videos = [rng.normal(size=(length, 7)).astype(np.float32) for length in (1, 3, 8, 17)]
        targets = [np.zeros(len(video), np.float32) for video in videos]
        x, _, mask = _pad_batch(videos, targets, np.arange(len(videos)))
        x.transpose(0, 2, 1)[~mask] = 100  # Padding contents must not be evidence.
        with TORCH.no_grad():
            batched = model(TORCH.from_numpy(x), TORCH.from_numpy(mask)).numpy()
            for index, video in enumerate(videos):
                single = model(TORCH.from_numpy(video.T.copy())[None]).numpy()[0]
                np.testing.assert_allclose(batched[index, :len(video)], single, rtol=0, atol=2e-6)
                np.testing.assert_array_equal(batched[index, len(video):], 0)

    def test_masked_loss_is_equal_video_not_equal_frame_and_padding_has_zero_gradient(self):
        logits = TORCH.tensor([[0.2, -0.3, 0.5, 1.0, -2.0],
                               [-0.7, 100, -100, 300, -300]], device="cpu", requires_grad=True)
        targets = TORCH.tensor([[0., 1., 1., 0., 0.], [1., 999., 999., 999., 999.]], device="cpu")
        mask = TORCH.tensor([[True] * 5, [True, False, False, False, False]], device="cpu")
        actual = _masked_video_loss(TORCH, logits, targets, mask)
        weight = TORCH.tensor(POSITIVE_WEIGHT, device="cpu")
        long_loss = TORCH.nn.functional.binary_cross_entropy_with_logits(
            logits[0], targets[0], pos_weight=weight)
        short_loss = TORCH.nn.functional.binary_cross_entropy_with_logits(
            logits[1, :1], targets[1, :1], pos_weight=weight)
        self.assertAlmostEqual(float(actual.detach()), float(((long_loss + short_loss) / 2).detach()),
                               places=7)
        actual.backward()
        np.testing.assert_array_equal(logits.grad.numpy()[~mask.numpy()], 0)

    def test_fixed_seed_repeatability_no_epoch_selection(self):
        repeated = train_temporal(self.videos, self.labels, self.rates, 17, steps=12)
        self.assertEqual(json.dumps(self.bundle, sort_keys=True), json.dumps(repeated, sort_keys=True))
        changed = train_temporal(self.videos, self.labels, self.rates, 18, steps=2)
        self.assertNotEqual(self.bundle["convs"][0]["weight"], changed["convs"][0]["weight"])

    def test_train_only_statistics_do_not_depend_on_labels_or_prediction_inputs(self):
        resampled = [_resample_features(x, rate)[0] for x, rate in zip(self.videos, self.rates)]
        mean, std = _fit_normalization(resampled)
        np.testing.assert_array_equal(self.bundle["mean"], mean)
        np.testing.assert_array_equal(self.bundle["std"], std)
        changed = train_temporal(self.videos, [1 - y for y in self.labels], self.rates, 17, steps=1)
        self.assertEqual(self.bundle["mean"], changed["mean"])
        self.assertEqual(self.bundle["std"], changed["std"])
        self.assertEqual(self.bundle["std"][-1], 1)

    def test_no_evidence_and_single_class_training_remain_finite(self):
        videos = [np.zeros((1, 3), np.float32), np.zeros((21, 3), np.float32)]
        for target in (0.0, 1.0):
            labels = [np.full(len(video), target, np.float32) for video in videos]
            bundle = train_temporal(videos, labels, [24.0, 8.0], seed=np.int64(4), steps=4)
            np.testing.assert_array_equal(bundle["mean"], [0, 0, 0])
            np.testing.assert_array_equal(bundle["std"], [1, 1, 1])
            self.assertTrue(np.isfinite(bundle["training"]["final_loss"]))
            result = predict_temporal(bundle, np.zeros((29, 3)), 29.97)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.all((result >= 0) & (result <= 1)))
            self.assertLess(float(np.max(np.abs(result - self.torch_reference(
                bundle, np.zeros((29, 3)), 29.97)))), 2e-6)

    def test_more_than_eight_videos_uses_uniform_video_balanced_minibatches(self):
        rng = np.random.default_rng(7)
        videos = [rng.normal(size=(i + 1, 3)).astype(np.float32) for i in range(10)]
        labels = [np.arange(len(video)) % 2 for video in videos]
        bundle = train_temporal(videos, labels, [24.] * 10, seed=8, steps=3)
        self.assertEqual(bundle["training"]["batch_size"], 8)
        self.assertEqual(bundle["training"]["video_count"], 10)
        self.assertEqual(bundle["training"]["sampling"], "uniform_videos_without_replacement_per_step")
        self.assertTrue(np.isfinite(bundle["training"]["final_loss"]))

    def test_training_restores_cpu_rng_threads_and_determinism_flags(self):
        old_threads = TORCH.get_num_threads()
        old_deterministic = TORCH.are_deterministic_algorithms_enabled()
        old_warn_only = TORCH.is_deterministic_algorithms_warn_only_enabled()
        torch_state = TORCH.get_rng_state().clone()
        numpy_state = np.random.get_state()
        try:
            TORCH.set_num_threads(4)
            TORCH.use_deterministic_algorithms(False, warn_only=True)
            train_temporal(self.videos, self.labels, self.rates, seed=9, steps=2)
            self.assertEqual(TORCH.get_num_threads(), 4)
            self.assertFalse(TORCH.are_deterministic_algorithms_enabled())
            self.assertTrue(TORCH.is_deterministic_algorithms_warn_only_enabled())
            self.assertTrue(TORCH.equal(torch_state, TORCH.get_rng_state()))
            current = np.random.get_state()
            self.assertEqual(numpy_state[0], current[0])
            np.testing.assert_array_equal(numpy_state[1], current[1])
            self.assertEqual(numpy_state[2:], current[2:])
        finally:
            TORCH.set_num_threads(old_threads)
            TORCH.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)


if __name__ == "__main__":
    unittest.main(verbosity=2)
