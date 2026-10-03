"""Focused CPU tests; no dataset, weights, GPU, or optional test dependencies."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from optimized_motion_features import FEATURE_NAMES, context_features, extract_video_features


class FakeCapture:
    def __init__(self, frames, fps=24.0, reported_count=None, read_error=False):
        self.frames = iter(frames)
        self.fps = fps
        self.reported_count = len(frames) if reported_count is None else reported_count
        self.read_error = read_error
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else self.reported_count

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            if self.read_error:
                raise cv2.error("synthetic decode failure")
            return False, None

    def release(self):
        self.released = True


def feature(result, name):
    return result["signals"][:, result["feature_names"].index(name)]


class VideoFeatureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="motion_features_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(42)
        texture = rng.integers(30, 185, size=(64, 96), dtype=np.uint8)
        self.base = cv2.GaussianBlur(texture, (3, 3), 0.7)
        self.base = cv2.cvtColor(self.base, cv2.COLOR_GRAY2BGR)

    def write_video(self, name, frames, fps=24.0):
        path = self.root / (name + ".avi")
        height, width = frames[0].shape[:2]
        writer = None
        for codec in ("MJPG", "FFV1"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if writer.isOpened():
                break
            writer.release()
        self.assertTrue(writer.isOpened(), "OpenCV needs a local MJPG or FFV1 encoder for integration tests")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        self.assertGreater(path.stat().st_size, 0)
        return path

    def translated_frames(self, step, count=15):
        return [cv2.warpAffine(self.base, np.float32([[1, 0, step * i], [0, 1, 0]]),
                               (96, 64), borderMode=cv2.BORDER_WRAP) for i in range(count)]

    def fake_result(self, frames, **kwargs):
        placeholder = self.root / "mock.avi"
        placeholder.write_bytes(b"mock input")
        cap = FakeCapture(frames, **kwargs)
        with patch("optimized_motion_features.cv2.VideoCapture", return_value=cap):
            result = extract_video_features(placeholder)
        self.assertTrue(cap.released)
        return result

    def test_static_schema_all_frames_fps_and_explicit_missing_motion(self):
        path = self.write_video("static", [self.base] * 12, fps=20.0)
        result = extract_video_features(path)
        self.assertEqual(result["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(result["signals"].shape, (12, 26))
        self.assertEqual(result["signals"].dtype, np.float32)
        self.assertTrue(np.isfinite(result["signals"]).all())
        self.assertAlmostEqual(result["fps"], 20.0, places=3)
        self.assertTrue(result["validity"]["valid"])
        self.assertTrue(result["validity"]["complete_decode"])
        self.assertTrue(result["validity"]["frame_count_matches"])
        self.assertEqual(result["metadata"]["decoded_frame_count"], 12)
        self.assertEqual(result["metadata"]["reported_frame_count"], 12)
        np.testing.assert_allclose(result["frame_times_seconds"], np.arange(12) / 20.0)
        np.testing.assert_array_equal(feature(result, "pair_valid"), [0] + [1] * 11)
        np.testing.assert_array_equal(feature(result, "second_order_valid"), [0, 0] + [1] * 10)
        np.testing.assert_array_equal(feature(result, "motion_valid"), [0] + [1] * 11)
        np.testing.assert_array_equal(feature(result, "acceleration_valid"), [0, 0] + [1] * 10)
        self.assertFalse(result["validity"]["feature_valid"][0, FEATURE_NAMES.index("flow_median")])
        self.assertTrue(result["validity"]["feature_valid"][0, FEATURE_NAMES.index("luma_std")])
        self.assertLess(float(feature(result, "flow_median")[1:].max()), 0.005)
        self.assertLess(float(feature(result, "frame_diff_p90")[1:].max()), 0.01)

    def test_camera_pan_has_absolute_motion_and_low_residual_not_an_error_score(self):
        static = extract_video_features(self.write_video("static", [self.base] * 15))
        pan = extract_video_features(self.write_video("pan", self.translated_frames(1)))
        fast = extract_video_features(self.write_video("fast", self.translated_frames(3)))
        self.assertEqual(static["feature_names"], pan["feature_names"])
        camera = float(np.median(feature(pan, "camera_magnitude")[2:]))
        residual = float(np.median(feature(pan, "residual_median")[2:]))
        self.assertGreater(camera, 0.12)
        self.assertLess(residual, camera * 0.25)
        self.assertGreater(float(np.median(feature(pan, "camera_dx")[2:])), 0.12)
        self.assertLess(abs(float(np.median(feature(pan, "camera_dy")[2:]))), 0.02)
        self.assertGreater(float(np.median(feature(fast, "flow_median")[2:])),
                           float(np.median(feature(pan, "flow_median")[2:])) * 2.0)
        self.assertGreater(float(np.median(feature(pan, "flow_median")[2:])),
                           float(np.median(feature(static, "flow_median")[2:])) + 0.12)
        self.assertNotIn("error_score", pan)
        expanded = context_features(pan["signals"], pan["feature_names"], fps=pan["fps"])
        np.testing.assert_array_equal(expanded["signals"][:, :26], pan["signals"])

    def test_local_motion_survives_global_camera_subtraction(self):
        frames = []
        patch_texture = self.base[16:40, 12:36].copy()
        for i in range(10):
            frame = self.base.copy()
            x = 12 + i * 2
            frame[16:40, x:x + 24] = patch_texture
            frames.append(frame)
        result = extract_video_features(self.write_video("local", frames))
        self.assertGreater(float(np.median(feature(result, "residual_p99")[2:])), 0.1)
        self.assertGreater(float(np.max(feature(result, "residual_active_fraction")[2:])), 0.02)
        self.assertGreater(float(np.max(feature(result, "residual_outlier_fraction")[2:])), 0.01)
        self.assertLess(float(np.median(feature(result, "camera_magnitude")[2:])), 0.05)
        self.assertTrue(np.all((feature(result, "residual_active_fraction") >= 0)
                               & (feature(result, "residual_active_fraction") <= 1)))

    def test_single_frame_flash_is_visible_in_diff_brightness_and_second_order(self):
        frames = [self.base.copy() for _ in range(17)]
        frames[8] = np.clip(self.base.astype(np.int16) + 70, 0, 255).astype(np.uint8)
        result = extract_video_features(self.write_video("flash", frames))
        difference = feature(result, "frame_diff_median")
        self.assertGreater(float(difference[8]), 0.20)
        self.assertGreater(float(difference[9]), 0.20)
        self.assertLess(float(difference[[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]].max()), 0.03)
        self.assertGreater(float(feature(result, "luma_change")[8]), 0.20)
        self.assertLess(float(feature(result, "luma_change")[9]), -0.20)
        self.assertGreater(float(feature(result, "frame_second_diff_p90")[9]), 0.40)
        self.assertLess(float(feature(result, "luma_second_change")[9]), -0.40)

    def test_sharpness_and_blur_use_fixed_scale(self):
        sharp = self.base
        blurred = cv2.GaussianBlur(sharp, (11, 11), 2.5)
        result = extract_video_features(self.write_video("blur", [sharp] * 3 + [blurred] * 3))
        laplacian = feature(result, "laplacian_variance")
        blur = feature(result, "blur_score")
        self.assertLess(float(laplacian[4]), float(laplacian[1]) * 0.15)
        self.assertGreater(float(blur[4]), float(blur[1]))
        self.assertTrue(np.all((blur >= 0) & (blur <= 1)))

    def test_fps_changes_speed_and_acceleration_units_not_rank_scaling(self):
        frames = self.translated_frames(1, count=8)
        slow = extract_video_features(self.write_video("slow_fps", frames, fps=12.0))
        fast = extract_video_features(self.write_video("fast_fps", frames, fps=24.0))
        np.testing.assert_allclose(feature(fast, "camera_magnitude"),
                                   feature(slow, "camera_magnitude") * 2, atol=1e-6)
        np.testing.assert_allclose(feature(fast, "camera_acceleration"),
                                   feature(slow, "camera_acceleration") * 4, atol=1e-6)

    def test_resize_preserves_aspect_ratio_without_upscaling(self):
        large = cv2.resize(self.base, (320, 180))
        result = extract_video_features(self.write_video("large", [large] * 3))
        self.assertEqual(result["metadata"]["source_size"], (320, 180))
        self.assertEqual(result["metadata"]["analysis_size"], (96, 54))
        small = self.base[:32, :48]
        result = extract_video_features(self.write_video("small", [small] * 3))
        self.assertEqual(result["metadata"]["analysis_size"], (48, 32))

    def test_single_frame_and_empty_corrupt_missing_inputs_keep_schema(self):
        single = extract_video_features(self.write_video("one", [self.base]))
        self.assertEqual(single["signals"].shape, (1, 26))
        self.assertTrue(single["validity"]["valid"])
        self.assertFalse(single["validity"]["motion_valid"].any())
        self.assertFalse(single["validity"]["acceleration_valid"].any())
        empty = self.root / "empty.avi"
        empty.touch()
        corrupt = self.root / "corrupt.avi"
        corrupt.write_bytes(b"this is not a video container")
        for path, status in ((empty, "empty_file"), (corrupt, "open_failed"),
                             (self.root / "missing.avi", "not_a_file")):
            with self.subTest(status=status):
                result = extract_video_features(path)
                self.assertEqual(result["signals"].shape, (0, 26))
                self.assertEqual(result["signals"].dtype, np.float32)
                self.assertEqual(result["feature_names"], single["feature_names"])
                self.assertFalse(result["validity"]["valid"])
                self.assertEqual(result["validity"]["status"], status)
                context = context_features(result["signals"], result["feature_names"], fps=result["fps"])
                self.assertEqual(context["signals"].shape, (0, 158))

    def test_filenames_do_not_affect_values(self):
        first = self.write_video("normal", self.translated_frames(1, count=4))
        second = self.root / "label_anomalous_999.avi"
        shutil.copyfile(first, second)
        np.testing.assert_array_equal(extract_video_features(first)["signals"],
                                      extract_video_features(second)["signals"])

    def test_reported_count_is_checked_but_never_limits_reads(self):
        for reported in (1, 8):
            with self.subTest(reported=reported):
                result = self.fake_result([self.base] * 3, reported_count=reported)
                self.assertEqual(result["signals"].shape[0], 3)
                self.assertFalse(result["validity"]["frame_count_matches"])
                self.assertFalse(result["validity"]["complete_decode"])
                self.assertFalse(result["validity"]["valid"])
                self.assertIn("frame_count_mismatch", result["metadata"]["issues"])

    def test_unknown_count_does_not_claim_complete_decode(self):
        result = self.fake_result([self.base] * 3, reported_count=0)
        self.assertIsNone(result["validity"]["frame_count_matches"])
        self.assertIsNone(result["validity"]["complete_decode"])
        self.assertTrue(result["validity"]["valid"])

    def test_invalid_fps_is_explicit_and_features_remain_finite(self):
        for fps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                result = self.fake_result([self.base] * 3, fps=fps)
                self.assertEqual(result["fps"], 24.0)
                self.assertEqual(result["metadata"]["fps_source"], "fallback")
                self.assertFalse(result["validity"]["fps_valid"])
                self.assertFalse(result["validity"]["valid"])
                self.assertFalse(result["validity"]["motion_valid"].any())
                self.assertFalse(result["validity"]["acceleration_valid"].any())
                motion_index = FEATURE_NAMES.index("flow_median")
                self.assertFalse(result["validity"]["feature_valid"][:, motion_index].any())
                context = context_features(result["signals"], result["feature_names"])
                rank_index = context["feature_names"].index("flow_median__midrank")
                self.assertFalse(context["validity"]["feature_valid"][:, rank_index].any())
                self.assertTrue(np.isfinite(result["signals"]).all())

    def test_decode_exception_retains_partial_frames_and_releases_capture(self):
        result = self.fake_result([self.base] * 2, reported_count=5, read_error=True)
        self.assertEqual(len(result["signals"]), 2)
        self.assertFalse(result["validity"]["valid"])
        self.assertEqual(result["metadata"]["read_status"], "read_error")
        self.assertIn("read_error", result["metadata"]["issues"])

    def test_no_decodable_or_malformed_frame_is_invalid(self):
        for frames in ([], [None], [self.base.astype(np.float32)]):
            with self.subTest(count=len(frames)):
                result = self.fake_result(frames)
                self.assertEqual(result["signals"].shape, (0, 26))
                self.assertFalse(result["validity"]["valid"])

    def test_flow_failure_preserves_quality_and_masks_temporal_flow(self):
        invalid_flow = np.full(self.base.shape[:2] + (2,), np.nan, dtype=np.float32)
        with patch("optimized_motion_features.cv2.calcOpticalFlowFarneback", return_value=invalid_flow):
            result = self.fake_result([self.base] * 4)
        self.assertEqual(result["metadata"]["flow_failure_count"], 3)
        self.assertFalse(result["validity"]["valid"])
        self.assertFalse(result["validity"]["motion_valid"].any())
        self.assertTrue(result["validity"]["frame_valid"].all())
        self.assertTrue(result["validity"]["pair_valid"][1:].all())
        self.assertTrue(np.isfinite(result["signals"]).all())

    def test_camera_acceleration_is_vector_change_in_seconds_units(self):
        flows = []
        for dx in (1.0, 3.0, 3.0):
            flow = np.zeros(self.base.shape[:2] + (2,), dtype=np.float32)
            flow[:, :, 0] = dx
            flows.append(flow)
        with patch("optimized_motion_features.cv2.calcOpticalFlowFarneback", side_effect=flows):
            result = self.fake_result([self.base] * 4)
        expected = 2.0 * 24.0 ** 2 / np.hypot(96, 64)
        np.testing.assert_allclose(feature(result, "camera_acceleration"), [0, 0, expected, 0], rtol=1e-6)
        np.testing.assert_array_equal(feature(result, "residual_median"), np.zeros(4))
        np.testing.assert_array_equal(feature(result, "residual_acceleration"), np.zeros(4))

    def test_flow_failure_recovery_requires_two_new_consecutive_estimates(self):
        flow = np.zeros(self.base.shape[:2] + (2,), dtype=np.float32)
        invalid = np.full_like(flow, np.nan)
        with patch("optimized_motion_features.cv2.calcOpticalFlowFarneback", side_effect=[flow, invalid, flow, flow]):
            result = self.fake_result([self.base] * 5)
        np.testing.assert_array_equal(feature(result, "motion_valid"), [0, 1, 0, 1, 1])
        np.testing.assert_array_equal(feature(result, "acceleration_valid"), [0, 0, 0, 0, 1])
        context = context_features(result["signals"], result["feature_names"], fps=result["fps"],
                                   validity=result["validity"]["feature_valid"])
        index = context["feature_names"].index("flow_median__midrank")
        np.testing.assert_array_equal(context["validity"]["feature_valid"][:, index], [False, True, False, True, True])

    def test_dimension_change_resets_temporal_comparison(self):
        resized = cv2.resize(self.base, (192, 128))
        result = self.fake_result([self.base, self.base, resized, resized])
        np.testing.assert_array_equal(feature(result, "pair_valid"), [0, 1, 0, 1])
        np.testing.assert_array_equal(feature(result, "second_order_valid"), [0, 0, 0, 0])
        self.assertEqual(result["metadata"]["frame_size_change_count"], 1)
        self.assertFalse(result["validity"]["valid"])


class ContextFeatureTests(unittest.TestCase):
    def test_absolute_columns_unchanged_and_average_tie_ranks(self):
        signals = np.asarray([[0, 7], [0, 7], [2, 7], [2, 7]], dtype=np.float32)
        original = signals.copy()
        result = context_features(signals, ["value", "constant"], fps=24)
        np.testing.assert_array_equal(result["signals"][:, :2], original)
        np.testing.assert_array_equal(signals, original)
        np.testing.assert_allclose(feature(result, "value__midrank"), [1 / 6, 1 / 6, 5 / 6, 5 / 6])
        np.testing.assert_array_equal(feature(result, "constant__midrank"), [0.5] * 4)
        np.testing.assert_array_equal(feature(result, "constant__robust_z"), [0] * 4)
        self.assertEqual(result["signals"].dtype, np.float32)
        self.assertTrue(np.isfinite(result["signals"]).all())

    def test_rolling_seconds_fps_and_robust_spike(self):
        values = np.full((41, 1), 2.0, dtype=np.float32)
        values[20, 0] = 12
        low = context_features(values, ["sample"], fps=5, window_seconds=(1.0,))
        high = context_features(values, ["sample"], fps=15, window_seconds=(1.0,))
        self.assertEqual(low["metadata"]["window_frames"], [5])
        self.assertEqual(high["metadata"]["window_frames"], [15])
        self.assertEqual(low["feature_names"], high["feature_names"])
        self.assertEqual(float(feature(low, "sample__rolling_median_1s")[20]), 2.0)
        self.assertGreater(float(feature(low, "sample__rolling_z_1s")[20]), 5.0)
        self.assertEqual(float(feature(low, "sample__rolling_z_1s")[0]), 0.0)
        self.assertFalse(low["metadata"]["fps_assumed"])
        self.assertTrue(low["validity"]["timing_valid"])
        assumed = context_features(values, ["sample"])
        self.assertTrue(assumed["metadata"]["fps_assumed"])
        self.assertFalse(assumed["validity"]["timing_valid"])

    def test_missing_first_motion_is_excluded_not_ranked_as_low_motion(self):
        signals = np.asarray([[0, 0], [5, 1], [5, 1]], dtype=np.float32)
        result = context_features(signals, ["flow_median", "motion_valid"], fps=10)
        np.testing.assert_array_equal(feature(result, "flow_median__midrank"), [0, 0.5, 0.5])
        np.testing.assert_array_equal(feature(result, "flow_median__robust_z"), [0, 0, 0])
        self.assertNotIn("motion_valid__midrank", result["feature_names"])
        index = result["feature_names"].index("flow_median__midrank")
        np.testing.assert_array_equal(result["validity"]["feature_valid"][:, index], [False, True, True])

    def test_nan_inf_and_explicit_validity_are_masked_without_mutation(self):
        signals = np.asarray([[1], [np.nan], [np.inf], [3]], dtype=np.float32)
        result = context_features(signals, ["x"], fps=20, validity=[[False], [True], [True], [True]])
        self.assertTrue(np.isfinite(result["signals"]).all())
        self.assertTrue(np.isnan(signals[1, 0]))
        np.testing.assert_array_equal(result["validity"]["feature_valid"][:, 0], [False, False, False, True])
        np.testing.assert_array_equal(feature(result, "x__midrank"), [0, 0, 0, 0.5])

    def test_all_invalid_and_singleton_context_are_finite(self):
        for signals in (np.asarray([[np.nan]], dtype=np.float32), np.asarray([[3]], dtype=np.float32)):
            with self.subTest(value=signals[0, 0]):
                result = context_features(signals, ["x"], fps=2000, window_seconds=(100,))
                self.assertTrue(np.isfinite(result["signals"]).all())
                self.assertEqual(result["signals"].shape, (1, 5))

    def test_finite_float32_extremes_do_not_overflow_rolling_statistics(self):
        limit = np.finfo(np.float32).max
        signals = np.asarray([[limit], [limit], [-limit], [-limit]], dtype=np.float32)
        with np.errstate(all="raise"):
            result = context_features(signals, ["extreme"], fps=24)
        self.assertTrue(np.isfinite(result["signals"]).all())
        np.testing.assert_array_equal(result["signals"][:, :1], signals)

    def test_input_validation(self):
        cases = [
            lambda: context_features(np.zeros(3), ["x"]),
            lambda: context_features(np.zeros((2, 2)), ["x", "x"]),
            lambda: context_features(np.zeros((2, 2)), ["x"]),
            lambda: context_features(np.zeros((2, 1)), ["x"], fps=0),
            lambda: context_features(np.zeros((2, 1)), ["x"], window_seconds=(0,)),
            lambda: context_features(np.zeros((2, 1)), ["x"], window_seconds=(1, 1)),
            lambda: context_features(np.zeros((2, 1)), ["x"], validity=np.zeros((3, 1))),
            lambda: extract_video_features("unused", long_edge=4),
            lambda: extract_video_features("unused", fallback_fps=float("nan")),
        ]
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
