"""CPU reliability and fixed-unit tests; all temporary writes stay isolated."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import optimized_local_motion as motion
import build_optimization_local_motion as builder


class Capture:
    def __init__(self, frames, fps=24.0, count=None, fail_at=None):
        self.frames, self.fps = list(frames), fps
        self.count = len(frames) if count is None else count
        self.index, self.fail_at, self.released = 0, fail_at, False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else self.count if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    def read(self):
        if self.index == self.fail_at:
            raise cv2.error("injected_read_failure")
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def column(result, name):
    return result["signals"][:, motion.FEATURE_NAMES.index(name)]


def valid(result, name):
    return result["validity"]["feature_valid"][:, motion.FEATURE_NAMES.index(name)]


class LocalMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Parent's existence does not interfere with builder's production output.
        builder.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="tests-", dir=builder.OUTPUT_ROOT)
        cls.root = Path(cls.temp.name)
        cls.placeholder = cls.root / "input.bin"
        cls.placeholder.write_bytes(b"test")
        rng = np.random.default_rng(20261002)
        cls.gray = rng.integers(0, 256, (72, 96), dtype=np.uint8)
        cls.frames = [cv2.cvtColor(cls.gray, cv2.COLOR_GRAY2BGR)] * 5

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def extract(self, frames=None, *, flow=None, **capture_args):
        cap = Capture(self.frames if frames is None else frames, **capture_args)
        with patch.object(motion.cv2, "VideoCapture", return_value=cap):
            if flow is None:
                result = motion.extract_video_features(self.placeholder)
            else:
                with patch.object(motion.cv2, "calcOpticalFlowFarneback", side_effect=flow):
                    result = motion.extract_video_features(self.placeholder)
        self.assertTrue(cap.released)
        self.assertEqual(result["signals"].dtype, np.float32)
        self.assertTrue(np.isfinite(result["signals"]).all())
        self.assertTrue(np.all(result["signals"][~result["validity"]["feature_valid"]] == 0))
        return result

    @staticmethod
    def object_flow(step=1.0):
        flow = np.zeros((72, 96, 2), np.float32)
        flow[24:48, 8:28, 0] = step
        return flow

    def test_schema_static_first_two_and_no_active_direction(self):
        result = self.extract()
        self.assertEqual(result["signals"].shape, (5, len(motion.FEATURE_NAMES)))
        self.assertEqual(set(motion.FEATURE_NAMES), set(motion.FEATURE_UNITS))
        self.assertEqual(len(set(motion.FEATURE_NAMES)), len(motion.FEATURE_NAMES))
        self.assertFalse(valid(result, "residual_p90")[0])
        self.assertFalse(valid(result, "acceleration_p90")[:2].any())
        self.assertTrue(valid(result, "residual_p90")[1:].all())
        self.assertTrue(valid(result, "acceleration_p90")[2:].all())
        self.assertFalse(valid(result, "active_centroid_x").any())
        self.assertFalse(valid(result, "direction_x").any())
        self.assertTrue(result["validity"]["complete_decode"])
        self.assertEqual(result["metadata"]["decoded_frame_count"], 5)

    def test_low_texture_is_unobservable_not_zero_normal(self):
        result = self.extract([np.zeros_like(self.frames[0])] * 4)
        self.assertFalse(valid(result, "residual_p90").any())
        self.assertFalse(column(result, "motion_valid").any())
        self.assertEqual(result["metadata"]["texture_unobservable_count"], 3)
        self.assertTrue(valid(result, "observed_fraction")[1:].all())

    def test_object_localization_direction_and_fixed_fps_scaling(self):
        low = self.extract(flow=lambda *a, **k: self.object_flow(), fps=12)
        high = self.extract(flow=lambda *a, **k: self.object_flow(), fps=24)
        self.assertGreater(column(low, "tile_10_active_fraction")[1], 0.2)
        self.assertLess(column(low, "active_centroid_x")[1], 0.3)
        self.assertGreater(column(low, "direction_x")[1], 0.95)
        np.testing.assert_allclose(column(high, "residual_p99")[1:], column(low, "residual_p99")[1:] * 2, rtol=1e-5)
        negative = self.extract(flow=lambda *a, **k: self.object_flow(-1))
        self.assertLess(column(negative, "direction_x")[1], -0.95)
        self.assertNotIn("anomaly", " ".join(negative["feature_names"]))
        self.assertTrue(column(negative, "motion_valid")[1:].all())

    def test_vector_acceleration_scales_with_fps_squared(self):
        def run(fps):
            fields = iter([self.object_flow(0.5), self.object_flow(1.0), self.object_flow(1.5)])
            return self.extract(self.frames[:4], flow=lambda *a, **k: next(fields), fps=fps)
        low, high = run(12), run(24)
        self.assertGreater(column(low, "tile_10_acceleration_p90")[2], 0.1)
        np.testing.assert_allclose(column(high, "tile_10_acceleration_p90")[2:], column(low, "tile_10_acceleration_p90")[2:] * 4, rtol=1e-5)

    def test_robust_affine_removes_camera_and_preserves_object(self):
        geometry = motion._geometry((72, 96))
        coeff = np.array([[0.4, -0.2], [0.3, 0.2], [1.0, 0.5]])
        flow = (geometry[0] @ coeff).astype(np.float32)
        flow[24:48, 8:28, 0] += 2
        support = geometry[1]
        fitted, model, reliable, fraction, _ = motion._camera(flow, support, geometry)
        self.assertEqual(model, "affine")
        self.assertTrue(reliable)
        self.assertGreater(fraction, 0.8)
        np.testing.assert_allclose(fitted, coeff, atol=0.04)
        result = self.extract(flow=lambda *a, **k: flow.copy())
        self.assertGreater(column(result, "tile_10_residual_p90")[1], 0.2)
        self.assertLess(column(result, "tile_12_residual_p90")[1], 0.03)

    def test_translation_fallback_is_flagged_and_model_switch_masks_accel(self):
        def forced(flow, support, geometry):
            coeff = np.zeros((3, 2)); coeff[2, 0] = 1
            return coeff, "median_translation", True, 1.0, 0.0
        with patch.object(motion, "_camera", side_effect=forced):
            result = self.extract(flow=lambda *a, **k: np.ones((72, 96, 2), np.float32) * [1, 0])
        self.assertTrue(column(result, "camera_fallback_used")[1:].all())
        self.assertFalse(valid(result, "camera_strain_xx").any())
        geometry = motion._geometry((72, 96))
        with patch.object(motion.np.linalg, "lstsq", side_effect=np.linalg.LinAlgError("injected")):
            _, model, reliable, _, _ = motion._camera(np.ones((72, 96, 2), np.float32), geometry[1], geometry)
        self.assertEqual(model, "median_translation")
        self.assertTrue(reliable)
        results = iter([(np.zeros((3, 2)), "affine", True, 1., 0.),
                        (np.zeros((3, 2)), "median_translation", True, 1., 0.),
                        (np.zeros((3, 2)), "median_translation", True, 1., 0.)])
        with patch.object(motion, "_camera", side_effect=lambda *a: next(results)):
            result = self.extract(self.frames[:4], flow=lambda *a, **k: self.object_flow())
        self.assertEqual(column(result, "camera_model_changed")[2], 1)
        self.assertFalse(valid(result, "acceleration_p90")[2])
        self.assertTrue(valid(result, "acceleration_p90")[3])

    def test_bad_fps_short_empty_and_missing(self):
        for fps in (0, -1, float("nan"), float("inf")):
            result = self.extract(fps=fps)
            self.assertGreater(result["fps"], 0)
            self.assertFalse(result["validity"]["fps_valid"])
            self.assertFalse(valid(result, "residual_p90").any())
        one = self.extract(self.frames[:1])
        self.assertFalse(valid(one, "residual_p90").any())
        empty = self.extract([])
        self.assertEqual(empty["signals"].shape, (0, len(motion.FEATURE_NAMES)))
        self.assertFalse(empty["validity"]["valid"])
        missing = motion.extract_video_features(self.root / "absent.mp4")
        self.assertEqual(missing["validity"]["status"], "not_a_file")
        for kwargs in ({"long_edge": 8}, {"long_edge": True}, {"fallback_fps": 0}, {"residual_active_threshold": float("nan")}):
            with self.assertRaises(ValueError):
                motion.extract_video_features(self.placeholder, **kwargs)

    def test_bad_frame_size_change_and_read_failure_reset_history(self):
        result = self.extract([self.frames[0], None, self.frames[0], self.frames[0], self.frames[0]])
        self.assertEqual(result["metadata"]["bad_frame_count"], 1)
        self.assertFalse(valid(result, "residual_p90")[2])
        self.assertFalse(valid(result, "acceleration_p90")[3])
        larger = cv2.resize(self.frames[0], (192, 144))
        result = self.extract([self.frames[0], self.frames[0], larger, larger])
        self.assertEqual(column(result, "source_size_changed")[2], 1)
        self.assertFalse(valid(result, "residual_p90")[2])
        self.assertFalse(valid(result, "acceleration_p90")[3])
        partial = self.extract(fail_at=2)
        self.assertEqual(len(partial["signals"]), 2)
        self.assertFalse(partial["validity"]["complete_decode"])
        self.assertEqual(partial["metadata"]["read_status"], "read_error")
        mismatch = self.extract(count=99)
        self.assertFalse(mismatch["validity"]["frame_count_matches"])
        unknown = self.extract(count=float("nan"))
        self.assertIsNone(unknown["validity"]["complete_decode"])

    def test_invalid_flow_masks_and_next_acceleration(self):
        calls = iter([self.object_flow(), np.full((72, 96, 2), np.nan, np.float32), self.object_flow()])
        result = self.extract(self.frames[:4], flow=lambda *a, **k: next(calls))
        self.assertEqual(column(result, "flow_failed")[2], 1)
        self.assertFalse(valid(result, "residual_p90")[2])
        self.assertFalse(valid(result, "acceleration_p90")[3])

    def test_incoherent_camera_is_unreliable_and_residuals_are_masked(self):
        rng = np.random.default_rng(99)
        field = rng.normal(0, 5, (72, 96, 2)).astype(np.float32)
        geometry = motion._geometry((72, 96))
        _, model, reliable, _, _ = motion._camera(field, geometry[1], geometry)
        self.assertEqual(model, "median_translation")
        self.assertFalse(reliable)
        result = self.extract(flow=lambda *a, **k: field)
        self.assertFalse(column(result, "camera_compensation_valid").any())
        self.assertFalse(valid(result, "residual_p90").any())
        self.assertFalse(valid(result, "acceleration_p90").any())

    def test_real_flow_local_object_has_location_and_signed_direction(self):
        frames = []
        texture = self.frames[0][24:44, 8:28].copy()
        for i in range(8):
            frame = self.frames[0].copy()
            frame[24:44, 8 + 2*i:28 + 2*i] = texture
            frames.append(frame)
        result = self.extract(frames)
        observable = valid(result, "direction_x")
        self.assertTrue(observable[2:].any())
        self.assertGreater(float(np.median(column(result, "direction_x")[observable])), 0.4)
        self.assertLess(float(np.median(column(result, "active_centroid_x")[observable])), 0.5)
        self.assertGreater(float(column(result, "tile_10_residual_p90")[2:].max()), 0.1)

    def test_same_source_determinism_prefix_and_backend_restore(self):
        threads, opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
        first, second = self.extract(), self.extract()
        np.testing.assert_array_equal(first["signals"], second["signals"])
        np.testing.assert_array_equal(first["validity"]["feature_valid"], second["validity"]["feature_valid"])
        prefix = self.extract(self.frames[:3])
        np.testing.assert_array_equal(prefix["signals"], first["signals"][:3])
        self.assertEqual(cv2.getNumThreads(), threads)
        self.assertEqual(cv2.ocl.useOpenCL(), opencl)

    def test_real_codec_and_builder_fresh_resume_corruption_refusal(self):
        video = self.root / "real.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 12., (96, 72))
        self.assertTrue(writer.isOpened(), "existing OpenCV MJPG encoder required")
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        first = motion.extract_video_features(video)
        second = motion.extract_video_features(video)
        np.testing.assert_array_equal(first["signals"], second["signals"])
        self.assertTrue(first["validity"]["valid"])
        source = self.root / "source.json"
        source.write_text(json.dumps({"rows": [{"input_path": str(video), "fps": 12.,
            "frames": 5, "sha256": builder.digest(video), "labels": {"NEVER_USE": 999}}]}), encoding="utf-8")
        out = self.root / "cache"
        result = builder.build(source, out, expected_rows=1)
        self.assertEqual(result["status"], "complete")
        resumed = builder.build(source, out, expected_rows=1, resume=True)
        self.assertEqual(resumed["videos"][0]["cache_status"], "verified_resume")
        self.assertNotIn("labels", json.dumps(resumed))
        with self.assertRaises(FileExistsError):
            builder.build(source, out, expected_rows=1)
        with self.assertRaises(ValueError):
            builder.build(source, out, expected_rows=1, resume=True, long_edge=160)
        artifact = out / "v000.npz"
        payload = artifact.read_bytes()
        artifact.write_bytes(payload + b"corrupt")
        refused = builder.build(source, out, expected_rows=1, resume=True)
        self.assertEqual(refused["status"], "incomplete")
        self.assertEqual(artifact.read_bytes(), payload + b"corrupt")
        artifact.write_bytes(payload)
        video.write_bytes(video.read_bytes() + b"source_changed")
        refused = builder.build(source, out, expected_rows=1, resume=True)
        self.assertEqual(refused["status"], "incomplete")
        self.assertEqual(refused["failures"][0]["error_code"], "source_sha_mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
