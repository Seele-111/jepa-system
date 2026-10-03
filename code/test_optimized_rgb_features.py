"""Synthetic time/index/alignment tests; no real videos, downloads or GPU work.

WSL: python -B -m unittest discover -s /mnt/e/jepa-system/code
     -p test_optimized_rgb_features.py -v
Temporary test artifacts stay inside the designated rgb_features output tree.
"""
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from optimized_rgb_features import (
    Budget, BudgetExceeded, DEFAULT_OUTPUT, FEATURE_DIM, VideoSpec,
    _warnings, anchor_frame_indices, extract_anchor_features,
    interpolate_anchor_features, load_video_specs, temporal_window_indices,
    transform_rgb_frames,
)


class TemporalIndexTests(unittest.TestCase):
    def test_quarter_second_centers_and_last_real_frame(self):
        centers = anchor_frame_indices(125, 24, 0.25)
        np.testing.assert_array_equal(centers, np.r_[np.arange(0, 121, 6), 124])

    def test_point_four_second_centers_use_time_not_fixed_stride(self):
        for fps in (8, 10, 16, 24, 30):
            centers = anchor_frame_indices(125, fps, 0.4)
            self.assertEqual(centers[0], 0)
            self.assertEqual(centers[-1], 124)
            np.testing.assert_allclose(centers[:-1] / fps, np.arange(len(centers) - 1) * 0.4, atol=0.5 / fps)
            self.assertTrue(np.all(np.diff(centers) > 0))

    def test_fractional_fps_has_no_accumulating_rounding_drift(self):
        fps = 30000 / 1001
        centers = anchor_frame_indices(600, fps, 0.25)
        np.testing.assert_allclose(centers[:-1] / fps, np.arange(len(centers) - 1) * 0.25, atol=0.5 / fps)

    def test_anchor_deduplication_and_one_frame_video(self):
        np.testing.assert_array_equal(anchor_frame_indices(1, 10), [0])
        np.testing.assert_array_equal(anchor_frame_indices(4, 1, 0.25), np.arange(4))

    def test_one_second_window_scales_with_source_fps(self):
        for fps in (8, 10, 16, 24, 30):
            x = temporal_window_indices(100, fps, np.array([50]))
            self.assertEqual(x["sample_frame_indices"].shape, (1, 16))
            self.assertEqual(x["sample_frame_indices"][0, -1] - x["sample_frame_indices"][0, 0], fps)
            self.assertFalse(x["padding_mask"].any())
            np.testing.assert_allclose(x["sample_offsets_sec"][[0, -1]], [-0.5, 0.5])

    def test_endpoint_padding_is_copied_and_recorded(self):
        x = temporal_window_indices(61, 30, np.array([0, 30, 60]))
        raw = x["sample_frame_indices_unclipped"]
        np.testing.assert_array_equal(raw[0], np.arange(-15, 16, 2))
        np.testing.assert_array_equal(x["sample_frame_indices"], np.clip(raw, 0, 60))
        np.testing.assert_array_equal(x["padding_mask"], (raw < 0) | (raw > 60))
        self.assertEqual(x["padding_mask"][0].sum(), 8)
        self.assertEqual(x["padding_mask"][-1].sum(), 8)
        self.assertFalse(x["padding_mask"][1].any())

    def test_very_short_video_is_not_discarded(self):
        for n in (1, 2, 4):
            centers = anchor_frame_indices(n, 24)
            x = temporal_window_indices(n, 24, centers)
            self.assertTrue(np.all((x["sample_frame_indices"] >= 0) & (x["sample_frame_indices"] < n)))
            self.assertTrue(x["padding_mask"].any())

    def test_low_fps_duplicate_samples_are_not_all_padding(self):
        x = temporal_window_indices(100, 8, np.array([50]))
        self.assertLess(len(np.unique(x["sample_frame_indices"])), 16)
        self.assertFalse(x["padding_mask"].any())

    def test_invalid_timing_and_centers_fail_explicitly(self):
        for n, fps in ((0, 24), (5, 0), (5, -1), (5, float("nan")), (5, float("inf"))):
            with self.subTest(n=n, fps=fps), self.assertRaises(ValueError):
                anchor_frame_indices(n, fps)
        for centers in (np.array([]), np.array([1.2]), np.array([-1]), np.array([5])):
            with self.assertRaises(ValueError):
                temporal_window_indices(5, 10, centers)
        with self.assertRaises(ValueError):
            anchor_frame_indices(10, 10, 0)


class AlignmentTests(unittest.TestCase):
    def test_linear_feature_alignment_to_true_t_and_512_dimensions(self):
        centers = np.array([0, 3, 8])
        channels = np.arange(FEATURE_DIM, dtype=np.float32) / 100
        sparse = centers[:, None].astype(np.float32) + channels[None, :]
        features = interpolate_anchor_features(sparse, centers, 9)
        self.assertEqual(features.shape, (9, 512))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(features, np.arange(9)[:, None] + channels[None, :], atol=1e-6)
        np.testing.assert_array_equal(features[centers], sparse)

    def test_nonlinear_anchors_are_preserved_and_local_interpolation_is_correct(self):
        features = interpolate_anchor_features(np.array([[1], [9], [3]], np.float32), np.array([0, 2, 5]), 6)
        np.testing.assert_array_equal(features[:, 0], [1, 5, 9, 7, 5, 3])

    def test_single_real_frame(self):
        sparse = np.ones((1, 512), dtype=np.float32)
        np.testing.assert_array_equal(interpolate_anchor_features(sparse, np.array([0]), 1), sparse)

    def test_invalid_coverage_order_shape_or_nonfinite_data_is_rejected(self):
        for centers in (np.array([1, 5]), np.array([0, 4]), np.array([5, 0]), np.array([0, 0, 5])):
            with self.assertRaises(ValueError):
                interpolate_anchor_features(np.ones((len(centers), 2)), centers, 6)
        with self.assertRaises(ValueError):
            interpolate_anchor_features(np.ones((1, 2)), np.array([0, 5]), 6)
        with self.assertRaises(ValueError):
            interpolate_anchor_features(np.array([[np.nan], [1]]), np.array([0, 5]), 6)


class MetadataTests(unittest.TestCase):
    def test_inventory_only_uses_name_and_frame_count(self):
        DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic-test-", dir=DEFAULT_OUTPUT) as tmp:
            card = Path(tmp) / "card.json"
            card.write_text(json.dumps({"videos": [{"video_name": "synthetic.mp4", "frame_count": 7,
                            "labels": "DO_NOT_USE", "annotation_file": "/must/not/be/read",
                            "positive_frames": 999}]}), encoding="utf-8")
            self.assertEqual(load_video_specs(card), [VideoSpec("synthetic.mp4", 7)])
            for names in (["../escape.mp4"], ["a.mp4", "A.mp4"]):
                card.write_text(json.dumps({"videos": [{"video_name": name, "frame_count": 7} for name in names]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_video_specs(card)

    def test_frame_count_mismatch_is_reported_not_silently_truncated(self):
        info = {"frame_count": 8, "reported_frame_count": 9}
        warnings = _warnings(info, 7)
        self.assertEqual(len(warnings), 2)
        self.assertEqual(info["frame_count"], 8)

    def test_budget_is_explicitly_bounded(self):
        with self.assertRaises(BudgetExceeded):
            Budget(0.01, time.perf_counter() - 1).check()
        Budget(10, time.perf_counter()).check()


class PreprocessingAndBatchTests(unittest.TestCase):
    """CPU-only installed-runtime checks; no model weights or real videos read."""

    @classmethod
    def setUpClass(cls):
        import torch
        from torchvision.models.video import R3D_18_Weights
        cls.torch = torch
        torch.set_num_threads(2)
        cls.runtime = {"torch": torch, "transform": R3D_18_Weights.KINETICS400_V1.transforms(), "device": "cpu"}

    def test_cached_spatial_preprocessing_matches_baseline_per_clip(self):
        rgb = np.random.default_rng(0).integers(0, 256, (5, 24, 32, 3), dtype=np.uint8)
        indices = temporal_window_indices(5, 10, np.array([0, 2, 4]))["sample_frame_indices"]
        prepared = transform_rgb_frames(self.runtime, rgb)
        for row in indices:
            baseline = self.runtime["transform"](self.torch.from_numpy(rgb[row]).permute(0, 3, 1, 2))
            cached = prepared[self.torch.from_numpy(row)].permute(1, 0, 2, 3)
            self.torch.testing.assert_close(cached, baseline, rtol=0, atol=0)
        self.assertEqual(tuple(prepared.shape), (5, 3, 112, 112))

    def test_color_channels_and_normalization_match_rgb_preset(self):
        rgb = np.empty((1, 20, 30, 3), dtype=np.uint8)
        rgb[:] = [255, 128, 0]
        prepared = transform_rgb_frames(self.runtime, rgb).numpy()
        transform = self.runtime["transform"]
        expected = (np.array([1, 128 / 255, 0]) - transform.mean) / transform.std
        np.testing.assert_allclose(prepared[0, :, 40, 40], expected, atol=1e-6)

    def test_batches_use_the_declared_temporal_indices(self):
        torch = self.torch
        class SyntheticModel:
            def __call__(self, batch):
                self_outer.assertFalse(torch.is_grad_enabled())
                return batch.mean(dim=(1, 2, 3, 4))[:, None].repeat(1, 512)
        self_outer = self
        runtime = {**self.runtime, "model": SyntheticModel()}
        frames = torch.arange(7, dtype=torch.float32)[:, None, None, None].expand(7, 3, 4, 4)
        indices = temporal_window_indices(7, 8, np.array([0, 3, 6]))["sample_frame_indices"]
        errors = []
        features, batches = extract_anchor_features(runtime, frames, indices, 2, Budget(30, time.perf_counter()), errors.append)
        np.testing.assert_allclose(features[:, 0], indices.mean(axis=1))
        self.assertEqual(features.shape, (3, 512))
        self.assertEqual([(b["anchor_start"], b["anchor_stop"]) for b in batches], [(0, 2), (2, 3)])
        self.assertEqual(errors, [])

    def test_failed_inference_batch_is_recorded_and_not_faked(self):
        class FailedModel:
            def __call__(self, batch):
                raise RuntimeError("synthetic inference failure")
        runtime = {**self.runtime, "model": FailedModel()}
        frames = self.torch.zeros((2, 3, 4, 4))
        indices = np.zeros((2, 16), dtype=np.int64)
        errors = []
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            extract_anchor_features(runtime, frames, indices, 2, Budget(30, time.perf_counter()), errors.append)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["stage"], "inference_batch")
        self.assertEqual((errors[0]["anchor_start"], errors[0]["anchor_stop"]), (0, 2))


if __name__ == "__main__":
    unittest.main()
