"""NumPy-only tests: no torch/cv2 imports, checkpoints, GPU, or disk artifacts.

Run under WSL with -B (also prevents touching historical __pycache__ files):
    CUDA_VISIBLE_DEVICES='' /home/zzy/vjepa2-main/vjepa-env/bin/python -B \
        -m unittest discover -s /mnt/e/jepa-system/code \
        -p test_optimized_jepa_extractor.py -v
"""
from contextlib import contextmanager, nullcontext
import io
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import optimized_jepa_extractor as extractor


def frames_for(ids):
    return [np.full((2, 2, 3), fid, dtype=np.uint8) for fid in ids]


class FakeCapture:
    def __init__(self, total=10, failed=(), remaps=None, positions=None, seek_fail=(), fps=25):
        self.total, self.failed, self.fps = total, set(failed), fps
        self.remaps, self.positions = remaps or {}, positions or {}
        self.seek_fail = set(seek_fail)
        self.requested = self.next_pos = 0
        self.read_ids = []
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {7: self.total, 5: self.fps, 1: self.next_pos}[prop]

    def set(self, prop, value):
        assert prop == 1
        self.requested = int(value)
        return self.requested not in self.seek_fail

    def read(self):
        requested = self.requested
        self.read_ids.append(requested)
        if requested in self.failed:
            return False, None
        actual = self.remaps.get(requested, requested)
        self.next_pos = self.positions.get(requested, actual + 1)
        return True, frames_for([actual])[0]

    def release(self):
        self.released = True


class NumpyVideoBackend:
    """Synthetic encoder/predictor/target contract, never a production fallback."""
    grid = 2

    def __init__(self, scale=1.0, zero=False):
        self.scale, self.zero = scale, zero
        self.prepared_members, self.context_calls, self.predictor_calls = [], [], []
        self.stats = {"synthetic_test_only": True}

    def prepare(self, frames):
        ids = np.asarray([float(frame[0, 0, 0]) for frame in frames])
        self.prepared_members.append(ids.reshape(-1, 2).copy())
        tubelet = ids.reshape(-1, 2).mean(axis=1) * self.scale
        self.targets = np.repeat(np.stack([tubelet, 2 * tubelet], axis=1), self.grid**2, axis=0)
        return len(tubelet)

    def predict_errors(self, context_ids, target_ids):
        if np.intersect1d(context_ids, target_ids).size:
            raise AssertionError("masked context must not include prediction targets")
        # Separately produced masked context, not a slice supplied to predictor
        # from the unmasked target tensor. Record both argument sets for checks.
        context = np.stack([context_ids, context_ids + 0.25], axis=1)
        self.context_calls.append(context_ids.copy())
        predicted = self.predictor(context, context_ids, target_ids)
        target = self.targets[target_ids]
        return np.zeros(len(target_ids)) if self.zero else np.linalg.norm(predicted - target, axis=-1)

    def predictor(self, context, context_ids, target_ids):
        assert len(context) == len(context_ids)
        self.predictor_calls.append(target_ids.copy())
        return np.zeros((len(target_ids), 2))


class NumpyImageBackend:
    grid = 2

    def __init__(self, zero=False):
        self.seeds, self.zero = [], zero
        self.stats = {"synthetic_test_only": True}

    def batch_errors(self, frames, seed):
        self.seeds.append(seed)
        b = len(frames)
        masks = np.repeat(np.asarray([[0, 1], [1, 3]])[:, None, :], b, axis=1)
        values = np.asarray([float(frame[0, 0, 0]) for frame in frames])
        errors = values[None, :, None] + masks
        return masks, np.zeros_like(errors, dtype=float) if self.zero else errors


class Tensor:
    """Tiny NumPy tensor shim used to execute the REAL adapter wiring without torch."""
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def size(self, dim):
        return self.value.shape[dim]

    def to(self, *args, **kwargs):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.value, axis=dim))

    def index_select(self, dim, indices):
        return Tensor(np.take(self.value, indices.value, axis=dim))

    def norm(self, dim):
        return Tensor(np.linalg.norm(self.value, axis=dim))

    def mean(self, dim):
        return Tensor(self.value.mean(axis=dim))

    def reshape(self, *shape):
        return Tensor(self.value.reshape(*shape))

    def __getitem__(self, indices):
        return Tensor(self.value[indices])

    def __sub__(self, other):
        return Tensor(self.value - other.value)


def fake_torch():
    seeds = []
    return SimpleNamespace(
        bfloat16="fake-bfloat16", long=np.int64,
        inference_mode=lambda: nullcontext(), autocast=lambda **kwargs: nullcontext(),
        as_tensor=lambda value, **kwargs: Tensor(value),
        cuda=SimpleNamespace(synchronize=lambda device: None),  # No native CUDA call exists.
        random=SimpleNamespace(fork_rng=lambda **kwargs: nullcontext(),
                               default_generator=SimpleNamespace(manual_seed=seeds.append)),
        seeds=seeds,
    )


class AdapterVideoScorer:
    def __init__(self):
        self.encoder = self.predictor = None
        self.loads, self.masked_calls, self.target_calls = 0, 0, 0
        self.preprocessed = []

    def _load_encoder(self):
        self.loads += 1

        def encode(video, masks=None, training=False):
            assert training
            if masks is None:
                self.target_calls += 1
                n_tokens = len(video) // 2 * 4
                self.target_tokens = Tensor(np.full((1, n_tokens, 3), [3., 4., 0.]))
                return self.target_tokens
            self.masked_calls += 1
            return Tensor(np.full((1, masks[0].shape[1], 3), 7.))
        self.encoder = encode

    def _load_predictor(self):
        def predict(context, masks_x, masks_y):
            # Distinctive masked-encoder value catches unmasked target reuse.
            assert np.all(context.value == 7.)
            assert masks_x[0].shape[1] == context.shape[1]
            return Tensor(np.zeros((1, masks_y[0].shape[1], 3))), Tensor(np.full(context.shape, 999.))
        predict.num_patches = 256
        self.predictor = predict

    def preprocess(self, frames):
        self.preprocessed.append([frame.copy() for frame in frames])
        return frames

    def _free_predictor(self):
        self.predictor = None

    def _free_encoder(self):
        self.encoder = None

    def compute(self, *args, **kwargs):
        raise AssertionError("upstream compute must not be used")


class AdapterImageScorer:
    def __init__(self, mask_seed=0):
        self.context_encoder = None
        self.mask_seed, self.context_calls, self.target_calls = mask_seed, 0, 0

    def _load_models(self):
        def target(imgs):
            self.target_calls += 1
            return Tensor(np.full((imgs.shape[0], 4, 2), [2., 4.]))

        def context(imgs, masks_x=None):
            assert masks_x is not None
            self.context_calls += 1
            return Tensor(np.full((imgs.shape[0], masks_x[0].shape[1], 2), 7.))

        def predictor(context, masks_enc, masks_pred):
            assert np.all(context.value == 7.)
            return Tensor(np.zeros((len(masks_pred) * context.shape[0], masks_pred[0].shape[1], 2)))

        def collate(tensors):
            b = len(tensors)
            imgs = Tensor(np.zeros((b, 3, 2, 2)))
            enc = [Tensor(np.repeat([[2]], b, axis=0))]
            pred = [Tensor(np.repeat([ids], b, axis=0)) for ids in ([0, 1], [1, 3])]
            return imgs, enc, pred

        self.context_encoder, self.target_encoder, self.predictor = context, target, predictor
        collate._itr_counter=SimpleNamespace(value=-1,get_lock=lambda:nullcontext())
        self.mask_collator = collate

    def _preprocess(self, frames):
        return frames

    def _free_models(self):
        self.context_encoder = self.target_encoder = self.predictor = None

    def compute(self, *args, **kwargs):
        raise AssertionError("upstream normalized compute must not be used")


class SamplingTests(unittest.TestCase):
    def test_uniform_indices_and_failed_reads_stay_bound_to_actual_frames(self):
        cap = FakeCapture(total=10, failed=(1, 7))
        sample = extractor.sample_video("fake.mp4", extractor.Config(max_frames=6, max_keyframes=4),
                                        capture_factory=lambda _: cap)
        np.testing.assert_array_equal(sample.frame_ids, [0, 3, 5, 9])
        np.testing.assert_array_equal(sample.keyframe_ids, [0, 3, 6, 9])
        self.assertEqual([int(f[0, 0, 0]) for f in sample.frames], sample.frame_ids.tolist())
        self.assertEqual([int(f[0, 0, 0]) for f in sample.keyframes], sample.keyframe_ids.tolist())
        self.assertEqual(cap.read_ids, [0, 1, 3, 5, 6, 7, 9])  # Shared requests decoded once.
        self.assertEqual([e["requested_id"] for e in sample.metadata["failures"]], [1, 7])
        self.assertTrue(cap.released)

    def test_seek_remap_uses_decoder_id_not_requested_id_and_deduplicates(self):
        cap = FakeCapture(total=10, remaps={3: 4, 6: 4})
        sample = extractor.sample_video("fake", extractor.Config(max_frames=4, max_keyframes=4),
                                        capture_factory=lambda _: cap)
        np.testing.assert_array_equal(sample.frame_ids, [0, 4, 9])
        self.assertEqual(sample.metadata["seek_remaps"], [{"requested_id": 3, "actual_id": 4}])
        self.assertEqual(sample.metadata["failures"][0]["reason"], "duplicate_decoded_frame")

    def test_seek_failure_and_invalid_positions_do_not_manufacture_ids(self):
        cap = FakeCapture(total=4, seek_fail=(0,), positions={1: 0, 2: 2.25})
        sample = extractor.sample_video("fake", extractor.Config(max_frames=4, max_keyframes=4),
                                        capture_factory=lambda _: cap)
        np.testing.assert_array_equal(sample.frame_ids, [3])
        self.assertEqual([e["reason"] for e in sample.metadata["failures"]],
                         ["seek_failed", "unverifiable_frame_id", "unverifiable_frame_id"])

    def test_unknown_count_is_rejected_and_capture_released(self):
        cap = FakeCapture(total=0)
        with self.assertRaisesRegex(ValueError, "frame count unavailable"):
            extractor.sample_video("fake", extractor.Config(), capture_factory=lambda _: cap)
        self.assertTrue(cap.released)

    def test_unknown_fps_is_not_replaced_by_invented_fps(self):
        cap = FakeCapture(total=4, fps=0)
        sample = extractor.sample_video("fake", extractor.Config(max_frames=4), capture_factory=lambda _: cap)
        self.assertIsNone(sample.fps)

    def test_sampling_limit_cannot_generate_duplicate_requested_indices(self):
        np.testing.assert_array_equal(extractor.uniform_frame_ids(3, 64), [0, 1, 2])
        self.assertEqual(extractor.uniform_frame_ids(0, 32).size, 0)


class VideoEvidenceTests(unittest.TestCase):
    def test_short_tubelets_skip_all_backend_calls_and_mark_insufficient(self):
        class NoCalls:
            grid = 2
            def prepare(self, *args):
                raise AssertionError("short clips must not load/encode")
            def predict_errors(self, *args):
                raise AssertionError("short clips must not predict")
        for length in range(4):
            with self.subTest(length=length):
                result = extractor.run_vjepa(frames_for(range(length)), list(range(length)),
                                             NoCalls(), extractor.Config(bidirectional=True))
                self.assertEqual(result.metadata["status"], "insufficient_evidence")
                self.assertFalse(result.valid_mask.any())
                self.assertTrue(np.isnan(result.raw_errors).all())
                self.assertEqual(result.patch_counts.sum(), 0)

    def test_windows_are_bounded_by_effective_tubelets_and_cover_tail(self):
        self.assertEqual(extractor.plan_windows(2, 6, 2), (1, 1, [0]))
        self.assertEqual(extractor.plan_windows(5, 6, 2), (2, 2, [0, 1]))
        for n in range(2, 33):
            context, target, starts = extractor.plan_windows(n, 6, 2)
            covered = set()
            for start in starts:
                self.assertGreater(context, 0)
                self.assertLessEqual(start + context + target, n)
                covered.update(range(start + context, start + context + target))
            self.assertEqual(covered, set(range(context, n)))
            reverse = {n - 1 - i for i in covered}
            self.assertEqual(covered | reverse, set(range(n)))

    def test_short_real_eligible_video_actually_calls_predictor(self):
        backend = NumpyVideoBackend()
        result = extractor.run_vjepa(frames_for([2, 4, 6, 8]), [2, 4, 6, 8], backend, extractor.Config())
        self.assertEqual(result.metadata["effective_context_tubelets"], 1)
        self.assertEqual(result.metadata["effective_target_tubelets"], 1)
        np.testing.assert_array_equal(result.valid_mask, [False, True])
        self.assertTrue(backend.context_calls)
        self.assertTrue(backend.predictor_calls)
        self.assertAlmostEqual(result.raw_errors[1], 7 * np.sqrt(5), places=5)
        self.assertTrue(np.isnan(result.raw_errors[0]))

    def test_absolute_scale_survives_while_relative_scores_remain_compatible(self):
        frames, ids = frames_for(range(2, 18, 2)), list(range(2, 18, 2))
        one = extractor.run_vjepa(frames, ids, NumpyVideoBackend(scale=1), extractor.Config())
        ten = extractor.run_vjepa(frames, ids, NumpyVideoBackend(scale=10), extractor.Config())
        np.testing.assert_allclose(ten.raw_errors[ten.valid_mask], 10 * one.raw_errors[one.valid_mask], rtol=1e-6)
        np.testing.assert_allclose(ten.relative_scores[ten.valid_mask], one.relative_scores[one.valid_mask], rtol=1e-6)
        self.assertFalse(one.valid_mask[0])
        self.assertTrue(np.isnan(one.raw_heatmaps[~one.patch_valid_mask]).all())

    def test_true_zero_error_is_valid_not_missing(self):
        result = extractor.run_vjepa(frames_for(range(8)), list(range(8)),
                                     NumpyVideoBackend(zero=True), extractor.Config())
        self.assertTrue(result.valid_mask.any())
        self.assertTrue((result.raw_errors[result.valid_mask] == 0).all())
        self.assertTrue((result.relative_scores[result.valid_mask] == 0).all())
        self.assertTrue(np.isnan(result.raw_errors[~result.valid_mask]).all())

    def test_complementary_half_masks_cover_all_spatial_patches(self):
        result = extractor.run_vjepa(frames_for(range(12)), list(range(12)), NumpyVideoBackend(),
                                     extractor.Config(mask_passes=2))
        self.assertTrue(result.patch_valid_mask[result.valid_mask].all())
        self.assertFalse(result.patch_valid_mask[~result.valid_mask].any())
        self.assertEqual(result.metadata["mask_ratio_effective"], 0.5)

    def test_bidirectional_mapping_preserves_pairs_and_drops_odd_before_reverse(self):
        ids = [1, 5, 9, 13, 19, 23, 31]
        backend = NumpyVideoBackend()
        result = extractor.run_vjepa(frames_for(ids), ids, backend,
                                     extractor.Config(bidirectional=True, mask_ratio=1.0))
        np.testing.assert_array_equal(backend.prepared_members[0], [[1, 5], [9, 13], [19, 23]])
        np.testing.assert_array_equal(backend.prepared_members[1], [[23, 19], [13, 9], [5, 1]])
        np.testing.assert_array_equal(result.valid_mask, [True, True, True])
        np.testing.assert_allclose(result.raw_errors, np.asarray([3, 11, 21]) * np.sqrt(5), rtol=1e-6)
        self.assertEqual(result.metadata["dropped_unpaired_frame_ids"], [31])
        np.testing.assert_array_equal(result.direction_valid_mask, [[False, True, True], [True, True, False]])
        self.assertTrue(np.isnan(result.direction_raw_errors[0, 0]))
        self.assertTrue(np.isnan(result.direction_raw_errors[1, 2]))

    def test_direction_fusion_is_count_weighted_not_mean_of_means(self):
        class Directional(NumpyVideoBackend):
            def predict_errors(self, context_ids, target_ids):
                d = len(self.prepared_members)
                # Per-direction patch errors deliberately unequal.
                return np.full(len(target_ids), 2 if d == 1 else 8, dtype=float)
        result = extractor.run_vjepa(frames_for(range(16)), list(range(16)), Directional(),
                                     extractor.Config(bidirectional=True, context_tubelets=1, target_tubelets=3))
        counts = result.direction_counts
        expected = np.nansum(result.direction_raw_errors * counts, axis=0) / counts.sum(axis=0)
        np.testing.assert_allclose(result.raw_errors, expected)
        unequal = (counts[0] > 0) & (counts[1] > 0) & (counts[0] != counts[1])
        self.assertTrue(unequal.any())
        self.assertTrue(np.any(result.raw_errors[unequal] != 5))

    def test_seed_reproducibility_and_other_seed_changes_masks(self):
        frames, ids = frames_for(range(16)), list(range(16))
        a, b, c = NumpyVideoBackend(), NumpyVideoBackend(), NumpyVideoBackend()
        for backend, seed in ((a, 11), (b, 11), (c, 12)):
            extractor.run_vjepa(frames, ids, backend, extractor.Config(seed=seed))
        for x, y in zip(a.predictor_calls, b.predictor_calls):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(any(not np.array_equal(x, y) for x, y in zip(a.predictor_calls, c.predictor_calls)))

    def test_encoder_token_mapping_mismatch_is_not_silently_trimmed(self):
        class WrongCount(NumpyVideoBackend):
            def prepare(self, frames):
                return super().prepare(frames) - 1
        with self.assertRaisesRegex(RuntimeError, "decoded mapping"):
            extractor.run_vjepa(frames_for(range(8)), list(range(8)), WrongCount(), extractor.Config())

    def test_nonfinite_or_wrong_shape_prediction_errors_fail_loudly(self):
        for bad in (np.asarray([np.nan, np.nan]), np.zeros(1), np.asarray([-1., 0.])):
            backend = NumpyVideoBackend()
            backend.predict_errors = lambda ctx, tgt, bad=bad: bad
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                extractor.run_vjepa(frames_for(range(4)), list(range(4)), backend, extractor.Config())

    def test_duplicate_or_noninteger_frame_ids_are_rejected(self):
        for ids in ([0, 0, 2, 3], [3, 2, 1, 0], [0., 1., 2., 3.]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                extractor.run_vjepa(frames_for(range(4)), ids, NumpyVideoBackend(), extractor.Config())


class ImageEvidenceTests(unittest.TestCase):
    def test_raw_scores_are_before_normalization_and_overlap_counts_are_retained(self):
        result = extractor.run_ijepa(frames_for([2, 4]), [2, 4], NumpyImageBackend(), extractor.Config())
        # 0,1,1,3 are four observations, not three equally weighted unique patches.
        np.testing.assert_allclose(result.raw_errors, [3.25, 5.25])
        np.testing.assert_allclose(result.relative_scores, [0, 2 / (2 + 1e-6)])
        np.testing.assert_array_equal(result.patch_counts[0], [[1, 2], [0, 1]])
        self.assertTrue(np.isnan(result.raw_heatmaps[:, 1, 0]).all())
        self.assertFalse(result.patch_valid_mask[:, 1, 0].any())
        np.testing.assert_allclose(result.raw_heatmaps[0, 0], [2, 3])

    def test_constant_legacy_ijepa_score_is_preserved_not_claimed_as_probability(self):
        result = extractor.run_ijepa(frames_for([4, 4]), [1, 3], NumpyImageBackend(), extractor.Config())
        np.testing.assert_allclose(result.raw_errors, [5.25, 5.25])
        np.testing.assert_allclose(result.relative_scores, result.raw_errors)
        semantics = extractor._normalization_metadata(result)
        self.assertFalse(semantics["relative_is_calibrated_probability"])
        self.assertTrue(semantics["constant_ijepa_may_be_outside_0_1"])
        self.assertFalse(semantics["raw_per_video_scaling"])

    def test_batches_retain_exact_keyframe_order_and_original_seed_offsets(self):
        backend = NumpyImageBackend()
        result = extractor.run_ijepa(frames_for([2, 7, 12]), [2, 7, 12], backend,
                                     extractor.Config(seed=17, ijepa_batch_size=2))
        self.assertEqual(backend.seeds, [17, 19])
        np.testing.assert_allclose(result.raw_errors, [3.25, 8.25, 13.25])
        self.assertEqual(result.metadata["keyframe_ids"], [2, 7, 12])

    def test_empty_keyframes_skip_backend(self):
        result = extractor.run_ijepa([], [], None, extractor.Config())
        self.assertEqual(result.metadata["status"], "insufficient_evidence")
        self.assertEqual(result.raw_errors.size, 0)

    def test_mask_shape_and_patch_bounds_mismatches_are_not_zipped_away(self):
        cases = [(np.zeros((2, 1, 2)), np.zeros((2, 1, 2))),  # Not integer masks.
                 (np.full((2, 1, 2), 4, dtype=int), np.zeros((2, 1, 2))),
                 (np.zeros((2, 1, 2), dtype=int), np.zeros((2, 1, 1)))]
        for ids, errors in cases:
            backend = NumpyImageBackend()
            backend.batch_errors = lambda frames, seed, ids=ids, errors=errors: (ids, errors)
            with self.subTest(shape=ids.shape), self.assertRaisesRegex(RuntimeError, "masks/errors"):
                extractor.run_ijepa(frames_for([3]), [3], backend, extractor.Config())


class RealAdapterWiringTests(unittest.TestCase):
    def test_real_v_adapter_uses_masked_encoder_and_predictor_target_output(self):
        # Exercise production adapter operations through NumPy tensors; no torch,
        # CUDA, checkpoint constructors, or filesystem access can occur.
        legacy = SimpleNamespace(torch=fake_torch(), DEVICE="fake-cpu", __file__="fake-v-loader",
                                 ENCODER_CKPT="not-loaded", FULL_CKPT="not-loaded",
                                 VJEPASurprise=AdapterVideoScorer)
        adapter = extractor._RealVJEPA(legacy)
        adapter.grid = 2
        bgr = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
        with patch.object(extractor, "VJEPA_FEATURE_DIM", 3):
            result = extractor.run_vjepa([bgr] * 6, list(range(6)), adapter,
                                         extractor.Config(bidirectional=True, mask_passes=2))
        self.assertTrue(result.valid_mask.all())
        np.testing.assert_allclose(result.raw_errors, 5.)  # Norm([3,4,0]) not context-output 999.
        self.assertEqual(adapter.scorer.loads, 1)
        self.assertEqual(adapter.scorer.target_calls, 2)  # Reverse recomputed, not flipped full targets.
        self.assertEqual(adapter.scorer.masked_calls, 2)  # One context forward per identical mask/direction.
        np.testing.assert_array_equal(adapter.scorer.preprocessed[0][0][0, 0], [30, 20, 10])
        adapter.close()
        self.assertIsNone(adapter.scorer.encoder)
        self.assertIsNone(adapter.scorer.predictor)

    def test_real_i_adapter_uses_layernorm_target_and_retains_smooth_l1_raw(self):
        layernorm_calls = []
        def layer_norm(tensor, dims):
            layernorm_calls.append(dims)
            x = tensor.value
            normalized = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)
            return Tensor(normalized)

        def smooth_l1(predicted, target, reduction, beta):
            self.assertEqual(reduction, "none")
            self.assertEqual(beta, 1.0)
            delta = np.abs(predicted.value - target.value)
            return Tensor(np.where(delta < beta, 0.5 * delta**2 / beta, delta - 0.5 * beta))

        def apply_masks(target, masks):
            return Tensor(np.concatenate([
                np.stack([target.value[b, mask.value[b]] for b in range(target.shape[0])])
                for mask in masks], axis=0))

        functions = SimpleNamespace(layer_norm=layer_norm, smooth_l1_loss=smooth_l1)
        checkpoint = SimpleNamespace(exists=lambda: True, __str__=lambda: "not-loaded")
        legacy = SimpleNamespace(torch=fake_torch(), F=functions, DEVICE="fake-cpu", __file__="fake-i-loader",
                                 CKPT_SLIM_BF16=checkpoint, CKPT_FULL="not-loaded",
                                 IJEPASurprise=AdapterImageScorer)
        masks_module = ModuleType("src.masks.utils")
        masks_module.apply_masks = apply_masks
        with patch.dict(sys.modules, {"src.masks.utils": masks_module}):
            adapter = extractor._RealIJEPA(legacy, seed=9)
            adapter.grid = 2
            result = extractor.run_ijepa(frames_for([1, 3]), [1, 3], adapter, extractor.Config(seed=9))
        self.assertEqual(layernorm_calls, [(2,)])
        self.assertEqual(adapter.scorer.context_calls, 1)
        self.assertEqual(adapter.scorer.target_calls, 1)
        self.assertEqual(legacy.torch.seeds, [9])
        np.testing.assert_allclose(result.raw_errors, 0.5 / (1 + 1e-5), rtol=1e-6)
        np.testing.assert_allclose(result.relative_scores, result.raw_errors)
        np.testing.assert_array_equal(result.patch_counts[0], [[1, 2], [0, 1]])
        adapter.close()
        self.assertIsNone(adapter.scorer.context_encoder)

    def test_predictor_shape_mismatch_never_silently_truncates(self):
        expected = (1, 3, 2)
        target_output = np.ones(expected)
        context_output = np.full((1, 5, 2), 999)
        self.assertIs(extractor.unwrap_prediction((target_output, context_output), expected), target_output)
        self.assertIs(extractor.unwrap_prediction([target_output], expected), target_output)
        with self.assertRaisesRegex(RuntimeError, "predictor shape"):
            extractor.unwrap_prediction(np.ones((1, 2, 2)), expected)
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            extractor.unwrap_prediction([target_output, context_output], expected)

    def test_legacy_imports_restore_conflicting_packages_paths_and_bytecode_policy(self):
        old_src = ModuleType("src")
        old_src_child = ModuleType("src.models")
        old_app = ModuleType("app")
        original_path, original_bytecode = sys.path[:], sys.dont_write_bytecode
        legacy = ModuleType("fake-legacy")
        entered = []

        def load(module):
            self.assertNotIn("src", sys.modules)
            self.assertNotIn("app", sys.modules)
            self.assertTrue(sys.dont_write_bytecode)
            sys.modules["src"] = ModuleType("new-src")
            sys.modules["src.newly_loaded"] = ModuleType("new-child")
            entered.append(module)

        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=load))
        with patch.dict(sys.modules, {"src": old_src, "src.models": old_src_child, "app": old_app}):
            with patch.object(extractor.importlib.util, "spec_from_file_location", return_value=spec), \
                    patch.object(extractor.importlib.util, "module_from_spec", return_value=legacy):
                with self.assertRaisesRegex(RuntimeError, "scope-test"):
                    with extractor.isolated_legacy_module("vjepa") as actual:
                        self.assertIs(actual, legacy)
                        self.assertIsNot(sys.modules["src"], old_src)
                        raise RuntimeError("scope-test")
            self.assertIs(sys.modules["src"], old_src)
            self.assertIs(sys.modules["src.models"], old_src_child)
            self.assertIs(sys.modules["app"], old_app)
            self.assertNotIn("src.newly_loaded", sys.modules)
        self.assertEqual(entered, [legacy])
        self.assertEqual(sys.path, original_path)
        self.assertEqual(sys.dont_write_bytecode, original_bytecode)


class OutputAndCliTests(unittest.TestCase):
    def make_factory(self, calls, zero=False):
        @contextmanager
        def factory(kind, config):
            calls.append(kind)
            yield NumpyVideoBackend(zero=zero) if kind == "vjepa" else NumpyImageBackend(zero=zero)
        return factory

    def test_failed_sampling_ids_and_uncovered_masks_reach_json_and_npz(self):
        cap = FakeCapture(total=10, failed=(1, 7))
        calls = []
        arrays, metadata = extractor.extract(
            "fake-video", extractor.Config(max_frames=6, max_keyframes=4),
            backend_factory=self.make_factory(calls), capture_factory=lambda _: cap)
        self.assertEqual(calls, ["vjepa", "ijepa"])
        self.assertEqual(metadata["status"], "ok")
        np.testing.assert_array_equal(arrays["tubelet_frame_ids"], [[0, 3], [5, 9]])
        np.testing.assert_array_equal(np.flatnonzero(arrays["physics_valid_mask"]), [5, 9])
        np.testing.assert_array_equal(np.flatnonzero(arrays["corruption_valid_mask"]), [0, 3, 6, 9])
        self.assertTrue(np.isnan(arrays["physics_raw"][[0, 1, 2, 3, 4, 6, 7, 8]]).all())
        self.assertIsNone(metadata["signals"]["physics_raw"][0])
        self.assertIsNone(metadata["signals"]["corruption_raw"][1])
        self.assertEqual(metadata["coverage"]["direct_vjepa_frames"], 2)
        self.assertFalse(metadata["profile"]["full_video_interpolation"])
        self.assertEqual(metadata["schema_version"], 1)
        self.assertIn("timings", metadata["runtime"])
        self.assertIn("process_peak_rss_bytes", metadata["runtime"]["memory"])
        serialized = json.dumps(metadata, allow_nan=False)
        self.assertNotIn(": NaN", serialized)
        self.assertNotIn(": Infinity", serialized)
        # Real NPZ writer/reader exercised in RAM: no output or __pycache__ files.
        handle = io.BytesIO()
        np.savez_compressed(handle, **arrays)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            self.assertTrue(all(archive[name].dtype.kind != "O" for name in archive.files))
            embedded = json.loads(str(archive["metadata_json"]))
            self.assertEqual(embedded, metadata)
            np.testing.assert_array_equal(archive["signals_valid_mask"], arrays["signals_valid_mask"])
            self.assertEqual(embedded["array_schema"]["vjepa_raw_heatmaps"]["shape"],
                             list(archive["vjepa_raw_heatmaps"].shape))

    def test_short_video_skips_v_weights_but_keeps_honest_i_evidence(self):
        calls = []
        arrays, metadata = extractor.extract(
            "short", extractor.Config(max_frames=3, max_keyframes=3, bidirectional=True),
            backend_factory=self.make_factory(calls), capture_factory=lambda _: FakeCapture(total=3))
        self.assertEqual(calls, ["ijepa"])
        self.assertEqual(metadata["status"], "insufficient_evidence")
        self.assertFalse(metadata["runtime"]["models"]["vjepa"]["loaded"])
        self.assertFalse(arrays["physics_valid_mask"].any())
        self.assertTrue(np.isnan(arrays["physics_raw"]).all())
        self.assertTrue(arrays["corruption_valid_mask"].all())
        np.testing.assert_array_equal(arrays["dropped_unpaired_frame_ids"], [2])

    def test_all_decode_failures_skip_all_models_and_do_not_create_normal_scores(self):
        calls = []
        arrays, metadata = extractor.extract(
            "unreadable", extractor.Config(max_frames=4, max_keyframes=4),
            backend_factory=self.make_factory(calls),
            capture_factory=lambda _: FakeCapture(total=4, failed=range(4)))
        self.assertEqual(calls, [])
        self.assertEqual(metadata["status"], "insufficient_evidence")
        self.assertFalse(arrays["signals_valid_mask"].any())
        self.assertTrue(np.isnan(arrays["signals_raw"]).all())
        self.assertEqual(arrays["vjepa_raw_errors"].size, 0)

    def test_true_model_failure_emits_explicit_error_not_proxy_fallback(self):
        @contextmanager
        def factory(kind, config):
            if kind == "vjepa":
                raise RuntimeError("synthetic load failure")
            yield NumpyImageBackend()
        arrays, metadata = extractor.extract(
            "fake", extractor.Config(max_frames=6, max_keyframes=3), backend_factory=factory,
            capture_factory=lambda _: FakeCapture(total=6))
        self.assertEqual(metadata["status"], "error")
        self.assertEqual(metadata["profile"]["vjepa"]["error"]["type"], "RuntimeError")
        self.assertFalse(metadata["jepa_usage"]["proxy_fallback"])
        self.assertFalse(arrays["physics_valid_mask"].any())
        self.assertTrue(np.isnan(arrays["physics_raw"]).all())
        self.assertEqual(int(arrays["corruption_valid_mask"].sum()), 3)

    def test_dense_binding_does_not_fill_intervals_or_hide_zero_scores(self):
        dense, valid = extractor.bind_dense_evidence(12, np.asarray([[2, 9]]), np.asarray([0.]), np.asarray([True]))
        np.testing.assert_array_equal(np.flatnonzero(valid), [2, 9])
        self.assertEqual(dense[2], 0.)
        self.assertEqual(dense[9], 0.)
        self.assertTrue(np.isnan(dense[~valid]).all())

    def test_unknown_fps_serializes_null_timestamps_and_no_fake_rate(self):
        arrays, metadata = extractor.extract(
            "fake", extractor.Config(max_frames=4), backend_factory=self.make_factory([]),
            capture_factory=lambda _: FakeCapture(total=4, fps=0))
        self.assertTrue(np.isnan(arrays["timestamps_sec"]).all())
        self.assertEqual(metadata["signals"]["timestamps_sec"], [None] * 4)
        self.assertIsNone(metadata["video"]["fps"])
        self.assertFalse(metadata["video"]["fps_fallback_used"])

    def test_cli_defaults_single_direction_and_explicit_reverse_flag(self):
        parser = extractor.make_parser()
        base = ["--video", "fake.mp4", "--output", "/unused"]
        args = parser.parse_args(base)
        self.assertFalse(args.bidirectional)
        self.assertEqual(args.seed, 0)
        self.assertTrue(parser.parse_args(base + ["--bidirectional"]).bidirectional)
        self.assertEqual(parser.parse_args(base + ["--mask-passes", "2"]).mask_passes, 2)

    def test_invalid_cli_config_is_rejected_before_model_loading(self):
        cases = [{"max_frames": 65}, {"max_frames": 0}, {"max_keyframes": 0},
                 {"seed": -1}, {"mask_ratio": 0}, {"mask_ratio": np.nan},
                 {"mask_ratio": 1.1}, {"mask_passes": 0}, {"ijepa_batch_size": 0}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                extractor.Config(**kwargs)

    def test_cli_status_exit_codes_without_inference_or_writes(self):
        for status, expected_code in (("ok", 0), ("insufficient_evidence", 2),
                                      ("partial_evidence", 2), ("error", 1)):
            metadata = {"status": status, "coverage": {}}
            with self.subTest(status=status), \
                    patch.object(extractor, "extract", return_value=({}, metadata)), \
                    patch.object(extractor, "write_outputs") as writer, \
                    patch.object(extractor.Path, "exists", return_value=False), \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(extractor.main(["--video", "fake", "--output", "/unused"]), expected_code)
                writer.assert_called_once()

    def test_existing_outputs_are_rejected_before_any_inference(self):
        with patch.object(extractor, "extract") as inference, \
                patch.object(extractor.Path, "exists", return_value=True), \
                patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(extractor.main(["--video", "fake", "--output", "/unused"]), 1)
            inference.assert_not_called()
        with patch.object(extractor.Path, "exists", return_value=True):
            with self.assertRaises(FileExistsError):
                extractor.write_outputs("/unused", {}, {})


if __name__ == "__main__":
    unittest.main()
