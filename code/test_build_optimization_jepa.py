"""Small CPU/NumPy core tests for the real batch adapter; no videos or disk writes.

Run only the two new files, not the repository's GPU/data tests, with Python -B.
Confirmed production bugs remain strict failures (never expectedFailure).
"""
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np

import build_optimization_jepa as adapter
from optimized_jepa_extractor import Evidence


CORRECTED_NAMES = [prefix + "_" + suffix for prefix in ("v", "i") for suffix in (
    "absolute_mean", "patch_std", "patch_p90", "patch_p99", "patch_fraction",
    "direct_observation", "interpolation_range_supported")]


def synthetic_arrays():
    """Three anchors per route: the middle one has NO real observation.

    Patch NaNs are outside each observed spatial mask. Absolute error is an
    authoritative token-observation-weighted score, not mean(unique heatmap).
    """
    result = {
        "frame_ids": np.arange(12, dtype=np.int64),
        "tubelet_frame_ids": np.asarray([[0, 2], [4, 6], [8, 10]], dtype=np.int64),
        "keyframe_ids": np.asarray([1, 5, 9], dtype=np.int64),
    }
    for kind, raw, heat in (
        ("vjepa", [2., np.nan, 7.], [[[1., 3.], [np.nan, np.nan]],
                                    [[np.nan, np.nan], [np.nan, np.nan]],
                                    [[5., 9.], [np.nan, np.nan]]]),
        ("ijepa", [6., np.nan, 10.], [[[5., 7.], [np.nan, np.nan]],
                                      [[np.nan, np.nan], [np.nan, np.nan]],
                                      [[8., 12.], [np.nan, np.nan]]]),
    ):
        result[kind + "_raw_errors"] = np.asarray(raw, dtype=np.float32)
        result[kind + "_raw_heatmaps"] = np.asarray(heat, dtype=np.float32)
        result[kind + "_patch_valid_mask"] = np.asarray(
            [[[True, True], [False, False]], [[False, False], [False, False]],
             [[True, True], [False, False]]], dtype=bool)
        result[kind + "_valid_mask"] = np.asarray([True, False, True])
    return result


def evidence_from_arrays(arrays, kind):
    observed = arrays[kind + "_valid_mask"].copy()
    raw = arrays[kind + "_raw_errors"].copy()
    counts = arrays[kind + "_patch_valid_mask"].astype(np.int64)
    return Evidence(
        kind=kind, raw_errors=raw, relative_scores=np.where(observed, 0.5, np.nan).astype(np.float32),
        valid_mask=observed, raw_heatmaps=arrays[kind + "_raw_heatmaps"].copy(),
        patch_valid_mask=arrays[kind + "_patch_valid_mask"].copy(), patch_counts=counts,
        metadata={"status": "ok", "directions": ["forward"]},
        direction_raw_errors=raw[None].copy(), direction_valid_mask=observed[None].copy(),
        direction_counts=counts.sum(axis=(1, 2))[None],
    )


class CorrectedFeatureTests(unittest.TestCase):
    def test_patch_descriptors_keep_absolute_error_and_known_patch_statistics(self):
        arrays = synthetic_arrays()
        rows, observed = adapter.patch_descriptors(arrays, "vjepa")
        self.assertEqual(rows.shape, (3, 5))
        self.assertEqual(rows.dtype, np.float32)
        np.testing.assert_array_equal(observed, [True, False, True])
        np.testing.assert_allclose(rows[0], [2., 1., 2.8, 2.98, 0.5], rtol=1e-6)
        np.testing.assert_allclose(rows[2], [7., 2., 8.6, 8.96, 0.5], rtol=1e-6)
        # Unobserved placeholder descriptors must be excluded by the route mask,
        # not mistaken for a zero-error interpolation anchor.
        np.testing.assert_array_equal(rows[1], np.zeros(5))

    def test_corrected_schema_is_fourteen_named_float32_columns(self):
        values, names = adapter.corrected_frame_features(synthetic_arrays())
        self.assertEqual(values.shape, (12, 14))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(names, CORRECTED_NAMES)
        self.assertEqual(len(set(names)), 14)
        self.assertTrue(np.isfinite(values).all())
        # Anchor scores are absolute, not rescaled separately for this video.
        self.assertEqual(values[1, names.index("v_absolute_mean")], 2.)
        self.assertEqual(values[9, names.index("v_absolute_mean")], 7.)
        self.assertEqual(values[1, names.index("i_absolute_mean")], 6.)
        self.assertEqual(values[9, names.index("i_absolute_mean")], 10.)

    def test_irregular_tubelet_members_use_mean_anchors_not_filled_intervals(self):
        arrays = synthetic_arrays()
        arrays["tubelet_frame_ids"] = np.asarray([[1, 6], [2, 4], [7, 11]], dtype=np.int64)
        # Only rows 0 and 2 are observed: centers 3.5 and 9, NOT requested
        # frame ranges [1..6]/[7..11], nor the unobserved center at 3.
        values, names = adapter.corrected_frame_features(arrays)
        by_name = {name: values[:, i] for i, name in enumerate(names)}
        expected = np.interp(np.arange(12), [3.5, 9.], [2., 7.])
        np.testing.assert_allclose(by_name["v_absolute_mean"], expected, rtol=1e-6)
        np.testing.assert_array_equal(np.flatnonzero(by_name["v_direct_observation"]), [1, 6, 7, 11])
        np.testing.assert_array_equal(by_name["v_interpolation_range_supported"], [0] + [1] * 11)
        np.testing.assert_allclose(by_name["i_absolute_mean"], np.interp(np.arange(12), [1, 9], [6, 10]))
        np.testing.assert_array_equal(np.flatnonzero(by_name["i_direct_observation"]), [1, 9])
        np.testing.assert_array_equal(by_name["i_interpolation_range_supported"], [0] + [1] * 9 + [0, 0])
        # Edge-extended values are modelling assumptions, with support explicitly false.
        self.assertEqual(by_name["i_absolute_mean"][11], 10.)
        self.assertEqual(by_name["i_interpolation_range_supported"][11], 0.)

    def test_unobserved_rows_and_masked_out_nan_patches_cannot_poison_interpolation(self):
        arrays = synthetic_arrays()
        expected, names = adapter.corrected_frame_features(arrays)
        for kind in ("vjepa", "ijepa"):
            # Even stale true spatial bits do not turn an unobserved frame into evidence.
            arrays[kind + "_patch_valid_mask"][1] = True
            arrays[kind + "_raw_heatmaps"][1] = np.inf
            arrays[kind + "_raw_errors"][1] = np.nan
            arrays[kind + "_raw_heatmaps"][[0, 2], 1, :] = np.inf
        actual, actual_names = adapter.corrected_frame_features(arrays)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_names, names)
        self.assertTrue(np.isfinite(actual).all())
        # Missing middle anchors do not inject zero dips.
        self.assertAlmostEqual(float(actual[5, 0]), 4.5)
        self.assertAlmostEqual(float(actual[5, 7]), 8.)

    def test_absolute_scale_changes_errors_and_patch_statistics_not_masks(self):
        arrays = synthetic_arrays()
        before, names = adapter.corrected_frame_features(arrays)
        scaled = deepcopy(arrays)
        for kind in ("vjepa", "ijepa"):
            scaled[kind + "_raw_errors"] *= 11
            scaled[kind + "_raw_heatmaps"] *= 11
        after, names_after = adapter.corrected_frame_features(scaled)
        self.assertEqual(names_after, names)
        for start in (0, 7):
            np.testing.assert_allclose(after[:, start:start+4], before[:, start:start+4] * 11, rtol=2e-6)
            np.testing.assert_array_equal(after[:, start+4:start+7], before[:, start+4:start+7])

    def test_single_zero_error_anchor_still_has_direct_and_support_evidence(self):
        arrays = synthetic_arrays()
        arrays["tubelet_frame_ids"][0] = [2, 6]
        arrays["keyframe_ids"][0] = 4
        for kind in ("vjepa", "ijepa"):
            arrays[kind + "_valid_mask"][:] = [True, False, False]
            arrays[kind + "_raw_errors"][0] = 0.
            arrays[kind + "_raw_heatmaps"][0, 0, :] = 0.
        values, names = adapter.corrected_frame_features(arrays)
        self.assertTrue(np.isfinite(values).all())
        for start in (0, 7):
            np.testing.assert_array_equal(values[:, start:start+4], np.zeros((12, 4)))
            np.testing.assert_array_equal(values[:, start+4], np.full(12, 0.5))
        np.testing.assert_array_equal(np.flatnonzero(values[:, names.index("v_direct_observation")]), [2, 6])
        np.testing.assert_array_equal(np.flatnonzero(values[:, names.index("v_interpolation_range_supported")]), np.arange(2, 7))
        np.testing.assert_array_equal(np.flatnonzero(values[:, names.index("i_direct_observation")]), [4])
        np.testing.assert_array_equal(np.flatnonzero(values[:, names.index("i_interpolation_range_supported")]), [4])

    def test_completely_missing_either_model_is_rejected_not_zero_imputed(self):
        for kind in ("vjepa", "ijepa"):
            with self.subTest(kind=kind):
                arrays = synthetic_arrays()
                arrays[kind + "_valid_mask"][:] = False
                arrays[kind + "_raw_errors"][:] = np.nan
                with self.assertRaisesRegex(ValueError, "no observed " + kind):
                    adapter.corrected_frame_features(arrays)

    def test_nonfinite_observed_error_or_patch_is_rejected(self):
        for kind in ("vjepa", "ijepa"):
            for corruption in ("error", "patch"):
                with self.subTest(kind=kind, corruption=corruption):
                    arrays = synthetic_arrays()
                    if corruption == "error":
                        arrays[kind + "_raw_errors"][0] = np.inf
                    else:
                        arrays[kind + "_raw_heatmaps"][0, 0, 0] = np.nan
                    with self.assertRaisesRegex(ValueError, "invalid corrected feature values"):
                        adapter.corrected_frame_features(arrays)


class BatchStageIntegrityTests(unittest.TestCase):
    def make_stage(self, kind="vjepa"):
        arrays = synthetic_arrays()
        evidence = evidence_from_arrays(arrays, kind)
        sampled = SimpleNamespace(frame_ids=np.asarray([0, 2, 4, 6, 8, 10]),
                                  keyframe_ids=arrays["keyframe_ids"])
        row = {"sha256": "synthetic-source-digest", "name": "synthetic-only"}
        handle = io.BytesIO()
        adapter.dump_part(handle, evidence, sampled, "synthetic-profile", row, 0.25,
                          {"test_backend": "NumPy", "gpu_used": False})
        return handle, evidence, sampled, row

    def test_stage_roundtrip_is_pickle_free_preserves_binding_and_rejects_stale_provenance(self):
        handle, evidence, sampled, row = self.make_stage()
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            self.assertTrue(all(archive[name].dtype.kind != "O" for name in archive.files))
            np.testing.assert_array_equal(archive["sampled_frame_ids"], sampled.frame_ids)
            np.testing.assert_array_equal(archive["keyframe_ids"], sampled.keyframe_ids)
            self.assertEqual(json.loads(str(archive["metadata_json"])), {
                "signature": "synthetic-profile", "source_sha256": row["sha256"], "kind": "vjepa",
                "metadata": evidence.metadata, "elapsed_seconds": 0.25,
                "model_runtime": {"test_backend": "NumPy", "gpu_used": False},
            })
        handle.seek(0)
        restored, metadata = adapter.read_part(handle, "vjepa", "synthetic-profile", row)
        self.assertEqual(restored.kind, evidence.kind)
        for attr in ("raw_errors", "relative_scores", "valid_mask", "raw_heatmaps", "patch_valid_mask",
                     "patch_counts", "direction_raw_errors", "direction_valid_mask", "direction_counts"):
            np.testing.assert_array_equal(getattr(restored, attr), getattr(evidence, attr))
        self.assertEqual(metadata["elapsed_seconds"], 0.25)
        for signature, source in (("stale-profile", row["sha256"]), ("synthetic-profile", "different-source")):
            with self.subTest(signature=signature, source=source):
                handle.seek(0)
                with self.assertRaisesRegex(ValueError, "stale true JEPA stage"):
                    adapter.read_part(handle, "vjepa", signature, {**row, "sha256": source})

    def test_stage_cannot_be_relabelled_as_the_other_model_with_same_profile_and_source(self):
        # Both stages intentionally share a profile/source signature; the stored
        # kind must also be validated, or I/V error units can silently be swapped.
        handle, _, _, row = self.make_stage(kind="ijepa")
        handle.seek(0)
        with self.assertRaises(ValueError):
            adapter.read_part(handle, "vjepa", "synthetic-profile", row)


if __name__ == "__main__":
    unittest.main()
