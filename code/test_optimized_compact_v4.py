"""Mechanism and portable parity tests for new experiment-only compact features."""
import copy
import importlib.util
import unittest
import numpy as np
from optimized_compact_features_v4 import TILE_FIELDS, RECIPES, feature_view_v4, fit_rgb_pca, content_weights
from optimized_compact_model_v4 import fit_compact_member, export_compact_member, predict_compact_member


def record(seed=0, frames=60):
    rng = np.random.default_rng(seed)
    motion = rng.uniform(.2, 2, (frames, 26)).astype(np.float32); motion[:, 22:] = 1
    corrected = rng.uniform(.2, 2, (frames, 14)).astype(np.float32); corrected[:, [5, 6, 12, 13]] = 1
    names = [f'local/g{i}' for i in range(32)]
    for r in range(3):
        for c in range(3):
            names.extend(f'local/tile_{r}{c}_{f}' for f in ['observed_fraction'] + list(TILE_FIELDS))
    names.extend(f'local/flag{i}_valid' for i in range(11))
    local = rng.uniform(.2, 2, (frames, 284)).astype(np.float32); local[:, 131:] = 1
    labels = np.zeros(frames, np.uint8); labels[frames//3:2*frames//3] = 1
    return {'sha256': f'content{seed}', 'fps': 24., 'frames': frames, 'labels': labels,
            'motion': motion, 'motion_names': [f'm{i}' for i in range(26)],
            'corrected': corrected, 'corrected_names': [f'c{i}' for i in range(14)],
            'local': local, 'local_names': names + [n + '_support_flag' for n in names],
            'rgb': rng.normal(size=(frames, 6)).astype(np.float32)}


class CompactFeatures(unittest.TestCase):
    def test_schema_single_frame_and_fps(self):
        for count in [1, 2, 60]:
            r = record(frames=count)
            for fps in [5, 24, 60]:
                r['fps'] = fps
                for recipe in [RECIPES[0], RECIPES[3]]:
                    x, v, names = feature_view_v4(recipe, r)
                    self.assertEqual(x.shape, (count, len(names)))
                    self.assertEqual(len(names), len(set(names)))
                    self.assertTrue(np.isfinite(x).all() and np.isfinite(v).all())

    def test_no_labels_or_metadata_in_transform(self):
        r = record(); a = feature_view_v4(RECIPES[0], r)
        q = copy.deepcopy(r); q['labels'][:] = 1; q.update(name='completely different', generator='unused', sha256='new')
        b = feature_view_v4(RECIPES[0], q)
        for j in range(2):
            np.testing.assert_array_equal(a[j], b[j])

    def test_missing_observation_not_an_acceleration(self):
        r = record(); r['local'][:, 0] = 1
        r['local'][20:24, 142] = 0; r['local'][20:24, 0] = 1000
        x, _, n = feature_view_v4(RECIPES[0], r)
        np.testing.assert_array_equal(x[:, n.index('local/g0/delta_per_second')], 0)
        self.assertTrue(np.all(x[:, n.index('local/g0/contrast_0.2s')] == 0))
        self.assertTrue(np.all(x[:, n.index('local/g0/future_minus_past_0.3s')] == 0))

    def test_tile_pool_ignores_unsupported_values(self):
        r = record(); q = copy.deepcopy(r)
        col = r['local_names'].index('local/tile_00_residual_p90')
        r['local'][:, col + 142] = 0; q['local'][:, col + 142] = 0; q['local'][:, col] = 1e9
        a, _, n = feature_view_v4(RECIPES[0], r); b, _, _ = feature_view_v4(RECIPES[0], q)
        ids = [j for j, name in enumerate(n) if name.startswith('local/tile_pool/residual_p90/')]
        np.testing.assert_array_equal(a[:, ids], b[:, ids])

    def test_training_pca_content_mass_and_holdout_invariance(self):
        records = [record(i) for i in range(3)]
        p = fit_rgb_pca(records, [0, 1]); records[2]['rgb'] *= 1e6
        self.assertEqual(p, fit_rgb_pca(records, [0, 1]))
        records.append(copy.deepcopy(records[0])); q = fit_rgb_pca(records, [0, 1, 3])
        np.testing.assert_allclose(p['mean'], q['mean'], atol=1e-7)
        np.testing.assert_allclose(p['components'], q['components'], atol=1e-7)
        self.assertEqual(content_weights(records, [0, 1, 3]), {0: .5, 1: 1., 3: .5})

    def test_bad_inputs(self):
        r = record()
        with self.assertRaises(ValueError): feature_view_v4('unknown', r)
        with self.assertRaises(ValueError): feature_view_v4(RECIPES[1], r)
        q = copy.deepcopy(r); q['fps'] = float('nan')
        with self.assertRaises(ValueError): feature_view_v4(RECIPES[0], q)
        q = copy.deepcopy(r); q['local'][:, 142] = 2
        with self.assertRaises(ValueError): feature_view_v4(RECIPES[0], q)


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'sklearn training environment required')
class CompactModels(unittest.TestCase):
    def test_fit_partition_leakage(self):
        records = [record(i) for i in range(4)]
        with self.assertRaises(ValueError): fit_compact_member(RECIPES[0], records, [0, 1], [1, 2], 1)
        with self.assertRaises(ValueError): fit_compact_member(RECIPES[0], records, [0, 1], [2], 1, full_fit=True)

    def test_et_and_hgb_portable_predictions(self):
        records = [record(i) for i in range(4)]
        records[0]['labels'][:] = 0
        for recipe in [RECIPES[1], RECIPES[2]]:
            fp, vp, state = fit_compact_member(recipe, records, [0, 1, 2], [3], 133)
            member = export_compact_member(state)
            f, v = predict_compact_member(member, records[3])
            np.testing.assert_allclose(f, fp[3], rtol=0, atol=2e-6)
            self.assertAlmostEqual(v, vp[3], delta=2e-6)


if __name__ == '__main__': unittest.main()
