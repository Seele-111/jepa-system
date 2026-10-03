"""Focused audit regression tests: no real training or experiment output writes."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_video_gate_v6 as audit


def fixture():
    records = [{'sha256': f'{i:064x}', 'generator': str(i % 2), 'event_count': i % 3,
                'frames': 2, 'fps': 24., 'labels': np.asarray([0, i % 2])} for i in range(36)]
    train, val = list(range(30)), list(range(30, 36))
    parts = []
    for k, iv in enumerate(audit.grouped_folds(records, train, 3, 20261002)):
        it = [i for i in train if i not in set(iv)]
        deep = []
        for d, dv in enumerate(audit.grouped_folds(records, it, 3, 20261006 + k * 31)):
            dt = [i for i in it if i not in set(dv)]
            deep.append({'fit': dt, 'validation': dv, 'member_fit_proof': [
                {'recipe': r, 'seed': 20261006 + k * 31 + d + j * 1000,
                 'fit_content_sha256': audit.content(records, dt), 'transform_sha256': 'a' * 64}
                for j, r in enumerate(audit.MEMBERS)]})
        states = {g: {'kind': g, 'fit_content_sha256': audit.content(records, it)} for g in audit.GATES}
        parts.append({'fit': it, 'validation': iv, 'deep_partitions': deep, 'gate_states': states})
    outer = {g: {'kind': g, 'fit_content_sha256': audit.content(records, train)} for g in audit.GATES}
    return records, {'fold': 0, 'train': train, 'validation': val, 'inner_partitions': parts,
                     'outer_gate_states': outer, 'outer_gate_seed': 20267006}


class GateAuditTests(unittest.TestCase):
    def test_logistic_precision_regression_and_sklearn(self):
        x = np.asarray([.5000, .5001, .5002, .5003], np.float32)
        mean, scale = np.mean(x.astype(np.float64)), np.std(x.astype(np.float64))
        state = {'mean': [mean], 'scale': [scale], 'coefficients': [1.],
                 'intercept': 0., 'feature_names': ['test']}
        bad = [audit.logistic_reference(state, [v], False) for v in x]
        good = [audit.logistic_reference(state, [v]) for v in x]
        self.assertGreater(max(abs(a - b) for a, b in zip(bad, good)), 2e-6)
        if importlib.util.find_spec('sklearn') is not None:
            model = audit.sklearn_model(state)
            for v, expected in zip(x, good):
                actual = model.predict_proba(audit.standardized(state, [v])[None])[0, 1]
                self.assertLess(abs(actual - expected), 1e-14)

    def test_features_do_not_read_labels_and_schema(self):
        class ForbiddenLabel:
            def __array__(self, *args, **kwargs):
                raise AssertionError('labels must not be read by feature calculation')
        record = {'fps': 24., 'corrected': np.arange(56, dtype=np.float32).reshape(4, 14),
                  'labels': ForbiddenLabel()}
        features, names = audit.features_reference(np.asarray([.2, .4, .6, .8]), .5, record)
        self.assertEqual(len(names), 41)
        self.assertEqual(features.shape, (41,))
        self.assertTrue(np.isfinite(features).all())

    def test_duplicate_content_is_kept_together_and_rejected_across_split(self):
        records, _ = fixture()
        records[1]['sha256'] = records[0]['sha256']
        folds = audit.grouped_folds(records, list(range(36)), 3, 7)
        self.assertTrue(any(0 in f and 1 in f for f in folds))
        e = audit.Evidence(Path.cwd())
        with self.assertRaisesRegex(audit.AuditError, 'content leakage'):
            audit.check_split(e, records, [0, 1], [0], [1], 'duplicate')

    def test_deep_proof_and_outer_gate_fit_content(self):
        records, row = fixture()
        e = audit.Evidence(Path.cwd())
        audit.verify_partitions(e, records, 0, row['train'], row['validation'], row)
        bad = deepcopy(row)
        bad['inner_partitions'][0]['deep_partitions'][0]['member_fit_proof'][0]['seed'] += 1
        with self.assertRaisesRegex(audit.AuditError, 'deep member fit proof'):
            audit.verify_partitions(e, records, 0, row['train'], row['validation'], bad)
        bad = deepcopy(row)
        bad['outer_gate_states'][audit.GATES[0]]['fit_content_sha256'].append(records[30]['sha256'])
        with self.assertRaisesRegex(audit.AuditError, 'outer gate fit-content'):
            audit.verify_partitions(e, records, 0, row['train'], row['validation'], bad)

    def test_v5_v6_inner_partition_alignment(self):
        records, parent = fixture()
        row = {'fold': 0, 'recipe': audit.MEMBERS[0], 'train': parent['train'],
               'validation': parent['validation'], 'outer_seed': 20261101,
               'inner_partitions': [{k: p[k] for k in ['fit', 'validation']} | {'seed': 20261002 + j}
                                    for j, p in enumerate(parent['inner_partitions'])],
               'fit_evidence': [{'partition': j, 'fit_content_sha256': audit.content(records, p['fit'])}
                                for j, p in enumerate(parent['inner_partitions'])]
                               + [{'partition': 'outer', 'fit_content_sha256': audit.content(records, parent['train'])}]}
        e = audit.Evidence(Path.cwd())
        with patch.object(audit, 'load_asset', return_value=(row, {})):
            audit.verify_v5(e, Path.cwd(), {}, 0, audit.MEMBERS[0], parent, records)
        bad = deepcopy(row)
        bad['inner_partitions'][0], bad['inner_partitions'][1] = bad['inner_partitions'][1], bad['inner_partitions'][0]
        with patch.object(audit, 'load_asset', return_value=(bad, {})):
            with self.assertRaisesRegex(audit.AuditError, 'inner content partition'):
                audit.verify_v5(e, Path.cwd(), {}, 0, audit.MEMBERS[0], parent, records)

    def test_signature_npz_hash_and_output_nonoverwrite(self):
        with tempfile.TemporaryDirectory(prefix='gate-audit-test-') as directory:
            root = Path(directory).resolve()
            folder = root / 'training'
            folder.mkdir()
            npz = folder / '0_gates.npz'
            np.savez(npz, inner_frame_0=np.asarray([.2, .8]), inner_video_0=np.asarray(.5))
            row = {'training_receipt_sha256': 'receipt', 'npz_sha256': audit.file_hash(npz)}
            row['signature'] = audit.object_hash(row)
            audit.write_new(folder / '0_gates.json', row)
            e = audit.Evidence(root)
            records = [{'frames': 2}]
            keys = {'inner_frame_0', 'inner_video_0'}
            audit.load_asset(e, root, '0_gates', {'receipt_sha256': 'receipt'}, keys, records)
            with self.assertRaises(FileExistsError):
                audit.write_new(folder / '0_gates.json', {})
            with npz.open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(audit.AuditError, 'hash mismatch'):
                audit.load_asset(audit.Evidence(root), root, '0_gates', {'receipt_sha256': 'receipt'}, keys, records)
            with self.assertRaisesRegex(audit.AuditError, 'asset changed'):
                e.unchanged()

    def test_npz_coverage_rejects_extra_key(self):
        with tempfile.TemporaryDirectory(prefix='gate-audit-test-') as directory:
            root = Path(directory).resolve()
            (root / 'training').mkdir()
            path = root / 'training/0_gates.npz'
            np.savez(path, inner_frame_0=np.asarray([.2, .8]), unexpected_video_0=np.asarray(.5))
            row = {'training_receipt_sha256': 'receipt', 'npz_sha256': audit.file_hash(path)}
            row['signature'] = audit.object_hash(row)
            audit.write_new(path.with_suffix('.json'), row)
            with self.assertRaisesRegex(audit.AuditError, 'key coverage'):
                audit.load_asset(audit.Evidence(root), root, '0_gates', {'receipt_sha256': 'receipt'},
                                 {'inner_frame_0'}, [{'frames': 2}])

    def test_threshold_equality_and_interval_impact(self):
        def decode(frame, fps, vp, config):
            return [(0, len(frame) - 1)] if vp >= config['video_threshold'] else []
        api = {'duration': decode, 'recall': decode}
        configs = [{'kind': 'duration-logit-v4', 'video_threshold': .4},
                   {'kind': 'recall-stable-v2', 'video_threshold': 0}]
        samples = [{'fold': 0, 'inner': None, 'role': 'outer_validation', 'row': 0,
                    'old': .4, 'reference': .4 - 1e-7, 'frame': np.ones(3)}]
        result = audit.interval_impact(api, [{'fps': 24.}], configs, samples)
        self.assertEqual(result['threshold_flips']['0.4'], 1)
        self.assertEqual(result['threshold_flips']['0.0'], 0)
        self.assertEqual(result['changed_predictions'], 1)

    def test_independent_forest_reference(self):
        state = {'model': {'kind': 'forest', 'n_features': 1, 'trees': [
            {'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [0, -2, -2],
             'threshold': [.5, -2., -2.], 'value': [0., .2, .8]}]}}
        self.assertEqual(audit.forest_reference(state, [.5]), float(np.float32(.2)))
        self.assertEqual(audit.forest_reference(state, [.6]), float(np.float32(.8)))

    def test_extract_does_not_import_or_run_training(self):
        with tempfile.TemporaryDirectory(prefix='gate-audit-test-') as directory:
            path = Path(directory) / 'source.py'
            path.write_text('raise RuntimeError("top-level")\n'
                            'def fit():\n    raise RuntimeError("training")\n'
                            'def apply(x):\n    return x + 1\n', encoding='utf-8')
            ns = {}
            audit.extract_functions(path, ns, ['apply'])
            self.assertNotIn('fit', ns)
            self.assertEqual(ns['apply'](2), 3)


if __name__ == '__main__':
    unittest.main()
