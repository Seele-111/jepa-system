"""No-fit tests of independent V13R audit contracts and fail-closed behavior."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_spatial_mask_repair_v13r as R
from audit_spatial_mask_repair_v13r import A


def seal(row):
    row['signature'] = A.value_hash({k: v for k, v in row.items() if k != 'signature'})
    return row


def records_and_outer():
    records = []
    for i in range(12):
        y = np.array([0, 1, 1, 0], bool) if i % 3 else np.zeros(4, bool)
        records.append({'sha256': A.sha(str(i).encode()), 'frames': 4, 'fps': 10., 'labels': y,
                        'event_count': len(A.label_spans(y)), 'generator': 'fixture', 'name': str(i)})
    records[1]['sha256'] = records[0]['sha256']  # Exact-content alias, conflicting labels retained.
    groups = A.content_groups(records, list(range(len(records))))
    return records, A.seeded_folds(records, groups, 5, R.SEED)


def metadata(scope=0):
    records, outer = records_and_outer()
    val = [] if scope == 5 else outer[scope]
    train = [i for i in range(len(records)) if i not in val]
    parts = A.expected_partitions(records, train, scope)
    names = ['motion/a', 'v_spatial/positive_mass']
    receipt = {'receipt_sha256': 'a' * 64}
    parent = {'scope': scope, 'final_selection': scope == 5, 'recipe': R.RECIPE, 'train': train,
              'validation': val, 'inner_partitions': parts, 'npz_sha256': 'b' * 64}
    fits = parts + ([] if scope == 5 else [{'fit': train, 'validation': val, 'seed': R.SEED + scope * 53 + 99}])
    proofs = [{'partition': k if k < 3 else 'outer', 'seed': p['seed'],
               'fit_content_sha256': sorted({records[i]['sha256'] for i in p['fit']}),
               'transform_sha256': A.value_hash(R.repaired_transform(names)),
               'video_model_sha256': 'c' * 64, 'max_portable_error': 1e-8}
              for k, p in enumerate(fits)]
    row = {**parent, 'training_receipt_sha256': receipt['receipt_sha256'],
           'parent_frame_npz_sha256': parent['npz_sha256'], 'npz_sha256': 'd' * 64,
           'frame_probabilities_reused_unchanged': True, 'fit_evidence': proofs}
    return records, outer, receipt, parent, names, seal(row)


def raw_fixture():
    heat = np.array([[np.nan, 4, np.nan], [2, np.nan, 9]], np.float32)
    mask = np.isfinite(heat)
    a = {'frame_ids': np.arange(6), 'timestamps_sec': np.arange(6) / 10.,
         'tubelet_frame_ids': np.array([[0, 1], [4, 5]]), 'keyframe_ids': np.array([0, 5])}
    for p in ('vjepa', 'ijepa'):
        a[p + '_raw_heatmaps'] = np.stack([heat, heat * 2])
        a[p + '_patch_valid_mask'] = np.stack([mask, mask])
        a[p + '_patch_counts'] = np.stack([mask, mask]).astype(np.int64)
        a[p + '_valid_mask'] = np.array([True, True])
    return a


class HashAndIO(unittest.TestCase):
    def test_canonical_receipt_tampering_or_wrong_pin_is_rejected(self):
        receipt = {'parent': 'a' * 64}
        receipt['receipt_sha256'] = A.value_hash(receipt)
        R.signed_object(receipt, 'receipt_sha256', 'fixture', receipt['receipt_sha256'])
        changed = {**receipt, 'parent': 'b' * 64}
        with self.assertRaises(A.AuditError):
            R.signed_object(changed, 'receipt_sha256', 'fixture')
        with self.assertRaises(A.AuditError):
            R.signed_object(receipt, 'receipt_sha256', 'fixture', 'c' * 64)

    def test_bitwise_rejects_signed_zero_and_dtype(self):
        a = np.array([0., .5], np.float32)
        R.bitwise_equal(a, a.copy(), 'same')
        for b in (np.array([-0., .5], np.float32), a.astype(np.float64)):
            self.assertTrue(np.array_equal(a, b))
            with self.assertRaises(A.AuditError):
                R.bitwise_equal(a, b, 'numerically equal is not bitwise')

    def test_input_path_escape_and_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = A.Evidence(tmp)
            Path(tmp, 'input.json').write_text('{}', encoding='utf-8')
            for path in ('../outside', 'C:/outside', '/outside', 'a/../../b'):
                with self.assertRaises(A.AuditError):
                    e.read(path)
            with self.assertRaises(A.AuditError):
                e.read('input.json', 'a' * 64)

    def test_evidence_recheck_detects_concurrent_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp, 'input'); file.write_bytes(b'first')
            e = A.Evidence(tmp); e.read('input')
            file.write_bytes(b'changed')
            with self.assertRaises(A.AuditError):
                e.recheck()

    def test_exclusive_write_and_cli_overwrite_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, 'audit.json')
            R.write_exclusive(output, {'status': 'pending'})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                R.write_exclusive(output, {'status': 'passed'})
            with patch.object(R, 'run_audit') as audit:
                with self.assertRaises(FileExistsError):
                    R.main(['--output', str(output)])
                audit.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_pending_cli_is_not_success_and_complete_flag_returns_two(self):
        pending = {'status': 'pending', 'rows': 77, 'video_fit_count': 20,
                   'frame_probability_bitwise': {'status': 'pending'},
                   'missing_artifacts_at_start': ['final_selection/5.npz']}
        with tempfile.TemporaryDirectory() as tmp, patch.object(R, 'run_audit', return_value=pending):
            output = Path(tmp, 'pending.json')
            with redirect_stdout(StringIO()):
                rc = R.main(['--output', str(output), '--require-complete'])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(output.read_text('utf-8'))['status'], 'pending')
            self.assertNotEqual(R.pending_result(['final'])['status'], 'passed')

    def test_failure_cannot_write_success_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, 'failed.json')
            with patch.object(R, 'run_audit', side_effect=A.AuditError('tampered parent')):
                with self.assertRaises(A.AuditError):
                    R.main(['--output', str(output)])
            self.assertFalse(output.exists())

    def test_cli_does_not_create_missing_parent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(R, 'run_audit') as audit:
            with self.assertRaises(FileNotFoundError):
                R.main(['--output', str(Path(tmp, 'absent/audit.json'))])
            audit.assert_not_called()


class MaskSemantics(unittest.TestCase):
    def test_masked_coordinate_peak_not_compressed_flat_index(self):
        a = raw_fixture()
        s = R.reference_heat_statistics(a['vjepa_raw_heatmaps'][0], a['vjepa_patch_valid_mask'][0])
        np.testing.assert_array_equal(s[12:14], [1., 1.])
        self.assertEqual(s[0], .5)
        self.assertEqual(s[1], 15.)
        self.assertAlmostEqual(s[2], 8., places=6)
        self.assertAlmostEqual(s[3], 8.9, places=5)

    def test_real_anchor_center_support_and_FPS_alignment(self):
        values, mask, names, proof = R.reference_raw_spatial(raw_fixture(), 6, 10.)
        self.assertEqual(values.shape, (6, 34)); self.assertEqual(len(names), 34)
        np.testing.assert_array_equal(mask[:, 0], [False, True, True, True, True, False])
        self.assertTrue(mask[:, 17:].all())
        self.assertEqual(proof[0]['first_center_frame'], .5)
        self.assertEqual(proof[0]['last_center_frame'], 4.5)
        self.assertEqual(proof[0]['unsupported_frames'], 2)

    def test_patch_mask_counts_or_observed_mask_tamper_rejected(self):
        for field in ('vjepa_patch_counts', 'vjepa_valid_mask', 'vjepa_patch_valid_mask'):
            with self.subTest(field=field):
                a = raw_fixture()
                if field == 'vjepa_patch_counts': a[field][0, 0, 1] = 0
                elif field == 'vjepa_valid_mask': a[field][0] = False
                else: a[field] = a[field].astype(np.int64)
                with self.assertRaises(A.AuditError):
                    R.reference_raw_spatial(a, 6, 10.)

    def test_invalid_fps_timestamp_or_anchor_ids_rejected(self):
        for mutate in (lambda a: a['timestamps_sec'].__setitem__(2, .5),
                       lambda a: a['tubelet_frame_ids'].__setitem__((0, 0), 7),
                       lambda a: a.update(keyframe_ids=a['keyframe_ids'].astype(np.float64)),
                       lambda a: a.update(keyframe_ids=np.array([5, 0]))):
            a = raw_fixture(); mutate(a)
            with self.assertRaises(A.AuditError):
                R.reference_raw_spatial(a, 6, 10.)
        with self.assertRaises(A.AuditError):
            R.reference_raw_spatial(raw_fixture(), 6, 0.)

    def test_masked_mean_population_std_percentiles_and_empty(self):
        x = np.array([[1, 4], [2, 5], [10000, 6]], np.float32)
        mask = np.array([[True, False], [True, False], [False, False]])
        expected = np.array([[1.5, 0], [.5, 0], [1.1, 0], [1.9, 0]], np.float32)
        np.testing.assert_allclose(R.reference_video_statistics(x, mask).reshape(4, 2), expected, rtol=0, atol=1e-7)
        x[2, 0] = np.nan
        np.testing.assert_allclose(R.reference_video_statistics(x, mask).reshape(4, 2), expected, rtol=0, atol=1e-7)
        x[0, 0] = np.nan
        with self.assertRaises(A.AuditError):
            R.reference_video_statistics(x, mask)

    def test_no_fit_label_free_frame_and_video_invariance(self):
        from test_spatial_jepa_v13 import spatial_record
        from optimized_spatial_jepa_v13 import feature_view_spatial
        from optimized_spatial_jepa_v13r import feature_view_spatial_repaired
        record = spatial_record(0)
        record['spatial_support'][:3] = False
        clean = {k: v for k, v in record.items() if k not in R.FORBIDDEN}
        before = feature_view_spatial_repaired(record)
        after = feature_view_spatial_repaired(clean)
        for k in (0, 1): R.bitwise_equal(before[k], after[k], 'no labels/identity/RGB')
        R.bitwise_equal(after[0], feature_view_spatial(R.RECIPE, clean)[0], 'frame frozen')
        changed = {**clean, 'spatial': clean['spatial'].copy()}
        changed['spatial'][~clean['spatial_support']] += 10000
        modified = feature_view_spatial_repaired(changed)
        for k in (0, 1): R.bitwise_equal(after[k], modified[k], 'unsupported invariance')


class PartitionAndProvenance(unittest.TestCase):
    def test_all23_fit_partitions_seeds_and_transform_digests(self):
        count = 0
        for scope in range(6):
            records, outer, receipt, parent, names, row = metadata(scope)
            proofs = R.verify_repair_metadata(row, scope, records, outer, receipt, parent, names)
            count += len(proofs)
            for proof in proofs:
                self.assertEqual(proof['estimator_random_state'], proof['seed'] + 1100)
                self.assertEqual(proof['model_state_status'], 'recorded_digest_only_not_replayed')
        self.assertEqual(count, 23)

    def test_resigned_seed_content_transform_or_parent_tamper_rejected(self):
        edits = [lambda r: r['fit_evidence'][0].update(seed=r['fit_evidence'][0]['seed'] + 1),
                 lambda r: r['fit_evidence'][0].update(fit_content_sha256=[]),
                 lambda r: r['fit_evidence'][0].update(transform_sha256='f' * 64),
                 lambda r: r.update(parent_frame_npz_sha256='f' * 64),
                 lambda r: r.update(training_receipt_sha256='f' * 64),
                 lambda r: r.update(frame_probabilities_reused_unchanged=False),
                 lambda r: r['fit_evidence'][0].update(partition='outer'),
                 lambda r: r['fit_evidence'][0].update(video_model_sha256='invalid')]
        for edit in edits:
            records, outer, receipt, parent, names, row = metadata()
            edit(row); seal(row)
            with self.assertRaises(A.AuditError):
                R.verify_repair_metadata(row, 0, records, outer, receipt, parent, names)

    def test_recorded_portable_error_nan_negative_or_large_rejected(self):
        for err in (float('nan'), -.1, 3e-6):
            records, outer, receipt, parent, names, row = metadata()
            row['fit_evidence'][0]['max_portable_error'] = err
            if np.isfinite(err): seal(row)
            with self.assertRaises((A.AuditError, ValueError)):
                R.verify_repair_metadata(row, 0, records, outer, receipt, parent, names)

    def test_same_SHA_alias_cannot_cross_fit_predict(self):
        records, _ = records_and_outer()
        self.assertFalse(np.array_equal(records[0]['labels'], records[1]['labels']))
        with self.assertRaisesRegex(A.AuditError, 'SHA content leakage'):
            A.check_partition(records, [0] + list(range(2, 12)), [1], list(range(12)), 'alias')
        np.testing.assert_array_equal(A.content_weights(records, [0, 1]), [.5, .5])

    def test_wrong_training_jobs_or_new_frame_fit_claim_rejected(self):
        row = {'status': 'complete', 'new_video_fits': 3, 'new_frame_fits': 0,
               'results': [{'scope': 5, 'final_selection': True, 'seconds': 1.}]}
        self.assertEqual(R.verify_training_report(row, True)['status'], 'passed')
        for edit in (lambda r: r.update(new_frame_fits=1),
                     lambda r: r['results'][0].update(scope=4),
                     lambda r: r.update(new_video_fits=20)):
            changed = deepcopy(row); edit(changed)
            with self.assertRaises(A.AuditError): R.verify_training_report(changed, True)


class CacheBinding(unittest.TestCase):
    def check_cache(self, tmp, scope=0, mutate=None, update_digest=True):
        records, outer, receipt, parent, names, row = metadata(scope)
        arrays, maps = {}, []
        for kind, ids in (('inner', row['train']), ('outer', row['validation'])):
            frame = {i: np.array([0, .2, .6, .8], np.float32) for i in ids}
            video = {i: .7 for i in ids}
            maps.append(A.Probabilities(frame, video))
            for i in ids:
                arrays[f'{kind}_frame_{i}'] = frame[i].copy()
                arrays[f'{kind}_video_{i}'] = np.asarray(video[i])
        if mutate: mutate(arrays, row)
        buf = BytesIO(); np.savez_compressed(buf, **arrays); data = buf.getvalue()
        if update_digest: row['npz_sha256'] = A.sha(data)
        seal(row)
        path = Path(tmp) / R.artifact_path(R.OUT, scope)
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        path.with_suffix('.json').write_text(json.dumps(row), encoding='utf-8')
        return R.verify_repair_cache(A.Evidence(tmp), scope, records, outer, receipt, tuple(maps), parent, names)

    def test_valid_nested_and_final_cache_and_exact_counts(self):
        total = Counter()
        with tempfile.TemporaryDirectory() as tmp:
            for scope in range(6):
                ip, op, row, proofs, counts = self.check_cache(tmp, scope)
                self.assertEqual(len(proofs), 3 if scope == 5 else 4)
                total.update(counts)
                if scope == 5:
                    self.assertFalse(op.frame)
                    self.assertEqual(set(ip.frame), set(range(12)))
        self.assertEqual(total['inner'], 48)
        self.assertEqual(total['outer'], 12)
        self.assertEqual(total['final_inner'], 12)

    def test_signed_zero_or_dtype_difference_rejected_even_when_resigned(self):
        for change in ('zero', 'dtype'):
            with tempfile.TemporaryDirectory() as tmp:
                def mutate(arrays, row):
                    key = next(k for k in arrays if k.startswith('inner_frame_'))
                    if change == 'zero': arrays[key][0] = -0.
                    else: arrays[key] = arrays[key].astype(np.float64)
                with self.assertRaisesRegex(A.AuditError, 'bitwise mismatch'):
                    self.check_cache(tmp, mutate=mutate)

    def test_fullfit_keys_missing_keys_or_bad_probability_rejected(self):
        for change in ('fullfit', 'missing', 'invalid'):
            with tempfile.TemporaryDirectory() as tmp:
                def mutate(arrays, row):
                    key = next(k for k in arrays if k.startswith('inner_frame_'))
                    if change == 'fullfit': arrays['full_frame_0'] = np.zeros(4, np.float32)
                    elif change == 'missing': arrays.pop(key)
                    else: arrays[key][0] = 1.01
                with self.assertRaises(A.AuditError): self.check_cache(tmp, 5, mutate=mutate)

    def test_npz_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(A.AuditError):
            self.check_cache(tmp, update_digest=False)


class IndependentSelection(unittest.TestCase):
    def test_candidate_tie_order_and_reversed_candidate_order(self):
        rows = [{'candidate': c, 'stable_utility': 1., 'key': [1., 1., 0, 0, -1, 0]}
                for c in (R.RECIPE, 'control')]
        self.assertEqual(R.rank_spatial(rows)['candidate'], R.RECIPE)
        with self.assertRaises(A.AuditError): R.rank_spatial(rows[::-1])

    def test_choice_reads_only_supplied_inner_records_and_final_seed(self):
        class Poison:
            def __getitem__(self, key): raise AssertionError('held-out record read')
        records, _ = records_and_outer()
        selected_records = [records[2], records[3], Poison()]
        p = A.Probabilities({0: np.array([0, .9, .9, 0]), 1: np.zeros(4)}, {0: .9, 1: .1})
        def decoder(frame, fps, video, cfg):
            return [(1, 2)] if video > cfg.get('video_threshold', 0) and max(frame) >= cfg['threshold'] else []
        with patch.object(A, 'independent_choice', wraps=A.independent_choice) as choose:
            chosen, rows = R.independent_rows(selected_records, [0, 1], p, p, 5, (decoder, decoder))
        self.assertEqual(choose.call_count, 2)
        self.assertEqual(chosen['candidate'], R.RECIPE)
        self.assertEqual(len(rows), 2)
        for call in choose.call_args_list:
            self.assertEqual(call.args[1], [0, 1]); self.assertEqual(call.args[3], 20261007)

    def test_extra_outer_or_fullfit_probability_not_accepted_as_inner(self):
        records, _ = records_and_outer()
        p = A.Probabilities({0: np.zeros(4), 1: np.zeros(4)}, {0: .1, 1: .1})
        with self.assertRaises(A.AuditError):
            R.independent_rows(records, [0], p, p, 0, (None, None))

    def test_no_production_runner_imports_or_choose_calls_in_auditor(self):
        tree = ast.parse(Path(R.__file__).read_text('utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name.startswith('run_') for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('run_'))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ('choose', 'select_all', 'final_select', 'fit'))


if __name__ == '__main__':
    unittest.main()
