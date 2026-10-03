"""Substantive tamper and golden-statistic tests; no model training or fixtures in outputs."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import audit_feature_blocks_v10 as A


def records_fixture():
    records = []
    for i in range(12):
        labels = ([0] * 8 if i % 3 == 0 else
                  [0, 1, 1, 0, 0, 0, 0, 0] if i % 3 == 1 else
                  [1, 1, 0, 0, 1, 1, 0, 0])
        records.append({'sha256': f'{i + 1:064x}', 'name': f'row-{i}', 'frames': 8, 'fps': 10.,
            'generator': 'g' + str(i % 2), 'labels': np.asarray(labels, bool),
            'event_count': len(A.label_spans(labels)),
            'rgb': (np.arange(48).reshape(8, 6) * .1 + i).astype(np.float32)})
    records.append({**records[1], 'name': 'alias-1', 'labels': np.array([0, 0, 1, 0, 0, 0, 0, 0], bool)})
    return records


def sign(row):
    row.pop('signature', None)
    row['signature'] = A.value_hash(row)
    return row


def metadata_fixture(records, outer, *, scope=0, final=False, parent=False):
    val = [] if final else outer[scope]
    train = [i for i in range(len(records)) if i not in set(val)]
    parts = A.expected_partitions(records, train, scope)
    fit_parts = parts + ([] if final else [{'fit': train, 'validation': val}])
    row = {'recipe': A.PARENT_RICH if parent else A.RECIPES[0],
        'fold' if parent else 'scope': scope, 'train': train, 'validation': val,
        'inner_partitions': parts, 'outer_seed': None if final else A.SEED + scope * 53 + 99,
        'training_receipt_sha256': A.PARENT_PIN if parent else A.RECEIPT_PIN,
        'npz_sha256': 'a' * 64, 'fit_evidence': []}
    if not parent:
        row['final_selection'] = final
    for k, part in enumerate(fit_parts):
        row['fit_evidence'].append({'partition': k if k < 3 else 'outer',
            'fit_content_sha256': sorted({records[i]['sha256'] for i in part['fit']}),
            'transform_sha256': 'b' * 64, 'frame_features': 2, 'video_features': 2})
    return sign(row)


def arrays_fixture(records, metadata):
    arrays = {}
    for kind, ids in (('inner', metadata['train']), ('outer', metadata['validation'])):
        for i in ids:
            arrays[f'{kind}_frame_{i}'] = np.linspace(.1, .8, records[i]['frames'], dtype=np.float32)
            arrays[f'{kind}_video_{i}'] = np.asarray(.5)
    return arrays


def npz_bytes(arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


class EvidenceTests(unittest.TestCase):
    def test_duplicate_json_and_nonfinite_constants_rejected(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(data=data), self.assertRaises(A.AuditError):
                A.strict_json(data, 'fixture')

    def test_paths_cannot_escape_windows_or_posix_root(self):
        with tempfile.TemporaryDirectory() as temp:
            for value in ('../x', '/etc/x', 'C:/x', 'C:\\x', 'a/../../x', 'a\\x', 'a:x'):
                with self.subTest(value=value), self.assertRaises(A.AuditError):
                    A.safe_path(temp, value)

    def test_hash_and_recheck_catch_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'source.py'
            target.write_bytes(b'original')
            evidence = A.Evidence(temp)
            self.assertEqual(evidence.read('source.py', A.sha(b'original')), b'original')
            target.write_bytes(b'altered')
            with self.assertRaisesRegex(A.AuditError, 'SHA256 mismatch'):
                evidence.recheck()

    def test_missing_input_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(A.MissingArtifact):
                A.run_audit(temp)

    def test_missing_deployment_not_partial_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / A.OUT
            path.mkdir(parents=True)
            (path / 'report.json').write_text('{"status":"complete"}', encoding='utf-8')
            with self.assertRaisesRegex(A.MissingArtifact, 'deployment_selection'):
                A.run_audit(temp)

    def test_rewritten_self_signed_receipt_cannot_replace_trust_anchor(self):
        row = {'sources': {}, 'receipt_sha256': 'bad'}
        row['receipt_sha256'] = A.value_hash({'sources': {}})
        class Fake:
            def json(self, *_args):
                return row
        with self.assertRaisesRegex(A.AuditError, 'trust anchor'):
            A.verify_receipts(Fake())

    def test_duplicate_archive_members_fail(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('x.npy', b'one')
            # Duplicate keys are a substantive overwrite/coverage ambiguity.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr('x.npy', b'two')
        with self.assertRaisesRegex(A.AuditError, 'duplicate'):
            A.read_archive(buffer.getvalue(), 'duplicate')

    def test_object_and_oversized_header_fail_before_np_load(self):
        object_data = npz_bytes({'x': np.array([{'unsafe': True}], object)})
        with self.assertRaises(A.AuditError):
            A.read_archive(object_data, 'object')
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header,
            {'descr': '<f8', 'fortran_order': False, 'shape': (10 ** 12,)})
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('huge.npy', header.getvalue())
        with patch.object(A.np, 'load', side_effect=AssertionError('must not allocate')):
            with self.assertRaisesRegex(A.AuditError, 'bound'):
                A.read_archive(buffer.getvalue(), 'huge')

    def test_boolean_is_not_a_numeric_metadata_identity(self):
        with self.assertRaises(A.AuditError):
            A.compare(True, 1, 'coverage')
        with self.assertRaises(A.AuditError):
            A.compare(float('nan'), .3, 'utility')


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.records = records_fixture()
        self.ids = list(range(len(self.records)))
        self.outer = A.seeded_folds(self.records, A.content_groups(self.records, self.ids), 5, A.SEED)
        self.metadata = metadata_fixture(self.records, self.outer)

    def verify(self, row, *, parent=False, final=False):
        return A.verify_training_metadata(row, A.RICH if parent else A.RECIPES[0],
            5 if final else 0, self.records, self.outer, parent=parent, final=final)

    def test_valid_nested_parent_and_final_metadata(self):
        self.verify(self.metadata)
        self.verify(metadata_fixture(self.records, self.outer, parent=True), parent=True)
        self.verify(metadata_fixture(self.records, self.outer, scope=5, final=True), final=True)

    def test_alias_split_rejected_even_with_disjoint_row_ids(self):
        with self.assertRaisesRegex(A.AuditError, 'SHA content leakage'):
            A.check_partition(self.records, [1], [12], [1, 12], 'alias')

    def test_each_inner_row_once_and_content_aliases_together(self):
        parts = A.expected_partitions(self.records, self.ids, 5)
        self.assertEqual(sorted(i for part in parts for i in part['validation']), self.ids)
        self.assertEqual(sum(1 for part in parts if 1 in part['validation'] and 12 in part['validation']), 1)
        for part in parts:
            self.assertFalse({self.records[i]['sha256'] for i in part['fit']} &
                             {self.records[i]['sha256'] for i in part['validation']})

    def test_resigned_partition_seed_or_coverage_tampering_rejected(self):
        for field in ('seed', 'duplicate', 'hole', 'train', 'outer_seed'):
            row = deepcopy(self.metadata)
            if field == 'seed':
                row['inner_partitions'][0]['seed'] += 1
            elif field == 'duplicate':
                row['inner_partitions'][0]['validation'].append(row['inner_partitions'][0]['validation'][0])
            elif field == 'hole':
                row['inner_partitions'][0]['validation'].pop()
            elif field == 'train':
                row['train'] = row['train'][1:]
            else:
                row['outer_seed'] += 1
            with self.subTest(field=field), self.assertRaises(A.AuditError):
                self.verify(sign(row))

    def test_resigned_fit_content_or_partition_proof_tampering_rejected(self):
        for field in ('content', 'partition', 'count'):
            row = deepcopy(self.metadata)
            if field == 'content':
                row['fit_evidence'][0]['fit_content_sha256'][0] = 'f' * 64
            elif field == 'partition':
                row['fit_evidence'][0]['partition'] = 1
            else:
                row['fit_evidence'].pop()
            with self.subTest(field=field), self.assertRaises(A.AuditError):
                self.verify(sign(row))

    def test_receipt_and_signature_tampering_rejected(self):
        row = deepcopy(self.metadata)
        row['signature'] = 'c' * 64
        with self.assertRaisesRegex(A.AuditError, 'signature'):
            self.verify(row)
        row = deepcopy(self.metadata)
        row['training_receipt_sha256'] = A.PARENT_PIN
        with self.assertRaisesRegex(A.AuditError, 'receipt'):
            self.verify(sign(row))

    def test_final_cache_cannot_claim_an_outer_validation(self):
        row = metadata_fixture(self.records, self.outer, scope=5, final=True)
        row['validation'] = [0]
        with self.assertRaises(A.AuditError):
            self.verify(sign(row), final=True)


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.records = records_fixture()
        ids = list(range(len(self.records)))
        outer = A.seeded_folds(self.records, A.content_groups(self.records, ids), 5, A.SEED)
        self.metadata = metadata_fixture(self.records, outer)
        self.arrays = arrays_fixture(self.records, self.metadata)
        self.frame = next(k for k in self.arrays if k.startswith('inner_frame_'))
        self.video = self.frame.replace('frame', 'video')

    def load(self, arrays, metadata=None):
        data = npz_bytes(arrays)
        row = deepcopy(self.metadata if metadata is None else metadata)
        row['npz_sha256'] = A.sha(data)
        return A.probability_maps(data, row, self.records, 'fixture')

    def test_valid_probability_heads_exact_coverage(self):
        inner, outer = self.load(self.arrays)
        self.assertEqual(set(inner.frame), set(self.metadata['train']))
        self.assertEqual(set(inner.video), set(inner.frame))
        self.assertEqual(set(outer.frame), set(self.metadata['validation']))

    def test_npz_byte_tamper_not_hidden_by_metadata_signature(self):
        data = npz_bytes(self.arrays)
        row = {**self.metadata, 'npz_sha256': A.sha(data)}
        with self.assertRaisesRegex(A.AuditError, 'SHA256'):
            A.probability_maps(data + b'tamper', row, self.records, 'fixture')

    def test_rehashed_invalid_probabilities_are_rejected(self):
        cases = {
            'wrong_length': np.array([.5], np.float32),
            'nan': np.full(8, np.nan), 'infinity': np.full(8, np.inf),
            'negative': np.full(8, -.01), 'over_one': np.full(8, 1.01),
            'boolean': np.ones(8, bool), 'complex': np.ones(8, complex),
        }
        for name, bad in cases.items():
            arrays = dict(self.arrays); arrays[self.frame] = bad
            with self.subTest(name=name), self.assertRaises(A.AuditError):
                self.load(arrays)

    def test_rehashed_wrong_video_scalar_shape_is_rejected(self):
        arrays = dict(self.arrays); arrays[self.video] = np.array([.5])
        with self.assertRaisesRegex(A.AuditError, 'probability tensor'):
            self.load(arrays)

    def test_missing_or_extra_probability_key_is_rejected(self):
        arrays = dict(self.arrays); arrays.pop(self.frame)
        with self.assertRaisesRegex(A.AuditError, 'OOF coverage'):
            self.load(arrays)
        arrays = {**self.arrays, 'outer_frame_999': np.ones(8)}
        with self.assertRaisesRegex(A.AuditError, 'OOF coverage'):
            self.load(arrays)

    def test_final_probability_has_no_outer_or_extra_row(self):
        metadata = {'train': list(range(len(self.records))), 'validation': []}
        arrays = arrays_fixture(self.records, metadata)
        inner, outer = self.load(arrays, metadata)
        self.assertFalse(outer.frame or outer.video)
        arrays['outer_video_0'] = np.asarray(.5)
        with self.assertRaisesRegex(A.AuditError, 'OOF coverage'):
            self.load(arrays, metadata)


class StatisticTests(unittest.TestCase):
    def test_greedy_match_first_gt_on_equal_iou_and_prediction_order(self):
        labels = np.array([1, 1, 0, 1, 1], bool)
        first = A.row_statistics([(0, 4), (0, 1)], labels)
        reverse = A.row_statistics([(0, 1), (0, 4)], labels)
        self.assertEqual(first[:6].tolist(), [1, 1, 1, 1, 1, 1])
        self.assertEqual(reverse[:6].tolist(), [2, 0, 0, 1, 1, 1])

    def test_inclusive_iou_point_five_threshold(self):
        result = A.row_statistics([(0, 3)], [1, 1, 0, 0])
        self.assertEqual(result.tolist(), [1, 0, 0, 1, 0, 0, 2, 2, 0, 0, 0, 1])

    def test_overlap_is_union_for_frames_not_double_counted(self):
        result = A.row_statistics([(0, 1), (0, 1)], [1, 1, 0, 0])
        self.assertEqual(result.tolist(), [1, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0, 2])

    def test_normal_false_positive_and_positive_empty(self):
        self.assertEqual(A.row_statistics([(1, 2)], [0] * 4).tolist(),
                         [0, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0, 1])
        self.assertEqual(A.row_statistics([], [0, 1, 1, 0]).tolist(),
                         [0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0])

    def test_invalid_intervals_fail_rather_than_silently_clip(self):
        for pred in ([(-1, 1)], [(1, 4)], [(2, 1)], [(True, 1)], [(1.5, 2)]):
            with self.subTest(pred=pred), self.assertRaises(A.AuditError):
                A.row_statistics(pred, [0] * 4)

    def test_utility_fixed_coefficients_and_zero_strata(self):
        stats = [2, 1, 1, 1, 2, 2, 3, 1, 2, 1, 1, 4]
        self.assertAlmostEqual(float(A.utility(stats, 2, 3)), 4 / 15, places=14)
        self.assertEqual(float(A.utility(np.zeros(12), 0, 0)), 0.)

    def test_grouped_metrics_empty_subsets_and_serialized_prediction_tamper(self):
        records = records_fixture()[:2]
        pred = {0: [], 1: [(1, 2)]}
        result = A.grouped_metrics(records, pred)
        self.assertEqual(result['all']['iou_0.5']['f1'], 1.)
        self.assertEqual(result['multi_event']['predicted_segments'], 0)
        rows = A.prediction_rows(records, pred)
        self.assertEqual(A.read_prediction_rows(rows, records, 'golden'), pred)
        rows[1]['is_normal'] = True
        with self.assertRaises(A.AuditError):
            A.read_prediction_rows(rows, records, 'tampered')
        broken = deepcopy(result); broken['all']['iou_0.5']['tp'] += 1
        with self.assertRaises(A.AuditError):
            A.compare(broken, result, 'metric')

    def test_guards_cannot_be_relaxed_via_pass_flag(self):
        records = records_fixture()[:2]
        base = A.grouped_metrics(records, {0: [], 1: [(1, 2)]})
        guards = A.promotion_guards(base, base)
        self.assertEqual(guards, {'event_f1_03_no_worse': True, 'event_f1_05_improved_02': False,
            'frame_f1_drop_at_most_005': True, 'normal_fp_no_worse': True, 'positive_empty_improved_3': False})
        with self.assertRaises(A.AuditError):
            A.compare({key: True for key in guards}, guards, 'guards')


class SelectionTests(unittest.TestCase):
    def single_record(self, labels):
        return {'sha256': '1' * 64, 'name': 'one', 'labels': np.array(labels, bool),
                'event_count': len(A.label_spans(labels)), 'frames': len(labels), 'fps': 10.}

    def test_bootstrap_golden_counts_and_alias_content_mass(self):
        records = [{'sha256': str(i), 'event_count': 1} for i in range(3)]
        records.append(dict(records[0]))
        counts = A.bootstrap_counts(records, [0, 1, 2, 3], 123, 4)
        self.assertEqual(counts.tolist(), [[1, 1, 1, 1], [2, 0, 1, 2], [2, 1, 0, 2], [1, 1, 1, 1]])
        self.assertEqual(A.content_weights(records, [0, 1, 2, 3]).tolist(), [.5, 1., 1., .5])
        np.testing.assert_array_equal(counts @ A.content_weights(records, [0, 1, 2, 3]), np.full(4, 3.))

    def test_prediction_dedup_retains_first_config_ordinal(self):
        records = [self.single_record([0, 1, 1, 0])]
        p = A.Probabilities({0: np.zeros(4)}, {0: .5})
        def same(*_args):
            return [(1, 2)]
        result = A.independent_choice(records, [0], p, A.SEED, (same, same))
        self.assertEqual(len(result['shortlist']), 1)
        self.assertEqual(result['config'], A.configurations()[0])
        self.assertEqual(result['key'][-1], 0)
        self.assertAlmostEqual(result['stable_utility'], 1.1)

    def test_top8_and_config_tie_order_golden(self):
        labels = [0] * 16; labels[7] = labels[8] = 1
        records = [self.single_record(labels)]
        p = A.Probabilities({0: np.zeros(16)}, {0: .5})
        configs = [{'kind': 'duration-logit-v4', 'ordinal': i} for i in range(16)]
        def point(_frame, _fps, _video, cfg):
            return [(cfg['ordinal'], cfg['ordinal'])]
        result = A.independent_choice(records, [0], p, A.SEED, (point, point), configs)
        self.assertEqual([r['config']['ordinal'] for r in result['shortlist']], [7, 8, 0, 1, 2, 3, 4, 5])
        self.assertEqual(result['config']['ordinal'], 7)
        self.assertAlmostEqual(result['bootstrap_q20'], 1. + 1. / 15)

    def test_q20_and_stable_utility_not_pooled_only(self):
        records = [{**self.single_record([0, 1, 0, 0]), 'sha256': str(i)} for i in range(3)]
        p = A.Probabilities({i: np.full(4, i, np.float32) for i in range(3)}, {i: .5 for i in range(3)})
        def proposal(frame, *_args):
            return ([(1, 1)], [], [(3, 3)])[int(frame[0])]
        result = A.independent_choice(records, [0, 1, 2], p, A.SEED, (proposal, proposal),
                                      [{'kind': 'duration-logit-v4'}])
        self.assertAlmostEqual(result['pooled_utility'], .34, places=14)
        self.assertAlmostEqual(result['bootstrap_q20'], -.1, places=14)
        self.assertAlmostEqual(result['stable_utility'], .23, places=14)

    def test_recipe_tie_order_and_declared_secondary_key(self):
        rows = [{'candidate': r, 'stable_utility': .4, 'key': [.4, .5, -1., -2., -3., -4]}
                for r in A.RECIPES]
        self.assertEqual(A.rank_candidates(rows)['candidate'], A.RECIPES[0])
        rows[2]['key'][1] = .6
        self.assertEqual(A.rank_candidates(rows)['candidate'], A.RECIPES[2])
        with self.assertRaises(A.AuditError):
            A.rank_candidates(list(reversed(rows)))

    def test_no_held_out_records_can_enter_selection(self):
        records = [self.single_record([0, 1, 0, 0]), self.single_record([0, 0, 1, 0])]
        p = A.Probabilities({0: np.zeros(4)}, {0: .5})
        def malicious(reader, *_args):
            return reader[1]
        with patch.object(A, 'decode', side_effect=malicious):
            with self.assertRaisesRegex(A.AuditError, 'held-out'):
                A.independent_choice(records, [0], p, A.SEED, (None, None))

    def test_inner_selection_requires_every_recipe_and_row(self):
        records = [self.single_record([0, 1, 0, 0])]
        p = A.Probabilities({0: np.zeros(4)}, {})
        with self.assertRaisesRegex(A.AuditError, 'probability coverage'):
            A.independent_choice(records, [0], p, A.SEED, (None, None))
        with self.assertRaisesRegex(A.AuditError, 'recipe coverage'):
            A.select_inner(records, [0], {}, 5, (None, None))

    def test_final_selection_tampering_including_outer_override_fails(self):
        chosen = {'candidate': A.RECIPES[0], 'stable_utility': .23, 'bootstrap_q20': -.1,
                  'config': {'kind': 'duration-logit-v4'}, 'shortlist': []}
        rows = [chosen]
        parts = [{'fit': [1], 'validation': [0], 'seed': 12}]
        outer = [A.RECIPES[0]] * 5
        saved = {'status': 'complete', 'role': 'inner3_only_deployment_choice_not_validation',
            'uses_outer_probabilities': False, 'uses_outer_metrics_for_selection': False,
            'selection_function': 'run_feature_blocks_v10.select_inner', 'coverage_per_row': 1,
            'inner_partitions': parts, 'chosen': chosen, 'all_inner_selections': rows,
            'outer_choices_diagnostic_only': outer, 'same_choice_all_outer': True,
            'training_receipt_sha256': A.RECEIPT_PIN}
        A.verify_deployment_selection(saved, chosen, rows, parts, outer)
        for mutation in ('candidate', 'q20', 'utility', 'all_rows', 'partition', 'uses_outer', 'coverage'):
            broken = deepcopy(saved)
            if mutation == 'candidate':
                broken['chosen']['candidate'] = A.RECIPES[-1]
            elif mutation == 'q20':
                broken['chosen']['bootstrap_q20'] = .4
            elif mutation == 'utility':
                broken['chosen']['stable_utility'] = .34
            elif mutation == 'all_rows':
                broken['all_inner_selections'] = []
            elif mutation == 'partition':
                broken['inner_partitions'][0]['seed'] += 1
            elif mutation == 'uses_outer':
                broken['uses_outer_metrics_for_selection'] = True
            else:
                broken['coverage_per_row'] = 4
            with self.subTest(mutation=mutation), self.assertRaises(A.AuditError):
                A.verify_deployment_selection(broken, chosen, rows, parts, outer)


class PCAEvidenceTests(unittest.TestCase):
    def test_pca_covariance_only_reads_fit_rgb_and_content_not_labels(self):
        class LabelForbidden(dict):
            def __getitem__(self, key):
                if key == 'labels':
                    raise AssertionError('PCA accessed labels')
                return super().__getitem__(key)
        records = [LabelForbidden(sha256='a', rgb=np.array([[0, 0], [2, 0]], np.float32)),
                   LabelForbidden(sha256='b', rgb=np.array([[4, 0], [6, 0]], np.float32)),
                   LabelForbidden(sha256='c', rgb=np.full((2, 2), 10 ** 6, np.float32))]
        result = A.reconstruct_pca(records, [0, 1])
        self.assertEqual(result['mean'], [3., 0.])
        self.assertEqual(result['components'], [[1., 0.], [0., 1.]])
        records[2]['rgb'] *= -100
        self.assertEqual(result, A.reconstruct_pca(records, [0, 1]))
        # An alias must not double its content's covariance mass.
        records.append(dict(records[0]))
        self.assertEqual(result, A.reconstruct_pca(records, [0, 1, 3]))

    def test_transform_digest_and_feature_dimensions_are_verified(self):
        proof = A.TransformEvidence.__new__(A.TransformEvidence)
        proof.records = [{'sha256': 'a', 'rgb': np.eye(2, dtype=np.float32)}]
        proof.pcas, proof.proofs = {}, []
        proof.fresh = lambda *_args: (np.ones((2, 2)), np.ones(2), ['motion/x', 'local/y'])
        evidence = {'partition': 0, 'transform_sha256': A.value_hash({
            'feature_view': 'compact-feature-blocks-v10', 'pca': None,
            'frame_feature_names': ['motion/x', 'local/y']}), 'frame_features': 2, 'video_features': 2}
        proof.verify(A.RECIPES[0], [0], evidence)
        self.assertEqual(proof.proofs[0]['PCA'], 'absent')
        for key, value in (('transform_sha256', 'f' * 64), ('frame_features', 3), ('video_features', 3)):
            broken = {**evidence, key: value}
            with self.subTest(key=key), self.assertRaises(A.AuditError):
                proof.verify(A.RECIPES[0], [0], broken)

    def test_rich_feature_or_name_mismatch_cannot_be_hidden(self):
        proof = A.TransformEvidence.__new__(A.TransformEvidence)
        proof.records = [{'sha256': 'a'}]; proof.pcas = {(0,): {'mean': [0]}}
        good = (np.ones((2, 1)), np.ones(1), ['x'])
        proof.fresh = lambda *_args: good
        proof.legacy = lambda *_args: good
        self.assertTrue(proof.compatibility([0])['frame_and_video_features_bitwise_equal'])
        for bad in ((np.zeros((2, 1)), np.ones(1), ['x']),
                    (np.ones((2, 1)), np.zeros(1), ['x']),
                    (np.ones((2, 1)), np.ones(1), ['changed'])):
            proof.legacy = lambda *_args: bad
            with self.assertRaises(A.AuditError):
                proof.compatibility([0])


class CLITests(unittest.TestCase):
    def test_missing_inputs_write_failed_report_and_exit_two(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'audit-new.json'
            with patch.object(A, 'run_audit', side_effect=A.MissingArtifact('missing deployment_selection.json')):
                self.assertEqual(A.main(['--output', str(target)]), 2)
            report = json.loads(target.read_text('utf-8'))
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error_kind'], 'missing_artifact')
            self.assertFalse(report['deployment_approved'])

    def test_output_is_exclusive_and_does_not_run_again(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'already.json'
            target.write_bytes(b'original')
            with patch.object(A, 'run_audit') as replay:
                self.assertEqual(A.main(['--output', str(target)]), 2)
                replay.assert_not_called()
            self.assertEqual(target.read_bytes(), b'original')

    def test_reserved_output_cannot_impersonate_frozen_input(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'deployment_selection.json'
            with patch.object(A, 'run_audit') as replay:
                self.assertEqual(A.main(['--output', str(target)]), 2)
                replay.assert_not_called()
            self.assertFalse(target.exists())

    def test_inconsistent_selection_and_unexpected_runtime_exit_one(self):
        with tempfile.TemporaryDirectory() as temp:
            for j, error in enumerate((A.AuditError('selection differs'), RuntimeError('decoder failed'))):
                target = Path(temp) / f'failed-{j}.json'
                with patch.object(A, 'run_audit', side_effect=error):
                    self.assertEqual(A.main(['--output', str(target)]), 1)
                self.assertEqual(json.loads(target.read_text('utf-8'))['status'], 'failed')




class V11Tests(unittest.TestCase):
    def record(self, normal=False, name='row'):
        y = np.zeros(8, bool)
        if not normal:
            y[1:3] = True
        return {'name': name, 'sha256': name, 'labels': y, 'event_count': 0 if normal else 1,
                'frames': len(y), 'fps': 10.}

    def runtime_fixture(self):
        from test_optimized_compact_v4 import record
        from optimized_feature_blocks_v10 import feature_view_blocks
        from optimized_compact_features_v4 import feature_view_v4
        r = record(frames=8); r['fps'] = 10.
        # Fixed identity projection is a fixture, NOT a call to a model/PCA fitter.
        width = r['rgb'].shape[1]
        pca = {'mean': [0.] * width, 'components': np.eye(width, dtype=np.float32).tolist()}
        x, v, names = feature_view_blocks(A.RICH, r, pca)
        primary = {'recipe': A.RICH, 'weight': 1.,
                   'transform': {'feature_view': 'compact-feature-blocks-v10',
                                 'pca': pca, 'frame_feature_names': names},
                   'frame_model': {'kind': 'constant', 'n_features': x.shape[1], 'probability': .7},
                   'video_model': {'kind': 'constant', 'n_features': len(v), 'probability': .8}}
        cfg = {'kind': 'duration-logit-v4', 'threshold': .5, 'transition_seconds': .03,
               'min_seconds': .12, 'video_threshold': 0}
        b = A.v11_cached_score_bundle(primary, cfg, A.review_configurations()[1])
        # Verify the reviewer really has a usable v4 transform/model schema, before
        # any injection. This prevents empty dicts from masquerading as leaf models.
        rx, rv, rn = feature_view_v4(A.REVIEWER, r, pca)
        self.assertEqual(b['proposal_verifier']['member']['transform']['frame_feature_names'], rn)
        self.assertEqual(rx.shape[1], b['proposal_verifier']['member']['frame_model']['n_features'])
        self.assertEqual(len(rv), b['proposal_verifier']['member']['video_model']['n_features'])
        return r, b

    def test_local_q75_inclusive_linear_interpolation_and_peak(self):
        primary = np.full(5, .7)
        reviewer = np.array([1., .1, .2, .8, 1.])
        keep, trace = A.independent_review([(1, 3)], primary, reviewer, A.review_configurations()[3])
        self.assertEqual(keep, [(1, 3)])
        self.assertAlmostEqual(trace[0]['review_q75'], .5)
        self.assertFalse(trace[0]['rescued'])
        self.assertEqual(A.independent_review([(1, 3)], primary, reviewer, A.review_configurations()[5])[0], [])

    def test_outside_candidate_cannot_affect_review_or_rescue(self):
        primary, reviewer = np.full(8, .7), np.full(8, .2)
        cfg = A.review_configurations()[4]  # minimum .45, rescue .8
        original = A.independent_review([(2, 4)], primary, reviewer, cfg)
        primary[[0, 1, 5, 6, 7]] = 1.; reviewer[[0, 1, 5, 6, 7]] = 1.
        self.assertEqual(original, A.independent_review([(2, 4)], primary, reviewer, cfg))
        primary[4] = .8
        kept, trace = A.independent_review([(2, 4)], primary, reviewer, cfg)
        self.assertEqual(kept, [(2, 4)])
        self.assertTrue(trace[0]['rescued'])

    def test_review_cannot_add_or_shift_and_control_is_identity(self):
        proposals = [(0, 1), (4, 7)]
        self.assertEqual(A.independent_review(proposals, np.zeros(8), np.zeros(8),
                                            A.review_configurations()[0])[0], proposals)
        accepted, _ = A.independent_review(proposals, np.zeros(8), np.r_[np.ones(2), np.zeros(6)],
                                          A.review_configurations()[1])
        self.assertEqual(accepted, [(0, 1)])

    def test_review_invalid_config_probabilities_and_spans_fail(self):
        configs = [dict(A.review_configurations()[0], minimum=False),
                   dict(A.review_configurations()[0], minimum=.9),
                   dict(A.review_configurations()[1], extra='not-frozen')]
        for cfg in configs:
            with self.subTest(cfg=cfg), self.assertRaises(A.AuditError):
                A.independent_review([(0, 1)], np.zeros(8), np.zeros(8), cfg)
        for q in (np.zeros(7), np.full(8, np.nan), np.full(8, 1.01)):
            with self.assertRaises(A.AuditError):
                A.independent_review([(0, 1)], np.zeros(8), q, A.review_configurations()[1])
        with self.assertRaises(A.AuditError):
            A.independent_review([(1, 8)], np.zeros(8), np.zeros(8), A.review_configurations()[1])

    def test_stageA_stageB_golden_filter_and_policy_ordinal_dedup(self):
        records = [self.record(True, 'normal'), self.record(False, 'positive')]
        primary = A.Probabilities({i: np.full(8, .8, np.float32) for i in (0, 1)}, {0: .8, 1: .8})
        reviewer = A.Probabilities({0: np.zeros(8), 1: np.full(8, .6)}, {0: .5, 1: .5})
        def proposal(*_args):
            return [(1, 2)]
        result = A.independent_review_choice(records, [0, 1], primary, reviewer, 5, (proposal, proposal))
        self.assertEqual(result['candidate'], 'et_proposals_hgb_local_review')
        self.assertEqual(result['config']['proposal'], A.configurations()[0])
        self.assertEqual(result['config']['review'], A.review_configurations()[1])
        self.assertEqual([r['config']['review'] for r in result['shortlist']], A.review_configurations()[:2])
        self.assertEqual(result['stats'], [1., 0., 0., 1., 0., 0., 2., 0., 0., 0., 0., 1.])
        self.assertAlmostEqual(result['stable_utility'], 1.1)
        self.assertAlmostEqual(result['base_selection']['stable_utility'], 13. / 30.)
        # Tampered Stage A, Stage B utility, and trace must differ from the golden evidence.
        for field in ('base_selection', 'bootstrap_q20', 'shortlist'):
            broken = deepcopy(result)
            if field == 'base_selection':
                broken[field]['config']['threshold'] = .6
            elif field == 'bootstrap_q20':
                broken[field] = .99
            else:
                broken[field].pop()
            with self.subTest(field=field), self.assertRaises(A.AuditError):
                A.compare(broken, result, 'StageA/StageB')

    def test_v11_no_held_out_record_access_and_review_head_coverage(self):
        records = [self.record(False, 'fit'), self.record(False, 'outer')]
        p = A.Probabilities({0: np.full(8, .8, np.float32)}, {0: .8})
        q = A.Probabilities({0: np.full(8, .6, np.float32)}, {0: .5})
        def malicious(reader, *_args):
            return reader[1]
        with patch.object(A, 'review_predictions', side_effect=malicious):
            with self.assertRaisesRegex(A.AuditError, 'held-out'):
                A.independent_review_choice(records, [0], p, q, 5, (lambda *_: [(1, 2)],) * 2)
        q.video[1] = .4
        with self.assertRaisesRegex(A.AuditError, 'coverage'):
            A.independent_review_choice(records, [0], p, q, 5, (None, None))

    def test_v4_parent_metadata_uses_v4_not_v8_receipt(self):
        records = records_fixture()
        outer = A.seeded_folds(records, A.content_groups(records, list(range(len(records)))), 5, A.SEED)
        row = metadata_fixture(records, outer, parent=True)
        row['recipe'], row['training_receipt_sha256'] = A.REVIEWER, A.V4_RECEIPT_PIN
        sign(row)
        A.verify_training_metadata(row, A.REVIEWER, 0, records, outer, parent=True,
                                   parent_recipe=A.REVIEWER, parent_receipt=A.V4_RECEIPT_PIN)
        row['training_receipt_sha256'] = A.PARENT_PIN
        sign(row)
        with self.assertRaises(A.AuditError):
            A.verify_training_metadata(row, A.REVIEWER, 0, records, outer, parent=True,
                                       parent_recipe=A.REVIEWER, parent_receipt=A.V4_RECEIPT_PIN)

    def test_runtime_active_branch_uses_identical_local_evidence(self):
        path = A.ROOT / 'code/optimized_detector.py'
        detector = A.module_from_bytes(path.read_bytes(), path, '_test_detector_active')
        record, bundle = self.runtime_fixture()
        p = np.array([.1, .8, .8, .1, .1, .1, .1, .1], np.float32)
        q = np.zeros(8, np.float32)
        cfg = bundle['decoder']
        # The unpatched leaf models are also usable; cached scores below isolate
        # active review mechanics rather than weaken the current validator.
        for m in (bundle['members'][0], bundle['proposal_verifier']['member']):
            fp, vp, boundary = detector.predict_member(m, record)
            self.assertEqual(fp.shape, (8,))
            np.testing.assert_array_equal(fp, np.full(8, m['frame_model']['probability'], np.float32))
            self.assertEqual(vp, float(np.float32(m['video_model']['probability'])))
            self.assertIsNone(boundary)
        def probabilities(m, _r):
            return (q if m['recipe'] == A.REVIEWER else p), .8, None
        self.assertEqual(detector.required_channels(bundle), {'rgb', 'motion', 'corrected', 'local'})
        raw = A.decoder_functions()[0](p, record['fps'], .8, cfg)
        self.assertTrue(raw)
        # The 77-row empirical replay does not necessarily contain an ET-peak
        # rescue. Explicit positive/negative controls cover identity/filter/rescue
        # in the actual runtime, against the independent implementation.
        for ordinal in (0, 1, 2):
            policy = A.review_configurations()[ordinal]
            bundle['proposal_verifier']['review'] = policy
            with self.subTest(policy=policy), patch.object(detector, 'predict_member', side_effect=probabilities):
                baseline = detector.predict_record({k: v for k, v in bundle.items()
                                                    if k != 'proposal_verifier'}, record)
                actual = detector.predict_record(bundle, record)
            expected, trace = A.independent_review(raw, p, q, policy)
            self.assertEqual(actual['intervals'], expected)
            self.assertEqual(actual['proposal_review_trace'], trace)
            self.assertEqual(actual['intervals'], [] if ordinal == 1 else raw)
            if ordinal == 2:
                self.assertTrue(any(row['rescued'] for row in actual['proposal_review_trace']))
            np.testing.assert_array_equal(actual['frame_probabilities'], p)
            np.testing.assert_array_equal(actual['score'], baseline['score'])
            self.assertEqual(actual['video_probability'], baseline['video_probability'])

    def test_detector_rejects_direct_nested_verifier_and_ensemble_reviewer(self):
        path = A.ROOT / 'code/optimized_detector.py'
        detector = A.module_from_bytes(path.read_bytes(), path, '_test_detector_validation')
        invalid = {'proposal_verifier': {'schema_version': 'proposal-verifier-v11',
                    'member': {'proposal_verifier': {}}, 'review': A.review_configurations()[1]}}
        with self.assertRaises(ValueError):
            detector._validated_proposal_verifier(invalid)
        ensemble = {'members': [{'recipe': A.REVIEWER,
                     'transform': {'feature_view': 'compact-fps-context-v4'}}]}
        invalid['proposal_verifier']['member'] = ensemble
        # The repaired contract rejects the container directly, before treating it
        # as a leaf. Retain this as a regression test for the original P2 bypass.
        with self.assertRaisesRegex(ValueError, 'leaf member'):
            detector._validated_proposal_verifier(invalid)

    def test_hardened_revision_preserves_export_pin_and_exact_scope(self):
        frozen = (A.ROOT / A.V11_FROZEN_DETECTOR).read_bytes()
        current = (A.ROOT / 'code/optimized_detector.py').read_bytes()
        review = A.verify_v11_detector_revision(frozen, current)
        self.assertTrue(review['P2_guard_revision_reviewed'])
        self.assertEqual(review['frozen_export_detector_sha256'], A.V11_DETECTOR_PIN)
        self.assertEqual(review['current_detector_sha256'], A.sha(current))
        self.assertNotEqual(review['frozen_export_detector_sha256'], review['current_detector_sha256'])
        self.assertTrue(review['module_AST_equal_excluding_reviewed_guard_bodies'])
        self.assertFalse(A.verify_v11_detector_revision(frozen, frozen)['P2_guard_revision_reviewed'])

    def test_hardened_revision_rejects_unknown_or_reanchored_sources(self):
        frozen = (A.ROOT / A.V11_FROZEN_DETECTOR).read_bytes()
        current = (A.ROOT / 'code/optimized_detector.py').read_bytes()
        with self.assertRaisesRegex(A.AuditError, 'unreviewed current detector'):
            A.verify_v11_detector_revision(frozen, current + b'\n# unreviewed change\n')
        with self.assertRaisesRegex(A.AuditError, 'frozen export detector'):
            A.verify_v11_detector_revision(current, current)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'code').mkdir()
            (root / 'code/optimized_detector.py').write_bytes(current)
            with self.assertRaisesRegex(A.MissingArtifact, 'sources/optimized_detector'):
                A.read_v11_detector(A.Evidence(root))

    def test_even_new_pin_cannot_hide_signature_or_inference_changes(self):
        frozen = (A.ROOT / A.V11_FROZEN_DETECTOR).read_bytes()
        current = (A.ROOT / 'code/optimized_detector.py').read_bytes()
        variants = [current.replace(b'def _validated_proposal_verifier(bundle):',
                                    b'def _validated_proposal_verifier(bundle, extra=None):'),
                    current.replace(b"'video_probability':video_probability", b"'video_probability':0.123")]
        kind = A.V11_HARDENED_PINS[A.sha(current)]
        for changed in variants:
            self.assertNotEqual(current, changed)
            with self.subTest(digest=A.sha(changed)), patch.dict(A.V11_HARDENED_PINS, {A.sha(changed): kind}):
                with self.assertRaisesRegex(A.AuditError, 'outside reviewed'):
                    A.verify_v11_detector_revision(frozen, changed)

    def test_independent_rejection_matrix_catches_original_p2_and_keeps_positive_controls(self):
        _record, valid = self.runtime_fixture()
        path = A.ROOT / 'code/optimized_detector.py'
        current = A.module_from_bytes(path.read_bytes(), path, '_test_hardened_matrix')
        result = A.audit_v11_verifier_contract(current, valid)
        self.assertEqual(result['rejected_case_count'], 39)
        self.assertEqual(len(set(result['rejected_cases'])), 39)
        self.assertEqual(result['supported_decoder_positive_controls'], 2)
        frozen_path = A.ROOT / A.V11_FROZEN_DETECTOR
        frozen = A.module_from_bytes(frozen_path.read_bytes(), frozen_path, '_test_unhardened_matrix')
        with self.assertRaisesRegex(A.AuditError, 'mutation accepted.*reviewer_nonleaf_members'):
            A.audit_v11_verifier_contract(frozen, valid)
        # A blanket refusal must never look like a correct fail-closed contract.
        with patch.object(current, '_validated_proposal_verifier', side_effect=ValueError('always reject')):
            with self.assertRaisesRegex(ValueError, 'always reject'):
                A.audit_v11_verifier_contract(current, valid)

    def test_public_predict_rejects_composition_risks_without_touching_reviewer(self):
        record, valid = self.runtime_fixture()
        path = A.ROOT / 'code/optimized_detector.py'
        current = A.module_from_bytes(path.read_bytes(), path, '_test_public_hardened')
        # Exercise public inference on real usable portable schemas, not just the
        # internal guard. All malformed combinations must fail before reviewer use.
        focus = {'reviewer_nonleaf_members', 'reviewer_nonleaf_decoder', 'reviewer_nonleaf_proposal_verifier',
                 'reviewer_missing_frame_model', 'reviewer_missing_video_model', 'reviewer_unverified_feature_view',
                 'base_calibration', 'base_video_gate', 'base_boundary_models',
                 'reviewer_calibration', 'reviewer_video_gate', 'reviewer_boundary_models',
                 'primary_calibration', 'primary_video_gate', 'primary_boundary_models',
                 'primary_multiple_members', 'primary_nonunit_weight', 'decoder_boundary_enabled',
                 'review_minimum_nonfinite', 'review_rescue_over_one'}
        selected = [(name, row) for name, row in A.v11_verifier_risk_cases(valid) if name in focus]
        self.assertEqual(len(selected), len(focus))
        original = current.predict_member
        for name, row in selected:
            with self.subTest(case=name), patch.object(current, 'predict_member', wraps=original) as called:
                with self.assertRaises(ValueError):
                    current.predict_record(row, record)
                self.assertNotIn(A.REVIEWER, [c.args[0].get('recipe') for c in called.call_args_list])

    def test_v11_mode_dispatches_and_missing_report_never_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'v11-audit.json'
            with patch.object(A, 'run_audit') as v10, patch.object(A, 'run_v11_audit',
                    side_effect=A.MissingArtifact('missing v11 report')) as v11:
                self.assertEqual(A.main(['--mode', 'v11', '--output', str(target)]), 2)
                v10.assert_not_called(); v11.assert_called_once()
            report = json.loads(target.read_text('utf-8'))
            self.assertEqual(report['schema_version'], 'independent-feature-block-audit-v11')
            self.assertEqual(report['status'], 'failed')




class V12Tests(unittest.TestCase):
    def record(self, normal=False, name='row'):
        y = np.zeros(8, bool)
        if not normal:
            y[1:3] = True
        return {'name': name, 'sha256': name, 'labels': y,
                'event_count': 0 if normal else 1, 'frames': 8, 'fps': 10.}

    def boundary_metadata(self, records, outer, final=False):
        scope = 5 if final else 0
        row = metadata_fixture(records, outer, scope=scope, final=final)
        row.pop('recipe'); row.pop('outer_seed')
        row['training_receipt_sha256'] = A.V12_RECEIPT_PIN
        parts = row['inner_partitions'] + ([] if final else [
            {'fit': row['train'], 'validation': row['validation'], 'seed': A.SEED + 99}])
        for proof, part in zip(row['fit_evidence'], parts):
            proof.update(seed=part['seed'] + 2200, models_sha256='c' * 64,
                         base_content_mass=float(len({records[i]['sha256'] for i in part['fit']})))
        return sign(row)

    def test_v12_resigned_metadata_tampering_is_not_hidden_by_hashes(self):
        records = records_fixture()
        outer = A.seeded_folds(records, A.content_groups(records, list(range(len(records)))), 5, A.SEED)
        receipt = {'receipt_sha256': A.V12_RECEIPT_PIN}
        row = self.boundary_metadata(records, outer)
        self.assertEqual(len(A.verify_v12_head_metadata(row, 0, records, outer, receipt)), 4)
        for mutation in ('partition', 'inner_seed', 'fit_content', 'head_seed', 'content_mass',
                         'models_hash', 'transform_hash', 'receipt', 'fit_count'):
            broken = deepcopy(row)
            if mutation == 'partition':
                broken['inner_partitions'][0]['validation'].append(
                    broken['inner_partitions'][1]['validation'][0])
            elif mutation == 'inner_seed':
                broken['inner_partitions'][0]['seed'] += 1
            elif mutation == 'fit_content':
                broken['fit_evidence'][0]['fit_content_sha256'].pop()
            elif mutation == 'head_seed':
                broken['fit_evidence'][0]['seed'] -= 2200
            elif mutation == 'content_mass':
                broken['fit_evidence'][0]['base_content_mass'] += .5
            elif mutation == 'models_hash':
                broken['fit_evidence'][0]['models_sha256'] = 'not-a-hash'
            elif mutation == 'transform_hash':
                broken['fit_evidence'][0]['transform_sha256'] = ''
            elif mutation == 'receipt':
                broken['training_receipt_sha256'] = A.RECEIPT_PIN
            else:
                broken['fit_evidence'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(A.AuditError):
                A.verify_v12_head_metadata(sign(broken), 0, records, outer, receipt)

    def test_v12_final_cannot_consume_outer_validation_or_other_scope(self):
        records = records_fixture()
        outer = A.seeded_folds(records, A.content_groups(records, list(range(len(records)))), 5, A.SEED)
        row = self.boundary_metadata(records, outer, final=True)
        receipt = {'receipt_sha256': A.V12_RECEIPT_PIN}
        self.assertEqual(len(A.verify_v12_head_metadata(row, 5, records, outer, receipt, final=True)), 3)
        for key, value in (('scope', 4), ('validation', [0]), ('final_selection', False)):
            broken = deepcopy(row); broken[key] = value
            with self.subTest(key=key), self.assertRaises(A.AuditError):
                A.verify_v12_head_metadata(sign(broken), 5, records, outer, receipt, final=True)

    def test_v12_training_ledger_duplicate_missing_or_count_must_fail(self):
        reports = {
            'training_report.json': {'status': 'complete', 'boundary_pair_fits': 20,
                'results': [{'scope': i, 'final_selection': False, 'seconds': 1.} for i in range(5)]},
            'final_training_report.json': {'status': 'complete', 'boundary_pair_fits': 3,
                'results': [{'scope': 5, 'final_selection': True, 'seconds': 1.}]}}
        class Fake:
            def __init__(self, evidence):
                self.evidence = evidence
            def json(self, path):
                if path.name not in self.evidence:
                    raise A.MissingArtifact('missing ' + path.name)
                return self.evidence[path.name]
        self.assertEqual(len(A.verify_v12_training_reports(Fake(reports))), 2)
        for mutation in ('duplicate', 'missing', 'wrong_count', 'not_complete', 'wrong_final'):
            broken = deepcopy(reports)
            if mutation == 'duplicate':
                broken['training_report.json']['results'][1]['scope'] = 0
            elif mutation == 'missing':
                broken.pop('final_training_report.json')
            elif mutation == 'wrong_count':
                broken['training_report.json']['boundary_pair_fits'] = 21
            elif mutation == 'not_complete':
                broken['final_training_report.json']['status'] = 'running'
            else:
                broken['final_training_report.json']['results'][0]['final_selection'] = False
            with self.subTest(mutation=mutation), self.assertRaises(A.AuditError):
                A.verify_v12_training_reports(Fake(broken))

    def test_v12_stageB_golden_and_resigned_selection_change_fail(self):
        records = [self.record(True, 'normal'), self.record(False, 'positive')]
        primary = A.Probabilities({i: np.full(8, .8, np.float32) for i in (0, 1)}, {0: .8, 1: .8})
        heads = {0: (np.zeros(8, np.float32), np.zeros(8, np.float32)),
                 1: (np.full(8, .6, np.float32), np.full(8, .6, np.float32))}
        def proposals(*_args):
            return [(1, 2)]
        def unchanged(intervals, *_args):
            return list(intervals)
        result = A.independent_boundary_choice(records, [0, 1], primary, heads, 5,
                                               (proposals, proposals), unchanged)
        self.assertEqual(result['candidate'], 'compact_boundary_refine_accept')
        self.assertEqual(result['config']['boundary'], A.v12_policies()[2])
        self.assertEqual([row['key'][-1] for row in result['shortlist']], [0, -2])
        self.assertAlmostEqual(result['stable_utility'], 1.1)
        self.assertAlmostEqual(result['bootstrap_q20'], 1.1)
        for key in ('stats', 'shortlist', 'base_selection', 'config'):
            broken = deepcopy(result)
            if key == 'stats':
                broken[key][3] += 1
            elif key == 'shortlist':
                broken[key].reverse()
            elif key == 'base_selection':
                broken[key]['config']['threshold'] = .6
            else:
                broken['config']['boundary']['minimum_boundary_evidence'] = .5
            with self.subTest(key=key), self.assertRaises(A.AuditError):
                A.compare(broken, result, 'V12 StageA/StageB')

    def test_v12_experiment_alias_dispatch_and_missing_inputs_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'v12-audit.json'
            with patch.object(A, 'run_v12_audit', side_effect=A.MissingArtifact('missing V12 heads')) as replay:
                self.assertEqual(A.main(['--experiment', 'v12', '--output', str(target)]), 2)
                replay.assert_called_once()
            report = json.loads(target.read_text('utf-8'))
            self.assertEqual(report['schema_version'], 'independent-feature-block-audit-v12')
            self.assertEqual(report['status'], 'failed')

    def test_v12_policy_grid_is_exactly_ten_and_ordered(self):
        self.assertEqual(A.v12_policies(), [
            {'boundary_seconds': 0., 'minimum_boundary_evidence': 0.},
            {'boundary_seconds': .2, 'minimum_boundary_evidence': 0.},
            {'boundary_seconds': .2, 'minimum_boundary_evidence': .3},
            {'boundary_seconds': .2, 'minimum_boundary_evidence': .5},
            {'boundary_seconds': .4, 'minimum_boundary_evidence': 0.},
            {'boundary_seconds': .4, 'minimum_boundary_evidence': .3},
            {'boundary_seconds': .4, 'minimum_boundary_evidence': .5},
            {'boundary_seconds': .6, 'minimum_boundary_evidence': 0.},
            {'boundary_seconds': .6, 'minimum_boundary_evidence': .3},
            {'boundary_seconds': .6, 'minimum_boundary_evidence': .5},
        ])

    def test_v12_endpoint_evidence_requires_both_endpoints_and_is_local(self):
        primary = np.zeros(8, np.float32)
        start = np.array([.9, .9, .2, .2, .2, .2, .2, .2], np.float32)
        end = np.array([.2, .2, .2, .2, .2, .2, .8, .8], np.float32)
        policy = A.v12_policies()[2]  # .2 seconds, evidence .3
        def unchanged(intervals, *_args):
            return list(intervals)
        self.assertEqual(A.independent_boundary_apply([(1, 2)], (start, end), 10., policy, unchanged)[0], [])
        start[0] = end[0] = 1.
        # Values outside the candidate (including a high end peak) cannot change q/evidence.
        end[6:] = 1.
        self.assertEqual(A.independent_boundary_apply([(1, 2)], (start, end), 10., policy, unchanged)[0], [])
        end[2] = .8
        self.assertEqual(A.independent_boundary_apply([(1, 2)], (start, end), 10., policy, unchanged)[0], [(1, 2)])

    def test_v12_zero_boundary_policy_does_not_change_intervals(self):
        from optimized_boundary_head import refine_intervals
        policy = A.v12_policies()[0]
        accepted, trace = A.independent_boundary_apply([(2, 3)],
            (np.full(8, .5, np.float32), np.full(8, .5, np.float32)), 10., policy,
            refine_intervals)
        self.assertEqual(accepted, [(2, 3)])
        self.assertEqual(trace[0]['span'], [2, 3])

    def test_v12_choice_is_inner_only_and_does_not_read_outer_content(self):
        records = [self.record(False, str(i)) for i in range(5)]
        records[0]['labels'][:] = 0; records[0]['event_count'] = 0
        ids = [0, 1, 2, 3]
        primary = A.Probabilities({i: np.full(8, .8, np.float32) for i in ids}, {i: .8 for i in ids})
        heads = {i: (np.full(8, .6, np.float32), np.full(8, .6, np.float32)) for i in ids}
        def proposals(*_args):
            return [(1, 2)]
        def same(intervals, *_args):
            return list(intervals)
        before = A.independent_boundary_choice(records, ids, primary, heads, 5,
            (proposals, proposals), same)
        records[4]['labels'][:] = 1; records[4]['event_count'] = 4; records[4]['sha256'] = 'outer-mutated'
        after = A.independent_boundary_choice(records, ids, primary, heads, 5,
            (proposals, proposals), same)
        self.assertEqual(before, after)
        with self.assertRaisesRegex(A.AuditError, 'coverage'):
            A.independent_boundary_choice(records, ids, primary,
                {**heads, 4: heads[0]}, 5, (proposals, proposals), same)

    def test_v12_boundary_cache_requires_float32_start_end_exact_keys(self):
        records = [self.record(False, str(i)) for i in range(2)]
        metadata = {'train': [0], 'validation': [1]}
        arrays = {'inner_start_0': np.full(8, .4, np.float32),
                  'inner_end_0': np.full(8, .5, np.float32),
                  'outer_start_1': np.full(8, .6, np.float32),
                  'outer_end_1': np.full(8, .7, np.float32)}
        data = npz_bytes(arrays); metadata['npz_sha256'] = A.sha(data)
        inner, outer = A.v12_boundary_arrays(data, metadata, records, 'valid')
        self.assertEqual(set(inner), {0}); self.assertEqual(set(outer), {1})
        for bad in ({**arrays, 'outer_start_99': np.full(8, .2, np.float32)},
                    {**arrays, 'inner_start_0': np.full(8, .4, np.float64)},
                    {**arrays, 'outer_end_1': np.full(8, np.nan, np.float32)},
                    {**arrays, 'outer_end_1': np.full(8, 1.1, np.float32)}):
            broken = npz_bytes(bad); metadata['npz_sha256'] = A.sha(broken)
            with self.subTest(keys=list(bad)), self.assertRaises(A.AuditError):
                A.v12_boundary_arrays(broken, metadata, records, 'bad')

    def test_v12_detector_has_no_implicit_unvalidated_adapter(self):
        text = (A.ROOT / 'code/optimized_detector.py').read_text('utf-8')
        self.assertNotIn('compact-boundary-v12', text)
        self.assertNotIn('minimum_boundary_evidence', text)
        # A V12 selection must not be reported as deployment-ready merely because
        # its statistical report exists.

    def test_v12_source_review_is_bounded_to_raw_features_and_fixed_head(self):
        # This test checks the current source-level contract without starting a fit.
        source = (A.ROOT / 'code/optimized_compact_boundary_v12.py').read_text('utf-8')
        head = (A.ROOT / 'code/optimized_boundary_head.py').read_text('utf-8')
        runner = (A.ROOT / 'code/run_compact_boundary_v12.py').read_text('utf-8')
        self.assertIn('fit_rgb_pca(records,train)', source.replace(' ', ''))
        self.assertIn('feature_view_blocks(RECIPE,records[i],pca)', source.replace(' ', ''))
        self.assertIn('_fit_head(x,starts,base,seed)', source.replace(' ', ''))
        self.assertIn('_fit_head(x,ends,base,seed+1)', source.replace(' ', ''))
        self.assertIn('n_estimators=128,max_depth=7,min_samples_leaf=5', head.replace(' ', ''))
        self.assertIn("part['seed']+2200", runner.replace(' ', ''))
        self.assertIn("new_frame_model_fits':0", runner.replace(' ', ''))

    def test_v12_independent_report_is_not_a_promotion_signal(self):
        path = A.ROOT / 'output/algorithm-opt-v12-compact-boundary/independent_feature_blocks_audit_v12_pass.json'
        if not path.exists():
            self.skipTest('V12 audit report not present')
        report = json.loads(path.read_text('utf-8'))
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['deployment_approved'])
        self.assertFalse(report['statistical_promotion_passed'])
        self.assertFalse(report['detector_compatibility']['detector_supports_v12_schema'])


if __name__ == '__main__':
    unittest.main()
