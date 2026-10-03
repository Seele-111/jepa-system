"""Focused tests for the independent crossed v8/v9 audit; no model fitting."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import audit_crossed_experiments_v8_v9 as A


def records_fixture():
    labels = ([0, 1, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0], [0, 0, 0], [1])
    hashes = ('a', 'a', 'b', 'c', 'd')
    return [{'sha256': h * 64, 'labels': np.asarray(y, bool), 'frames': len(y), 'fps': 30.,
             'event_count': len(A.B.label_spans(y)), 'name': f'fixture-{i}', 'generator': 'fixture'}
            for i, (h, y) in enumerate(zip(hashes, labels))]


class CrossedAuditTests(unittest.TestCase):
    def test_utility_uses_both_point30_penalties_and_fractional_mass(self):
        stats = np.zeros(12)
        stats[9], stats[10] = .5, .25
        with patch.object(A.B, 'score_components', side_effect=AssertionError('old utility forbidden')):
            result = A.components(stats, 1., 1.)
        self.assertAlmostEqual(result['utility'], -.225)
        self.assertAlmostEqual(A.components(stats, .5, .5)['utility'], -.45)
        self.assertEqual(A.components(np.zeros(12), 0, 0)['utility'], 0.)

    def test_actual_frozen_weights_match_independent_raw_normal4(self):
        # Only pure weight functions are AST-extracted; no training module import.
        event, cost = A.objective_functions(A.B.Evidence(A.ROOT))
        records, ids = records_fixture(), [0, 1, 2, 3]
        guarded = A.TrainRecords(records, ids)
        w2, w4 = event(guarded, ids), cost(guarded, ids)
        np.testing.assert_allclose(w2, A.independent_event_weights(guarded, ids, 2.), rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(w4, A.independent_event_weights(guarded, ids, 4.), rtol=2e-13, atol=2e-12)
        labels = np.concatenate([records[i]['labels'] for i in ids])
        np.testing.assert_allclose(w2[labels], w4[labels], atol=2e-12)
        self.assertAlmostEqual(w4[labels].sum(), w4[~labels].sum())
        self.assertAlmostEqual(w4.mean(), 1.)
        self.assertGreater(w4[5], w2[5])  # conflicting normal alias
        self.assertLess(w4[0], w2[0])  # abnormal background is not doubled
        changed = deepcopy(records)
        changed[4]['labels'] = object()
        np.testing.assert_array_equal(cost(A.TrainRecords(changed, ids), ids), w4)
        with self.assertRaises(A.B.AuditError):
            cost(A.TrainRecords(records, ids), ids + [4])

    def test_actual_training_ast_is_objective_only_and_detects_model_change(self):
        e = A.B.Evidence(A.ROOT)
        _, base = A.functions(e, A.B.V4, 'optimized_compact_model_v4.py')
        _, event = A.functions(e, A.V8.folder, 'optimized_event_compact_v8.py')
        _, cost = A.functions(e, A.V9.folder, 'optimized_normal_cost_v9.py')
        b = A.normalized_fit(base['fit_compact_member'], 'v4')
        v8 = A.normalized_fit(event['fit_event_compact_member'], 'v8')
        v9 = A.normalized_fit(cost['fit_normal_cost_member'], 'v9')
        self.assertTrue(A.same_ast(b, v8))
        self.assertTrue(A.same_ast(v8, v9))
        changed = deepcopy(event['fit_event_compact_member'])
        for n in ast.walk(changed):
            if isinstance(n, ast.Constant) and n.value == 192:
                n.value = 193
        self.assertFalse(A.same_ast(v8, A.normalized_fit(changed, 'v8')))
        self.assertEqual(A.source_review(e, A.V8)['status'], 'passed')
        self.assertEqual(A.source_review(e, A.V9)['status'], 'passed')

    def test_normalized_fit_does_not_hide_extra_objective_side_effect(self):
        _, event = A.functions(A.B.Evidence(A.ROOT), A.V8.folder, 'optimized_event_compact_v8.py')
        changed = deepcopy(event['fit_event_compact_member'])
        index = next(i for i, n in enumerate(changed.body) if A.assignment_name(n) == 'weights')
        changed.body.insert(index + 1, ast.parse("weights += records[predict[0]]['labels']").body[0])
        with self.assertRaises(A.B.AuditError):
            A.normalized_fit(changed, 'v8')

    def test_crossed_control_rejects_transform_or_partition_mismatch(self):
        p = A.B.Probabilities({0: np.asarray([.5], np.float32)}, {0: .4})
        meta = {'train': [0], 'validation': [1], 'inner_partitions': [], 'outer_seed': 1,
                'fit_evidence': [{'transform_sha256': 'a' * 64, 'fit_content_sha256': ['b' * 64]}]}
        original = (p, p, meta)
        self.assertTrue(A.control_comparison([], original, original)['all_partitions_seeds_fit_content_and_transform_proofs_equal'])
        for field, value in (('outer_seed', 2), ('train', [1]), ('fit_evidence', [])):
            changed = deepcopy(meta)
            changed[field] = value
            with self.assertRaises(A.B.AuditError):
                A.control_comparison([], (p, p, changed), original)

    def test_chooser_top8_precedes_stability_and_never_reads_holdout(self):
        records = [{'labels': np.asarray([1]), 'fps': 30.}, {'labels': np.asarray([1, 1]), 'fps': 30.}, {}]
        context = SimpleNamespace(indices=[0, 1], weights=np.asarray([.5, .5]), normal_mass=0., positive_mass=1.,
                                  normal=np.zeros(2), positive=np.ones(2), weighted_counts=np.tile([1., 0.], (32, 1)),
                                  counts=np.tile([2, 0], (32, 1)), seed=17)
        probabilities = A.B.Probabilities({0: np.ones(1), 1: np.ones(2)}, {0: 1., 1: 1.})
        configs = [{'kind': 'test', 'ordinal': k} for k in range(40)]
        decoders = {'test': lambda frame, fps, video, cfg: [(cfg['ordinal'], cfg['ordinal'])]}
        def statistics(pred, labels):
            ordinal = pred[0][0]
            pooled = 1. - .01 * ordinal
            first = .99 if ordinal == 8 else 0.
            s = np.zeros(12)
            s[0] = first if len(labels) == 1 else 2 * pooled - first
            return s
        def utility(s, n, p):
            return {'utility': float(s[0]), 'event_f1_05': 0.}
        with patch.object(A.B, 'configurations', return_value=configs), patch.object(A.B, 'row_statistics', side_effect=statistics), patch.object(A, 'components', side_effect=utility):
            choice, detail = A.choose_independently(A.TrainRecords(records, [0, 1]), context, probabilities, decoders)
        self.assertEqual(choice['config']['ordinal'], 0)
        self.assertEqual([r['config']['ordinal'] for r in choice['shortlist']], list(range(8)))
        self.assertNotIn('bootstrap_q20', detail['all_configurations'][8])
        self.assertGreater(.75 * .92 + .25 * .99, choice['stable_utility'])
        self.assertEqual(detail['bootstrap_replicates'], 32)

    def test_duplicate_predictions_keep_first_config_and_timeout_is_explicit(self):
        records = records_fixture()
        ids = [0, 1, 2, 3]
        guarded = A.TrainRecords(records, ids)
        context = A.B.selection_context(guarded, ids, 19, empty_penalty=.30)
        probabilities = A.B.Probabilities({i: np.zeros(records[i]['frames']) for i in ids}, {i: 0. for i in ids})
        decoders = {name: lambda *args: [] for name in ('duration-logit-v4', 'recall-stable-v2')}
        choice, detail = A.choose_independently(guarded, context, probabilities, decoders)
        self.assertEqual(detail['unique_prediction_sets'], 1)
        self.assertEqual(choice['key'][-1], 0)
        self.assertTrue(all(r['duplicate_of_ordinal'] == 0 for r in detail['all_configurations'][1:]))
        with self.assertRaises(TimeoutError):
            A.choose_independently(guarded, context, probabilities, decoders, deadline=-1)

    def test_write_is_exclusive_and_readiness_does_not_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for spec in A.EXPERIMENTS:
                (root / spec.folder).mkdir(parents=True)
                for name in ('protocol.json', 'training_receipt.json', 'training_report.json', 'report.json'):
                    (root / spec.folder / name).write_text('{"status":"complete"}', encoding='utf-8')
            with self.assertRaises(A.B.AuditError):
                A.readiness(root, False)
            A.readiness(root, True)
            path = A.write_report(root, A.V8, {'status': 'fixture'})
            with self.assertRaises(FileExistsError):
                A.write_report(root, A.V8, {'status': 'changed'})
            self.assertIn('fixture', path.read_text('utf-8'))
            with self.assertRaises(A.B.AuditError):
                A.readiness(root, True)


if __name__ == '__main__':
    unittest.main()
