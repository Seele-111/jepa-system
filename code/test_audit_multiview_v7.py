"""Independent v7 policy/integrity tests; no runner execution or DP re-enumeration."""
import ast
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_multiview_v7 as a


def record(i, positive=True):
    labels = np.zeros(30, dtype=bool)
    if positive:
        labels[10:20] = True
    return {'sha256': f'{i:064x}', 'name': f'row{i}', 'generator': 'synthetic',
            'frames': 30, 'fps': 20., 'event_count': int(positive), 'labels': labels}


def policy_row(ordinal, feasible=True, violation=0., pooled=.6, stable=.5, f05=.4):
    return {'config': {'ordinal': ordinal}, 'pooled_utility': pooled, 'stable_utility': stable,
            'bootstrap_q20': 4 * stable - 3 * pooled,
            'error_budget': {'feasible': feasible, 'normalized_violation': violation},
            'key': [int(feasible), -violation, pooled, f05, -2., -1., -3., -ordinal]}


def fixture(predictions):
    rows = [record(0, False), record(1), record(2)]
    p = a.independent.Probabilities({i: np.full(30, i, np.float32) for i in (0, 1)}, {0: .9, 1: .9})
    grid = [{'kind': 'synthetic', 'id': i} for i in range(len(predictions))]
    def decoder(values, fps, video, cfg):
        return predictions[cfg['id']][int(values[0])]
    return rows, p, grid, {'synthetic': decoder}


class BudgetPolicy(unittest.TestCase):
    def test_exact_budget_boundaries_are_feasible(self):
        s = np.zeros(12)
        s[9], s[10] = 3, 2
        budget = a.error_budget(s, 10, 10)
        self.assertTrue(budget['feasible'])
        self.assertEqual(budget['normalized_violation'], 0)
        self.assertEqual(budget['normal_false_positive_rate'], .30)
        self.assertEqual(budget['positive_empty_rate'], .20)

    def test_normalized_excess_sum_and_zero_mass(self):
        s = np.zeros(12)
        s[9], s[10] = 6, 4
        budget = a.error_budget(s, 10, 10)
        self.assertFalse(budget['feasible'])
        self.assertAlmostEqual(budget['normalized_violation'], 2.)
        self.assertEqual(a.components(np.zeros(12), 0, 0)['utility'], 0.)
        self.assertTrue(a.error_budget(np.zeros(12), 0, 0)['feasible'])

    def test_both_penalties_are_point30_and_fractional_mass_is_not_clamped(self):
        s = np.zeros(12)
        s[9] = .5
        self.assertEqual(a.components(s, .5, 1)['utility'], -.30)
        s[9], s[10] = 0, .5
        self.assertEqual(a.components(s, 1, .5)['utility'], -.30)
        s[9] = .5
        self.assertEqual(a.components(s, .5, .5)['utility'], -.60)

    def test_feasible_wins_over_higher_utility_infeasible(self):
        high = policy_row(0, False, .01, pooled=1., stable=1.)
        low = policy_row(1, True, pooled=-1., stable=-1.)
        selected = a.shortlist_and_select([high, low])
        self.assertEqual(selected['config']['ordinal'], 1)
        self.assertTrue(selected['error_budget']['feasible'])

    def test_fallback_minimizes_violation_before_utility_or_q20(self):
        low_violation = policy_row(0, False, .25, pooled=-.8, stable=-.8)
        high_utility = policy_row(1, False, .5, pooled=.9, stable=100.)
        chosen = a.shortlist_and_select([low_violation, high_utility])
        self.assertEqual(chosen['config']['ordinal'], 0)
        self.assertFalse(chosen['error_budget']['feasible'])

    def test_top8_is_budget_pooled_not_stable_and_stable_ninth_cannot_win(self):
        rows = [policy_row(i, pooled=1. - i * .01, stable=.4) for i in range(10)]
        rows[8]['stable_utility'] = 100.
        chosen = a.shortlist_and_select(rows)
        self.assertEqual([r['config']['ordinal'] for r in chosen['shortlist']], list(range(8)))
        self.assertEqual(chosen['config']['ordinal'], 0)

    def test_violation_priority_applies_again_at_final_shortlist_rank(self):
        rows = [policy_row(0, False, .5, pooled=1., stable=100.),
                policy_row(1, False, .2, pooled=-1., stable=-1.)]
        self.assertEqual(a.shortlist_and_select(rows)['config']['ordinal'], 1)

    def test_candidate_final_tie_uses_config_ordinal_then_candidate_order(self):
        rows = []
        for candidate in a.CANDIDATES:
            row = policy_row(3)
            row['candidate'] = candidate
            rows.append(row)
        self.assertEqual(a.rank_candidates(rows)[0]['candidate'], 'compact_control')
        rows[1]['key'][-1] = -2
        self.assertEqual(a.rank_candidates(rows)[0]['candidate'], 'event_control')
        rows[2]['error_budget'] = {'feasible': True, 'normalized_violation': 0.}
        rows[0]['error_budget'] = rows[1]['error_budget'] = {'feasible': False, 'normalized_violation': .1}
        self.assertEqual(a.rank_candidates(rows)[0]['candidate'], 'equal_multiview')


class ChoiceReplay(unittest.TestCase):
    def test_independent_statistics_and_feasible_choice_integration(self):
        predictions = [{0: [(10, 19)], 1: [(10, 19)]}, {0: [], 1: []}, {0: [], 1: [(0, 0)]}]
        rows, p, grid, decoders = fixture(predictions)
        context = a.independent.selection_context(rows, [0, 1], 101, .30)
        chosen, detail = a.choose_independently(rows, context, p, decoders, grid)
        self.assertEqual(chosen['config']['id'], 2)
        self.assertTrue(chosen['error_budget']['feasible'])
        self.assertLess(chosen['pooled_utility'], detail['all_configurations'][0]['pooled_utility'])
        self.assertEqual(detail['feasible_unique_configurations'], 1)
        self.assertFalse(detail['explicit_fallback_used'])
        self.assertEqual(len(detail['all_configurations'][0]['bootstrap_utilities']), 32)

    def test_infeasible_fallback_is_explicit_and_not_claimed_feasible(self):
        rows, p, grid, decoders = fixture([{0: [(10, 19)], 1: [(10, 19)]}, {0: [], 1: []}])
        chosen, detail = a.choose_independently(rows, a.independent.selection_context(rows, [0, 1], 101, .30), p, decoders, grid)
        self.assertEqual(chosen['config']['id'], 0)
        self.assertFalse(chosen['error_budget']['feasible'])
        self.assertTrue(detail['explicit_fallback_used'])
        self.assertAlmostEqual(chosen['error_budget']['normalized_violation'], (1 - .3) / .3)

    def test_duplicate_prediction_uses_earliest_ordinal(self):
        pred = {0: [], 1: [(10, 19)]}
        rows, p, grid, decoders = fixture([pred, pred])
        chosen, detail = a.choose_independently(rows, a.independent.selection_context(rows, [0, 1], 7, .30), p, decoders, grid)
        self.assertEqual(chosen['config']['id'], 0)
        self.assertEqual(len(chosen['shortlist']), 1)
        self.assertEqual(detail['all_configurations'][1]['duplicate_of_ordinal'], 0)
        self.assertEqual(chosen['pooled_utility'], 1.1)
        self.assertEqual(chosen['bootstrap_q20'], 1.1)

    def test_unlisted_outer_label_probability_and_metadata_perturbation_invariance(self):
        rows, p, grid, decoders = fixture([{0: [], 1: [(10, 19)]}, {0: [(0, 0)], 1: [(10, 19)]}])
        before = a.choose_independently(rows, a.independent.selection_context(rows, [0, 1], 7, .30), p, decoders, grid)
        rows[2]['labels'][:] = False
        rows[2].update(sha256=rows[0]['sha256'], fps=999., event_count=0)
        p.frame[2] = np.full(30, .999, np.float32)
        p.video[2] = 0
        after = a.choose_independently(rows, a.independent.selection_context(rows, [0, 1], 7, .30), p, decoders, grid)
        self.assertEqual(before, after)


class FusionAndEvidence(unittest.TestCase):
    def test_fixed_linear_fusion_float32_and_no_mutation(self):
        left = a.independent.Probabilities({0: np.array([.2, .6], np.float32)}, {0: .4})
        right = a.independent.Probabilities({0: np.array([.8, .4], np.float32)}, {0: .8})
        before = left.frame[0].copy()
        fused = a.fusion(left, right, [0])
        np.testing.assert_array_equal(fused.frame[0], np.array([.5, .5], np.float32))
        self.assertEqual(fused.frame[0].dtype, np.dtype('float32'))
        self.assertAlmostEqual(fused.video[0], .6)
        np.testing.assert_array_equal(left.frame[0], before)

    def test_invalid_fusion_heads_shape_nan_range_and_duplicate_indices_rejected(self):
        p = a.independent.Probabilities({0: np.array([.2], np.float32)}, {0: .4})
        bad = [a.independent.Probabilities({0: np.array([.2, .3], np.float32)}, {0: .4}),
               a.independent.Probabilities({0: np.array([np.nan], np.float32)}, {0: .4}),
               a.independent.Probabilities({0: np.array([1.1], np.float32)}, {0: .4}),
               a.independent.Probabilities({0: np.array([.3], np.float32)}, {}),
               a.independent.Probabilities({0: np.array([.3], np.float32)}, {0: 1.1})]
        for q in bad:
            with self.assertRaises(a.AuditError): a.fusion(p, q, [0])
        with self.assertRaises(a.AuditError): a.fusion(p, p, [0, 0])

    def test_pair_requires_all_shared_content_partition_seed_proofs_but_not_same_transform(self):
        left = {'fold': 0, 'train': [0, 1], 'validation': [2], 'outer_seed': 100,
                'inner_partitions': [{'fit': [0], 'validation': [1], 'seed': 1}],
                'fit_evidence': [{'partition': k if k < 3 else 'outer', 'fit_content_sha256': ['a' * 64],
                                 'transform_sha256': 'b' * 64} for k in range(4)]}
        right = deepcopy(left)
        for proof in right['fit_evidence']: proof['transform_sha256'] = 'c' * 64
        a.verify_parent_pair(left, right)
        for key in ('outer_seed', 'train', 'inner_partitions', 'fit_content', 'truncated_proofs'):
            changed = deepcopy(right)
            if key == 'outer_seed': changed[key] += 1
            if key == 'train': changed[key] = [0]
            if key == 'inner_partitions': changed[key][0]['seed'] += 1
            if key == 'fit_content': changed['fit_evidence'][0]['fit_content_sha256'] = []
            if key == 'truncated_proofs': changed['fit_evidence'].pop()
            with self.assertRaises(a.AuditError): a.verify_parent_pair(left, changed)

    def test_receipt_must_match_human_supplied_pin_before_other_evidence(self):
        class FakeEvidence:
            def json(self, path): return {'receipt_sha256': 'a' * 64}
        with self.assertRaises(a.AuditError): a.verify_v7_receipt(FakeEvidence())
        self.assertEqual(len(a.expected_inputs()), 22)
        self.assertEqual(len(a.SOURCES), 20)

    def test_static_source_lock_and_refusal_if_outer_decode_moved_before_choice(self):
        source = (a.ROOT / 'code' / a.RUNNER).read_text(encoding='utf-8')
        review = a.source_review_text(source)
        self.assertEqual(review['freeze_blockers_detected'], [])
        self.assertLess(review['primary_lock_line'], review['first_primary_outer_decode_line'])
        original = "        chosen=max(rows,key=lambda r:(*selection_key(r),-CANDIDATES.index(r['candidate'])))"
        predicted = "        pred=decode_predictions(records,meta['validation'],outer[chosen['candidate']],chosen['config']);selected.update(pred)"
        self.assertIn(original + '\n' + predicted, source)
        bad = source.replace(original + '\n' + predicted, predicted + '\n' + original)
        with self.assertRaises(a.AuditError): a.source_review_text(bad)

    def test_missing_inputs_no_wait_no_output_and_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(a.IncompleteArtifacts): a.complete_inputs(root, False)
            with self.assertRaises(a.IncompleteArtifacts): a.complete_inputs(root, True)
            self.assertFalse((root / a.OUT / a.OUTPUT_NAME).exists())
            (root / a.OUT).mkdir(parents=True)
            path = a.write_report(root, {'status': 'fixture'})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): a.write_report(root, {'status': 'other'})
            self.assertEqual(path.read_bytes(), before)

    def test_no_runner_choose_utility_or_statistics_import(self):
        tree = ast.parse(Path(a.__file__).read_text(encoding='utf-8'))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        forbidden = ('run_', 'optimized_video_statistics', 'optimized_compact_model', 'optimized_event_training')
        self.assertFalse([name for name in imports if name and name.startswith(forbidden)])


if __name__ == '__main__':
    unittest.main()
