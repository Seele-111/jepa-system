"""Synthetic independent-auditor tests; no training files or DP re-enumeration."""
import ast
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import audit_compact_selection_v4 as a


def record(i, positive=True, frames=10):
    labels = np.zeros(frames, dtype=bool)
    if positive:
        labels[2:8] = True
    return {'sha256': f'{i:064x}', 'name': f'row{i}', 'generator': 'synthetic',
            'frames': frames, 'fps': 10., 'event_count': int(positive), 'labels': labels}


def archive(values):
    stream = BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


class IndependentStatistics(unittest.TestCase):
    def test_inclusive_threshold_and_dense_frame_counts(self):
        labels = np.zeros(10, bool)
        labels[:3] = True
        expected = [1, 0, 0, 0, 1, 1, 3, 7, 0, 0, 0, 1]
        np.testing.assert_array_equal(a.row_statistics([(0, 9)], labels), expected)

    def test_greedy_order_and_first_gt_tie_are_preserved(self):
        labels = np.zeros(8, bool)
        labels[1:3] = labels[5:7] = True
        first = a.row_statistics([(1, 6), (1, 2)], labels)
        reverse = a.row_statistics([(1, 2), (1, 6)], labels)
        np.testing.assert_array_equal(first[:3], [1, 1, 1])
        np.testing.assert_array_equal(reverse[:3], [2, 0, 0])
        np.testing.assert_array_equal(first[3:6], [1, 1, 1])

    def test_frame_union_clamps_and_does_not_double_count_overlap(self):
        s = a.row_statistics([(-3, 2), (2, 9)], [0, 1, 1, 0, 1])
        np.testing.assert_array_equal(s[6:9], [3, 2, 0])

    def test_empty_and_normal_statistics(self):
        np.testing.assert_array_equal(a.row_statistics([], [0, 0]), np.zeros(12, int))
        s = a.row_statistics([(0, 0)], [0, 0])
        np.testing.assert_array_equal(s[9:], [1, 0, 1])
        s = a.row_statistics([], [0, 1])
        np.testing.assert_array_equal(s[9:], [0, 1, 0])
        self.assertEqual(a.label_spans([1, 1, 0, 1]), [(0, 1), (3, 3)])

    def test_fractional_normal_mass_and_zero_denominators(self):
        stats = np.zeros(12)
        stats[9] = .5
        self.assertEqual(a.score_components(stats, .5, 1)['utility'], -.2)
        stats[9], stats[10] = 0, .5
        self.assertEqual(a.score_components(stats, 1, .5)['utility'], -.15)
        self.assertEqual(a.score_components(np.zeros(12), 0, 0)['utility'], 0)


class IndependentGrouping(unittest.TestCase):
    def test_alias_bootstrap_and_rng_order_golden(self):
        rows = [{'sha256': 'x', 'event_count': 0}, {'sha256': 'x', 'event_count': 1},
                {'sha256': 'y', 'event_count': 1}, {'sha256': 'z', 'event_count': 1}]
        counts = a.bootstrap_counts(rows, [0, 1, 2, 3], 17, 3)
        np.testing.assert_array_equal(counts, [[1, 1, 0, 2], [1, 1, 2, 0], [1, 1, 1, 1]])
        weights = a.content_weights(rows, [0, 1, 2, 3])
        np.testing.assert_array_equal(weights, [.5, .5, 1, 1])
        np.testing.assert_array_equal(counts @ weights, [3, 3, 3])

    def test_seeded_group_folds_hold_out_aliases_together(self):
        rows = [record(i, i % 3 != 0) for i in range(16)]
        rows[15]['sha256'] = rows[1]['sha256']
        folds = a.grouped_folds(rows, list(range(16)), 5, a.SEED)
        self.assertEqual(sorted(i for f in folds for i in f), list(range(16)))
        self.assertEqual(next(k for k, f in enumerate(folds) if 1 in f),
                         next(k for k, f in enumerate(folds) if 15 in f))
        self.assertEqual(folds, a.grouped_folds(rows, list(range(16)), 5, a.SEED))
        self.assertNotEqual(folds, a.grouped_folds(rows, list(range(16)), 5, a.SEED + 1))
        for fold in range(5):
            train, val, parts = a.expected_partitions(rows, folds, fold)
            self.assertEqual(sorted(i for p in parts for i in p['validation']), train)
            self.assertEqual([p['seed'] for p in parts], [a.SEED + fold * 53 + k for k in range(3)])

    def test_frozen_outer_consolidates_alias_without_reassigning_other_rows(self):
        rows = [record(i, i % 3 != 0) for i in range(18)]
        rows[17]['sha256'] = rows[1]['sha256']
        original, consolidated = a.frozen_outer_replay(rows)
        old_owner = {i: k for k, ids in enumerate(original) for i in ids}
        new_owner = {i: k for k, ids in enumerate(consolidated) for i in ids}
        self.assertEqual(new_owner[17], old_owner[1])
        for i in range(17):
            self.assertEqual(new_owner[i], old_owner[i])
        self.assertEqual(sorted(i for ids in consolidated for i in ids), list(range(18)))

    def test_reject_content_leakage_even_without_row_overlap(self):
        rows = [record(0), record(0)]
        with self.assertRaisesRegex(a.AuditError, 'content leakage'):
            a.check_isolation(rows, [0], [1], [0, 1], 'test')


class IndependentChoice(unittest.TestCase):
    def perfect_fixture(self):
        rows = [record(0, False), record(1), record(2)]
        p = a.Probabilities({0: np.zeros(10, np.float32), 1: np.ones(10, np.float32)}, {0: .9, 1: .9})
        def decoder(values, fps, video, config):
            return [(2, 7)] if values[0] else []
        decoders = {kind: decoder for kind in ('duration-logit-v4', 'recall-stable-v2')}
        return rows, p, decoders

    def test_all40_deduplicate_earliest_and_known_perfect_utility(self):
        rows, p, decoders = self.perfect_fixture()
        choice, detail = a.independent_choice(rows, a.selection_context(rows, [0, 1], 7), p, decoders)
        self.assertEqual(detail['configurations_evaluated'], 40)
        self.assertEqual(detail['unique_prediction_sets'], 1)
        self.assertEqual(choice['config'], a.configurations()[0])
        self.assertEqual(choice['pooled_utility'], 1.1)
        self.assertEqual(choice['bootstrap_q20'], 1.1)
        self.assertEqual(len(choice['shortlist']), 1)
        self.assertTrue(all(row['duplicate_of_ordinal'] == 0 for row in detail['all_configurations'][1:]))

    def test_unlisted_outer_labels_metadata_scores_do_not_affect_choice(self):
        rows, p, decoders = self.perfect_fixture()
        before = a.independent_choice(rows, a.selection_context(rows, [0, 1], 7), p, decoders)
        rows[2]['labels'][:] = False
        rows[2].update(event_count=0, fps=999., sha256=rows[0]['sha256'])
        p.frame[2] = np.full(10, .999, np.float32)
        p.video[2] = 0
        after = a.independent_choice(rows, a.selection_context(rows, [0, 1], 7), p, decoders)
        self.assertEqual(before, after)

    def test_pooled_top8_precedes_q20_and_excludes_stable_ninth(self):
        rows = [record(0, frames=80)]
        rows[0]['labels'][:] = True
        p = a.Probabilities({0: np.zeros(80, np.float32)}, {0: .9})
        grid = a.configurations()
        def decoder(values, fps, video, config):
            return [(grid.index(config), 79)]
        decoders = {kind: decoder for kind in ('duration-logit-v4', 'recall-stable-v2')}
        q = [100. if ordinal == 8 else 0. for ordinal in range(40)]
        with patch.object(a.np, 'quantile', side_effect=q):
            chosen, detail = a.independent_choice(rows, a.selection_context(rows, [0], 3), p, decoders)
        self.assertEqual([row['config'] for row in chosen['shortlist']], grid[:8])
        self.assertEqual(chosen['config'], grid[0])
        excluded = detail['all_configurations'][8]
        self.assertFalse(excluded['in_pooled_top8'])
        self.assertGreater(excluded['stable_utility'], chosen['stable_utility'])

    def test_candidate_ties_use_protocol_order_not_outer_best(self):
        rows = [{'candidate': name, 'stable_utility': .5,
                 'key': [.6, .4, -2., -1., -3., 0], 'config': a.configurations()[0]}
                for name, _ in a.CANDIDATES]
        primary, ranking = a.rank_candidates(rows)
        self.assertEqual(primary['candidate'], a.CANDIDATES[0][0])
        rows[2]['stable_utility'] = .51
        self.assertEqual(a.rank_candidates(rows)[0]['candidate'], a.CANDIDATES[2][0])
        self.assertEqual(ranking[0]['candidate_ordinal'], 0)


class IntegrityAndContracts(unittest.TestCase):
    def test_probability_extra_key_wrong_shape_nonfinite_and_range_rejected(self):
        rows = [record(0), record(1)]
        metadata = {'train': [0], 'validation': [1]}
        values = {'inner_frame_0': np.full(10, .5, np.float32), 'inner_video_0': np.asarray(.8),
                  'outer_frame_1': np.full(10, .4, np.float32), 'outer_video_1': np.asarray(.9)}
        inner, outer = a.probability_maps(archive(values), metadata, rows, 'fixture')
        self.assertEqual(set(inner.frame), {0})
        self.assertEqual(set(outer.frame), {1})
        for key, replacement in [('extra', np.asarray(0)), ('inner_frame_0', np.full(9, .5)),
                                 ('inner_video_0', np.asarray(np.nan)), ('outer_frame_1', np.full(10, 1.1))]:
            changed = dict(values)
            changed[key] = replacement
            with self.assertRaises(a.AuditError):
                a.probability_maps(archive(changed), metadata, rows, 'fixture')

    def test_metadata_seed_and_fit_evidence_rejected_even_if_resigned(self):
        rows = [record(i) for i in range(8)]
        folds = a.grouped_folds(rows, list(range(8)), 5, a.SEED)
        partition = a.expected_partitions(rows, folds, 0)
        train, val, parts = partition
        metadata = {'fold': 0, 'recipe': a.RECIPE_NAMES[0], 'train': train, 'validation': val,
                    'inner_partitions': parts, 'outer_seed': a.SEED + 99,
                    'training_receipt_sha256': a.RECEIPT_PIN, 'seconds': 1.,
                    'runtime': {'numpy': 'fixture', 'sklearn': 'fixture'},
                    'fit_evidence': [{'partition': k if k < 3 else 'outer',
                                     'fit_content_sha256': sorted({rows[i]['sha256'] for i in p['fit']}),
                                     'transform_sha256': 'a' * 64}
                                    for k, p in enumerate(parts + [{'fit': train}])]}
        metadata['signature'] = a.value_hash(metadata)
        a.verify_training_metadata(metadata, a.RECIPE_NAMES[0], 0, partition, rows, a.RECEIPT_PIN)
        for field in ('outer_seed', 'inner_seed', 'fit_content', 'receipt'):
            changed = deepcopy(metadata)
            if field == 'outer_seed': changed['outer_seed'] += 1
            if field == 'inner_seed': changed['inner_partitions'][0]['seed'] += 1
            if field == 'fit_content': changed['fit_evidence'][0]['fit_content_sha256'] = []
            if field == 'receipt': changed['training_receipt_sha256'] = 'b' * 64
            changed['signature'] = a.value_hash({k: v for k, v in changed.items() if k != 'signature'})
            with self.assertRaises(a.AuditError):
                a.verify_training_metadata(changed, a.RECIPE_NAMES[0], 0, partition, rows, a.RECEIPT_PIN)

    def test_strict_json_rejects_duplicate_nan_and_overflow(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.assertRaises(a.AuditError): a.strict_json(text, 'fixture')

    def test_missing_readiness_no_output_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(a.IncompleteArtifacts): a.complete_inputs(root, False)
            with self.assertRaises(a.IncompleteArtifacts): a.complete_inputs(root, True)
            self.assertFalse((root / a.V4 / a.OUTPUT_NAME).exists())
            (root / a.V4).mkdir(parents=True)
            target = a.write_report(root, {'status': 'fixture'})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError): a.write_report(root, {'status': 'other'})
            self.assertEqual(target.read_bytes(), before)

    def test_evidence_detects_file_change_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'x').write_bytes(b'first')
            evidence = a.Evidence(root)
            evidence.read(Path('x'))
            (root / 'x').write_bytes(b'second')
            with self.assertRaises(a.AuditError): evidence.recheck()
            for relative in ('../escape', 'C:/escape', '/escape', 'a\\b'):
                with self.assertRaises(a.AuditError): a.safe_path(root, relative)

    def test_report_float_tolerance_and_boolean_type_contract(self):
        comparison = a.Comparisons()
        comparison.compare(.5 + 1e-13, .5, 'float')
        self.assertEqual(comparison.mismatch_count, 0)
        comparison.compare(1, True, 'boolean')
        comparison.compare(.5 + 1e-8, .5, 'float')
        self.assertEqual(comparison.mismatch_count, 2)

    def test_guard_exact_boundaries(self):
        baseline = {'iou_0.3': {'f1': .5}, 'iou_0.5': {'f1': .3}, 'frame': {'f1': .6},
                    'normal': {'false_positive_videos': 5}, 'positive_videos_without_candidate': 15}
        primary = deepcopy(baseline)
        primary['iou_0.5']['f1'] += .02
        primary['frame']['f1'] -= .005
        primary['positive_videos_without_candidate'] -= 3
        self.assertTrue(all(a.promotion_guards(primary, baseline).values()))
        primary['normal']['false_positive_videos'] = 6
        self.assertFalse(a.promotion_guards(primary, baseline)['normal_fp_no_worse'])

    def test_no_runner_statistics_or_training_imports_in_verifier(self):
        tree = ast.parse(Path(a.__file__).read_text(encoding='utf-8'))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        forbidden = ('run_compact', 'run_event', 'optimized_event_training', 'run_recall', 'run_algorithm', 'run_optimization',
                     'optimized_video_statistics', 'optimized_grouped_training', 'optimized_compact_model')
        self.assertFalse([name for name in imports if name and name.startswith(forbidden)])



class V5Adaptation(unittest.TestCase):
    def test_v5_has_15_new_caches_and_its_own_candidates_receipt_and_namespace(self):
        experiment = a.EXPERIMENTS['v5']
        self.assertEqual(5 * len(experiment.recipes), 15)
        self.assertEqual(len(experiment.candidates), 5)
        self.assertEqual(len(experiment.sources), 16)
        self.assertIsNone(experiment.fast_recipe)
        self.assertEqual(experiment.empty_penalty, .30)
        self.assertNotEqual(experiment.receipt_pin, a.V4_EXPERIMENT.receipt_pin)
        self.assertNotEqual(experiment.folder, a.V4_EXPERIMENT.folder)
        self.assertEqual(a.V4_EXPERIMENT.empty_penalty, .15)

    def test_v5_uses_doubled_empty_penalty_for_pooled_and_each_bootstrap(self):
        rows = [record(0, False), record(1)]
        p = a.Probabilities({i: np.zeros(10, np.float32) for i in range(2)}, {0: .9, 1: .9})
        decoders = {kind: (lambda *args: []) for kind in ('duration-logit-v4', 'recall-stable-v2')}
        for penalty in (.15, .30):
            context = a.selection_context(rows, [0, 1], 17, penalty)
            choice, detail = a.independent_choice(rows, context, p, decoders)
            self.assertEqual(choice['pooled_utility'], -penalty)
            self.assertEqual(choice['bootstrap_q20'], -penalty)
            self.assertEqual(detail['positive_empty_penalty'], penalty)
            np.testing.assert_array_equal(detail['all_configurations'][0]['bootstrap_utilities'],
                                          np.full(32, -penalty))

    def test_v5_candidate_ranking_still_uses_inner_key_and_fixed_order(self):
        choices = [{'candidate': name, 'stable_utility': .5,
                    'key': [.6, .4, -2., -1., -3., 0], 'config': a.configurations()[0]}
                   for name, _ in a.V5_EXPERIMENT.candidates]
        primary, ranking = a.rank_candidates(choices, a.V5_EXPERIMENT.candidates)
        self.assertEqual(primary['candidate'], 'event_corrected_motion_et')
        choices[-1]['key'][1] = .41
        self.assertEqual(a.rank_candidates(choices, a.V5_EXPERIMENT.candidates)[0]['candidate'],
                         'old_et_recall_control')

    def test_v5_exclusive_output_cannot_touch_existing_v4_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / a.V4).mkdir(parents=True)
            (root / a.V5_EXPERIMENT.folder).mkdir(parents=True)
            v4 = a.write_report(root, {'status': 'v4_fixture'})
            before = v4.read_bytes()
            v5 = a.write_report(root, {'status': 'v5_fixture'}, a.V5_EXPERIMENT)
            self.assertNotEqual(v4, v5)
            self.assertEqual(v4.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                a.write_report(root, {'status': 'other'}, a.V5_EXPERIMENT)


if __name__ == '__main__':
    unittest.main()
