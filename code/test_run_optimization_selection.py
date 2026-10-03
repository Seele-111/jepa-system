r"""Tiny synthetic folds/caches/probabilities only; no real-data training or GPU.

Windows: python -B -m unittest discover -s E:\jepa-system\code -p test_run_optimization_selection.py -v
WSL uses the existing CPU sklearn environment for a tiny export parity check.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_optimization_selection as selection
from optimized_detector import predict_record
from optimized_feature_view import feature_view
from optimized_locator import decode, metrics, objective, portable_predict

HAS_SKLEARN = importlib.util.find_spec('sklearn') is not None


def config(**kwargs):
    value = {'threshold': .5, 'low_ratio': 1., 'smooth_seconds': 0., 'gap_seconds': 0.,
             'min_seconds': 0., 'video_threshold': 0., 'video_strength': 0., 'boundary_seconds': 0.}
    value.update(kwargs)
    return value


def synthetic_dataset():
    records = []
    rng = np.random.default_rng(7)
    for i in range(10):
        labels = np.array([0, 0, 1, 1, 0, 0], np.int64) if i % 2 else np.zeros(6, np.int64)
        records.append({'name': f'fake-{i}', 'sha256': hashlib.sha256(str(i).encode()).hexdigest(),
            'frames': 6, 'fps': 10., 'event_count': int(i % 2), 'generator': 'fake', 'prompt_id': str(i),
            'labels': labels, 'jepa': rng.random((6, 3)).astype(np.float32),
            'jepa_names': ['j0', 'j1', 'j2'], 'rank_motion': rng.random((6, 4)).astype(np.float32),
            'motion': rng.random((6, 2)).astype(np.float32), 'motion_names': ['m0', 'm1'],
            'corrected': rng.random((6, 3)).astype(np.float32), 'corrected_names': ['c0', 'c1', 'c2'],
            'rgb': rng.random((6, 4)).astype(np.float32)})
    rows = [{k: r[k] for k in ('name', 'sha256', 'frames', 'fps', 'event_count', 'generator', 'prompt_id')}
            for r in records]
    manifest = {'schema_version': 'algorithm-opt-dataset-v1', 'rows': rows,
                'role': 'development_nested_video_cv_not_blind',
                'feature_names': {'jepa': ['j0', 'j1', 'j2']},
                'protocol': {'outer_folds': 5, 'inner_folds': 3, 'seed': selection.SEED},
                'outer_folds': [[2 * f, 2 * f + 1] for f in range(5)]}
    return manifest, records


def pack(records, indices, good=False, boundary=False):
    fp = {i: records[i]['labels'].astype(np.float32) * .9 + .05 if good
          else np.zeros(len(records[i]['labels']), np.float32) for i in indices}
    return selection.Probabilities(fp, {i: .8 for i in indices},
        {i: (np.ones(6, np.float32) * .2, np.ones(6, np.float32) * .8) for i in indices} if boundary else None)


def synthetic_caches(manifest, records):
    caches = {}
    for recipe in selection.SINGLE_RECIPES:
        caches[recipe] = {}
        for fold, val in enumerate(manifest['outer_folds']):
            train, _ = selection.inner_splits(records, val, fold)
            caches[recipe][fold] = selection.FoldCache(
                pack(records, train, good=recipe == 'motion_et', boundary='boundary' in recipe),
                pack(records, val, good=recipe == 'motion_rf', boundary='boundary' in recipe))
    return caches


def simple_chooser(records, indices, fp, vp, bp=None):
    cfg = config()
    predictions = [decode(fp[i], records[i]['fps'], vp[i], cfg) for i in indices]
    return cfg, metrics(predictions, [records[i]['labels'] for i in indices])


def cache_row(manifest, records, recipe, fold, dataset_hash='dataset', signature='signature'):
    val = manifest['outer_folds'][fold]
    train, _ = selection.inner_splits(records, val, fold)
    result = {'fold': fold, 'recipe': recipe, 'validation_indices': val,
              'train_names': [records[i]['name'] for i in train],
              'validation_names': [records[i]['name'] for i in val],
              'dataset_manifest_sha256': dataset_hash, 'experiment_signature': signature}
    for prefix, indices in [('', val), ('inner_', train)]:
        p = pack(records, indices, good=True, boundary='boundary' in recipe)
        result[prefix + 'frame_probabilities'] = {str(i): p.frame[i].tolist() for i in indices}
        result[prefix + 'video_probabilities'] = {str(i): p.video[i] for i in indices}
        result[prefix + 'boundary_probabilities'] = None if p.boundary is None else {
            str(i): [v.tolist() for v in p.boundary[i]] for i in indices}
    return result


def write_fixture(root, manifest, records):
    selection.json_write(root / 'dataset_manifest.json', manifest)
    dataset_hash = selection.sha256(root / 'dataset_manifest.json')
    np.savez(root / 'dataset.npz', **{f'{key}_{i}': r[key] for i, r in enumerate(records)
                                  for key in ('jepa', 'rank_motion', 'labels')})
    source_paths = [root / 'dataset_manifest.json', root / 'dataset.npz']
    for key, folder in [('motion', 'motion_features'), ('rgb', 'rgb_features'), ('corrected', selection.CORRECTED_FOLDER)]:
        directory = root / folder
        directory.mkdir()
        profile = {'name': 'synthetic-' + key}
        if key == 'corrected':
            profile['i_mask_reuse_policy'] = 'explicit_batch_seed_reset_and_restore'
        if key != 'rgb':
            profile['feature_names'] = records[0][key + '_names']
            selection.json_write(directory / 'feature_names.json', profile['feature_names'])
        entries = []
        for i, r in enumerate(records):
            file = directory / f'v{i}.npz'
            np.savez(file, signals=r[key])
            entries.append({'name': r['name'], 'file': file.name, 'status': 'ok',
                            'source_sha256': r['sha256'], 'feature_sha256': selection.sha256(file)})
        selection.json_write(directory / 'manifest.json', {'profile': profile, 'videos': entries,
            'status': 'complete', 'dataset_manifest_sha256': dataset_hash})
        source_paths.append(directory / 'manifest.json')
    for folder, recipes in selection.REPORT_RECIPES.items():
        directory = root / folder
        snapshot = directory / 'sources' / 'fake_runner.py'
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text('# frozen synthetic producer\n', encoding='utf-8')
        sources = {str(p): selection.sha256(p) for p in source_paths}
        sources['/different/host/code/fake_runner.py'] = selection.sha256(snapshot)
        signature = hashlib.sha256(json.dumps({'sources': sources, 'recipes': list(recipes)}, sort_keys=True).encode()).hexdigest()
        report = {'schema_version': 'algorithm-opt-nested-cv-v1', 'protocol': manifest['protocol'],
            'dataset_manifest_sha256': dataset_hash, 'recipes': list(recipes),
            'experiment_signature': signature, 'experiment_sources': sources,
            'nested_selected_strategy': {}, 'family_oof': {},
            'selected_predictions': [{'name': r['name']} for r in records],
            'folds': [{'fold': fold, 'families': [{'recipe': r} for r in recipes]} for fold in range(5)]}
        selection.json_write(directory / 'report.json', report)
        for recipe in recipes:
            for fold in range(5):
                selection.json_write(directory / 'folds' / f'fold{fold}_{recipe}.json',
                    cache_row(manifest, records, recipe, fold, dataset_hash, signature))


class ProtocolAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.records = synthetic_dataset()

    def test_fixed_candidates_and_no_legacy_candidates(self):
        self.assertEqual(len(selection.CANDIDATES), 13)
        self.assertEqual(len(selection.SINGLE_RECIPES), 9)
        self.assertEqual([c.name for c in selection.FAST_CANDIDATES], ['motion_rf', 'motion_et'])
        self.assertNotIn('jepa_tcn', selection.SINGLE_RECIPES)
        self.assertEqual([tuple(w for _, w in c.members) for c in selection.CANDIDATES[-4:]],
                         [(.5, .5), (.75, .25), (.25, .75), (.4, .4, .2)])
        for members in [(('jepa_tcn', 1.),), (('motion_rf', .5),), (('motion_rf', -1.),),
                        (('motion_rf', .5), ('motion_rf', .5))]:
            with self.assertRaises(ValueError):
                selection.Candidate('bad', members)

    def test_inner_training_validation_and_outer_are_disjoint_and_complete(self):
        selection.validate_protocol(self.manifest, self.records)
        for f, outer in enumerate(self.manifest['outer_folds']):
            train, splits = selection.inner_splits(self.records, outer, f)
            self.assertEqual(len(splits), 3)
            self.assertEqual(sorted(i for s in splits for i in s['validation_indices']), train)
            for split in splits:
                a, b, c = map(set, (split['train_indices'], split['validation_indices'], outer))
                self.assertFalse(a & b or a & c or b & c)
                self.assertEqual(a | b, set(train))

    def test_rejects_changed_protocol_partial_outer_and_wrong_count(self):
        for field, value in [('inner_folds', 2), ('outer_folds', 4), ('seed', 0)]:
            manifest = deepcopy(self.manifest)
            manifest['protocol'][field] = value
            with self.assertRaises(selection.EvidenceError):
                selection.validate_protocol(manifest, self.records)
        manifest = deepcopy(self.manifest)
        manifest['outer_folds'][0].pop()
        with self.assertRaises(selection.EvidenceError):
            selection.validate_protocol(manifest, self.records)
        with self.assertRaises(selection.EvidenceError):
            selection.validate_protocol(self.manifest, self.records, 77)

    def test_fold_cache_rejects_stale_names_indices_and_inner_outer_leak(self):
        row = cache_row(self.manifest, self.records, 'motion_rf', 0)
        selection.validate_fold_row(row, 'motion_rf', 0, self.manifest, self.records, 'dataset', 'signature')
        for key, value in [('train_names', []), ('validation_indices', [1, 0]),
                           ('dataset_manifest_sha256', 'old'), ('experiment_signature', 'old')]:
            changed = deepcopy(row)
            changed[key] = value
            with self.assertRaises(selection.EvidenceError):
                selection.validate_fold_row(changed, 'motion_rf', 0, self.manifest, self.records, 'dataset', 'signature')
        row['inner_frame_probabilities']['0'] = [0.] * 6
        with self.assertRaisesRegex(selection.EvidenceError, 'OOF indices'):
            selection.validate_fold_row(row, 'motion_rf', 0, self.manifest, self.records, 'dataset', 'signature')

    def test_probability_guards_and_required_boundary_curves(self):
        row = cache_row(self.manifest, self.records, 'motion_rf', 0)
        for bad in [[float('nan')] * 6, [-.1] * 6, [1.1] * 6, [0.] * 5, [[0.]] * 6, ['0'] * 6]:
            changed = deepcopy(row)
            changed['frame_probabilities']['0'] = bad
            with self.assertRaises(selection.EvidenceError):
                selection.probabilities_from_row(changed, self.records, [0, 1])
        row['video_probabilities']['0'] = True
        with self.assertRaises(selection.EvidenceError):
            selection.probabilities_from_row(row, self.records, [0, 1])
        row = cache_row(self.manifest, self.records, 'corrected_motion_boundary_rf', 0)
        row['inner_boundary_probabilities'] = None
        with self.assertRaisesRegex(selection.EvidenceError, 'boundary'):
            selection.probabilities_from_row(row, self.records, list(range(2, 10)), True)

    def test_complete_synthetic_cache_load_and_source_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root, self.manifest, self.records)
            with patch.object(selection, 'EXPECTED_VIDEOS', 10):
                manifest, records, profiles, hashes = selection.load_dataset(root)
            caches, sources, experiments = selection.load_fold_caches(root, manifest, records, hashes)
            self.assertEqual(len(caches), 9)
            self.assertEqual(len(sources), 49)
            self.assertEqual(len(experiments), 4)
            self.assertEqual(set(profiles), {'motion', 'rgb', 'corrected'})
            self.assertEqual(records[0]['motion_names'], ['m0', 'm1'])

    def test_missing_fold_never_runs_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root, self.manifest, self.records)
            path = root / 'nested_fast_seed0' / 'folds' / 'fold4_motion_et.json'
            path.unlink()
            with self.assertRaisesRegex(selection.EvidenceError, 'refusing subset'):
                selection.load_fold_caches(root, self.manifest, self.records, {})

    def test_tampered_feature_source_or_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root, self.manifest, self.records)
            snapshot = root / 'nested_fast_seed0' / 'sources' / 'fake_runner.py'
            snapshot.write_text('# changed', encoding='utf-8')
            with self.assertRaisesRegex(selection.EvidenceError, 'hash mismatch'):
                selection.load_fold_caches(root, self.manifest, self.records, {})

    def test_json_duplicate_indices_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.json'
            path.write_text('{"0": 1, "0": 2}', encoding='utf-8')
            with self.assertRaisesRegex(selection.EvidenceError, 'duplicate'):
                selection.read_json(path)

    def test_default_cli_reads_nothing_without_confirmation(self):
        with patch.object(selection, 'run') as runner, patch('sys.stderr'):
            with self.assertRaises(SystemExit) as error:
                selection.main([])
        self.assertEqual(error.exception.code, 2)
        runner.assert_not_called()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.records = synthetic_dataset()
        self.caches = synthetic_caches(self.manifest, self.records)

    def test_objective_penalizes_normal_false_positives(self):
        labels = [np.zeros(6, np.int64), np.array([0, 0, 1, 1, 0, 0])]
        good = metrics([[], [(2, 3)]], labels)
        false_positive = metrics([[(0, 5)], [(2, 3)]], labels)
        self.assertEqual(objective(good), 1.)
        expected = .5 * (false_positive['iou_0.3']['f1'] + false_positive['iou_0.5']['f1']) - .2
        self.assertAlmostEqual(objective(false_positive), expected)
        self.assertLess(objective(false_positive), objective(good))

    def test_outer_good_family_cannot_win_inner_selection_and_all_results_retained(self):
        calls = []
        def spy(records, indices, fp, vp, bp):
            fold = len(calls) // 13
            self.assertFalse(set(indices) & set(self.manifest['outer_folds'][fold]))
            self.assertEqual(set(fp), set(indices))
            calls.append(indices[:])
            return simple_chooser(records, indices, fp, vp, bp)
        report, oof = selection.nested_selection(self.manifest, self.records, self.caches, chooser=spy)
        self.assertEqual(len(calls), 65)
        self.assertTrue(all(f['selected_candidate'] == 'motion_et' for f in report['folds']))
        self.assertEqual(report['primary_nested_selection']['metrics']['all']['iou_0.5']['f1'], 0.)
        self.assertEqual(report['fixed_candidate_oof']['motion_rf']['metrics']['all']['iou_0.5']['f1'], 1.)
        self.assertEqual(len(report['fixed_candidate_oof']), 13)
        self.assertEqual(report['primary_nested_selection']['no_candidates']['positive_videos'], 5)
        self.assertEqual(report['primary_nested_selection']['normal']['false_positive_videos'], 0)
        self.assertEqual(len(oof['motion_et'].frame), 10)

    def test_changing_outer_curves_never_changes_family_or_decoder(self):
        before, _ = selection.nested_selection(self.manifest, self.records, self.caches, chooser=simple_chooser)
        changed = deepcopy(self.caches)
        for recipe, folds in changed.items():
            for fold, saved in folds.items():
                saved.outer = pack(self.records, self.manifest['outer_folds'][fold],
                                   good=True, boundary='boundary' in recipe)
        after, _ = selection.nested_selection(self.manifest, self.records, changed, chooser=simple_chooser)
        self.assertEqual([(f['selected_candidate'], f['decoder']) for f in before['folds']],
                         [(f['selected_candidate'], f['decoder']) for f in after['folds']])

    def test_incomplete_in_memory_cache_is_rejected(self):
        del self.caches['motion_rf'][4]
        with self.assertRaisesRegex(selection.EvidenceError, 'all five'):
            selection.nested_selection(self.manifest, self.records, self.caches, chooser=simple_chooser)

    def test_blend_probabilities_and_boundary_mass_normalization(self):
        candidate = selection.Candidate('test-head-blend', (
            ('corrected_motion_boundary_rf', .2), ('corrected_motion_boundary_et', .3), ('motion_rf', .5)))
        packs = {'corrected_motion_boundary_rf': pack(self.records, [0], boundary=True),
                 'corrected_motion_boundary_et': pack(self.records, [0], boundary=True),
                 'motion_rf': pack(self.records, [0])}
        packs['corrected_motion_boundary_rf'].frame[0].fill(.1)
        packs['corrected_motion_boundary_et'].frame[0].fill(.3)
        packs['motion_rf'].frame[0].fill(.9)
        packs['corrected_motion_boundary_rf'].boundary[0][0].fill(.1)
        packs['corrected_motion_boundary_et'].boundary[0][0].fill(.9)
        blended = selection.blend_probabilities(candidate, packs, [0])
        np.testing.assert_allclose(blended.frame[0], .56, atol=1e-7)
        np.testing.assert_allclose(blended.boundary[0][0], .58, atol=1e-7)
        self.assertEqual(blended.frame[0].dtype, np.float32)
        del packs['motion_rf'].video[0]
        with self.assertRaisesRegex(selection.EvidenceError, 'partitions'):
            selection.blend_probabilities(candidate, packs, [0])

    def test_pre_boundary_single_search_is_semantically_identical(self):
        import ast
        import run_algorithm_optimization as runner
        grid = next(n for n in ast.parse(Path(runner.__file__).read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'decoder_grid')
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'runner.py'
            source.write_text(ast.unparse(grid) + '\n' + selection.LEGACY_NON_BOUNDARY_SEARCH, encoding='utf-8')
            self.assertTrue(selection.same_search_implementation(source, ['motion_rf']))
            self.assertFalse(selection.same_search_implementation(source, ['corrected_motion_boundary_rf']))
        env = {'decode': decode, 'metrics': metrics, 'objective': objective, 'decoder_grid': runner.decoder_grid}
        exec(selection.LEGACY_NON_BOUNDARY_SEARCH, env)
        p = self.caches['motion_et'][0].inner
        indices = list(p.frame)
        def tiny_grid():
            yield {k: v for k, v in config(min_seconds=.08).items()
                   if k not in ('video_threshold', 'video_strength', 'boundary_seconds')}
        with patch.object(runner, 'decoder_grid', side_effect=tiny_grid):
            env['decoder_grid'] = tiny_grid
            cfg, result = runner.choose_decoder(self.records, indices, p.frame, p.video)
            old_cfg, old_result = env['choose_decoder'](self.records, indices, p.frame, p.video)
        old_cfg['boundary_seconds'] = 0
        self.assertEqual((cfg, result), (old_cfg, old_result))

    def test_reused_single_search_matches_same_grid_recomputed_selection(self):
        from run_algorithm_optimization import choose_decoder
        def tiny_grid():
            yield {k: v for k, v in config(min_seconds=.08).items()
                   if k not in ('video_threshold', 'video_strength', 'boundary_seconds')}
        with patch('run_algorithm_optimization.decoder_grid', side_effect=tiny_grid):
            baseline, _ = selection.nested_selection(self.manifest, self.records, self.caches)
            for recipe, folds in self.caches.items():
                for fold, saved in folds.items():
                    train, _ = selection.inner_splits(self.records, self.manifest['outer_folds'][fold], fold)
                    cfg, result = choose_decoder(self.records, train, saved.inner.frame, saved.inner.video, saved.inner.boundary)
                    saved.inner_selection = {'config': cfg, 'metrics': result}
            with patch.object(selection, 'choose_decoder', wraps=choose_decoder) as chooser:
                reused, _ = selection.nested_selection(self.manifest, self.records, self.caches)
            self.assertEqual(chooser.call_count, 20)  # four blends, five outer folds
        self.assertEqual(baseline['primary_nested_selection'], reused['primary_nested_selection'])
        self.assertEqual([(f['selected_candidate'], f['decoder']) for f in baseline['folds']],
                         [(f['selected_candidate'], f['decoder']) for f in reused['folds']])

    def test_deployment_selection_marked_optimistic_and_not_validation(self):
        packs = {r: pack(self.records, list(range(10)), good=r == 'motion_rf', boundary='boundary' in r)
                 for r in selection.SINGLE_RECIPES}
        result = selection.deployment_selection(self.records, packs, chooser=simple_chooser)
        self.assertFalse(result['is_validation_score'])
        self.assertIn('optimistic', result['role'])
        self.assertEqual(len(result['candidates']), 13)
        self.assertEqual(result['selected']['candidate'], 'motion_rf')
        self.assertEqual(result['fast_selected']['candidate'], 'motion_rf')


class ConstantForest:
    """Minimal sklearn-like forest, used without fitting or installing sklearn."""
    def __init__(self, dimension, probability):
        self.n_features_in_ = dimension
        self.classes_ = np.array([0, 1])
        self.probability = probability
        tree = SimpleNamespace(children_left=np.array([-1]), children_right=np.array([-1]),
            feature=np.array([-2]), threshold=np.array([-2.]),
            value=np.array([[[1 - probability, probability]]]))
        self.estimators_ = [SimpleNamespace(tree_=tree)]

    def predict_proba(self, values):
        return np.tile([1 - self.probability, self.probability], (len(values), 1))


def fake_fit_state(recipe, records, boundary=False, temporal=None):
    pca = {'mean': [0.] * 4, 'components': np.eye(4).tolist()} if recipe.startswith('rgb_') else None
    views = [feature_view(recipe, r, pca) for r in records]
    x = [v[0] for v in views]
    video = np.stack([v[1] for v in views])
    transform = {'recipe': recipe, 'frame_feature_names': views[0][2], 'pca': pca,
                 'feature_view': 'shared-label-free-v1'}
    fm = ConstantForest(x[0].shape[1], .7) if temporal is None else temporal(x[0].shape[1])
    vm = ConstantForest(video.shape[1], .8)
    if temporal:
        from optimized_temporal_head import predict_temporal
        fp = {i: predict_temporal(fm, v, records[i]['fps']) for i, v in enumerate(x)}
    else:
        fp = {i: fm.predict_proba(v)[:, 1].astype(np.float32) for i, v in enumerate(x)}
    vp = {i: float(vm.predict_proba(v[None])[0, 1]) for i, v in enumerate(video)}
    heads = None
    bp = None
    if boundary:
        heads = {key: {'kind': 'constant', 'n_features': x[0].shape[1], 'probability': p}
                 for key, p in [('start_model', .2), ('end_model', .9)]}
        bp = {i: (np.full(len(v), .2, np.float32), np.full(len(v), .9, np.float32)) for i, v in enumerate(x)}
    return fp, vp, (fm, vm, transform, x, video,
                   {'boundary_models': heads, 'boundary_probabilities': bp})


class DeploymentParityTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.records = synthetic_dataset()
        self.records = self.records[:2]
        self.profiles = {key: {'name': 'actual-synthetic-' + key} for key in ('motion', 'rgb', 'corrected')}
        self.provenance = {'dataset_manifest_sha256': 'dataset', 'dataset_npz_sha256': 'npz',
                           'report_sources': {'report.json': 'hash'}, 'source_snapshot_sha256': {}}
        self.selection = {'role': 'optimistic_all_development_OOF_selection_not_validation', 'note': 'not validation'}

    def test_single_bundle_roundtrip_predict_record_and_fit_views(self):
        recipe = 'motion_rf'
        fp, vp, state = fake_fit_state(recipe, self.records)
        calls = []
        def predictor(bundle, record):
            calls.append(record['name'])
            return predict_record(bundle, record)
        member, pack, errors = selection.export_fitted_member(recipe, self.records, fp, vp, state, predictor)
        self.assertEqual(len(calls), 2)
        self.assertLess(errors['frame'], 2e-6)
        candidate = selection.Candidate(recipe, ((recipe, 1.),))
        bundle = selection.build_bundle(candidate, config(), {recipe: member}, self.records,
            self.manifest, self.profiles, self.provenance, self.selection)
        self.assertEqual(bundle['recipe'], recipe)
        self.assertEqual(bundle['members'][0]['weight'], 1.)
        self.assertEqual(bundle['feature_profiles'], self.profiles)
        self.assertFalse(bundle['deployment_selection']['is_validation_score'])
        result = selection.verify_bundle(bundle, candidate, self.records, {recipe: pack}, predictor)
        self.assertTrue(result['passed'])
        self.assertTrue(result['intervals_exact'])

    def test_boundary_blend_parity_uses_predict_record_real_refinement(self):
        recipe = 'corrected_motion_boundary_rf'
        members, packs = {}, {}
        for r, boundary in [(recipe, True), ('motion_et', False)]:
            fp, vp, state = fake_fit_state(r, self.records, boundary=boundary)
            members[r], packs[r], _ = selection.export_fitted_member(r, self.records, fp, vp, state)
        candidate = selection.Candidate('synthetic-boundary-blend', ((recipe, .25), ('motion_et', .75)))
        bundle = selection.build_bundle(candidate, config(boundary_seconds=.2), members, self.records,
            self.manifest, self.profiles, self.provenance, self.selection)
        self.assertEqual(bundle['recipe'], 'ensemble')
        p = selection.blend_probabilities(candidate, packs, [0, 1])
        np.testing.assert_allclose(p.boundary[0][0], .2)
        result = selection.verify_bundle(bundle, candidate, self.records, packs)
        self.assertTrue(result['passed'])

    def test_temporal_schema_alias_and_numpy_detector_dispatch_without_training(self):
        from test_optimized_temporal_head import synthetic_bundle
        recipe = 'rgb_motion_tcn'
        fp, vp, state = fake_fit_state(recipe, self.records, temporal=synthetic_bundle)
        member, pack, _ = selection.export_fitted_member(recipe, self.records, fp, vp, state)
        self.assertEqual(member['frame_model']['schema'], 'optimized-temporal-head-v1')
        self.assertEqual(member['frame_model']['schema_version'], 'optimized-temporal-head-v1')
        candidate = selection.Candidate(recipe, ((recipe, 1.),))
        bundle = selection.build_bundle(candidate, config(), {recipe: member}, self.records,
            self.manifest, self.profiles, self.provenance, self.selection)
        self.assertTrue(selection.verify_bundle(bundle, candidate, self.records, {recipe: pack})['passed'])

    def test_hard_parity_rejects_probability_and_feature_view_drift(self):
        fp, vp, state = fake_fit_state('motion_rf', self.records)
        wrong = deepcopy(fp)
        wrong[0][0] += .01
        with self.assertRaisesRegex(selection.ParityError, 'portable frame'):
            selection.export_fitted_member('motion_rf', self.records, wrong, vp, state)
        changed = deepcopy(state)
        changed[3][0][0, 0] += .01
        with self.assertRaisesRegex(selection.ParityError, 'frame view'):
            selection.export_fitted_member('motion_rf', self.records, fp, vp, changed)
        with self.assertRaisesRegex(selection.ParityError, 'six-tuple'):
            selection.export_fitted_member('motion_rf', self.records, fp, vp, state[:5])

    @unittest.skipUnless(HAS_SKLEARN, 'run tiny CPU sklearn check in existing WSL environment')
    def test_tiny_sklearn_fit_and_export_hard_parity(self):
        from sklearn.ensemble import ExtraTreesClassifier
        recipe = 'motion_et'
        _, _, state = fake_fit_state(recipe, self.records)
        _, _, transform, x, videos, extra = state
        fm = ExtraTreesClassifier(n_estimators=4, max_depth=2, n_jobs=1, random_state=5).fit(
            np.concatenate(x), np.concatenate([r['labels'] for r in self.records]))
        vm = ExtraTreesClassifier(n_estimators=4, max_depth=2, n_jobs=1, random_state=6).fit(videos, [0, 1])
        fp = {i: fm.predict_proba(v)[:, 1].astype(np.float32) for i, v in enumerate(x)}
        vp = {i: float(vm.predict_proba(v[None])[0, 1]) for i, v in enumerate(videos)}
        member, _, errors = selection.export_fitted_member(recipe, self.records, fp, vp,
            (fm, vm, transform, x, videos, extra))
        self.assertLess(errors['frame'], 2e-6)
        np.testing.assert_allclose(portable_predict(member['video_model'], videos), [vp[0], vp[1]], atol=2e-6)

    def test_runner_fits_only_selected_members_once_and_writes_only_json_models(self):
        manifest, records = synthetic_dataset()
        caches = synthetic_caches(manifest, records)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection.json_write(root / 'dataset_manifest.json', manifest)
            np.savez(root / 'dataset.npz', synthetic=np.zeros(1))
            def fitter(recipe, passed_records, train, predict, seed):
                self.assertEqual(train, list(range(10)))
                self.assertEqual(predict, train)
                return fake_fit_state(recipe, passed_records)
            with patch.object(selection, 'load_dataset', return_value=(manifest, records, self.profiles, {})), \
                 patch.object(selection, 'load_fold_caches', return_value=(caches, {}, {})), \
                 patch.object(selection, 'snapshot_sources', return_value={}), \
                 patch.object(selection, 'choose_decoder', side_effect=simple_chooser), \
                 patch.object(selection, 'fit_predict', side_effect=fitter) as fit, \
                 patch.dict('os.environ'):
                report = selection.run(root, deployment_fit=True)
            self.assertEqual([call.args[0] for call in fit.call_args_list], ['motion_rf'])
            self.assertEqual(report['status'], 'complete')
            output = root / 'deployment_selection'
            self.assertTrue((output / 'locator_bundle.json').is_file())
            self.assertTrue((output / 'fast_locator_bundle.json').is_file())
            self.assertFalse((root / 'models').exists())
            self.assertEqual(len(report['fixed_candidate_oof']), 13)

    def test_failed_attempt_saved_without_overwriting_previous_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {'status': 'complete', 'previous': True}
            selection.json_write(root / 'deployment_selection' / 'report.json', previous)
            with patch.object(selection, 'load_dataset', side_effect=selection.EvidenceError('missing fold')):
                with self.assertRaises(selection.EvidenceError):
                    selection.run(root)
            self.assertEqual(selection.read_json(root / 'deployment_selection' / 'report.json'), previous)
            failures = list((root / 'deployment_selection').glob('failure_*.json'))
            self.assertEqual(len(failures), 1)
            self.assertEqual(selection.read_json(failures[0])['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
