"""Small CPU-only tests: synthetic caches live entirely in BytesIO/dictionaries.

Run: python -B -m unittest test_optimized_training_provenance -v
No temporary files, training, GPU libraries, service calls or original cache edits.
"""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import optimized_training_provenance as p


def npz_bytes(arrays):
    stream = BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')


class MemoryInventory:
    """Real NPZ/JSON formats with 16 short rows, 15 groups and all 65 fits."""
    def __init__(self):
        self.root = p.ROOT / '__memory_only_provenance_fixture__'
        self.files = {}
        self.n = 16
        self.records = []
        rows = []
        for i in range(self.n):
            row = {'name': f'video_{i}.mp4', 'input_path': f'C:/fixture/video_{i}.mp4',
                   'sha256': p._sha(str(0 if i == 1 else i).encode()), 'frames': 4,
                   'fps': 10., 'generator': 'synthetic', 'prompt_id': f'prompt-{i}',
                   'event_count': int(i % 2 != 0)}
            rows.append(row)
            y = np.array([0, int(i % 2 != 0), int(i % 2 != 0), 0], np.int64)
            self.records.append({**row, 'labels': y, 'jepa': np.full((4, 1), .1, np.float32),
                                 'rank_motion': np.full((4, 1), .2, np.float32), 'jepa_names': ['j']})
        old = {'schema_version': 'algorithm-opt-dataset-v1', 'rows': rows,
               'feature_names': {'jepa': ['j'], 'rank_motion': ['r']}, 'excluded': []}
        self.put_json(p.OLD + '/dataset_manifest.json', old)
        old_sha = p._sha(self.files[p.OLD + '/dataset_manifest.json'])
        folds = [[i for i in range(self.n) if (0 if i == 1 else i) % 5 == k] for k in range(5)]
        self.manifest = {'schema_version': 'algorithm-opt-grouped-dataset-v2', 'rows': rows,
                         'v1_manifest_sha256': old_sha,
                         'protocol': {'outer_folds': 5, 'inner_folds': 3, 'seed': p.SEED},
                         'outer_folds': folds, 'content_groups': [[0, 1]] + [[i] for i in range(2, self.n)]}
        self.put_json(p.NEW + '/dataset_manifest.json', self.manifest)
        self.manifest_sha = p._sha(self.files[p.NEW + '/dataset_manifest.json'])
        base = {f'{k}_{i}': r[k] for i, r in enumerate(self.records) for k in ['labels', 'jepa', 'rank_motion']}
        self.files[p.OLD + '/dataset.npz'] = npz_bytes(base)
        self.snapshots = {}
        for name in p.LEGACY_SNAPSHOT_SHA256:
            data = ('frozen synthetic snapshot ' + name).encode()
            self.files[p.NEW + '/grouped_training/sources/' + name] = data
            self.files['code/' + name] = data
            self.snapshots[name] = p._sha(data)
        for name in p.NEW_TRAINING_SOURCES:
            self.files.setdefault(name, ('synthetic current source ' + name).encode())
        for name in ['optimized_motion_features.py', 'optimized_local_motion.py', 'build_optimization_local_motion.py']:
            self.files['code/' + name] = ('synthetic feature source ' + name).encode()
        for name in ['extractor_source_v2.py', 'adapter_source_v2.py']:
            self.files[p.OLD + '/corrected_jepa_features_v2/' + name] = ('synthetic ' + name).encode()
        self.feature_folders = {'motion': p.OLD + '/motion_features', 'rgb': p.OLD + '/rgb_features',
                                'corrected': p.OLD + '/corrected_jepa_features_v2', 'local': p.NEW + '/local_motion'}
        schemas = {'motion': 'absolute-motion-feature-cache-v1', 'rgb': 'optimized-frozen-r3d18-v1',
                   'corrected': 'corrected-true-jepa-feature-cache-v1', 'local': 'local-motion-cache-v1'}
        for group, folder in self.feature_folders.items():
            names = [group + '0', group + '1']
            profile = {'name': group, 'feature_names': names}
            if group == 'motion':
                profile['code_sha256'] = p._sha(self.files['code/optimized_motion_features.py'])
            if group == 'corrected':
                profile.update(extractor_sha256=p._sha(self.files[folder + '/extractor_source_v2.py']),
                               adapter_sha256=p._sha(self.files[folder + '/adapter_source_v2.py']),
                               i_mask_reuse_policy='explicit_batch_seed_reset_and_restore')
            if group == 'local':
                profile.update(extractor_sha256=p._sha(self.files['code/optimized_local_motion.py']),
                               builder_sha256=p._sha(self.files['code/build_optimization_local_motion.py']))
            if group == 'rgb':
                profile = {'schema_version': schemas[group], 'feature_dim': 2}
                profile_hash = p._sha(json_bytes(profile))
                profile['profile_id'] = profile_hash
                self.put_json(folder + '/feature_profile.json', profile)
            else:
                signature_profile = {k: v for k, v in profile.items() if k != 'feature_names'} if group == 'corrected' else profile
                profile_hash = p._digest_value(signature_profile) if group == 'local' else p._sha(json_bytes(signature_profile))
                self.put_json(folder + '/feature_names.json', names)
            entries = []
            for i, row in enumerate(rows):
                x = np.full((4, 2), .25 + i / 100, np.float32)
                arrays = {'features' if group == 'rgb' else 'signals': x, 'fps': np.asarray(10.)}
                entry = {'status': 'ok', 'fps': 10.}
                if group == 'rgb':
                    entry.update(name=row['name'], video_name=row['name'], npz_path=f'v{i:03d}.npz', frame_count=4,
                                 profile_id=profile_hash, source_size_bytes=100 + i, source_mtime_ns=1000 + i,
                                 source_path=row['input_path'])
                    embedded = {k: entry[k] for k in ['profile_id', 'fps', 'frame_count', 'source_size_bytes', 'source_mtime_ns', 'source_path']}
                    arrays.update(feature_profile_json=np.asarray(json.dumps(profile)),
                                  metadata_json=np.asarray(json.dumps(embedded)))
                else:
                    entry.update(file=f'v{i:03d}.npz', frames=4, source_sha256=row['sha256'])
                    arrays['frame_ids'] = np.arange(4)
                    if group == 'local':
                        entry.update(row_id=i, source={'input_path': row['input_path'], 'fps': 10.,
                                                     'frame_count': 4, 'sha256': row['sha256']},
                                     extractor_sha256=profile['extractor_sha256'])
                        arrays['feature_names'] = np.asarray(names)
                    else:
                        entry['name'] = row['name']
                    if group in ('motion', 'local'):
                        entry['profile_signature'] = profile_hash
                        arrays.update(feature_valid=np.ones((4, 2), bool), frame_times_seconds=np.arange(4) / 10.)
                filename = entry.get('file', entry.get('npz_path'))
                data = npz_bytes(arrays)
                self.files[folder + '/' + filename] = data
                if group != 'rgb':
                    entry['feature_sha256'] = p._sha(data)
                if group in ('motion', 'local'):
                    self.put_json(folder + '/' + filename.replace('.npz', '.json'), entry)
                entries.append(entry)
                self.records[i][group] = np.concatenate([x, np.ones_like(x)], axis=1) if group == 'local' else x
                if group != 'rgb':
                    self.records[i][group + '_names'] = ([f'local/{n}' for n in names]
                                                        + [f'local/{n}_support_flag' for n in names]) if group == 'local' else names
            fm = {'schema_version': schemas[group], 'profile': profile, 'videos': entries}
            if group == 'rgb':
                fm.update(summary={'run_status': 'completed_selected_inventory'},
                          features_by_name={e['name']: e['npz_path'] for e in entries})
            else:
                fm.update(status='complete', profile_signature=profile_hash)
                fm['source_manifest_sha256' if group == 'local' else 'dataset_manifest_sha256'] = old_sha
            if group == 'local':
                fm['source_projection_sha256'] = p._digest_value([e['source'] for e in entries])
                self.put_json(folder + '/profile.json', {k: fm[k] for k in
                                                       ['schema_version', 'profile', 'profile_signature', 'source_projection_sha256']})
            self.put_json(folder + '/manifest.json', fm)
        sources = {p.LEGACY_SOURCE_PREFIX + name: sha for name, sha in self.snapshots.items()}
        signature = self.manifest_sha + json.dumps(sources, sort_keys=True)
        for fold in range(5):
            for recipe in p.RECIPES:
                row = p._expected_partitions(self.manifest, fold, recipe)
                arrays = {}
                for kind, indices in [('inner', row['train']), ('outer', row['validation'])]:
                    for i in indices:
                        arrays[f'{kind}_frame_{i}'] = np.full(4, .4, np.float32)
                        arrays[f'{kind}_video_{i}'] = np.asarray(.5)
                        if 'boundary' in recipe:
                            arrays[f'{kind}_boundary_{i}'] = np.full((2, 4), .3, np.float32)
                base_path = self.fold_path(fold, recipe)
                self.files[base_path + '.npz'] = npz_bytes(arrays)
                row.update(signature=signature, sources=sources, seconds=1.,
                           npz_sha256=p._sha(self.files[base_path + '.npz']))
                self.put_json(base_path + '.json', row)

    @staticmethod
    def fold_path(fold=0, recipe='motion_rf'):
        return p.NEW + f'/grouped_training/folds/{fold}_{recipe}'

    def put_json(self, path, value):
        self.files[path] = json_bytes(value)


class TrainingProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = MemoryInventory()
        with patch.object(p, '_read_bytes', side_effect=lambda path: cls.fixture.files[Path(path).relative_to(cls.fixture.root).as_posix()]), patch.object(p, 'EXPECTED_ROWS', cls.fixture.n):
            inputs = p._collect_inputs(cls.fixture.root)[2]
            cls.fixture.inputs_sha = p._digest_value(dict(inputs))

    def setUp(self):
        self.files = dict(self.fixture.files)
        self.records = deepcopy(self.fixture.records)
        overrides = dict(EXPECTED_ROWS=self.fixture.n, LEGACY_MANIFEST_SHA256=self.fixture.manifest_sha,
                         LEGACY_SNAPSHOT_SHA256=self.fixture.snapshots, LEGACY_AUDITED_INPUTS_SHA256=self.fixture.inputs_sha)
        self.patchers = [patch.multiple(p, **overrides), patch.object(p, '_read_bytes', side_effect=self.read),
                         patch.object(p, '_cache_files', side_effect=self.cache_files)]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.evidence = p.audit_legacy_inputs(self.fixture.root)

    def read(self, path):
        key = Path(path).relative_to(self.fixture.root).as_posix()
        if key not in self.files:
            raise p.ProvenanceError('missing fixture artifact: ' + key)
        return self.files[key]

    def cache_files(self, root):
        prefix = p.NEW + '/grouped_training/folds/'
        return {key[len(prefix):] for key in self.files if key.startswith(prefix)}

    def metadata(self, fold=0, recipe='motion_rf'):
        return json.loads(self.files[self.fixture.fold_path(fold, recipe) + '.json'])

    def put_metadata(self, row, fold=0, recipe='motion_rf'):
        self.files[self.fixture.fold_path(fold, recipe) + '.json'] = json_bytes(row)

    def verify(self, fold=0, recipe='motion_rf', records=None):
        return p.verify_fold_cache(self.evidence, fold, recipe, records)

    def replace_probability_arrays(self, mutate, fold=0, recipe='motion_rf'):
        path = self.fixture.fold_path(fold, recipe)
        arrays = p._archive(self.files[path + '.npz'], 'fixture')
        mutate(arrays)
        data = npz_bytes(arrays)
        self.files[path + '.npz'] = data
        row = self.metadata(fold, recipe)
        row['npz_sha256'] = p._sha(data)
        self.put_metadata(row, fold, recipe)

    def test_positive_legacy_records_and_all_65(self):
        checked = self.verify(records=self.records)
        self.assertEqual(checked.mode, p.LEGACY_MODE)
        self.assertFalse(next(iter(checked.arrays.values())).flags.writeable)
        results = p.verify_all_cached_folds(self.evidence)
        self.assertEqual(len(results), 65)
        self.assertEqual(len({(r['fold'], r['recipe']) for r in results}), 65)
        report = p.audit_report(self.evidence, results)
        self.assertFalse(report['historical_pixel_to_probability_binding_proven'])
        self.assertEqual(report['attestation_scope'], 'current-files-only')
        self.assertEqual(report['features']['rgb']['original_declared_feature_hashes_verified'], 0)
        self.assertFalse(report['original_metadata_modified'])
        self.assertFalse(self.fixture.root.exists())

    def test_signature_tampering_refused(self):
        row = self.metadata()
        row['signature'] = 'STALE_SIGNATURE'
        self.put_metadata(row)
        with self.assertRaisesRegex(p.ProvenanceError, 'signature mismatch'):
            self.verify()

    def test_sources_and_coherent_signature_tampering_refused(self):
        for change in ['hash', 'prefix', 'missing', 'extra']:
            with self.subTest(change=change):
                row = json.loads(self.fixture.files[self.fixture.fold_path() + '.json'])
                key = next(iter(row['sources']))
                if change == 'hash':
                    row['sources'][key] = '0' * 64
                elif change == 'prefix':
                    row['sources']['/forged/code/' + key.rsplit('/', 1)[1]] = row['sources'].pop(key)
                elif change == 'missing':
                    row['sources'].pop(key)
                else:
                    row['sources']['/forged/extra.py'] = '0' * 64
                row['signature'] = self.fixture.manifest_sha + json.dumps(row['sources'], sort_keys=True)
                self.put_metadata(row)
                with self.assertRaisesRegex(p.ProvenanceError, 'sources mismatch'):
                    self.verify()

    def test_fixed_snapshot_tampering_refused(self):
        name = next(iter(self.fixture.snapshots))
        self.files[p.NEW + '/grouped_training/sources/' + name] = b'changed snapshot'
        with self.assertRaisesRegex(p.ProvenanceError, 'source changed'):
            self.verify()
        with self.assertRaisesRegex(p.ProvenanceError, 'snapshot mismatch'):
            p.audit_legacy_inputs(self.fixture.root)

    def test_fixed_snapshot_not_live_code_controls_legacy(self):
        self.files['code/optimized_locator.py'] = b'new inference-only implementation'
        self.verify()
        report = p.audit_report(self.evidence, p.verify_all_cached_folds(self.evidence))
        self.assertEqual(report['live_training_sources_differing_from_snapshot'], ['optimized_locator.py'])
        self.assertNotEqual(str(self.evidence.root), p.LEGACY_SOURCE_PREFIX)

    def test_each_pixel_feature_hash_tamper_refused(self):
        for group, folder in self.fixture.feature_folders.items():
            with self.subTest(group=group):
                self.files = dict(self.fixture.files)
                key = folder + '/v000.npz'
                values = p._archive(self.files[key], 'fixture')
                values['features' if group == 'rgb' else 'signals'][0, 0] += .1
                self.files[key] = npz_bytes(values)
                with self.assertRaisesRegex(p.ProvenanceError, 'bound input changed'):
                    self.verify()
                error = 'audit.*binding mismatch' if group == 'rgb' else 'declared feature hash mismatch'
                with self.assertRaisesRegex(p.ProvenanceError, error):
                    p.audit_legacy_inputs(self.fixture.root)

    def test_original_hash_missing_is_not_silently_backfilled(self):
        key = self.fixture.feature_folders['motion'] + '/manifest.json'
        manifest = json.loads(self.files[key])
        del manifest['videos'][0]['feature_sha256']
        self.files[key] = json_bytes(manifest)
        with self.assertRaisesRegex(p.ProvenanceError, 'missing original feature hash'):
            p.audit_legacy_inputs(self.fixture.root)

    def test_coherent_feature_manifest_sidecar_rehash_still_refused(self):
        folder = self.fixture.feature_folders['motion']
        key = folder + '/v000.npz'
        values = p._archive(self.files[key], 'fixture')
        values['signals'][0, 0] += .1
        self.files[key] = npz_bytes(values)
        manifest = json.loads(self.files[folder + '/manifest.json'])
        manifest['videos'][0]['feature_sha256'] = p._sha(self.files[key])
        self.files[folder + '/manifest.json'] = json_bytes(manifest)
        self.files[folder + '/v000.json'] = json_bytes(manifest['videos'][0])
        with self.assertRaisesRegex(p.ProvenanceError, 'audit.*binding mismatch'):
            p.audit_legacy_inputs(self.fixture.root)

    def test_base_dataset_hash_tamper_refused(self):
        key = p.OLD + '/dataset.npz'
        values = p._archive(self.files[key], 'fixture')
        values['jepa_0'][0, 0] += .1
        self.files[key] = npz_bytes(values)
        with self.assertRaisesRegex(p.ProvenanceError, 'bound input changed'):
            self.verify()
        with self.assertRaisesRegex(p.ProvenanceError, 'audit.*binding mismatch'):
            p.audit_legacy_inputs(self.fixture.root)

    def test_feature_source_provenance_tamper_refused(self):
        key = self.fixture.feature_folders['corrected'] + '/manifest.json'
        manifest = json.loads(self.files[key])
        manifest['videos'][0]['source_sha256'] = '0' * 64
        self.files[key] = json_bytes(manifest)
        with self.assertRaisesRegex(p.ProvenanceError, 'feature source SHA'):
            p.audit_legacy_inputs(self.fixture.root)

    def test_wrong_fold_recipe_and_boolean_fold_refused(self):
        for field, value in [('fold', 1), ('fold', False), ('recipe', 'motion_et'), ('content_holdout', 1)]:
            with self.subTest(field=field, value=value):
                row = json.loads(self.fixture.files[self.fixture.fold_path() + '.json'])
                row[field] = value
                self.put_metadata(row)
                with self.assertRaisesRegex(p.ProvenanceError, 'identity'):
                    self.verify()
        with self.assertRaisesRegex(p.ProvenanceError, 'requested fold'):
            self.verify(fold=False)
        with self.assertRaisesRegex(p.ProvenanceError, 'requested recipe'):
            self.verify(recipe='../motion_rf')

    def test_train_validation_partition_tamper_refused(self):
        row = self.metadata()
        row['train'][0], row['validation'][0] = row['validation'][0], row['train'][0]
        row['train'].sort()
        row['validation'].sort()
        self.put_metadata(row)
        with self.assertRaisesRegex(p.ProvenanceError, 'exact outer'):
            self.verify()

    def test_inner_seed_coverage_and_valid_repartition_refused(self):
        for change in ['seed', 'duplicate', 'repartition']:
            with self.subTest(change=change):
                row = json.loads(self.fixture.files[self.fixture.fold_path() + '.json'])
                if change == 'seed':
                    row['inner_partitions'][0]['seed'] += 1
                elif change == 'duplicate':
                    row['inner_partitions'][0]['validation'].append(row['inner_partitions'][0]['validation'][0])
                else:
                    a, b = row['inner_partitions'][:2]
                    a['validation'], b['validation'] = b['validation'], a['validation']
                    for part in [a, b]:
                        part['fit'] = [i for i in row['train'] if i not in part['validation']]
                self.put_metadata(row)
                with self.assertRaises(p.ProvenanceError):
                    self.verify()

    def test_inner_sha_leakage_even_with_complete_index_coverage_refused(self):
        row = self.metadata(fold=1)
        parts = row['inner_partitions']
        first = next(k for k, part in enumerate(parts) if 0 in part['validation'])
        second = (first + 1) % 3
        parts[first]['validation'].remove(1)
        parts[second]['validation'] = sorted(parts[second]['validation'] + [1])
        for part in parts:
            part['fit'] = [i for i in row['train'] if i not in part['validation']]
        self.put_metadata(row, fold=1)
        with self.assertRaisesRegex(p.ProvenanceError, 'content SHA leakage'):
            self.verify(fold=1)

    def test_manifest_group_leakage_refused(self):
        key = p.NEW + '/dataset_manifest.json'
        manifest = json.loads(self.files[key])
        manifest['outer_folds'][0].remove(1)
        manifest['outer_folds'][1] = sorted(manifest['outer_folds'][1] + [1])
        self.files[key] = json_bytes(manifest)
        with self.assertRaisesRegex(p.ProvenanceError, 'content SHA leakage'):
            p.audit_legacy_inputs(self.fixture.root)

    def test_probability_npz_hash_tamper_refused(self):
        key = self.fixture.fold_path() + '.npz'
        self.files[key] += b'altered bytes'
        with self.assertRaisesRegex(p.ProvenanceError, 'NPZ hash mismatch'):
            self.verify()

    def test_missing_extra_wrong_split_probability_coverage_refused(self):
        for change in ['missing', 'extra', 'wrong_split']:
            with self.subTest(change=change):
                self.files = dict(self.fixture.files)
                def mutate(arrays):
                    key = next(k for k in arrays if k.startswith('inner_frame_'))
                    if change == 'missing':
                        arrays.pop(key)
                    elif change == 'extra':
                        arrays['unexpected'] = np.asarray(.5)
                    else:
                        arrays[key.replace('inner', 'outer')] = arrays.pop(key)
                self.replace_probability_arrays(mutate)
                with self.assertRaisesRegex(p.ProvenanceError, 'coverage'):
                    self.verify()

    def test_video_probabilities_are_scalar_finite_and_bounded(self):
        for value in [np.asarray(np.nan), np.asarray(1.01), np.asarray([.5]), np.asarray(True)]:
            with self.subTest(value=str(value)):
                self.files = dict(self.fixture.files)
                def mutate(arrays):
                    arrays[next(k for k in arrays if k.startswith('inner_video_'))] = value
                self.replace_probability_arrays(mutate)
                with self.assertRaisesRegex(p.ProvenanceError, 'probability shape/range'):
                    self.verify()

    def test_boundary_shapes_ranges_and_coverage(self):
        recipe = 'corrected_motion_boundary_rf'
        self.verify(recipe=recipe)
        for value in [None, np.full((3, 4), .5), np.full((2, 4), np.inf), np.full((2, 4), -1.)]:
            with self.subTest(value=str(value)):
                self.files = dict(self.fixture.files)
                def mutate(arrays):
                    key = next(k for k in arrays if '_boundary_' in k)
                    if value is None:
                        arrays.pop(key)
                    else:
                        arrays[key] = value
                self.replace_probability_arrays(mutate, recipe=recipe)
                with self.assertRaises(p.ProvenanceError):
                    self.verify(recipe=recipe)

    def test_loader_in_memory_feature_and_row_identity_tampering_refused(self):
        self.records[0]['local'][0, 0] += .1
        with self.assertRaisesRegex(p.ProvenanceError, 'loader input differs'):
            self.verify(records=self.records)
        self.records = deepcopy(self.fixture.records)
        self.records[0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(p.ProvenanceError, 'loader row identity'):
            self.verify(records=self.records)

    def test_fabricated_extended_fields_refused_in_legacy_mode(self):
        row = self.metadata()
        row.update(signature_version=p.EXTENDED_VERSION, provenance={'fabricated': True})
        self.put_metadata(row)
        with self.assertRaisesRegex(p.ProvenanceError, 'fabricated extended provenance'):
            self.verify()
        with self.assertRaisesRegex(p.ProvenanceError, 'cannot upgrade legacy'):
            p.extended_signature_fields(self.evidence, 0, 'motion_rf', npz_sha256=row['npz_sha256'])

    def new_fit_row(self):
        evidence = p.collect_new_fit_inputs(self.fixture.root)
        old = self.metadata()
        fields = p.extended_signature_fields(evidence, 0, 'motion_rf', npz_sha256=old['npz_sha256'])
        row = p._expected_partitions(evidence.manifest, 0, 'motion_rf')
        row.update(fields, npz_sha256=old['npz_sha256'], seconds=1.)
        self.put_metadata(row)
        return evidence, fields['signature']

    def test_new_extended_signature_positive_and_complete_inputs(self):
        evidence, receipt = self.new_fit_row()
        checked = p.verify_fold_cache(evidence, 0, 'motion_rf', self.records, expected_signature=receipt)
        inputs = checked.metadata['provenance']['input_hashes']
        self.assertIn(p.OLD + '/dataset.npz', inputs)
        self.assertIn(self.fixture.feature_folders['rgb'] + '/v000.npz', inputs)
        self.assertIn('code/build_optimization_dataset.py', checked.metadata['sources'])
        self.assertEqual(checked.mode, p.NEW_FIT_MODE)

    def test_extended_requires_external_receipt_and_no_legacy_downgrade(self):
        evidence, receipt = self.new_fit_row()
        with self.assertRaisesRegex(p.ProvenanceError, 'trusted expected_signature'):
            p.verify_fold_cache(evidence, 0, 'motion_rf')
        with self.assertRaisesRegex(p.ProvenanceError, 'signature mismatch'):
            p.verify_fold_cache(evidence, 0, 'motion_rf', expected_signature='0' * 64)
        self.put_metadata(json.loads(self.fixture.files[self.fixture.fold_path() + '.json']))
        with self.assertRaisesRegex(p.ProvenanceError, 'signature version required'):
            p.verify_fold_cache(evidence, 0, 'motion_rf', expected_signature=receipt)

    def test_extended_signature_input_and_sources_tampering_refused(self):
        evidence, receipt = self.new_fit_row()
        original = self.metadata()
        for change in ['input', 'source', 'signature']:
            with self.subTest(change=change):
                row = deepcopy(original)
                if change == 'input':
                    row['provenance']['input_hashes'][p.OLD + '/dataset.npz'] = '0' * 64
                    row['signature'] = p._digest_value(row['provenance'])
                elif change == 'source':
                    row['sources']['code/build_optimization_dataset.py'] = '0' * 64
                else:
                    row['signature'] = '0' * 64
                self.put_metadata(row)
                with self.assertRaises(p.ProvenanceError):
                    p.verify_fold_cache(evidence, 0, 'motion_rf', expected_signature=receipt)

    def test_extended_inputs_cannot_change_between_capture_and_fit_finish(self):
        evidence = p.collect_new_fit_inputs(self.fixture.root)
        self.files['code/build_optimization_dataset.py'] += b'changed'
        with self.assertRaisesRegex(p.ProvenanceError, 'training source changed'):
            p.extended_signature_fields(evidence, 0, 'motion_rf', npz_sha256=self.metadata()['npz_sha256'])

    def test_batch_inventory_must_be_exactly_65_pairs(self):
        del self.files[self.fixture.fold_path() + '.json']
        with self.assertRaisesRegex(p.ProvenanceError, '65-cache inventory'):
            p.verify_all_cached_folds(self.evidence)

    def test_path_traversal_and_cross_platform_absolute_paths_refused(self):
        for relative in ['../outside.npz', '/tmp/outside.npz', 'C:/outside.npz',
                         '..' + chr(92) + 'outside.npz', 'dir' + chr(92) + 'file.npz']:
            with self.subTest(relative=relative), self.assertRaisesRegex(p.ProvenanceError, 'unsafe feature path'):
                p._safe_file(self.fixture.root, relative)

    def test_duplicate_json_keys_nonfinite_json_and_duplicate_npz_rejected(self):
        with self.assertRaises(p.ProvenanceError):
            p._json(b'{"signature":"a","signature":"b"}', 'fixture')
        with self.assertRaises(p.ProvenanceError):
            p._json(b'{"value": NaN}', 'fixture')
        stream = BytesIO()
        array_bytes = BytesIO()
        np.save(array_bytes, np.asarray(.5), allow_pickle=False)
        import warnings
        import zipfile
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr('value.npy', array_bytes.getvalue())
                archive.writestr('value.npy', array_bytes.getvalue())
        with self.assertRaisesRegex(p.ProvenanceError, 'invalid NPZ'):
            p._archive(stream.getvalue(), 'duplicate fixture')

    def test_verifiers_never_write_original_artifacts(self):
        before = dict(self.files)
        with patch.object(Path, 'write_text', side_effect=AssertionError('writes forbidden')):
            self.verify(records=self.records)
            p.verify_all_cached_folds(self.evidence)
            p.audit_legacy_inputs(self.fixture.root)
        self.assertEqual(self.files, before)


if __name__ == '__main__':
    unittest.main()
