"""Read-only provenance guards for the 65 grouped CPU probability caches.

Legacy use (explicitly audited, NOT a training-time full-input attestation)::

    evidence = audit_legacy_inputs()
    checked = verify_fold_cache(evidence, fold, recipe, records)
    # Build Probabilities from checked.arrays / checked.metadata; do not reopen NPZ.

New fits: collect_new_fit_inputs() BEFORE fitting, then extended_signature_fields()
AFTER fitting. Keep its signature in a trusted training receipt. Never add these
fields to an old cache: that cannot recover omitted historical input provenance.

Only stdlib and the existing NumPy dependency are imported. Source snapshots are
read and hashed, never imported/executed. No training, extraction or service calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = 'output/algorithm-opt-2026-10-02'
NEW = 'output/algorithm-opt-2026-10-02-v2'
SEED = 20261002
EXPECTED_ROWS = 77
RECIPES = (
    'motion_rf', 'motion_et', 'rgb_motion_tcn', 'corrected_motion_rf',
    'corrected_motion_et', 'corrected_motion_tcn', 'rgb_corrected_motion_rf',
    'corrected_motion_boundary_rf', 'corrected_motion_boundary_et',
    'noglobal_corrected_motion_et', 'noglobal_corrected_motion_rf',
    'local_corrected_motion_et', 'local_motion_et',
)
LEGACY_MODE = 'legacy-audited-verification-v1'
NEW_FIT_MODE = 'new-fit-inputs-v2'
EXTENDED_VERSION = 'grouped-training-provenance-v2'
LEGACY_SOURCE_PREFIX = '/mnt/e/jepa-system/code/'
# Observed ORIGINAL fields, independently checked against the fixed snapshots.
LEGACY_MANIFEST_SHA256 = '6c4761d94e6db5166d3a9ac291557076bbb196f0571c429597e8fa9dd0c2520e'
LEGACY_SNAPSHOT_SHA256 = {
    'optimized_grouped_training.py': '4ab8b589758f32469ffd08932f48ec7af37056d9016004862faa115b89da3ce4',
    'run_recall_training.py': '5ae7b2788bc8352923dc8a98bb55abfb628ca7c799173f86e3bf52b21c4a97e7',
    'run_algorithm_optimization.py': '2f01a39c5c127bf353b7b7485f7c76a6684aecee6eacc6af86345c83e4058f1d',
    'optimized_feature_view.py': '99a0887841a240a559bcc1bda386b95f3f5ed5a2a2b3f5d0aa6a33ec1e090f54',
    'optimized_feature_view_v2.py': 'bb8133c817fd958443ef94dad1f75fcd4fc66ae6611ca3d1b1a3c1374be78aa3',
    'optimized_locator.py': 'e1eab5162f43ed4b856dba0f1e73bc531076e331f8d66fba66272d7eebedb943',
    'optimized_temporal_head.py': 'd5f5a96e7f59955d4cf790788f1b28cceef093d01a65834b8297adeb8546ee6e',
    'optimized_boundary_head.py': 'f49de1549eb6faddb4cc56073ce156315fe96884f00aa2d7d09c1dabd0318f8a',
}
# AUDIT-TIME pin only. It is not an original training signature or feature origin.
# Covers manifests, dataset.npz, every consumed feature, names/profile/sidecars,
# and feature extractor provenance. See provenance_audit.json for its limitations.
LEGACY_AUDITED_INPUTS_SHA256 = '013b8637f6d073261d438956def803e10beb8fb9d704657ff1f543ed334fd0ca'
NEW_TRAINING_SOURCES = tuple('code/' + n for n in LEGACY_SNAPSHOT_SHA256) + (
    'code/build_optimization_dataset.py', 'code/run_optimization_selection.py',
    'code/optimized_training_provenance.py',
)
LEGACY_LIMITATIONS = (
    'Original training signature binds the grouped manifest and only eight source files; '
    'it omits dataset.npz and all pixel feature manifests/file hashes.',
    'dataset.npz and RGB per-file hashes below are audit-time observations, not '
    'recovered training-time attestations. RGB also lacks per-video source SHA256.',
    'The original source snapshot set omits build_optimization_dataset.py and '
    'run_optimization_selection.py; grouped partitions are independently replayed.',
    'Source videos, model checkpoints and raw JEPA intermediates are not reopened; '
    'this is cache provenance/compatibility verification, not proof of model fitting.',
    'Hashes are integrity anchors, not authenticated cryptographic signatures. '
    'A trusted audit pin/receipt is required; metadata alone is not a trust root.',
)


class ProvenanceError(ValueError):
    """Fail-closed cache refusal; checks remain active under python -O."""


def _require(condition, message):
    if not condition:
        raise ProvenanceError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _digest_value(value):
    return _sha(_canonical(value))


def _hash_string(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _read_bytes(path):
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ProvenanceError('missing/unreadable artifact: ' + str(path)) from exc


def _json(data, label):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key: ' + label)
            result[key] = value
        return result

    def invalid(value):
        raise ProvenanceError('nonfinite JSON number: ' + label)

    try:
        return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ProvenanceError('invalid JSON: ' + label) from exc


def _safe_file(folder, relative):
    _require(isinstance(relative, str) and relative, 'missing relative feature file')
    windows, posix = PureWindowsPath(relative), PurePosixPath(relative)
    _require(not windows.drive and not windows.is_absolute() and not posix.is_absolute()
             and chr(92) not in relative and ':' not in relative
             and '..' not in posix.parts, 'unsafe feature path')
    path = (folder / relative).resolve()
    _require(path.is_relative_to(folder.resolve()), 'feature path escapes cache folder')
    return path


def _array_digest(value):
    array = np.asarray(value)
    return _sha(_canonical({'shape': array.shape, 'dtype': array.dtype.str})
                + array.tobytes(order='C'))


def _archive(data, label):
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            _require(len(archive.files) == len(set(archive.files)),
                     'duplicate NPZ members: ' + label)
            return {k: archive[k] for k in archive.files}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ProvenanceError('invalid NPZ: ' + label) from exc


def _indices(value, n, label):
    _require(isinstance(value, list) and value and all(type(i) is int for i in value),
             'invalid indices: ' + label)
    _require(value == sorted(set(value)) and 0 <= value[0] and value[-1] < n,
             'duplicate/unsorted/out-of-range indices: ' + label)
    return value


def _content_groups(rows, indices):
    groups = {}
    for i in indices:
        groups.setdefault(rows[i]['sha256'], []).append(i)
    return list(groups.values())


def _grouped_folds(rows, indices, n_folds, seed):
    """Replay the frozen make_folds + grouped_folds without importing trainers."""
    groups = _content_groups(rows, indices)
    strata = defaultdict(list)
    for j, group in enumerate(groups):
        row = rows[group[0]]
        event = 'normal' if row['event_count'] == 0 else 'multi' if row['event_count'] > 1 else 'single'
        strata[(row['generator'], event)].append(j)
    rng = np.random.default_rng(seed)
    folds, offset = [[] for _ in range(n_folds)], 0
    for _, members in sorted(strata.items()):
        for j, member in enumerate(rng.permutation(members)):
            folds[(offset + j) % n_folds].extend(groups[int(member)])
        offset = (offset + len(members)) % n_folds
    return [sorted(f) for f in folds]


def _no_overlap(rows, a, b, label):
    _require(not set(a) & set(b), 'index leakage: ' + label)
    _require(not {rows[i]['sha256'] for i in a} & {rows[i]['sha256'] for i in b},
             'content SHA leakage: ' + label)


def _validate_manifest(manifest):
    _require(isinstance(manifest, dict)
             and manifest.get('schema_version') == 'algorithm-opt-grouped-dataset-v2',
             'unexpected grouped dataset schema')
    protocol = manifest.get('protocol', {})
    _require(all(type(protocol.get(k)) is int and protocol[k] == v
                 for k, v in [('outer_folds', 5), ('inner_folds', 3), ('seed', SEED)]),
             'unexpected outer5/inner3/seed protocol')
    rows = manifest.get('rows')
    _require(isinstance(rows, list) and len(rows) == EXPECTED_ROWS, 'dataset row coverage')
    for row in rows:
        _require(isinstance(row, dict) and _hash_string(row.get('sha256'))
                 and isinstance(row.get('name'), str) and isinstance(row.get('generator'), str)
                 and type(row.get('frames')) is int and row['frames'] > 0
                 and type(row.get('event_count')) is int and row['event_count'] >= 0
                 and type(row.get('fps')) in (int, float)
                 and np.isfinite(row['fps']) and row['fps'] > 0, 'invalid dataset row')
    _require(len({r['name'] for r in rows}) == len(rows), 'duplicate dataset names')
    prompts = [r['prompt_id'] for r in rows if r.get('prompt_id', 'unknown') != 'unknown']
    _require(len(set(prompts)) == len(prompts), 'duplicate known prompt groups')
    folds = manifest.get('outer_folds')
    _require(isinstance(folds, list) and len(folds) == 5, 'outer fold count')
    for fold, val in enumerate(folds):
        _indices(val, len(rows), 'outer ' + str(fold))
        train = [i for i in range(len(rows)) if i not in set(val)]
        _no_overlap(rows, train, val, 'outer ' + str(fold))
    _require(sorted(i for f in folds for i in f) == list(range(len(rows))), 'outer fold coverage')
    _require(manifest.get('content_groups') == _content_groups(rows, range(len(rows))),
             'manifest content groups')
    return rows


def _expected_partitions(manifest, fold, recipe):
    _require(type(fold) is int and 0 <= fold < 5, 'invalid requested fold')
    _require(recipe in RECIPES, 'invalid requested recipe')
    rows = manifest['rows']
    val = manifest['outer_folds'][fold]
    train = [i for i in range(len(rows)) if i not in set(val)]
    inner = []
    for k, validation in enumerate(_grouped_folds(rows, train, 3, SEED + fold * 31)):
        _indices(validation, len(rows), 'expected inner validation')
        fit = [i for i in train if i not in set(validation)]
        _no_overlap(rows, fit, validation, 'expected inner ' + str(k))
        inner.append({'fit': fit, 'validation': validation, 'seed': SEED + fold * 53 + k})
    return {'fold': fold, 'recipe': recipe, 'train': train, 'validation': val,
            'inner_partitions': inner, 'content_holdout': True}


def _validate_partitions(metadata, manifest, fold, recipe):
    expected = _expected_partitions(manifest, fold, recipe)
    _require(type(metadata.get('fold')) is int and metadata['fold'] == fold
             and metadata.get('recipe') == recipe and metadata.get('content_holdout') is True,
             'fold/recipe/content_holdout identity')
    n, rows = len(manifest['rows']), manifest['rows']
    for field in ['train', 'validation']:
        _indices(metadata.get(field), n, field)
        _require(metadata[field] == expected[field], 'exact outer ' + field + ' mismatch')
    inner = metadata.get('inner_partitions')
    _require(isinstance(inner, list) and len(inner) == 3, 'inner partition count')
    covered = []
    for k, part in enumerate(inner):
        _require(isinstance(part, dict) and set(part) == {'fit', 'validation', 'seed'},
                 'inner partition schema')
        fit = _indices(part['fit'], n, 'inner fit')
        val = _indices(part['validation'], n, 'inner validation')
        _require(type(part['seed']) is int and part['seed'] == expected['inner_partitions'][k]['seed'],
                 'inner seed mismatch')
        _no_overlap(rows, fit, val, 'inner ' + str(k))
        _no_overlap(rows, fit + val, metadata['validation'], 'inner vs outer')
        _require(sorted(fit + val) == metadata['train'], 'inner complement/coverage mismatch')
        _require(part == expected['inner_partitions'][k], 'fixed grouped inner partition mismatch')
        covered.extend(val)
    _require(sorted(covered) == metadata['train'], 'inner OOF coverage')
    return expected


@dataclass(frozen=True)
class TrainingEvidence:
    root: Path
    mode: str
    manifest_bytes: bytes
    sources: tuple[tuple[str, str], ...]
    input_hashes: tuple[tuple[str, str], ...]
    record_hashes: tuple[tuple[tuple[str, str], ...], ...]
    feature_summary_bytes: bytes
    inputs_sha256: str

    @property
    def manifest(self):
        return _json(self.manifest_bytes, 'bound manifest')

    @property
    def feature_summary(self):
        return _json(self.feature_summary_bytes, 'feature summary')


@dataclass(frozen=True)
class VerifiedFold:
    metadata: dict
    arrays: dict[str, np.ndarray]
    mode: str
    npz_sha256: str
    inputs_sha256: str


def _collect_inputs(root):
    """Verify existing declarations; measure omitted hashes separately, in memory."""
    root = Path(root).resolve()
    hashes = {}

    def read(relative):
        path = root / relative
        data = _read_bytes(path)
        hashes[path.relative_to(root).as_posix()] = _sha(data)
        return data

    def read_json(relative):
        return _json(read(relative), str(relative))

    manifest_bytes = read(NEW + '/dataset_manifest.json')
    manifest = _json(manifest_bytes, 'grouped manifest')
    rows = _validate_manifest(manifest)
    old = read_json(OLD + '/dataset_manifest.json')
    old_sha = hashes[OLD + '/dataset_manifest.json']
    _require(manifest.get('v1_manifest_sha256') == old_sha and old.get('rows') == rows,
             'base/grouped manifest linkage')
    base = _archive(read(OLD + '/dataset.npz'), 'base dataset.npz')
    _require(set(base) == {f'{k}_{i}' for i in range(len(rows)) for k in ['jepa', 'rank_motion', 'labels']},
             'base NPZ row/key coverage')
    record_hashes = [dict() for _ in rows]
    for i, row in enumerate(rows):
        y = base[f'labels_{i}']
        _require(y.shape == (row['frames'],) and y.dtype.kind in 'iu'
                 and np.isin(y, [0, 1]).all(), 'base labels schema')
        events = int(np.count_nonzero((y != 0) & np.r_[True, y[:-1] == 0]))
        _require(events == row['event_count'], 'base labels/event count mismatch')
        record_hashes[i]['labels'] = _array_digest(y)
        for key in ['jepa', 'rank_motion']:
            x = base[f'{key}_{i}']
            names = old['feature_names'][key]
            _require(x.dtype == np.float32 and x.shape == (row['frames'], len(names))
                     and np.isfinite(x).all(), 'base feature schema: ' + key)
            record_hashes[i][key] = _array_digest(x)
        record_hashes[i]['jepa_names'] = _digest_value(old['feature_names']['jepa'])
    del base
    summary = {}
    folders = [('motion', OLD + '/motion_features'), ('rgb', OLD + '/rgb_features'),
               ('corrected', OLD + '/corrected_jepa_features_v2'), ('local', NEW + '/local_motion')]
    schemas = {'motion': 'absolute-motion-feature-cache-v1', 'rgb': 'optimized-frozen-r3d18-v1',
               'corrected': 'corrected-true-jepa-feature-cache-v1', 'local': 'local-motion-cache-v1'}
    for group, folder in folders:
        fm = read_json(folder + '/manifest.json')
        _require(fm.get('schema_version') == schemas[group], 'feature manifest schema: ' + group)
        profile = fm.get('profile')
        _require(isinstance(profile, dict) and profile, 'feature profile: ' + group)
        if group == 'rgb':
            _require(fm.get('summary', {}).get('run_status') == 'completed_selected_inventory',
                     'RGB inventory incomplete')
            _require(read_json(folder + '/feature_profile.json') == profile, 'RGB profile artifact')
            projected = {k: v for k, v in profile.items() if k != 'profile_id'}
            profile_hash = _sha(json.dumps(projected, sort_keys=True, allow_nan=False).encode())
            _require(profile.get('profile_id') == profile_hash, 'RGB profile signature')
            names = None
        else:
            _require(fm.get('status') == 'complete', 'incomplete feature cache: ' + group)
            binding = 'source_manifest_sha256' if group == 'local' else 'dataset_manifest_sha256'
            _require(fm.get(binding) == old_sha, 'feature dataset binding: ' + group)
            names = read_json(folder + '/feature_names.json')
            _require(isinstance(names, list) and names and all(isinstance(x, str) for x in names)
                     and profile.get('feature_names') == names, 'feature name order: ' + group)
            projected = {k: v for k, v in profile.items() if k != 'feature_names'} if group == 'corrected' else profile
            profile_hash = (_digest_value(projected) if group == 'local' else
                            _sha(json.dumps(projected, sort_keys=True, allow_nan=False).encode()))
            _require(fm.get('profile_signature') == profile_hash, 'feature profile signature: ' + group)
        code_bindings = {
            'motion': [('code/optimized_motion_features.py', 'code_sha256')],
            'corrected': [(folder + '/extractor_source_v2.py', 'extractor_sha256'),
                          (folder + '/adapter_source_v2.py', 'adapter_sha256')],
            'local': [('code/optimized_local_motion.py', 'extractor_sha256'),
                      ('code/build_optimization_local_motion.py', 'builder_sha256')],
            'rgb': [],
        }
        for relative, field in code_bindings[group]:
            _require(_hash_string(profile.get(field)) and _sha(read(relative)) == profile[field],
                     'feature extractor snapshot/hash: ' + group + '/' + field)
        if group == 'corrected':
            _require(profile.get('i_mask_reuse_policy') == 'explicit_batch_seed_reset_and_restore',
                     'corrected I-mask provenance')
        entries = fm.get('videos')
        _require(isinstance(entries, list), 'feature entry list: ' + group)
        if group == 'local':
            _require(len(entries) == len(rows) and all(type(e.get('row_id')) is int
                     and e['row_id'] == i for i, e in enumerate(entries)), 'local row coverage/order')
            projected_sources = [{'input_path': r['input_path'], 'fps': r['fps'],
                                  'frame_count': r['frames'], 'sha256': r['sha256']} for r in rows]
            _require(fm.get('source_projection_sha256') == _digest_value(projected_sources),
                     'local source projection signature')
            contract = read_json(folder + '/profile.json')
            _require(all(contract.get(k) == fm.get(k) for k in contract), 'local profile contract')
            by_name = {row['name']: entry for row, entry in zip(rows, entries)}
        else:
            by_name = {}
            for entry in entries:
                name = entry.get('name')
                _require(isinstance(name, str) and name not in by_name, 'duplicate feature entries: ' + group)
                by_name[name] = entry
            expected_names = {r['name'] for r in rows}
            if group != 'rgb':
                _require(set(by_name) == expected_names, 'feature row coverage: ' + group)
            else:
                excluded = {e['video_name'] for e in old.get('excluded', [])}
                _require(set(by_name) == expected_names | excluded, 'RGB inventory name coverage')
                _require(all(by_name[n].get('status') == 'missing' and not by_name[n].get('npz_path')
                             for n in excluded), 'unexpected RGB excluded feature')
        attested = 0
        used_paths = set()
        for i, row in enumerate(rows):
            entry = by_name[row['name']]
            _require(entry.get('status') == 'ok', 'unsuccessful feature entry: ' + group)
            frame_field = 'frame_count' if group == 'rgb' else 'frames'
            _require(type(entry.get(frame_field)) is int and entry[frame_field] == row['frames']
                     and entry.get('fps') == row['fps'], 'feature frames/FPS: ' + group)
            if group != 'rgb':
                _require(entry.get('source_sha256') == row['sha256'], 'feature source SHA: ' + group)
                _require(_hash_string(entry.get('feature_sha256')), 'missing original feature hash: ' + group)
            if group in ('motion', 'local'):
                _require(entry.get('profile_signature') == profile_hash, 'per-row profile: ' + group)
            if group == 'local':
                _require(entry.get('source') == projected_sources[i]
                         and entry.get('extractor_sha256') == profile['extractor_sha256'], 'local source provenance')
            relative = entry.get('file', entry.get('npz_path'))
            path = _safe_file(root / folder, relative)
            _require(path.suffix == '.npz' and path not in used_paths, 'duplicate/non-NPZ feature path')
            used_paths.add(path)
            file_key = path.relative_to(root).as_posix()
            data = read(file_key)
            # Do not invent a declaration for RGB. Its measured hash joins only the audit/new-fit binding.
            if 'feature_sha256' in entry:
                _require(_hash_string(entry['feature_sha256']) and hashes[file_key] == entry['feature_sha256'],
                         'declared feature hash mismatch: ' + group + ' row ' + str(i))
                attested += 1
            if 'source_sha256' in entry:
                _require(entry['source_sha256'] == row['sha256'], 'declared feature source SHA')
            if group in ('motion', 'local'):
                side = read_json(path.with_suffix('.json').relative_to(root).as_posix())
                fields = ['file', 'status', 'frames', 'fps', 'source_sha256', 'feature_sha256', 'profile_signature']
                fields += ['name'] if group == 'motion' else ['row_id', 'source', 'extractor_sha256']
                _require(all(side.get(k) == entry.get(k) for k in fields), 'feature sidecar provenance: ' + group)
            values = _archive(data, group + ' feature row ' + str(i))
            key = 'features' if group == 'rgb' else 'signals'
            _require(key in values and ('signals' if key == 'features' else 'features') not in values,
                     'feature array identity: ' + group)
            x = values[key]
            dim = profile['feature_dim'] if group == 'rgb' else len(names)
            _require(x.dtype == np.float32 and x.shape == (row['frames'], dim)
                     and np.isfinite(x).all(), 'feature array schema: ' + group)
            fps = values.get('fps')
            _require(fps is not None and fps.shape == () and fps.dtype.kind in 'iuf'
                     and np.isclose(float(fps), row['fps'], rtol=1e-4, atol=1e-4), 'feature NPZ FPS')
            if group in ('motion', 'corrected', 'local'):
                _require('frame_ids' in values and np.array_equal(values['frame_ids'], np.arange(row['frames'])),
                         'feature frame coverage: ' + group)
            if group in ('motion', 'local'):
                mask = values.get('feature_valid')
                _require(mask is not None and mask.dtype == np.bool_ and mask.shape == x.shape
                         and not np.any(x[~mask] != 0), 'feature validity/mask: ' + group)
                times = values.get('frame_times_seconds')
                _require(times is not None and times.shape == (row['frames'],)
                         and np.allclose(times, np.arange(row['frames']) / float(fps), rtol=0, atol=1e-12),
                         'feature frame times: ' + group)
            if group == 'local':
                _require(values.get('feature_names') is not None
                         and values['feature_names'].tolist() == names, 'local NPZ feature names')
                x = np.concatenate([x, mask.astype(np.float32)], axis=1)
            if group == 'rgb':
                _require(entry.get('profile_id') == profile_hash
                         and _json(str(values['feature_profile_json'].item()), 'RGB NPZ profile') == profile,
                         'RGB embedded profile provenance')
                embedded = _json(str(values['metadata_json'].item()), 'RGB NPZ metadata')
                _require(all(embedded.get(k) == entry.get(k) for k in
                             ['profile_id', 'fps', 'frame_count', 'source_size_bytes', 'source_mtime_ns', 'source_path']),
                         'RGB embedded source metadata')
                _require(fm.get('features_by_name', {}).get(row['name']) == relative, 'RGB name/file mapping')
            record_hashes[i][group] = _array_digest(x)
            if names is not None:
                bound_names = (['local/' + n for n in names]
                               + ['local/' + n + '_support_flag' for n in names]) if group == 'local' else names
                record_hashes[i][group + '_names'] = _digest_value(bound_names)
        summary[group] = {'files': len(rows), 'original_declared_feature_hashes_verified': attested,
                          'audit_measured_feature_hashes': len(rows), 'manifest_sha256': hashes[folder + '/manifest.json'],
                          'profile_signature': profile_hash}
    frozen_hashes = tuple(sorted(hashes.items()))
    frozen_records = tuple(tuple(sorted(r.items())) for r in record_hashes)
    return root, manifest_bytes, frozen_hashes, frozen_records, _canonical(summary)


def audit_legacy_inputs(root=ROOT):
    """Verify fixed legacy snapshots AND the explicit 2026-10-02 audit-time pin.

    Rebuilding this pin is a new audit decision, never automatic cache repair.
    It cannot supply missing historical provenance. No metadata files are changed.
    """
    root, manifest, inputs, records, summary = _collect_inputs(root)
    _require(_sha(manifest) == LEGACY_MANIFEST_SHA256, 'legacy grouped manifest hash')
    for name, expected in LEGACY_SNAPSHOT_SHA256.items():
        _require(_sha(_read_bytes(root / NEW / 'grouped_training/sources' / name)) == expected,
                 'fixed training source snapshot mismatch: ' + name)
    inputs_sha = _digest_value(dict(inputs))
    _require(_hash_string(LEGACY_AUDITED_INPUTS_SHA256) and inputs_sha == LEGACY_AUDITED_INPUTS_SHA256,
             'legacy audited input binding mismatch; do not silently regenerate the audit pin')
    return TrainingEvidence(root, LEGACY_MODE, manifest, tuple(sorted(LEGACY_SNAPSHOT_SHA256.items())),
                            inputs, records, summary, inputs_sha)


def collect_new_fit_inputs(root=ROOT):
    """Capture inputs before a NEW fit; not authorization to relabel legacy caches."""
    root, manifest, inputs, records, summary = _collect_inputs(root)
    sources = tuple(sorted((name, _sha(_read_bytes(root / name))) for name in NEW_TRAINING_SOURCES))
    return TrainingEvidence(root, NEW_FIT_MODE, manifest, sources, inputs, records, summary,
                            _digest_value(dict(inputs)))


def _recheck_evidence(evidence):
    _require(evidence.mode in (LEGACY_MODE, NEW_FIT_MODE), 'unknown evidence mode')
    _require(evidence.inputs_sha256 == _digest_value(dict(evidence.input_hashes))
             and dict(evidence.input_hashes).get(NEW + '/dataset_manifest.json') == _sha(evidence.manifest_bytes),
             'inconsistent evidence input binding')
    if evidence.mode == LEGACY_MODE:
        _require(evidence.inputs_sha256 == LEGACY_AUDITED_INPUTS_SHA256
                 and _sha(evidence.manifest_bytes) == LEGACY_MANIFEST_SHA256
                 and dict(evidence.sources) == LEGACY_SNAPSHOT_SHA256,
                 'evidence differs from fixed legacy audit anchors')
    else:
        _require(set(dict(evidence.sources)) == set(NEW_TRAINING_SOURCES),
                 'incomplete extended training source coverage')
    for relative, expected in evidence.input_hashes:
        _require(_sha(_read_bytes(evidence.root / relative)) == expected,
                 'bound input changed: ' + relative)
    for name, expected in evidence.sources:
        path = (evidence.root / NEW / 'grouped_training/sources' / name
                if evidence.mode == LEGACY_MODE else evidence.root / name)
        _require(_sha(_read_bytes(path)) == expected, 'bound training source changed: ' + name)


def _validate_records(evidence, records):
    rows = evidence.manifest['rows']
    _require(len(records) == len(rows), 'loader record coverage')
    for i, (row, record, fingerprints) in enumerate(zip(rows, records, evidence.record_hashes)):
        _require(all(record.get(k) == v for k, v in row.items()), 'loader row identity: ' + str(i))
        for key, expected in fingerprints:
            _require(key in record, 'missing loader input: ' + key)
            actual = _digest_value(record[key]) if key.endswith('_names') else _array_digest(record[key])
            _require(actual == expected, 'loader input differs from audited feature: ' + key + ' row ' + str(i))


def _extended_payload(evidence, fold, recipe, npz_sha256):
    _require(_hash_string(npz_sha256), 'invalid probability NPZ hash')
    return {'schema_version': EXTENDED_VERSION,
            **_expected_partitions(evidence.manifest, fold, recipe),
            'grouped_manifest_sha256': _sha(evidence.manifest_bytes),
            'inputs_sha256': evidence.inputs_sha256, 'input_hashes': dict(evidence.input_hashes),
            'sources': dict(evidence.sources), 'npz_sha256': npz_sha256,
            'runtime': {'numpy': np.__version__},
            'purpose': 'new-fit-only; not retroactive legacy provenance'}


def extended_signature_fields(evidence, fold, recipe, *, npz_sha256):
    """Fields for a new metadata row; save signature separately as a trusted receipt.

    Caller must fit from this evidence's inputs between collection and this call.
    No writer/migration of existing metadata is provided.
    """
    _require(evidence.mode == NEW_FIT_MODE, 'cannot upgrade legacy audited evidence into a new fit')
    _recheck_evidence(evidence)
    payload = _extended_payload(evidence, fold, recipe, npz_sha256)
    return {'signature_version': EXTENDED_VERSION, 'signature': _digest_value(payload),
            'provenance': payload, 'sources': dict(evidence.sources)}


def _validate_probabilities(arrays, metadata, rows, recipe):
    expected_keys = set()
    boundary = 'boundary' in recipe
    for kind, indices in [('inner', metadata['train']), ('outer', metadata['validation'])]:
        for i in indices:
            for channel in (['frame', 'video', 'boundary'] if boundary else ['frame', 'video']):
                key = f'{kind}_{channel}_{i}'
                expected_keys.add(key)
                _require(key in arrays, 'missing probability coverage: ' + key)
                arr = arrays[key]
                shape = () if channel == 'video' else (2, rows[i]['frames']) if channel == 'boundary' else (rows[i]['frames'],)
                _require(arr.dtype.kind in 'iuf' and arr.shape == shape
                         and np.isfinite(arr).all() and np.all((arr >= 0) & (arr <= 1)),
                         'invalid probability shape/range: ' + key)
    _require(set(arrays) == expected_keys, 'extra/wrong-split probability coverage')


def _verify_fold(evidence, fold, recipe, expected_signature):
    _expected_partitions(evidence.manifest, fold, recipe)
    path = evidence.root / NEW / 'grouped_training/folds' / f'{fold}_{recipe}.npz'
    metadata = _json(_read_bytes(path.with_suffix('.json')), 'cache metadata')
    _require(isinstance(metadata, dict), 'cache metadata must be object')
    _validate_partitions(metadata, evidence.manifest, fold, recipe)
    data = _read_bytes(path)
    actual_hash = _sha(data)
    _require(_hash_string(metadata.get('npz_sha256')) and metadata['npz_sha256'] == actual_hash,
             'probability NPZ hash mismatch')
    if evidence.mode == LEGACY_MODE:
        _require('signature_version' not in metadata and 'provenance' not in metadata,
                 'legacy cache cannot carry fabricated extended provenance')
        sources = {LEGACY_SOURCE_PREFIX + name: sha for name, sha in evidence.sources}
        _require(metadata.get('sources') == sources, 'legacy training sources mismatch')
        signature = _sha(evidence.manifest_bytes) + json.dumps(sources, sort_keys=True)
        _require(metadata.get('signature') == signature, 'legacy manifest/snapshot signature mismatch')
    else:
        _require(_hash_string(expected_signature), 'extended cache requires a trusted expected_signature')
        _require(metadata.get('signature_version') == EXTENDED_VERSION, 'extended signature version required')
        payload = _extended_payload(evidence, fold, recipe, actual_hash)
        signature = _digest_value(payload)
        _require(metadata.get('sources') == dict(evidence.sources), 'extended sources mismatch')
        _require(metadata.get('provenance') == payload and metadata.get('signature') == signature
                 and signature == expected_signature, 'extended input/source/partition signature mismatch')
    arrays = _archive(data, 'probability cache')
    _validate_probabilities(arrays, metadata, evidence.manifest['rows'], recipe)
    for value in arrays.values():
        value.setflags(write=False)
    return VerifiedFold(metadata, arrays, evidence.mode, actual_hash, evidence.inputs_sha256)


def verify_fold_cache(evidence, fold, recipe, records=None, *, expected_signature=None):
    """Refuse stale/tampered metadata, input features, grouping or NPZ coverage.

    All bound inputs and snapshots are rehashed, even when evidence is reused.
    Returned arrays come from the SAME bytes whose hash was verified. Integrating
    loaders should consume these rather than opening the path a second time.
    """
    _recheck_evidence(evidence)
    if records is not None:
        _validate_records(evidence, records)
    return _verify_fold(evidence, fold, recipe, expected_signature)


def _cache_files(root):
    folder = root / NEW / 'grouped_training/folds'
    return {p.name for p in folder.iterdir() if p.suffix in ('.npz', '.json')}


def verify_all_cached_folds(evidence):
    """Bounded legacy batch audit: rehash shared inputs once before and after."""
    _require(evidence.mode == LEGACY_MODE, 'batch audit is only for the existing legacy cache')
    _recheck_evidence(evidence)
    expected = {f'{f}_{r}.{suffix}' for f in range(5) for r in RECIPES for suffix in ('json', 'npz')}
    _require(_cache_files(evidence.root) == expected, '65-cache inventory incomplete/extra')
    results = []
    for fold in range(5):
        for recipe in RECIPES:
            checked = _verify_fold(evidence, fold, recipe, None)
            results.append({'fold': fold, 'recipe': recipe, 'npz_sha256': checked.npz_sha256,
                            'inner_rows': len(checked.metadata['train']),
                            'outer_rows': len(checked.metadata['validation'])})
    _recheck_evidence(evidence)
    return results


def audit_report(evidence, results):
    manifest = evidence.manifest
    live_differences = []
    for name, expected in evidence.sources:
        if _sha(_read_bytes(evidence.root / 'code' / name)) != expected:
            live_differences.append(name)
    return {'schema_version': LEGACY_MODE, 'audited_at': datetime.now(timezone.utc).isoformat(),
            'status': 'verified', 'report_kind': 'current-attestation',
            'attestation_scope': 'current-files-only',
            'historical_pixel_to_probability_binding_proven': False,
            'training_origin_attested_for_omitted_fields': False,
            'original_metadata_modified': False, 'cache_count': len(results),
            'rows': len(manifest['rows']), 'unique_content_groups': len(manifest['content_groups']),
            'outer_fold_sizes': [len(f) for f in manifest['outer_folds']],
            'grouped_manifest_sha256': _sha(evidence.manifest_bytes),
            'base_dataset_npz_audit_time_sha256': dict(evidence.input_hashes)[OLD + '/dataset.npz'],
            'audit_time_inputs_sha256': evidence.inputs_sha256,
            'audit_time_input_file_count': len(evidence.input_hashes),
            'fixed_training_snapshot_sha256': dict(evidence.sources),
            'live_training_sources_differing_from_snapshot': live_differences,
            'features': evidence.feature_summary,
            'checks': ['fold/recipe identity', 'original manifest+fixed snapshot signature',
                       'exact sources including original WSL path namespace', 'NPZ SHA256',
                       'exact train/outer/seeded inner partitions', 'SHA holdout and full OOF coverage',
                       'probability keys, shapes, finite ranges including video/boundary',
                       'existing declared pixel provenance hashes/profile/source/sidecar checks',
                       'independent audit-time full input binding including base NPZ and RGB'],
            'limitations': list(LEGACY_LIMITATIONS),
            'new_signature_version': EXTENDED_VERSION,
            'new_signature_fields': ['grouped_manifest_sha256', 'input_hashes', 'inputs_sha256',
                                     'sources (including transitive splitter/loader/helper)',
                                     'fold/recipe/train/validation/inner_partitions/seeds',
                                     'npz_sha256', 'runtime.numpy', 'purpose'],
            'cache_npz_inventory_sha256': _digest_value({f"{r['fold']}_{r['recipe']}": r['npz_sha256'] for r in results})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true',
                        help='write only the explicitly authorized v2 provenance_audit.json')
    args = parser.parse_args(argv)
    evidence = audit_legacy_inputs()
    results = verify_all_cached_folds(evidence)
    report = audit_report(evidence, results)
    if args.write_report:
        target = ROOT / NEW / 'provenance_audit.json'
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + chr(10), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
