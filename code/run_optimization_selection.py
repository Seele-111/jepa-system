#!/usr/bin/env python3
"""Fixed, cache-only outer5/inner3 selection and CPU portable deployment export.

No candidate generation, outer-score pruning, feature extraction, GPU, or installs.
Execution requires --reports-complete; fitting also requires --deployment-fit.
Full-development OOF selection is optimistic and used ONLY for deployment.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from build_optimization_dataset import make_folds
from optimized_feature_view import feature_view
from optimized_locator import SCHEMA, metrics, objective, spans
from run_algorithm_optimization import (
    _group_metrics, _read_optional_features, choose_decoder,
    evaluate_probabilities, export_model, fit_predict,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'output' / 'algorithm-opt-2026-10-02'
SEED = 20261002
CORRECTED_FOLDER = 'corrected_jepa_features_v2'
EXPECTED_VIDEOS = 77
PARITY_ATOL = 2e-6
REPORT_RECIPES = {
    'nested_fast_seed0': ('motion_rf', 'motion_et'),
    'nested_temporal_seed0': ('rgb_motion_tcn',),
    'nested_corrected_v2_seed0': (
        'corrected_motion_rf', 'corrected_motion_et', 'corrected_motion_tcn',
        'rgb_corrected_motion_rf',
    ),
    'nested_boundary_v2_seed0': (
        'corrected_motion_boundary_rf', 'corrected_motion_boundary_et',
    ),
}
SINGLE_RECIPES = tuple(r for recipes in REPORT_RECIPES.values() for r in recipes)
EXCLUSIONS = {
    'corrected_jepa_features': 'v1 hidden I mask counter: cold/hot/order drift',
    'nested_corrected_seed0': 'uses invalid corrected v1 features',
    'corrected_temporal_seed0': 'uses invalid corrected v1 features',
    'jepa_tcn': 'diagnostic legacy JEPA only; never deployable',
}


@dataclass(frozen=True)
class Candidate:
    name: str
    members: tuple[tuple[str, float], ...]

    def __post_init__(self):
        recipes = [r for r, _ in self.members]
        weights = np.asarray([w for _, w in self.members], dtype=np.float64)
        if (not recipes or len(set(recipes)) != len(recipes)
                or not set(recipes) <= set(SINGLE_RECIPES)
                or not np.isfinite(weights).all() or np.any(weights <= 0)
                or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError('invalid fixed candidate members/weights')

    def as_dict(self):
        return {'name': self.name, 'members': [
            {'recipe': r, 'weight': w} for r, w in self.members]}


CANDIDATES = tuple(Candidate(r, ((r, 1.0),)) for r in SINGLE_RECIPES) + (
    Candidate('corrected_rf_et_50_50', (('corrected_motion_rf', .5), ('corrected_motion_et', .5))),
    Candidate('corrected_rf_et_75_25', (('corrected_motion_rf', .75), ('corrected_motion_et', .25))),
    Candidate('corrected_rf_et_25_75', (('corrected_motion_rf', .25), ('corrected_motion_et', .75))),
    Candidate('corrected_rf_et_rgb_tcn_40_40_20', (
        ('corrected_motion_rf', .4), ('corrected_motion_et', .4), ('rgb_motion_tcn', .2))),
)
FAST_CANDIDATES = tuple(c for c in CANDIDATES if c.name in ('motion_rf', 'motion_et'))


@dataclass
class Probabilities:
    frame: dict[int, np.ndarray]
    video: dict[int, float]
    boundary: dict[int, tuple[np.ndarray, np.ndarray]] | None = None


@dataclass
class FoldCache:
    inner: Probabilities
    outer: Probabilities
    inner_selection: dict | None = None


class EvidenceError(ValueError):
    """Incomplete/stale/cross-fold evidence; never run a subset."""


class ParityError(ValueError):
    """Portable inference differs from the final fitted state."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise EvidenceError('duplicate JSON key: ' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_pairs)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _indices(values, expected, label):
    if (not isinstance(values, list) or any(type(v) is not int for v in values)
            or len(set(values)) != len(values) or values != list(expected)):
        raise EvidenceError(label + ': indices/order differ from the fixed protocol')


def validate_protocol(manifest, records, expected_count=None):
    protocol = manifest.get('protocol', {})
    if (manifest.get('schema_version') != 'algorithm-opt-dataset-v1'
            or protocol.get('outer_folds') != 5 or protocol.get('inner_folds') != 3
            or protocol.get('seed') != SEED):
        raise EvidenceError('requires fixed outer5/inner3/20261002 dataset protocol')
    n = len(records)
    if not n or (expected_count is not None and n != expected_count):
        raise EvidenceError(f'expected {expected_count} development videos, got {n}')
    if len(set(r['name'] for r in records)) != n or len(manifest.get('rows', [])) != n:
        raise EvidenceError('duplicate names or manifest/record count mismatch')
    prompts = [r['prompt_id'] for r in records if r.get('prompt_id', 'unknown') != 'unknown']
    if len(set(prompts)) != len(prompts):
        raise EvidenceError('duplicate prompt groups require a new grouped protocol')
    for row, r in zip(manifest['rows'], records):
        y = np.asarray(r['labels'])
        if (row['name'] != r['name'] or y.ndim != 1 or not len(y)
                or not np.isin(y, [0, 1]).all() or r.get('frames') != len(y)
                or r['event_count'] != len(spans(y))
                or not np.isfinite(r['fps']) or r['fps'] <= 0):
            raise EvidenceError('invalid video/label/FPS metadata: ' + r['name'])
        digest = r.get('sha256', '')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise EvidenceError('invalid training video SHA256: ' + r['name'])
    folds = manifest.get('outer_folds', [])
    if (len(folds) != 5 or any(not isinstance(f, list) or not f for f in folds)
            or any(type(i) is not int for f in folds for i in f)
            or sorted(i for f in folds for i in f) != list(range(n))):
        raise EvidenceError('outer folds must partition all videos exactly once into five nonempty folds')
    if any(f != sorted(f) for f in folds):
        raise EvidenceError('outer fold order differs from fixed protocol')


def inner_splits(records, outer_indices, fold):
    held_out = set(outer_indices)
    train = [i for i in range(len(records)) if i not in held_out]
    local = make_folds([records[i] for i in train], 3, SEED + fold * 31)
    if len(local) != 3 or any(not f for f in local) or sorted(sum(local, [])) != list(range(len(train))):
        raise EvidenceError('inner folds must form three complete nonempty OOF partitions')
    splits = []
    for inner_fold, indices in enumerate(local):
        val = [train[j] for j in indices]
        fit = [i for i in train if i not in set(val)]
        if set(fit) & set(val) or (set(fit) | set(val)) != set(train) or (set(fit) | set(val)) & held_out:
            raise EvidenceError('inner/outer leakage')
        splits.append({'fold': inner_fold, 'train_indices': fit, 'validation_indices': val,
                       'seed': SEED + fold * 53 + inner_fold})
    return train, splits


def load_dataset(data_root):
    data_root = Path(data_root)
    manifest_path = data_root / 'dataset_manifest.json'
    manifest = read_json(manifest_path)
    dataset_hash = sha256(manifest_path)
    input_hashes = {str(manifest_path.resolve()): dataset_hash,
                    str((data_root / 'dataset.npz').resolve()): sha256(data_root / 'dataset.npz')}
    profiles, arrays, feature_names = {}, {}, {}
    rows = manifest['rows']
    for key, folder in [('motion', 'motion_features'), ('rgb', 'rgb_features'), ('corrected', CORRECTED_FOLDER)]:
        path = data_root / folder / 'manifest.json'
        cache_manifest = read_json(path)
        input_hashes[str(path.resolve())] = sha256(path)
        if cache_manifest.get('dataset_manifest_sha256', dataset_hash) != dataset_hash:
            raise EvidenceError('feature cache belongs to another dataset: ' + folder)
        if cache_manifest.get('status', 'complete') != 'complete':
            raise EvidenceError('feature cache incomplete: ' + folder)
        profile = cache_manifest.get('profile')
        if not isinstance(profile, dict) or not profile:
            raise EvidenceError('missing actual manifest feature profile: ' + folder)
        if key == 'corrected' and profile.get('i_mask_reuse_policy') != 'explicit_batch_seed_reset_and_restore':
            raise EvidenceError('corrected cache does not attest fixed v2 I mask-counter isolation')
        profiles[key] = profile
        arrays[key] = _read_optional_features(data_root, folder)
        missing = [r['name'] for r in rows if r['name'] not in arrays[key]]
        if missing:
            raise EvidenceError(f'{folder}: missing fixed videos {missing}; no subset allowed')
        name_path = data_root / folder / 'feature_names.json'
        if key != 'rgb':
            feature_names[key] = read_json(name_path)
            input_hashes[str(name_path.resolve())] = sha256(name_path)
            if profile.get('feature_names', feature_names[key]) != feature_names[key]:
                raise EvidenceError('feature_names/profile mismatch: ' + folder)
        entries = cache_manifest.get('videos', cache_manifest.get('records', cache_manifest.get('results', [])))
        if isinstance(entries, dict):
            entries = [dict(v, name=k) for k, v in entries.items()]
        by_name = {e.get('name', e.get('video_name')): e for e in entries}
        for row in rows:
            entry = by_name[row['name']]
            if entry.get('source_sha256', row['sha256']) != row['sha256']:
                raise EvidenceError('feature source video hash mismatch: ' + row['name'])
            file_path = path.parent / entry.get('file', entry.get('npz_path'))
            input_hashes[str(file_path.resolve())] = sha256(file_path)
    records = []
    with np.load(data_root / 'dataset.npz', allow_pickle=False) as archive:
        for i, row in enumerate(rows):
            r = {**row, 'jepa': archive[f'jepa_{i}'], 'rank_motion': archive[f'rank_motion_{i}'],
                 'labels': archive[f'labels_{i}'], 'jepa_names': manifest['feature_names']['jepa']}
            for key in arrays:
                r[key] = arrays[key][row['name']]
                if key in feature_names:
                    r[key + '_names'] = feature_names[key]
            for key in ('jepa', 'rank_motion', 'motion', 'rgb', 'corrected'):
                x = np.asarray(r[key])
                if x.ndim != 2 or x.shape[0] != len(r['labels']) or x.shape[1] < 1 or not np.isfinite(x).all():
                    raise EvidenceError('invalid/misaligned raw features: ' + key + ' ' + row['name'])
                if key in ('jepa', 'motion', 'corrected') and len(r[key + '_names']) != x.shape[1]:
                    raise EvidenceError('raw feature name order/dimension mismatch: ' + key)
            records.append(r)
    validate_protocol(manifest, records, EXPECTED_VIDEOS)
    return manifest, records, profiles, input_hashes


def _probability_array(value, length, label):
    arr = np.asarray(value)
    if (arr.dtype.kind not in 'iuf' or arr.shape != (length,) or not np.isfinite(arr).all()
            or np.any(arr < 0) or np.any(arr > 1)):
        raise EvidenceError('invalid probability curve: ' + label)
    return arr.astype(np.float32)


def _probability_map(value, indices, label):
    if not isinstance(value, dict) or set(value) != {str(i) for i in indices}:
        raise EvidenceError(label + ': OOF indices must match exactly, no subset/outer leakage')
    return value


def probabilities_from_row(row, records, indices, inner=False):
    prefix = 'inner_' if inner else ''
    frame = _probability_map(row.get(prefix + 'frame_probabilities'), indices, prefix + 'frame')
    video = _probability_map(row.get(prefix + 'video_probabilities'), indices, prefix + 'video')
    fp, vp = {}, {}
    for i in indices:
        fp[i] = _probability_array(frame[str(i)], len(records[i]['labels']), prefix + str(i))
        p = video[str(i)]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 1:
            raise EvidenceError('invalid video probability: ' + prefix + str(i))
        vp[i] = float(p)
    boundary = row.get(prefix + 'boundary_probabilities')
    bp = None
    if boundary is not None:
        boundary = _probability_map(boundary, indices, prefix + 'boundary')
        bp = {}
        for i in indices:
            curves = boundary[str(i)]
            if not isinstance(curves, list) or len(curves) != 2:
                raise EvidenceError('boundary probabilities need start/end curves')
            bp[i] = tuple(_probability_array(v, len(fp[i]), 'boundary') for v in curves)
    if ('boundary' in row['recipe']) != (bp is not None):
        raise EvidenceError('missing/unexpected boundary probabilities for ' + row['recipe'])
    return Probabilities(fp, vp, bp)


def validate_fold_row(row, recipe, fold, manifest, records, dataset_hash, signature):
    val = manifest['outer_folds'][fold]
    train, _ = inner_splits(records, val, fold)
    if row.get('fold') != fold or row.get('recipe') != recipe:
        raise EvidenceError('fold/recipe mismatch')
    if row.get('dataset_manifest_sha256') != dataset_hash or row.get('experiment_signature') != signature:
        raise EvidenceError('stale fold dataset/experiment signature')
    _indices(row.get('validation_indices'), val, 'outer')
    if row.get('train_indices') is not None:
        _indices(row['train_indices'], train, 'train')
    if (row.get('train_names') != [records[i]['name'] for i in train]
            or row.get('validation_names') != [records[i]['name'] for i in val]):
        raise EvidenceError('fold train/validation names mismatch')
    inner = probabilities_from_row(row, records, train, True)
    outer = probabilities_from_row(row, records, val)
    saved = deepcopy(row.get('selection'))
    if saved is not None:
        if inner.boundary is None:
            saved['config'].setdefault('boundary_seconds', 0)
        allowed = {'threshold': [.3, .4, .5, .6, .7, .8], 'low_ratio': [1., .7],
            'smooth_seconds': [0, .15], 'gap_seconds': [0, .12], 'min_seconds': [.08, .20],
            'video_strength': [0., .5], 'video_threshold': [0, .35, .5, .65, .75],
            'boundary_seconds': [0, .2, .4] if inner.boundary is not None else [0]}
        cfg = saved.get('config', {})
        if set(cfg) != set(allowed) or any(cfg[k] not in choices for k, choices in allowed.items()):
            raise EvidenceError('cached single decoder is outside the fixed same grid')
        _, checked = evaluate_probabilities(records, train, inner.frame, inner.video, cfg, inner.boundary)
        if checked != saved.get('metrics'):
            raise EvidenceError('cached inner selection metrics do not reproduce from inner OOF')
    return FoldCache(inner, outer, saved)



# The earlier valid motion/RGB/corrected singles predate boundary heads. Their
# search is identical to the current search when boundary_probs=None. Accept
# ONLY this exact recorded implementation, never arbitrary older code.
LEGACY_NON_BOUNDARY_SEARCH = '''
def choose_decoder(records,indices,frame_probs,video_probs):
    labels=[records[i]['labels'] for i in indices]
    best=None;best_key=None
    for config in decoder_grid():
        for strength in [0.0,0.5]:
            adjusted={**config,'video_strength':strength}
            base=[decode(frame_probs[i],records[i]['fps'],video_probs[i],adjusted) for i in indices]
            for vt in [0,.35,.5,.65,.75]:
                preds=[p if video_probs[i]>=vt else [] for i,p in zip(indices,base)]
                result=metrics(preds,labels)
                key=(objective(result),result['iou_0.5']['f1'],result['iou_0.3']['precision'],-result['predicted_segments'])
                if best_key is None or key>best_key:best_key=key;best=({**adjusted,'video_threshold':vt},result)
    return best
'''


def same_search_implementation(producer, recipes):
    def functions(text):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name in ('decoder_grid', 'choose_decoder')}
    saved = functions(Path(producer).read_text(encoding='utf-8'))
    live = functions(Path(__file__).with_name('run_algorithm_optimization.py').read_text(encoding='utf-8'))
    legacy = functions(LEGACY_NON_BOUNDARY_SEARCH)
    return (saved.get('decoder_grid') == live.get('decoder_grid') and
            (saved.get('choose_decoder') == live.get('choose_decoder') or
             (not any('boundary' in r for r in recipes) and
              saved.get('choose_decoder') == legacy['choose_decoder'])))


def _verify_experiment_sources(report, source_dir, data_root, input_hashes):
    sources = report.get('experiment_sources')
    if not isinstance(sources, dict) or not sources:
        raise EvidenceError('missing source-hashed experiment provenance')
    signature = hashlib.sha256(json.dumps({'sources': sources, 'recipes': report['recipes']}, sort_keys=True).encode()).hexdigest()
    if report.get('experiment_signature') != signature:
        raise EvidenceError('experiment signature does not match recorded sources/recipes')
    seen = set()
    for raw_path, expected in sources.items():
        path = raw_path.replace('\\', '/')
        parts = path.split('/')
        if 'corrected_jepa_features' in parts or any(old in parts for old in ('nested_corrected_seed0', 'corrected_temporal_seed0')):
            raise EvidenceError('forbidden corrected v1 experiment source')
        if path.endswith('.py'):
            actual = source_dir / 'sources' / parts[-1]
        elif parts[-1] in ('dataset_manifest.json', 'dataset.npz'):
            actual = data_root / parts[-1]
            seen.add(parts[-1])
        else:
            actual = None
            for folder in ('motion_features', 'rgb_features', CORRECTED_FOLDER):
                marker = '/' + folder + '/'
                if marker in path:
                    suffix = path.split(marker, 1)[1]
                    if '..' in suffix.split('/'):
                        raise EvidenceError('source path escapes feature cache')
                    actual = data_root / folder / suffix
                    seen.add(folder)
                    break
            if actual is None:
                raise EvidenceError('unrecognized experiment source: ' + raw_path)
        key = str(actual.resolve())
        digest = input_hashes.get(key)
        if digest is None:
            digest = sha256(actual)
            input_hashes[key] = digest
        if digest != expected:
            raise EvidenceError('experiment source hash mismatch: ' + str(actual))
        if parts[-1] == 'run_algorithm_optimization.py' and not same_search_implementation(actual, report['recipes']):
            raise EvidenceError('cached single search is not the same choose_decoder/grid')
    required = {'dataset_manifest.json', 'dataset.npz', 'motion_features'}
    if any(r.startswith('rgb_') for r in report['recipes']):
        required.add('rgb_features')
    if any('corrected' in r for r in report['recipes']):
        required.add(CORRECTED_FOLDER)
    if not required <= seen:
        raise EvidenceError('experiment omitted required input source hashes')
    return signature


def load_fold_caches(data_root, manifest, records, input_hashes):
    data_root = Path(data_root)
    paths = [data_root / folder / 'report.json' for folder in REPORT_RECIPES]
    paths += [data_root / folder / 'folds' / f'fold{fold}_{recipe}.json'
              for folder, recipes in REPORT_RECIPES.items() for recipe in recipes for fold in range(5)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise EvidenceError('reports incomplete; refusing subset selection. Missing: ' + ', '.join(missing))
    dataset_hash = sha256(data_root / 'dataset_manifest.json')
    caches, report_sources, experiments = {}, {}, {}
    for folder, recipes in REPORT_RECIPES.items():
        source_dir = data_root / folder
        path = source_dir / 'report.json'
        report = read_json(path)
        report_sources[str(path.resolve())] = sha256(path)
        if (report.get('schema_version') != 'algorithm-opt-nested-cv-v1'
                or report.get('dataset_manifest_sha256') != dataset_hash
                or report.get('protocol') != manifest['protocol']
                or not set(recipes) <= set(report.get('recipes', []))
                or 'nested_selected_strategy' not in report or 'family_oof' not in report):
            raise EvidenceError('incomplete/stale report: ' + folder)
        folds = report.get('folds', [])
        if len(folds) != 5 or sorted(f.get('fold', -1) for f in folds) != list(range(5)):
            raise EvidenceError('report must contain all five outer folds: ' + folder)
        for fold_summary in folds:
            if not set(recipes) <= {r['recipe'] for r in fold_summary.get('families', [])}:
                raise EvidenceError('report missing a fixed family: ' + folder)
        if [r.get('name') for r in report.get('selected_predictions', [])] != [r['name'] for r in records]:
            raise EvidenceError('report final OOF video coverage/order mismatch: ' + folder)
        signature = _verify_experiment_sources(report, source_dir, data_root, input_hashes)
        experiments[folder] = {'experiment_signature': signature, 'recipes': report['recipes']}
        for recipe in recipes:
            caches[recipe] = {}
            for fold in range(5):
                path = source_dir / 'folds' / f'fold{fold}_{recipe}.json'
                row = read_json(path)
                report_sources[str(path.resolve())] = sha256(path)
                caches[recipe][fold] = validate_fold_row(row, recipe, fold, manifest, records, dataset_hash, signature)
    return caches, report_sources, experiments


def blend_probabilities(candidate, probabilities, indices):
    expected = set(indices)
    for recipe, _ in candidate.members:
        p = probabilities[recipe]
        if set(p.frame) != expected or set(p.video) != expected or (p.boundary is not None and set(p.boundary) != expected):
            raise EvidenceError('cannot blend different OOF partitions')
    # Match predict_record's float32 frame arithmetic and normalized head weights.
    fp, vp, bp = {}, {}, None
    heads = [(w, probabilities[r].boundary) for r, w in candidate.members if probabilities[r].boundary is not None]
    if heads:
        bp = {}
    for i in indices:
        curves = [probabilities[r].frame[i] for r, _ in candidate.members]
        if len({c.shape for c in curves}) != 1:
            raise EvidenceError('cannot blend differently aligned curves')
        fp[i] = sum(w * probabilities[r].frame[i] for r, w in candidate.members).astype(np.float32)
        vp[i] = float(sum(w * probabilities[r].video[i] for r, w in candidate.members))
        if heads:
            mass = sum(w for w, _ in heads)
            bp[i] = tuple(sum(w * b[i][j] for w, b in heads) / mass for j in (0, 1))
    return Probabilities(fp, vp, bp)


def prediction_summary(records, predictions):
    if set(predictions) != set(range(len(records))):
        raise EvidenceError('final OOF predictions must cover every fixed video')
    normal = metrics([predictions[i] for i in range(len(records))], [r['labels'] for r in records])['normal']
    empty = [i for i in range(len(records)) if not predictions[i]]
    return {
        'normal': {**normal, 'false_positive_names': [r['name'] for i, r in enumerate(records)
                  if not np.any(r['labels']) and predictions[i]],
                  'correctly_empty_names': [r['name'] for i, r in enumerate(records)
                  if not np.any(r['labels']) and not predictions[i]]},
        'no_candidates': {'videos': len(empty), 'names': [records[i]['name'] for i in empty],
                          'normal_videos': int(sum(not np.any(records[i]['labels']) for i in empty)),
                          'positive_videos': sum(bool(np.any(records[i]['labels'])) for i in empty)},
    }


def predictions_json(records, predictions):
    return [{'index': i, 'name': r['name'], 'segments': [list(v) for v in predictions[i]],
             'has_candidates': bool(predictions[i]), 'is_normal': not bool(np.any(r['labels']))}
            for i, r in enumerate(records)]


def nested_selection(manifest, records, caches, candidates=CANDIDATES, chooser=None):
    """The chooser and family objective see INNER probabilities only."""
    chooser = chooser or choose_decoder
    validate_protocol(manifest, records)
    if set(caches) != set(SINGLE_RECIPES) or any(set(folds) != set(range(5)) for folds in caches.values()):
        raise EvidenceError('requires all nine fixed single families and all five folds')
    started = time.perf_counter()
    selected_predictions, fast_predictions = {}, {}
    fixed_predictions = {c.name: {} for c in candidates}
    fixed_folds = {c.name: [] for c in candidates}
    outer_oof = {r: Probabilities({}, {}, {} if 'boundary' in r else None) for r in SINGLE_RECIPES}
    folds = []
    for fold, val in enumerate(manifest['outer_folds']):
        train, inner = inner_splits(records, val, fold)
        candidate_rows = []
        selection_started = time.perf_counter()
        for candidate in candidates:
            p = blend_probabilities(candidate, {r: caches[r][fold].inner for r, _ in candidate.members}, train)
            saved = caches[candidate.members[0][0]][fold].inner_selection if len(candidate.members) == 1 else None
            if saved is not None:
                config, result = deepcopy(saved['config']), deepcopy(saved['metrics'])
            else:
                config, result = chooser(records, train, p.frame, p.video, p.boundary)
            candidate_rows.append({'candidate': candidate.name, 'decoder': config, 'metrics': result,
                                   'objective': objective(result), 'selection_source': 'inner_3fold_OOF_only',
                                   'cached_single_search_reused': saved is not None})
        # Stable declared order breaks exact objective ties; never use outer scores.
        selected = max(candidate_rows, key=lambda r: r['objective'])
        fast = max((r for r in candidate_rows if r['candidate'] in {c.name for c in FAST_CANDIDATES}), key=lambda r: r['objective'])
        selection_seconds = time.perf_counter() - selection_started
        for candidate, row in zip(candidates, candidate_rows):
            p = blend_probabilities(candidate, {r: caches[r][fold].outer for r, _ in candidate.members}, val)
            predictions, result = evaluate_probabilities(records, val, p.frame, p.video, row['decoder'], p.boundary)
            fixed_predictions[candidate.name].update(predictions)
            fixed_folds[candidate.name].append({'fold': fold, 'selection': row, 'validation': result})
            if candidate.name == selected['candidate']:
                selected_predictions.update(predictions)
            if candidate.name == fast['candidate']:
                fast_predictions.update(predictions)
        for recipe, pack in outer_oof.items():
            p = caches[recipe][fold].outer
            pack.frame.update(p.frame)
            pack.video.update(p.video)
            if pack.boundary is not None:
                pack.boundary.update(p.boundary)
        folds.append({'fold': fold, 'train_indices': train, 'validation_indices': val,
                      'inner_folds': inner, 'selected_candidate': selected['candidate'],
                      'decoder': selected['decoder'], 'fast_selected_candidate': fast['candidate'],
                      'fast_decoder': fast['decoder'], 'selection_seconds': selection_seconds,
                      'inner_candidate_selections': candidate_rows})
        print(f'outer {fold}: inner-selected {selected["candidate"]}; fast {fast["candidate"]}', flush=True)
    report = {
        'schema_version': 'algorithm-opt-nested-selection-v1',
        'protocol': manifest['protocol'], 'role': manifest.get('role'),
        'candidate_policy': {'fixed_before_outer_evaluation': True, 'candidates': [c.as_dict() for c in candidates],
                             'fast_candidates': [c.name for c in FAST_CANDIDATES],
                             'tie_break': 'first declared candidate at equal inner objective',
                             'exclusions': EXCLUSIONS, 'outer_score_pruning': False},
        'primary_nested_selection': {'selection_source': 'inner_3fold_OOF_only_for_family_and_decoder',
            'metrics': _group_metrics(records, selected_predictions),
            'predictions': predictions_json(records, selected_predictions),
            **prediction_summary(records, selected_predictions)},
        'fast_nested_selection': {'selection_source': 'inner_3fold_OOF_motion_RF_ET_only',
            'metrics': _group_metrics(records, fast_predictions),
            'predictions': predictions_json(records, fast_predictions),
            **prediction_summary(records, fast_predictions)},
        'fixed_candidate_oof': {c.name: {'members': c.as_dict()['members'],
            'metrics': _group_metrics(records, fixed_predictions[c.name]),
            'predictions': predictions_json(records, fixed_predictions[c.name]),
            'folds': fixed_folds[c.name], **prediction_summary(records, fixed_predictions[c.name])}
            for c in candidates},
        'folds': folds, 'meta_selection_seconds': time.perf_counter() - started,
        'evaluation_note': 'Only primary_nested_selection is the held-out nested family-selection estimate. '
            'All fixed candidate results are retained, including failures/no gains; do not choose candidates '
            'from their outer scores. Inspected development CV is not a fresh blind or external test.',
    }
    return report, outer_oof


def deployment_selection(records, outer_oof, candidates=CANDIDATES, chooser=None):
    chooser = chooser or choose_decoder
    indices = list(range(len(records)))
    rows = []
    started = time.perf_counter()
    for c in candidates:
        p = blend_probabilities(c, outer_oof, indices)
        config, result = chooser(records, indices, p.frame, p.video, p.boundary)
        rows.append({'candidate': c.name, 'decoder': config, 'optimistic_metrics': result,
                     'objective': objective(result)})
    best = max(rows, key=lambda r: r['objective'])
    fast = max((r for r in rows if r['candidate'] in {c.name for c in FAST_CANDIDATES}), key=lambda r: r['objective'])
    return {'role': 'optimistic_all_development_OOF_selection_not_validation',
            'is_validation_score': False, 'video_count': len(records),
            'note': 'Uses all development OOF labels to choose final family/config. These scores '
                    'are optimistic and must not be reported as validation or generalization.',
            'selected': best, 'fast_selected': fast, 'candidates': rows,
            'selection_seconds': time.perf_counter() - started}


def _hard_close(actual, expected, label, atol=PARITY_ATOL):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ParityError(label + ': shape/nonfinite mismatch')
    error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    if error >= atol:
        raise ParityError(f'{label}: max error {error} >= {atol}')
    return error


def export_fitted_member(recipe, records, fp, vp, state, predictor=None):
    from optimized_detector import predict_record
    predictor = predictor or predict_record
    if not isinstance(state, tuple) or len(state) != 6:
        raise ParityError('fit_predict state must be current six-tuple')
    fm, vm, transform, x_values, v_values, extra = state
    if not isinstance(extra, dict) or not {'boundary_models', 'boundary_probabilities'} <= set(extra):
        raise ParityError('fit state missing boundary extra')
    if recipe.endswith('_tcn'):
        if not isinstance(fm, dict) or fm.get('schema') != 'optimized-temporal-head-v1':
            raise ParityError('invalid fitted temporal dict')
        frame_model = deepcopy(fm)
        # Preserve temporal module schema and add detector dispatch discriminator.
        frame_model['schema_version'] = 'optimized-temporal-head-v1'
    else:
        frame_model = export_model(fm)
    member = {'weight': 1.0, 'recipe': recipe, 'transform': transform,
              'frame_model': frame_model, 'video_model': export_model(vm),
              'boundary_models': extra['boundary_models']}
    member = json.loads(json.dumps(member, allow_nan=False))
    empty_decoder = {'threshold': .5, 'low_ratio': 1., 'smooth_seconds': 0.,
                     'gap_seconds': 0., 'min_seconds': 0., 'video_threshold': 0., 'boundary_seconds': 0.}
    probe = {'schema_version': SCHEMA, 'recipe': recipe, 'members': [member], 'decoder': empty_decoder}
    errors = {'frame': 0., 'video': 0., 'boundary': 0., 'frame_view': 0., 'video_view': 0.}
    indices = list(range(len(records)))
    if set(fp) != set(indices) or set(vp) != set(indices) or len(x_values) != len(records) or len(v_values) != len(records):
        raise ParityError('final fit must predict all training videos for hard parity')
    bp = extra['boundary_probabilities']
    if ('boundary' in recipe) != (member['boundary_models'] is not None) or (member['boundary_models'] is not None) != (bp is not None):
        raise ParityError('boundary models/probabilities mismatch')
    for i, record in enumerate(records):
        x, v, names = feature_view(recipe, record, transform.get('pca'))
        if names != transform['frame_feature_names']:
            raise ParityError('feature name order mismatch')
        errors['frame_view'] = max(errors['frame_view'], _hard_close(x, x_values[i], 'frame view'))
        errors['video_view'] = max(errors['video_view'], _hard_close(v, v_values[i], 'video view'))
        out = predictor(probe, record)
        errors['frame'] = max(errors['frame'], _hard_close(out['frame_probabilities'], fp[i], 'portable frame'))
        errors['video'] = max(errors['video'], _hard_close(out['video_probability'], vp[i], 'portable video'))
        if not recipe.endswith('_tcn'):
            errors['frame'] = max(errors['frame'], _hard_close(out['frame_probabilities'], fm.predict_proba(x_values[i])[:, 1], 'sklearn frame'))
        errors['video'] = max(errors['video'], _hard_close(out['video_probability'], vm.predict_proba(v_values[i:i + 1])[:, 1][0], 'sklearn video'))
        if bp is not None:
            # predict_record hides curves, so check portable heads directly and
            # check their real decoder/refinement path through predict_record later.
            from optimized_boundary_head import predict_boundary_heads
            boundary = predict_boundary_heads(member['boundary_models'], x)
            for j in (0, 1):
                errors['boundary'] = max(errors['boundary'], _hard_close(boundary[j], bp[i][j], 'boundary head'))
    return member, Probabilities(fp, vp, bp), errors


def build_bundle(candidate, decoder, fitted_members, records, manifest, profiles, provenance, selection):
    members = [dict(deepcopy(fitted_members[r]), weight=w) for r, w in candidate.members]
    bundle = {
        'schema_version': SCHEMA, 'recipe': candidate.members[0][0] if len(members) == 1 else 'ensemble',
        'members': members, 'decoder': decoder,
        'raw_feature_names': {'jepa': records[0]['jepa_names'], 'motion': records[0]['motion_names'],
                              'corrected': records[0]['corrected_names']},
        'feature_profiles': deepcopy(profiles),
        'training_video_sha256': [r['sha256'] for r in records],
        'training_video_names': [r['name'] for r in records],
        'training_role': f'all {len(records)} inspected development videos; not blind; full-data fit',
        'deployment_selection': {'role': selection['role'], 'is_validation_score': False,
                                'candidate': candidate.name, 'note': selection['note']},
        'model_note': 'Offline non-causal locator; uncalibrated evidence probabilities. '
                      'Only selected members fitted on CPU; no full215 training or GPU extraction.',
        'evaluation_note': 'Validation is report.json primary_nested_selection, not optimistic '
                           'full-development deployment selection or final-fit parity probes. '
                           'Live cold/hot/batch GPU extraction parity is a separate owner check.',
        'dataset_manifest_sha256': provenance['dataset_manifest_sha256'],
        'dataset_npz_sha256': provenance['dataset_npz_sha256'],
        'report_sources': provenance['report_sources'],
        'source_snapshot_sha256': provenance['source_snapshot_sha256'],
        'protocol': manifest['protocol'],
        'trained_at': datetime.now().astimezone().isoformat(timespec='seconds'),
    }
    if len(members) == 1:
        for key in ('transform', 'frame_model', 'video_model', 'boundary_models'):
            bundle[key] = members[0][key]
    return json.loads(json.dumps(bundle, allow_nan=False))


def verify_bundle(bundle, candidate, records, fitted_probabilities, predictor=None):
    from optimized_detector import predict_record
    predictor = predictor or predict_record
    indices = list(range(len(records)))
    p = blend_probabilities(candidate, fitted_probabilities, indices)
    expected, _ = evaluate_probabilities(records, indices, p.frame, p.video, bundle['decoder'], p.boundary)
    errors = {'frame': 0., 'video': 0.}
    for i, record in enumerate(records):
        out = predictor(bundle, record)
        errors['frame'] = max(errors['frame'], _hard_close(out['frame_probabilities'], p.frame[i], 'bundle frame'))
        errors['video'] = max(errors['video'], _hard_close(out['video_probability'], p.video[i], 'bundle video'))
        if [tuple(v) for v in out['intervals']] != [tuple(v) for v in expected[i]]:
            raise ParityError('predict_record decoded/refined interval mismatch: ' + record['name'])
    return {'passed': True, 'videos': len(records), 'atol_strict': PARITY_ATOL,
            'max_absolute_errors': errors, 'intervals_exact': True,
            'note': 'Fitted-state NumPy/sklearn probabilities and decoder parity, not validation.'}


def snapshot_sources(output):
    names = ('run_optimization_selection.py', 'test_run_optimization_selection.py',
             'run_algorithm_optimization.py', 'build_optimization_dataset.py',
             'optimized_detector.py', 'optimized_locator.py', 'optimized_feature_view.py',
             'optimized_temporal_head.py', 'optimized_boundary_head.py')
    hashes = {}
    for name in names:
        source = Path(__file__).with_name(name)
        digest = sha256(source)
        target = Path(output) / 'sources' / digest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        if sha256(target) != digest:
            raise EvidenceError('source snapshot hash mismatch')
        hashes[str(target.resolve())] = digest
    return hashes


def run(data_root, deployment_fit=False):
    data_root = Path(data_root).resolve()
    output = data_root / 'deployment_selection'
    run_id = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f')
    history = output / 'runs' / (run_id + '_report.json')
    report = {'schema_version': 'algorithm-opt-nested-selection-v1', 'status': 'starting', 'run_id': run_id}
    started = time.perf_counter()
    try:
        manifest, records, profiles, input_hashes = load_dataset(data_root)
        caches, report_sources, experiments = load_fold_caches(data_root, manifest, records, input_hashes)
        source_hashes = snapshot_sources(output)
        provenance = {'dataset_manifest_sha256': sha256(data_root / 'dataset_manifest.json'),
                      'dataset_npz_sha256': sha256(data_root / 'dataset.npz'),
                      'report_sources': report_sources, 'source_snapshot_sha256': source_hashes}
        report, outer_oof = nested_selection(manifest, records, caches)
        report.update(provenance, run_id=run_id, status='nested_complete',
                      experiment_reports=experiments, input_source_sha256=input_hashes)
        json_write(history, report)
        json_write(output / 'report.json', report)
        selection = deployment_selection(records, outer_oof)
        report['deployment_selection'] = selection
        report['status'] = 'selection_complete'
        json_write(history, report)
        json_write(output / 'report.json', report)
        if deployment_fit:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            chosen = {c.name: c for c in CANDIDATES}
            main = chosen[selection['selected']['candidate']]
            fast = chosen[selection['fast_selected']['candidate']]
            needed = {r for c in (main, fast) for r, _ in c.members}
            fitted_members, fitted_probabilities, member_parity = {}, {}, {}
            indices = list(range(len(records)))
            report['deployment_fit'] = {'role': 'full_development_fit_not_validation', 'members': []}
            for recipe in SINGLE_RECIPES:
                if recipe not in needed:
                    continue
                print('CPU deployment fit:', recipe, flush=True)
                fit_started = time.perf_counter()
                fp, vp, state = fit_predict(recipe, records, indices, indices, SEED + 8800)
                member, pack, errors = export_fitted_member(recipe, records, fp, vp, state)
                fitted_members[recipe], fitted_probabilities[recipe] = member, pack
                member_parity[recipe] = errors
                report['deployment_fit']['members'].append({'recipe': recipe,
                    'seconds': time.perf_counter() - fit_started, 'parity_max_absolute_errors': errors})
                json_write(history, report)
                json_write(output / 'report.json', report)
            bundles = []
            for filename, candidate, choice in [('locator_bundle.json', main, selection['selected']),
                                                ('fast_locator_bundle.json', fast, selection['fast_selected'])]:
                bundle = build_bundle(candidate, choice['decoder'], fitted_members, records,
                                      manifest, profiles, provenance, selection)
                bundle['parity'] = verify_bundle(bundle, candidate, records, fitted_probabilities)
                bundle['parity']['member_max_absolute_errors'] = {r: member_parity[r] for r, _ in candidate.members}
                bundles.append((output / filename, bundle))
            # BOTH bundles must pass before writing either final model artifact.
            for path, bundle in bundles:
                json_write(path, bundle)
            report['deployment_fit']['artifacts'] = {str(p): sha256(p) for p, _ in bundles}
            report['status'] = 'complete'
        report['elapsed_seconds'] = time.perf_counter() - started
        json_write(history, report)
        json_write(output / 'report.json', report)
        return report
    except Exception as exc:
        report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        # Keep failed attempts and all candidate results; never delete old bundles
        # or replace a previous complete report when input validation fails.
        json_write(history, report)
        json_write(output / ('failure_' + run_id + '.json'), report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--reports-complete', action='store_true',
                        help='explicit confirmation all four reports and 45 valid folds are complete')
    parser.add_argument('--deployment-fit', action='store_true',
                        help='fit only selected members on all 77 development videos, CPU only')
    args = parser.parse_args(argv)
    if not args.reports_complete:
        parser.error('no data read: wait for completion notification, then pass --reports-complete')
    report = run(args.data_root, args.deployment_fit)
    print('status:', report['status'], flush=True)
    print('nested metrics:', json.dumps(report['primary_nested_selection']['metrics']['all']), flush=True)
    print('OPTIMISTIC deployment choice:', report['deployment_selection']['selected']['candidate'], flush=True)
    print('OPTIMISTIC fast choice:', report['deployment_selection']['fast_selected']['candidate'], flush=True)


if __name__ == '__main__':
    main()
