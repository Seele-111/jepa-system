#!/usr/bin/env python3
"""Read-only v6 asset audit, optionally precision-only v6r replay.

Local: python -B code/audit_video_gate_v6.py
WSL: /path/to/python -B /mnt/e/jepa-system/code/audit_video_gate_v6.py --repair
Only NEW independent_gate_audit.json files are written, never overwritten.
No fit, selector, performance metrics, original report parsing, or grid search.
NumPy is required; --repair requires sklearn for prediction, NOT fitting.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
import platform
import sys
import numpy as np

MEMBERS = ('event_rgb_corrected_motion_et', 'event_rgb_corrected_motion_hgb')
GATES = ('ridge_logistic_v6', 'shallow_forest_v6')
REVISION = 'float64-standardization-f32-v6r'
TOLERANCE = 2e-6


class AuditError(ValueError):
    pass


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.watched = {}
        self.checks = 0

    def require(self, condition, message):
        self.checks += 1
        if not condition:
            raise AuditError(message)

    def path(self, relative):
        path = (self.root / relative).resolve()
        self.require(path.is_relative_to(self.root), 'asset path escapes root: ' + str(relative))
        return path

    def digest(self, path):
        path = Path(path).resolve()
        digest = file_hash(path)
        previous = self.watched.setdefault(path, digest)
        self.require(previous == digest, 'asset changed during audit: ' + str(path))
        return digest

    def json(self, path):
        self.digest(path)
        return json.loads(Path(path).read_text('utf-8'))

    def signed(self, path, signature):
        value = self.json(path)
        self.require(object_hash({k: v for k, v in value.items() if k != signature})
                     == value[signature], 'invalid ' + signature + ': ' + str(path))
        return value

    def hash_map(self, entries):
        for relative, expected in entries.items():
            self.require(self.digest(self.path(relative)) == expected,
                         'receipt asset hash mismatch: ' + relative)

    def unchanged(self):
        for path, expected in self.watched.items():
            self.require(file_hash(path) == expected, 'asset changed after reading: ' + str(path))


def content(records, ids):
    return sorted({records[i]['sha256'] for i in ids})


def check_split(e, records, parent, fit, validation, name):
    e.require(all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(records)
                  for i in fit + validation), name + ': invalid row index')
    e.require(len(fit) == len(set(fit)) and len(validation) == len(set(validation)),
              name + ': repeated row index')
    e.require(not set(fit) & set(validation) and sorted(fit + validation) == sorted(parent),
              name + ': complement/coverage mismatch')
    e.require(not set(content(records, fit)) & set(content(records, validation)),
              name + ': content leakage')


def grouped_folds(records, ids, count, seed):
    """Independent reconstruction of SHA-grouped, stratified partitioning."""
    groups = {}
    for i in ids:
        groups.setdefault(records[i]['sha256'], []).append(i)
    grouped = list(groups.values())
    strata = {}
    for j, group in enumerate(grouped):
        row = records[group[0]]
        event = 'normal' if row['event_count'] == 0 else 'multi' if row['event_count'] > 1 else 'single'
        strata.setdefault((row['generator'], event), []).append(j)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    offset = 0
    for key in sorted(strata):
        ordered = rng.permutation(strata[key])
        for j, group in enumerate(ordered):
            folds[(offset + j) % count].extend(grouped[int(group)])
        offset = (offset + len(ordered)) % count
    return [sorted(fold) for fold in folds]


def balanced_weights(records, ids):
    counts = {}
    for i in ids:
        counts[records[i]['sha256']] = counts.get(records[i]['sha256'], 0) + 1
    weights = np.asarray([1 / counts[records[i]['sha256']] for i in ids], np.float64)
    labels = np.asarray([bool(np.any(records[i]['labels'])) for i in ids])
    if not labels.any() or labels.all():
        raise AuditError('gate training requires both video classes')
    weights[labels] *= weights[~labels].sum() / weights[labels].sum()
    return weights * len(weights) / weights.sum()


def rolling_reference(values, width):
    values = np.asarray(values, np.float32)
    width = max(1, min(width, len(values)))
    if width == 1:
        return values.copy()
    left = width // 2
    padded = np.pad(values, (left, width - 1 - left), mode='edge')
    return np.convolve(padded, np.ones(width, np.float64) / width, mode='valid').astype(np.float32)


def features_reference(frame, video, record):
    """Independent implementation of the frozen 41-dimensional feature contract."""
    p = np.asarray(frame, np.float32)
    fps = float(record['fps'])
    q = np.percentile(p, [10, 25, 50, 75, 90, 99])
    values = [float(video), float(p.mean()), float(p.std()), *q.tolist(), float(p.max()),
              float(np.sort(p)[-max(1, math.ceil(.1 * len(p))):].mean()), float(q[-1] - q[0])]
    names = ['raw_video_probability', 'frame_mean', 'frame_std']
    names += [f'frame_p{k}' for k in [10, 25, 50, 75, 90, 99]]
    names += ['frame_max', 'frame_top10_mean', 'frame_contrast_p99_p10']
    for seconds in [.2, .6]:
        z = rolling_reference(p, max(1, round(seconds * fps)))
        values.extend([float(z.max()), float(np.percentile(z, 90))])
        names.extend([f'frame_mean_{seconds}s_max', f'frame_mean_{seconds}s_p90'])
    for threshold in [.3, .5, .7]:
        mask = p >= threshold
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        longest = int((ends - starts + 1).max()) if len(starts) else 0
        values.extend([float(mask.mean()), longest / fps, float(len(starts))])
        names.extend([f'frame_mass_above_{threshold}', f'frame_longest_seconds_above_{threshold}',
                      f'frame_components_above_{threshold}'])
    corrected = np.asarray(record['corrected'], np.float32)
    for j in [0, 2, 3, 4, 7, 9, 10, 11]:
        values.extend([float(corrected[:, j].mean()), float(np.percentile(corrected[:, j], 90))])
        names.extend([f'corrected_{j}_mean', f'corrected_{j}_p90'])
    return np.asarray(values, np.float32), names


def standardized(state, x, training_path=True):
    dtype = np.float64 if training_path else np.float32
    mean, scale = np.asarray(state['mean'], dtype), np.asarray(state['scale'], dtype)
    x = np.asarray(x, np.float32)
    if mean.shape != x.shape or scale.shape != x.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise AuditError('invalid logistic standardizer')
    return np.clip((x - mean) / scale, -8, 8).astype(np.float32)


def logistic_reference(state, x, training_path=True):
    """Float64 fsum/sigmoid, with the fitter's explicit float32 z cast."""
    z = standardized(state, x, training_path)
    coef = np.asarray(state['coefficients'], np.float64)
    if coef.shape != z.shape or not np.isfinite(coef).all() or not math.isfinite(state['intercept']):
        raise AuditError('invalid logistic parameters')
    logit = math.fsum(float(a) * float(b) for a, b in zip(z, coef)) + float(state['intercept'])
    logit = max(-700., min(700., logit))
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1 + exp)


def forest_reference(state, x):
    """Independent scalar traversal and float64 aggregation of saved ET nodes."""
    model = state['model']
    x = np.asarray(x, np.float32)
    if model['n_features'] != len(x):
        raise AuditError('forest feature count mismatch')
    if model['kind'] == 'constant':
        return float(np.float32(model['probability']))
    if model['kind'] != 'forest' or not model['trees']:
        raise AuditError('unsupported gate forest')
    probabilities = []
    for tree in model['trees']:
        node = 0
        for _ in range(len(tree['left'])):
            if tree['left'][node] < 0:
                probabilities.append(float(tree['value'][node]))
                break
            branch = 'left' if float(x[tree['feature'][node]]) <= tree['threshold'][node] else 'right'
            node = tree[branch][node]
        else:
            raise AuditError('invalid forest topology')
    return float(np.float32(math.fsum(probabilities) / len(probabilities)))


def sklearn_model(state):
    """Reconstruct a fitted logistic predictor from saved coefficients; no fit."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=.3, solver='lbfgs', max_iter=1000)
    model.classes_ = np.asarray([0, 1])
    model.coef_ = np.asarray([state['coefficients']], np.float64)
    model.intercept_ = np.asarray([state['intercept']], np.float64)
    model.n_features_in_ = len(state['feature_names'])
    model.n_iter_ = np.asarray([0], np.int32)
    return model


def extract_functions(path, namespace, names, assignments=()):
    """Load ONLY hash-verified frozen inference functions, never imports/fit/CLI."""
    tree = ast.parse(Path(path).read_text('utf-8'), filename=str(path))
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    nodes = [future]
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets):
            nodes.append(node)
    if found != set(names):
        raise AuditError('missing frozen inference function: ' + str(path))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


def frozen_api(folder, repair_folder=None):
    ns = {'np': np, 'math': math, 'Real': Real, 'Mapping': Mapping}
    extract_functions(folder / 'optimized_locator.py', ns,
                      ['rolling_mean', 'spans', '_tree_values', 'portable_predict'])
    extract_functions(folder / 'optimized_video_gate_v6.py', ns, ['gate_features', 'apply_gate'])
    old_apply = ns['apply_gate']
    extract_functions(folder / 'optimized_recall_decoder.py', ns, ['effective_score', 'decode_v2'])
    duration = ast.parse((folder / 'optimized_duration_decoder_v4.py').read_text('utf-8'))
    extract_functions(folder / 'optimized_duration_decoder_v4.py', ns,
                      [n.name for n in duration.body if isinstance(n, ast.FunctionDef)],
                      ['_KIND', '_REQUIRED_FIELDS', '_ALLOWED_FIELDS', '_PROBABILITY_EPSILON', '_DURATION_EPSILON'])
    api = {'features': ns['gate_features'], 'apply': old_apply,
           'duration': ns['decode_duration'], 'recall': ns['decode_v2']}
    if repair_folder is not None:
        repair = dict(ns, original_apply_gate=old_apply)
        extract_functions(repair_folder / 'optimized_video_gate_v6r.py', repair,
                          ['apply_gate'], ['NUMERIC_REVISION'])
        api['repair'] = repair['apply_gate']
    return api


def verify_receipt(e, folder):
    receipt = e.signed(folder / 'training_receipt.json', 'receipt_sha256')
    e.require(e.digest(folder / 'protocol.json') == receipt['protocol_sha256'], 'protocol hash mismatch')
    for name, expected in receipt['sources'].items():
        e.require(e.digest(folder / 'sources' / name) == expected, 'frozen source hash mismatch: ' + name)
    e.require(e.digest(e.root / 'output/algorithm-opt-2026-10-02-v2/selection_audited/report.json')
              == receipt['baseline_report_sha256'], 'baseline byte hash mismatch')
    return receipt


def load_records(e):
    old = e.root / 'output/algorithm-opt-2026-10-02'
    manifest = e.json(e.root / 'output/algorithm-opt-2026-10-02-v2/dataset_manifest.json')
    original = e.json(old / 'dataset_manifest.json')
    e.require(manifest['rows'] == original['rows'], 'v1/v2 row identity mismatch')
    rows = manifest['rows']
    e.require(len(rows) == 77 and len(content(rows, range(len(rows)))) == 76, 'expected 77 rows/76 contents')
    corrected_root = old / 'corrected_jepa_features_v2'
    cm = e.json(corrected_root / 'manifest.json')
    e.require(cm['status'] == 'complete' and cm['dataset_manifest_sha256'] == e.digest(old / 'dataset_manifest.json'),
              'corrected feature cache dataset mismatch')
    entries = {entry['name']: entry for entry in cm['videos']}
    records = []
    e.digest(old / 'dataset.npz')
    with np.load(old / 'dataset.npz', allow_pickle=False) as labels:
        for i, row in enumerate(rows):
            entry = entries[row['name']]
            e.require(entry['source_sha256'] == row['sha256'], 'corrected source content mismatch')
            path = corrected_root / entry['file']
            e.require(e.digest(path) == entry['feature_sha256'], 'corrected NPZ hash mismatch')
            with np.load(path, allow_pickle=False) as data:
                x = np.asarray(data['signals'], np.float32)
                e.require(x.shape == (row['frames'], 14) and np.isfinite(x).all()
                          and float(data['fps']) == row['fps'], 'invalid corrected feature array')
            y = labels[f'labels_{i}'].copy()
            e.require(y.shape == (row['frames'],) and np.isin(y, [0, 1]).all(), 'invalid frame labels')
            records.append(dict(row, corrected=x, labels=y))
    covered = []
    for fold, val in enumerate(manifest['outer_folds']):
        train = [i for i in range(len(rows)) if i not in set(val)]
        check_split(e, records, list(range(len(rows))), train, val, f'outer{fold}')
        covered.extend(val)
    e.require(len(manifest['outer_folds']) == 5 and sorted(covered) == list(range(len(rows))), 'outer fold coverage')
    return manifest, records


def load_asset(e, folder, stem, receipt, expected, records):
    path = folder / 'training' / (stem + '.json')
    row = e.signed(path, 'signature')
    e.require(row['training_receipt_sha256'] == receipt['receipt_sha256'], 'training receipt link mismatch')
    npz = path.with_suffix('.npz')
    data = npz.read_bytes()
    e.require(e.digest(npz) == row['npz_sha256'] == hashlib.sha256(data).hexdigest(), 'training NPZ hash mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        e.require(set(archive.files) == expected, 'NPZ key coverage mismatch: ' + stem)
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, value in arrays.items():
        i = int(key.rsplit('_', 1)[1])
        shape = (records[i]['frames'],) if '_frame_' in key else ()
        e.require(value.dtype.kind in 'iuf' and value.shape == shape and np.isfinite(value).all()
                  and np.all((value >= 0) & (value <= 1)), 'invalid probability array: ' + key)
    return row, arrays


def expected_gate_keys(train, val, parts):
    keys = {f'{side}_frame_{i}' for side, ids in [('inner', train), ('outer', val)] for i in ids}
    keys.update(f'{side}_video_{gate}_{i}' for side, ids in [('inner', train), ('outer', val)]
                for i in ids for gate in ('ungated_control',) + GATES)
    keys.update(f'deep{k}_{head}_{i}' for k, part in enumerate(parts) for i in part['fit'] for head in ['frame', 'video'])
    return keys


def verify_partitions(e, records, fold, train, val, row):
    e.require(row['fold'] == fold and row['train'] == train and row['validation'] == val, 'v6 outer partition mismatch')
    expected_inner = grouped_folds(records, train, 3, 20261002 + fold * 31)
    e.require(len(row['inner_partitions']) == 3, 'inner partition count')
    for k, iv in enumerate(expected_inner):
        it = [i for i in train if i not in set(iv)]
        part = row['inner_partitions'][k]
        e.require(part['fit'] == it and part['validation'] == iv, 'v6 inner partition mismatch')
        check_split(e, records, train, it, iv, 'inner partition')
        ds = grouped_folds(records, it, 3, 20261006 + fold * 223 + k * 31)
        e.require(len(part['deep_partitions']) == 3, 'deep partition count')
        for d, dv in enumerate(ds):
            dt = [i for i in it if i not in set(dv)]
            dp = part['deep_partitions'][d]
            e.require(dp['fit'] == dt and dp['validation'] == dv, 'deep partition/seed mismatch')
            check_split(e, records, it, dt, dv, 'deep partition')
            e.require(not set(content(records, dt)) & set(content(records, dv + iv + val)), 'cross-level content leakage')
            e.require(len(dp['member_fit_proof']) == 2, 'deep fixed member proof count')
            for j, proof in enumerate(dp['member_fit_proof']):
                e.require(proof['recipe'] == MEMBERS[j] and proof['seed'] == 20261006 + fold * 223 + k * 31 + d + j * 1000
                          and proof['fit_content_sha256'] == content(records, dt), 'deep member fit proof mismatch')
                e.require(len(proof['transform_sha256']) == 64 and all(c in '0123456789abcdef' for c in proof['transform_sha256']), 'invalid transform hash')
        e.require(set(part['gate_states']) == set(GATES), 'inner gate state coverage')
        for kind, state in part['gate_states'].items():
            e.require(state['kind'] == kind and state['fit_content_sha256'] == content(records, it), 'inner gate fit-content mismatch')
            e.require(not set(state['fit_content_sha256']) & set(content(records, iv + val)), 'inner gate heldout overlap')
    e.require(row['outer_gate_seed'] == 20261006 + fold * 223 + 6000 and set(row['outer_gate_states']) == set(GATES), 'outer gate seed/state mismatch')
    for kind, state in row['outer_gate_states'].items():
        e.require(state['kind'] == kind and state['fit_content_sha256'] == content(records, train), 'outer gate fit-content != outer train')
        e.require(not set(state['fit_content_sha256']) & set(content(records, val)), 'outer gate contains outer validation')


def verify_v5(e, folder, receipt, fold, recipe, parent, records):
    train, val = parent['train'], parent['validation']
    keys = {f'{side}_{head}_{i}' for side, ids in [('inner', train), ('outer', val)] for i in ids for head in ['frame', 'video']}
    row, arrays = load_asset(e, folder, f'{fold}_{recipe}', receipt, keys, records)
    e.require(row['fold'] == fold and row['recipe'] == recipe and row['train'] == train and row['validation'] == val, 'v5/v6 outer row partition mismatch')
    e.require(len(row['inner_partitions']) == 3 and len(row['fit_evidence']) == 4, 'v5 proof coverage')
    for k, part in enumerate(row['inner_partitions']):
        original = parent['inner_partitions'][k]
        for key in ['fit', 'validation']:
            e.require(part[key] == original[key] and content(records, part[key]) == content(records, original[key]), 'v5/v6 inner content partition mismatch')
        e.require(part['seed'] == 20261002 + fold * 53 + k and row['fit_evidence'][k]['partition'] == k
                  and row['fit_evidence'][k]['fit_content_sha256'] == content(records, part['fit']), 'v5 inner fit provenance mismatch')
    e.require(row['outer_seed'] == 20261002 + fold * 53 + 99 and row['fit_evidence'][3]['partition'] == 'outer'
              and row['fit_evidence'][3]['fit_content_sha256'] == content(records, train), 'v5 outer fit provenance mismatch')
    return arrays


def difference_summary(rows):
    if not rows:
        return {'count': 0, 'max_abs_difference': 0., 'above_2e_6': 0}
    worst = max(rows, key=lambda r: r['difference'])
    return {'count': len(rows), 'max_abs_difference': worst['difference'],
            'above_2e_6': sum(r['difference'] > TOLERANCE for r in rows), 'worst': worst}


def interval_impact(api, records, configs, samples):
    """Only frozen gate thresholds/configs; no labels, scoring or selection."""
    thresholds = sorted({float(c['video_threshold']) for c in configs})
    flips = {str(t): 0 for t in thresholds}
    examples, changed_rows = [], set()
    changes = comparisons = 0
    for row in samples:
        for t in thresholds:
            flips[str(t)] += int((row['old'] >= t) != (row['reference'] >= t))
        for ordinal, cfg in enumerate(configs):
            comparisons += 1
            t = cfg['video_threshold']
            if (row['old'] >= t) == (row['reference'] >= t):
                continue
            decoder = api['duration'] if cfg['kind'] == 'duration-logit-v4' else api['recall']
            fps = records[row['row']]['fps']
            before = decoder(row['frame'], fps, row['old'], cfg)
            after = decoder(row['frame'], fps, row['reference'], cfg)
            if before != after:
                changes += 1
                changed_rows.add((row['fold'], row['role'], row.get('inner'), row['row']))
                if len(examples) < 10:
                    examples.append({k: v for k, v in row.items() if k != 'frame'} | {'config_ordinal': ordinal, 'before': before, 'after': after})
    return {'threshold_flips': flips, 'frozen_config_count': len(configs), 'row_config_comparisons': comparisons,
            'changed_predictions': changes, 'changed_context_rows': len(changed_rows), 'examples': examples,
            'scope': 'inner_validation and outer_validation only; all frozen configs, no selector/metrics'}


def evaluate_fold(e, records, fold, meta, arrays, api, configs, repaired=None):
    details, features_delta, saved_delta, forest_delta = [], 0., 0., 0.
    scaler_delta, repair_delta, sklearn_delta, repair_saved_delta = 0., 0., 0., 0.
    inference, repair_inference = [], []
    contexts = []
    for k, part in enumerate(meta['inner_partitions']):
        contexts.append((k, part['gate_states'], [
            ('inner_train_deep', part['fit'], f'deep{k}', None),
            ('inner_validation', part['validation'], 'inner', 'inner')]))
    contexts.append((None, meta['outer_gate_states'], [
        ('outer_train_inner_oof', meta['train'], 'inner', None),
        ('outer_validation', meta['validation'], 'outer', 'outer')]))
    for inner, states, roles in contexts:
        training_features = []
        for role, ids, side, saved_side in roles:
            training = saved_side is None
            model = sklearn_model(states[GATES[0]]) if repaired is not None else None
            repair_states = (repaired[0]['outer_gate_states'] if inner is None else repaired[0]['inner_partitions'][inner]['gate_states']) if repaired is not None else None
            for i in ids:
                frame = arrays[f'{side}_frame_{i}']
                video = float(arrays[f'{side}_video_{i}'] if side.startswith('deep') else arrays[f'{side}_video_ungated_control_{i}'])
                x, names = features_reference(frame, video, records[i])
                actual_x, actual_names = api['features'](frame, video, records[i])
                delta = float(np.max(np.abs(x.astype(np.float64) - actual_x)))
                features_delta = max(features_delta, delta)
                e.require(names == actual_names == states[GATES[0]]['feature_names'] == states[GATES[1]]['feature_names']
                          and delta == 0, 'independent gate feature mismatch')
                if training:
                    training_features.append(x)
                state = states[GATES[0]]
                old = api['apply'](state, frame, video, records[i])
                ref = logistic_reference(state, x)
                e.require(abs(old - logistic_reference(state, x, False)) <= 1e-12, 'old portable/reference mismatch')
                summary = {'fold': fold, 'inner': inner, 'role': role, 'row': i, 'old': old,
                           'reference': ref, 'difference': abs(old - ref)}
                details.append(summary)
                old_forest = api['apply'](states[GATES[1]], frame, video, records[i])
                forest_delta = max(forest_delta, abs(old_forest - forest_reference(states[GATES[1]], x)))
                if saved_side:
                    saved_delta = max(saved_delta, abs(old - float(arrays[f'{saved_side}_video_{GATES[0]}_{i}'])),
                                      abs(old_forest - float(arrays[f'{saved_side}_video_{GATES[1]}_{i}'])))
                    inference.append(dict(summary, frame=frame))
                if repaired is not None:
                    rstate = repair_states[GATES[0]]
                    probability = api['repair'](rstate, frame, video, records[i])
                    z = standardized(rstate, x)[None, :]
                    sk = float(model.predict_proba(z)[0, 1])
                    repair_delta = max(repair_delta, abs(probability - ref))
                    sklearn_delta = max(sklearn_delta, abs(probability - sk))
                    e.require(abs(api['repair'](repair_states[GATES[1]], frame, video, records[i]) - old_forest) <= 1e-12, 'repair changed forest inference')
                    if saved_side:
                        repair_saved_delta = max(repair_saved_delta, abs(probability - float(repaired[1][f'{saved_side}_video_{GATES[0]}_{i}'])))
                        repair_inference.append(dict(summary, old=probability, reference=sk, frame=frame))
            if training:
                w = balanced_weights(records, ids)
                matrix = np.stack(training_features)
                mean = np.average(matrix, axis=0, weights=w)
                scale = np.maximum(np.sqrt(np.average((matrix - mean) ** 2, axis=0, weights=w)), 1e-6)
                state = states[GATES[0]]
                d = max(float(np.max(np.abs(mean - state['mean']))), float(np.max(np.abs(scale - state['scale']))))
                scaler_delta = max(scaler_delta, d)
                e.require(d <= 1e-12, 'saved scaler differs from actual heldout training arrays')
    e.require(saved_delta <= TOLERANCE and forest_delta <= TOLERANCE, 'saved/original forest portable parity failed')
    by_role = {role: difference_summary([r for r in details if r['role'] == role]) for role in sorted({r['role'] for r in details})}
    result = {'fold': fold, 'partitions_and_provenance': 'passed', 'all_roles': difference_summary(details),
              'inference_only': difference_summary([{k: v for k, v in r.items() if k != 'frame'} for r in inference]),
              'by_role': by_role, 'feature_max_abs_difference': features_delta, 'scaler_max_abs_difference': scaler_delta,
              'saved_v6_recompute_max_abs_difference': saved_delta, 'forest_reference_max_abs_difference': forest_delta,
              'interval_impact': interval_impact(api, records, configs, inference)}
    if repaired is not None:
        e.require(max(repair_delta, sklearn_delta, repair_saved_delta) <= TOLERANCE, 'v6r independent parity failed')
        result['repair'] = {'helper_vs_training_reference_max_abs_difference': repair_delta,
                            'helper_vs_sklearn_max_abs_difference': sklearn_delta,
                            'saved_v6r_recompute_max_abs_difference': repair_saved_delta,
                            'sklearn_probability_comparisons': len(details),
                            'interval_impact_vs_sklearn': interval_impact(api, records, configs, repair_inference)}
    return result


def verify_repair(e, folder, receipt, fold, parent, original, records):
    row, arrays = load_asset(e, folder, f'{fold}_gates', receipt, set(original), records)
    e.require(row['numeric_revision'] == REVISION and row['parent_npz_sha256'] == parent['npz_sha256']
              and row['parent_metadata_sha256'] == e.digest(e.root / 'output/algorithm-opt-2026-10-02-v6/training' / f'{fold}_gates.json'), 'repair parent provenance mismatch')
    verify_partitions(e, records, fold, parent['train'], parent['validation'], row)
    for original_part, repaired_part in zip(parent['inner_partitions'] + [{'gate_states': parent['outer_gate_states']}],
                                           row['inner_partitions'] + [{'gate_states': row['outer_gate_states']}]):
        if 'deep_partitions' in original_part:
            e.require(original_part['deep_partitions'] == repaired_part['deep_partitions'], 'repair changed deep proof')
        for kind in GATES:
            state = repaired_part['gate_states'][kind]
            e.require(state.get('numeric_revision') == REVISION and {k: v for k, v in state.items() if k != 'numeric_revision'}
                      == original_part['gate_states'][kind], 'repair changed fitted gate parameters')
    for key in original:
        if '_video_ridge_logistic_v6_' not in key:
            e.require(np.array_equal(original[key], arrays[key]), 'repair changed nonlogistic source evidence: ' + key)
    return row, arrays


def run_audit(root, repair=False):
    e = Evidence(root)
    old, v5 = e.root / 'output/algorithm-opt-2026-10-02-v6', e.root / 'output/algorithm-opt-2026-10-02-v5'
    receipt = verify_receipt(e, old)
    r5 = verify_receipt(e, v5)
    e.require(receipt['v5_training_receipt_sha256'] == r5['receipt_sha256']
              and receipt['v5_inputs_sha256'] == r5['inputs_sha256'] == object_hash(r5['input_hashes']), 'v5/v6 input receipt link mismatch')
    e.hash_map(r5['input_hashes'])
    e.hash_map(r5['input_source_hashes'])
    e.hash_map(receipt['v5_artifacts'])
    actual_v5 = {p.relative_to(e.root).as_posix() for p in (v5 / 'training').iterdir() if p.suffix in ['.json', '.npz']}
    e.require(actual_v5 == set(receipt['v5_artifacts']), 'v5 sourcecache inventory mismatch')
    protocol = e.json(old / 'protocol.json')
    configs = protocol['decoder_configs']
    e.require(protocol['base_ensemble'] == [[MEMBERS[0], .5], [MEMBERS[1], .5]], 'base ensemble not fixed .5/.5')
    e.require(all(c['kind'] in ['duration-logit-v4', 'recall-stable-v2'] and c.get('video_strength', 0) == 0 for c in configs), 'unsupported gate-to-decoder contract')
    repair_folder, repair_receipt = None, None
    if repair:
        import sklearn
        repair_folder = e.root / 'output/algorithm-opt-2026-10-02-v6r'
        repair_receipt = verify_receipt(e, repair_folder)
        e.require(repair_receipt['parent_receipt_sha256'] == receipt['receipt_sha256'], 'repair receipt parent link mismatch')
        e.hash_map(repair_receipt['parent_artifacts'])
        rp = e.json(repair_folder / 'protocol.json')
        e.require(rp['parent_protocol_sha256'] == receipt['protocol_sha256'] and rp['decoder_configs'] == configs
                  and rp['numerical_change'] == REVISION, 'repair protocol changed numerical-only contract')
    manifest, records = load_records(e)
    api = frozen_api(old / 'sources', repair_folder / 'sources' if repair else None)
    folds = []
    for fold, val in enumerate(manifest['outer_folds']):
        train = [i for i in range(len(records)) if i not in set(val)]
        parts = [{'fit': [i for i in train if i not in set(iv)]} for iv in grouped_folds(records, train, 3, 20261002 + fold * 31)]
        meta, arrays = load_asset(e, old, f'{fold}_gates', receipt, expected_gate_keys(train, val, parts), records)
        verify_partitions(e, records, fold, train, val, meta)
        members = [verify_v5(e, v5, r5, fold, recipe, meta, records) for recipe in MEMBERS]
        for side, ids in [('inner', train), ('outer', val)]:
            for i in ids:
                frame = sum(.5 * m[f'{side}_frame_{i}'] for m in members).astype(np.float32)
                video = sum(.5 * float(m[f'{side}_video_{i}']) for m in members)
                e.require(np.array_equal(frame, arrays[f'{side}_frame_{i}'])
                          and video == float(arrays[f'{side}_video_ungated_control_{i}']), 'v6 evidence differs from fixed v5 sourcecache blend')
        repaired = verify_repair(e, repair_folder, repair_receipt, fold, meta, arrays, records) if repair else None
        folds.append(evaluate_fold(e, records, fold, meta, arrays, api, configs, repaired))
    e.unchanged()
    max_difference = max(f['all_roles']['max_abs_difference'] for f in folds)
    report = {'schema_version': 'independent-video-gate-asset-audit-v1',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'status': 'completed_with_original_v6_numeric_defect' if max_difference > TOLERANCE else 'passed',
              'role': 'numerical_and_provenance_audit_not_model_selection', 'root': str(e.root),
              'audit_script_sha256': file_hash(Path(__file__)), 'training_receipt_sha256': receipt['receipt_sha256'],
              'checks_passed': e.checks, 'watched_assets': len(e.watched), 'before_after_asset_hashes_equal': True,
              'data': {'rows': 77, 'contents': 76, 'outer_folds': 5, 'inner_partitions': 15,
                       'deep_partitions': 45, 'deep_member_proofs': 90, 'gate_states': 40},
              'method': {'reference': 'independent float64 fsum sigmoid; float64 mean/scale then clip.cast(float32)',
                         'frozen_source_only_inference': True, 'no_training': True, 'no_selector': True,
                         'no_outer_metric_report_parsed': True, 'thresholds': sorted({c['video_threshold'] for c in configs})},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__}, 'folds': folds,
              'original_v6_max_abs_difference_all_roles': max_difference,
              'original_v6_max_abs_difference_inference_only': max(f['inference_only']['max_abs_difference'] for f in folds),
              'original_v6_above_2e_6_inference_only': sum(f['inference_only']['above_2e_6'] for f in folds),
              'original_v6_interval_changes': sum(f['interval_impact']['changed_predictions'] for f in folds),
              'limitations': ['No base model refits; fit-content proofs are signed source/cache attestations, not independently retrained models.',
                              'Saved logistic coefficients are held fixed; reference verifies preprocessing/inference, not solver convergence.',
                              'OOF/full-fit evidence distribution shift is not eliminated or measured by this provenance audit.',
                              'All frozen decoder configs are compared without ground-truth scoring or selection.']}
    if repair:
        report['runtime']['sklearn'] = sklearn.__version__
        report['repair_receipt_sha256'] = repair_receipt['receipt_sha256']
        report['repair'] = {'status': 'passed', 'helper_vs_sklearn_max_abs_difference': max(f['repair']['helper_vs_sklearn_max_abs_difference'] for f in folds),
                            'helper_vs_training_reference_max_abs_difference': max(f['repair']['helper_vs_training_reference_max_abs_difference'] for f in folds),
                            'saved_v6r_recompute_max_abs_difference': max(f['repair']['saved_v6r_recompute_max_abs_difference'] for f in folds),
                            'sklearn_probability_comparisons': sum(f['repair']['sklearn_probability_comparisons'] for f in folds),
                            'coefficients_and_nonlogistic_evidence_unchanged': True}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repair', action='store_true', help='audit frozen v6r with sklearn; write its audit separately')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    original_output = root / 'output/algorithm-opt-2026-10-02-v6/independent_gate_audit.json'
    repair_output = root / 'output/algorithm-opt-2026-10-02-v6r/independent_gate_audit.json'
    outputs = ([original_output] if not original_output.exists() else []) + ([repair_output] if args.repair else [])
    if not outputs:
        parser.error('independent audit output already exists; existing reports are never overwritten')
    if any(p.exists() for p in outputs):
        parser.error('independent repair audit output already exists; existing reports are never overwritten')
    try:
        report = run_audit(root, args.repair)
    except (AuditError, OSError, ValueError, KeyError, ImportError) as exc:
        print('AUDIT FAILED:', str(exc), file=sys.stderr)
        return 1
    original = dict(report)
    original.pop('repair', None)
    original.pop('repair_receipt_sha256', None)
    original['folds'] = [{k: v for k, v in f.items() if k != 'repair'} for f in report['folds']]
    if original_output in outputs:
        write_new(original_output, original)
    if args.repair:
        write_new(repair_output, report)
    print(json.dumps({'status': report['status'], 'outputs': [str(p) for p in outputs],
                      'original_max_inference_difference': report['original_v6_max_abs_difference_inference_only'],
                      'original_interval_changes': report['original_v6_interval_changes'],
                      'repair': report.get('repair')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
