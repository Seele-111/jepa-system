#!/usr/bin/env python3
"""Bounded V13R audit: no fitting, extraction, services or production selection.

Only verified transforms/decoders are production dependencies. Statistics, masks,
ranking, bootstrap and choices are independently checked. Missing repair products
are pending, not passed. --output must name a new file (exclusive creation).
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import audit_feature_blocks_v10 as A
import audit_spatial_jepa_v13 as P

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('output/algorithm-opt-v13r-spatial-mask-repair')
PARENT = P.OUT
RAW = A.OLD / 'corrected_jepa_features_v2'
RECIPE, CONTROL, SEED = P.RECIPE, P.CONTROL, A.SEED
# Observed at source review, NOT independently recorded before repair training.
RECEIPT_PIN = '13586eae19cf3347e5ca8cb11d457743f45d504b9b1557c3c217986edf63d0e2'
AUDITOR_PINS = {
    'audit_feature_blocks_v10.py': 'f1ad533abe2bc46e454f73bfdd0ab007afdd8d83924b3ce5696ae54fddc97700',
    'audit_spatial_jepa_v13.py': '199fa598b6bcc3f67be1262b29114f2de8af861533f02cfc9637b819905dff83',
}
SOURCE_PINS = {
    'optimized_spatial_jepa_v13r.py': 'd4e17bbc26d631f8d665dfcfba6988d8c2f25859c26f9f391f28e064035e37fd',
    'run_spatial_jepa_v13r.py': 'd120d4c979ccf099a9176cae9eb1ed3a3c9b0092ffd87e547d9025f6bf075d92',
}
FIELDS = ('patch_support_fraction', 'positive_mass', 'p90', 'p99', 'centroid_x',
          'centroid_y', 'spread_x', 'spread_y', 'covariance_xy', 'anisotropy',
          'top10_mass_fraction', 'top25_mass_fraction', 'peak_x', 'peak_y',
          'edge_mass_fraction', 'center_mass_fraction', 'entropy_normalized')
FORBIDDEN = frozenset(('labels', 'sha256', 'source_sha256', 'name', 'generator',
                       'prompt_id', 'event_count', 'rgb', 'index', 'row_id'))
LIMITATIONS = [
    'Repeated development data, not blind/confirmatory evaluation or deployment approval.',
    'Independent statistics/selection reuse the pinned independent V10 auditor; '
    'production transforms and fixed decoders are audited dependencies, not independently rewritten.',
    'Exact-SHA isolation does not establish near-duplicate, generator, scene or prompt isolation.',
    'Original conflicting alias labels are preserved; only 14 normal annotation rows.',
    'Frame equality proves reuse of archived V13 tensors, not correctness or historical fit execution of V13.',
    'Repair video_model_sha256 is a recorded digest only: no model state bytes are saved '
    'to recompute it or replay fits/predictions. Valid digest syntax is NOT model-state verification.',
    'Transform digests bind configuration/name order, not the full numerical feature tensors '
    'or historical execution. Features are recomputed read-only without fitting.',
    'max_portable_error is checked as recorded metadata; portable estimator parity is NOT '
    'independently replayed without per-fit saved estimators. It is not a verified promotion parity guard.',
    'Receipt/source pins for V13R and auditor dependencies were observed at audit time, '
    'not independently committed before training/selection. Unkeyed signatures are consistency hashes.',
    'Raw cache hashes bind current bytes, not missing historical extraction/source authentication.',
    'Offline anchor interpolation/centered/future/global features are not causal streaming.',
    'Final inner-only scores are selection diagnostics, not an independent final validation score.',
    'No training, remote calls, production chooser calls, fresh-pixel inference, latency or deployment audit.',
]


def signed_object(row, field, label, pin=None):
    A.require(isinstance(row, dict) and A.is_hash(row.get(field)), label + ': missing digest')
    actual = A.value_hash({k: v for k, v in row.items() if k != field})
    A.require(actual == row[field] and (pin is None or actual == pin), label + ': canonical digest differs')
    return actual


def bitwise_equal(actual, expected, label):
    """Numerical array_equal is insufficient: dtype, endian and signed zero matter."""
    a, b = np.asarray(actual), np.asarray(expected)
    A.require(a.shape == b.shape and a.dtype.str == b.dtype.str
              and a.tobytes(order='C') == b.tobytes(order='C'), 'bitwise mismatch: ' + label)


def artifact_path(folder, scope):
    return folder / ('final_selection' if scope == 5 else 'training') / f'{scope}_{RECIPE}.npz'


def inventory(folder, recipe):
    return {(folder / ('final_selection' if s == 5 else 'training') / f'{s}_{recipe}.{ext}').as_posix()
            for s in range(6) for ext in ('json', 'npz')}


def verify_lineage(e):
    for name, pin in AUDITOR_PINS.items():
        e.read(Path('code') / name, pin)
    receipt = e.json(OUT / 'training_receipt.json')
    signed_object(receipt, 'receipt_sha256', 'repair receipt', RECEIPT_PIN)
    parent = e.json(PARENT / 'training_receipt.json')
    signed_object(parent, 'receipt_sha256', 'original V13 receipt', P.RECEIPT_PIN)
    v10, baseline = A.verify_receipts(e)  # Independent receipt verification only, no runner import.
    A.compare(receipt['parent_receipt_sha256'], parent['receipt_sha256'], 'repair/parent receipt')
    A.compare(parent['parents'], {'v8': v10['parent_receipt_sha256'], 'v10': v10['receipt_sha256']},
              'original/control parents')
    for key in ('input_hashes', 'inputs_sha256', 'raw_spatial', 'control_artifacts',
                'baseline_sha256', 'default_models'):
        A.compare(receipt[key], parent[key], 'repair/original/' + key)
    for key in ('input_hashes', 'inputs_sha256', 'input_sources', 'default_models'):
        A.compare(parent[key], v10[key], 'original/control/' + key)
    A.require(A.value_hash(receipt['input_hashes']) == receipt['inputs_sha256'] == A.INPUTS_PIN,
              'original labels/FPS input inventory differs')
    for rel in (A.OLD / 'dataset.npz', A.OLD / 'dataset_manifest.json', A.GROUPED / 'dataset_manifest.json'):
        A.require(rel.as_posix() in receipt['input_hashes'], 'missing original labels/FPS binding')
    A.compare(receipt['baseline_sha256'], A.BASELINE_PIN, 'repair/baseline pin')
    A.require(set(parent['sources']) == set(A.SOURCES) | {'optimized_spatial_jepa_v13.py', 'run_spatial_jepa_v13.py'},
              'original source inventory differs')
    A.require(set(receipt['sources']) == set(parent['sources']) | set(SOURCE_PINS), 'repair source inventory differs')
    for name, h in parent['sources'].items():
        A.compare(receipt['sources'][name], h, 'repair source inheritance/' + name)
        e.read(PARENT / 'sources' / name, h)
    for name, h in receipt['sources'].items():
        if name in SOURCE_PINS:
            A.compare(h, SOURCE_PINS[name], 'repair source review pin/' + name)
        e.read(Path('code') / name, h)
        e.read(OUT / 'sources' / name, h)
    A.require(set(receipt['control_artifacts']) == inventory(A.OUT, CONTROL), 'control cache inventory differs')
    A.require(set(receipt['reused_parent_frame_artifacts']) == inventory(PARENT, RECIPE),
              'original frame cache inventory differs')
    for section in ('control_artifacts', 'reused_parent_frame_artifacts'):
        for rel, h in receipt[section].items():
            e.read(rel, h)
    raw = e.json(RAW / 'manifest.json', receipt['raw_spatial']['manifest_sha256'])
    A.require(raw.get('status') == 'complete' and raw.get('label_free') is True
              and isinstance(raw.get('videos'), list) and len(raw['videos']) == 77, 'raw manifest differs')
    expected_raw = {(RAW / 'manifest.json').as_posix()} | {
        (RAW / r['raw_directory'] / 'signals.npz').as_posix() for r in raw['videos']}
    A.require(len(expected_raw) == 78 and set(receipt['raw_spatial']['files']) == expected_raw,
              'raw mask cache inventory differs')
    for rel, h in receipt['raw_spatial']['files'].items():
        e.read(rel, h)
    protocol = e.json(OUT / 'protocol.json', receipt['protocol_sha256'])
    original_protocol = e.json(PARENT / 'protocol.json', parent['protocol_sha256'])
    A.compare(protocol['schema_version'], 'spatial-jepa-mask-repair-v13r', 'repair/schema')
    A.compare(protocol['role'], 'repeated_development_not_blind_or_confirmatory', 'repair/role')
    A.compare(protocol['promotion_guards'], A.GUARD_SPEC, 'repair/guards')
    A.compare(original_protocol['promotion_guards'], protocol['promotion_guards'], 'original/guards')
    for key, value in (('new_nested_video_fits', 20), ('new_final_video_fits', 3), ('new_frame_fits', 0)):
        A.compare(protocol[key], value, 'repair/protocol/' + key)
    return receipt, parent, baseline, raw, protocol


def source_contract(e):
    """Static contract inspection: never import/execute either experiment runner."""
    src = e.read('code/optimized_spatial_jepa_v13r.py', SOURCE_PINS['optimized_spatial_jepa_v13r.py'])
    old = e.read('code/optimized_spatial_jepa_v13.py')
    runner = e.read('code/run_spatial_jepa_v13r.py', SOURCE_PINS['run_spatial_jepa_v13r.py'])
    tree, otree, rtree = ast.parse(src), ast.parse(old), ast.parse(runner)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    fitter = funcs['fit_repaired_video']
    def et_calls(node):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'ExtraTreesClassifier']
    calls = et_calls(fitter)
    expected = {'n_estimators': '128', 'max_depth': '4', 'min_samples_leaf': '2',
                'max_features': '0.75', 'class_weight': "'balanced'", 'n_jobs': '2',
                'random_state': 'seed + 1100'}
    A.require(len(calls) == 1, 'repair must fit only one video estimator')
    A.compare({k.arg: ast.unparse(k.value) for k in calls[0].keywords}, expected, 'video ET parameters')
    original = [c for c in et_calls(otree) if any(k.arg == 'n_estimators' and ast.unparse(k.value) == '128'
                                               for k in c.keywords)]
    A.require(len(original) == 1, 'original video ET parameters ambiguous')
    A.compare({k.arg: ast.unparse(k.value) for k in original[0].keywords}, expected, 'unchanged video ET')
    fit_calls = [ast.unparse(n) for n in ast.walk(fitter) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit']
    A.compare(fit_calls, ['model.fit(video, y, sample_weight=weights)'], 'only video fit')
    text = ast.unparse(fitter)
    for required in ("{records[i]['sha256'] for i in train} & {records[i]['sha256'] for i in predict}",
                     'content_weights(records, train)', "[int(np.any(records[i]['labels'])) for i in train]",
                     '[cw[i] * (2 if y[j] == 0 else 1) for j, i in enumerate(train)]'):
        A.require(required in text, 'fit train-only/content/weighting contract differs: ' + required)
    view = ast.unparse(funcs['feature_view_spatial_repaired'])
    A.require('frame, _, names = feature_view_spatial(RECIPE, record, None)' in view
              and '_, base, _ = feature_view_blocks(BASE_RECIPE, record, None)' in view,
              'frozen frame/no-PCA base transform differs')
    A.require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id == 'fit_rgb_pca' for n in ast.walk(tree)), 'repair calls RGB/PCA')
    rf = {n.name: n for n in rtree.body if isinstance(n, ast.FunctionDef)}
    task = ast.unparse(rf['fit_task'])
    A.require("arrays[f'{prefix}_frame_{i}'] = fp[i].copy()" in task
              and 'OLD.load_new(scope, records, OLD.check_receipt(), final)' in task,
              'frame parent/copy contract differs')
    A.require('fit_spatial_member' not in task and 'fit_block_member' not in task,
              'repair task unexpectedly refits a frame member')
    selection = ast.unparse(rf['select_rows'])
    A.require('V8.choose(records, ids, p, 20261002 + scope)' in selection,
              'production selection seed/procedure source differs')
    return {'status': 'passed', 'unchanged_video_ET': expected, 'new_frame_fit_calls': 0,
            'production_selection_executed': False,
            'scope': 'pinned_static_source_contract_not_historical_fit_replay'}


def reference_heat_statistics(heat, valid):
    """Independent 17-field semantics, including peak in the masked coordinate list."""
    h = np.asarray(heat, np.float64)
    mask = np.asarray(valid)
    A.require(h.ndim == 2 and mask.dtype == np.dtype(bool) and mask.shape == h.shape
              and np.isfinite(h[mask]).all() and np.all(h[mask] >= 0), 'invalid observed heat values/mask')
    out = np.zeros(17, np.float32)
    pos = np.argwhere(mask)
    if not len(pos):
        return out
    v = h[mask]
    xy = np.stack((2 * pos[:, 1] / max(1, h.shape[1] - 1) - 1,
                   2 * pos[:, 0] / max(1, h.shape[0] - 1) - 1), axis=1)
    total = float(v.sum())
    weight = v / total if total > 1e-12 else np.ones(len(v)) / len(v)
    center = weight @ xy
    delta = xy - center
    xx, yy = (float(np.sum(weight * delta[:, j] ** 2)) for j in range(2))
    cov = float(np.sum(weight * delta[:, 0] * delta[:, 1]))
    anis = math.sqrt(max(0., (xx - yy) ** 2 + 4 * cov ** 2)) / max(xx + yy, 1e-6)
    descending = np.sort(v)[::-1]
    fractions = [float(descending[:max(1, math.ceil(p * len(v)))].sum() / max(total, 1e-12))
                 for p in (.10, .25)]
    peak = xy[int(np.argmax(v))]
    edge = np.any(np.abs(xy) > .66, axis=1)
    central = (xy ** 2).sum(axis=1) <= .25
    entropy = -float(np.sum(weight * np.log(np.maximum(weight, 1e-12)))) / math.log(max(2, len(v)))
    out[:] = [mask.mean(), total, *np.percentile(v, [90, 99]), *center,
              math.sqrt(max(0., xx)), math.sqrt(max(0., yy)), cov, anis,
              *fractions, *peak, weight[edge].sum(), weight[central].sum(), entropy]
    return out


def reference_raw_spatial(arrays, frames, fps):
    A.require(type(frames) is int and frames > 0 and type(fps) in (int, float)
              and math.isfinite(fps) and fps > 0, 'invalid frame/FPS schema')
    A.require(arrays['frame_ids'].dtype.kind in 'iu' and arrays['frame_ids'].shape == (frames,)
              and np.array_equal(arrays['frame_ids'], np.arange(frames)), 'raw frame ids mismatch')
    ts = arrays['timestamps_sec']
    A.require(ts.shape == (frames,) and np.isfinite(ts).all()
              and np.allclose(ts, np.arange(frames) / fps, rtol=0, atol=1e-6), 'raw timestamps/FPS mismatch')
    values, supports, evidence = [], [], []
    for prefix, ids_key in (('vjepa', 'tubelet_frame_ids'), ('ijepa', 'keyframe_ids')):
        heat = arrays[prefix + '_raw_heatmaps']
        mask = arrays[prefix + '_patch_valid_mask']
        obs = arrays[prefix + '_valid_mask']
        counts = arrays[prefix + '_patch_counts']
        ids = arrays[ids_key]
        A.require(heat.ndim == 3 and min(heat.shape) > 0 and heat.dtype.kind == 'f'
                  and mask.dtype == np.dtype(bool) and mask.shape == heat.shape,
                  'raw heat/boolean patch-mask schema mismatch')
        A.require(counts.shape == heat.shape and counts.dtype.kind in 'iu'
                  and np.all(counts >= 0) and np.array_equal(mask, counts > 0), 'patch mask/count semantics mismatch')
        A.require(obs.dtype == np.dtype(bool) and obs.shape == (len(heat),)
                  and np.array_equal(obs, mask.reshape(len(heat), -1).any(axis=1)), 'anchor observed mask mismatch')
        A.require(np.isfinite(heat[mask]).all() and np.all(heat[mask] >= 0)
                  and np.all(~np.isfinite(heat[~mask])), 'raw supported/nonfinite-unobserved values mismatch')
        shape = (len(heat), 2) if prefix == 'vjepa' else (len(heat),)
        A.require(ids.dtype.kind in 'iu' and ids.shape == shape and np.all((ids >= 0) & (ids < frames)),
                  'raw integer/in-range anchor ids mismatch')
        centers = ids.astype(np.float64).mean(axis=1) if prefix == 'vjepa' else ids.astype(np.float64)
        if prefix == 'vjepa':
            A.require(np.all(ids[:, 0] <= ids[:, 1]), 'tubelet member order mismatch')
        loc = centers[obs]
        A.require(len(loc) > 0 and np.all(np.diff(loc) > 0), 'no/duplicate/nonmonotone observed anchors')
        stats = np.stack([reference_heat_statistics(h, m) for h, m in zip(heat, mask)])
        A.require(np.all(stats[obs, 1] > 0), 'zero-mass observed raw anchor (requires separate review)')
        dense = np.stack([np.interp(np.arange(frames), loc, stats[obs, j]) for j in range(17)], axis=1).astype(np.float32)
        # Support is between observed anchor CENTERS, not tubelet endpoints or heat nonzero values.
        keep = (np.arange(frames) >= loc[0]) & (np.arange(frames) <= loc[-1])
        support = np.repeat(keep[:, None], 17, axis=1)
        values.append(dense); supports.append(support)
        evidence.append({'model': prefix, 'anchors': len(heat), 'observed_anchors': int(obs.sum()),
                         'first_center_frame': float(loc[0]), 'last_center_frame': float(loc[-1]),
                         'unsupported_frames': int((~keep).sum())})
    names = [p + '_spatial/' + name for p in ('v', 'i') for name in FIELDS]
    return np.concatenate(values, axis=1), np.concatenate(supports, axis=1), names, evidence


def reference_video_statistics(values, support):
    """Population std + linear percentiles; compute moments in float64 independently."""
    x, mask = np.asarray(values, np.float32), np.asarray(support)
    A.require(x.ndim == 2 and len(x) > 0 and mask.dtype == np.dtype(bool) and mask.shape == x.shape
              and np.isfinite(x[mask]).all(), 'invalid masked video data')
    result = np.zeros((4, x.shape[1]), np.float32)
    for j in range(x.shape[1]):
        column = x[mask[:, j], j].astype(np.float64)
        if len(column):
            mean = float(column.sum() / len(column))
            std = math.sqrt(float(((column - mean) ** 2).sum() / len(column)))
            result[:, j] = [mean, std, *np.quantile(column, [.1, .9], method='linear')]
    return result.reshape(-1)


def close_float32(a, b, label):
    A.require(np.asarray(a).shape == np.asarray(b).shape and np.isfinite(a).all()
              and np.isfinite(b).all() and np.allclose(a, b, rtol=2e-6, atol=2e-6),
              'float32 reference mismatch: ' + label)


def verify_real_features(e, records, aliases, raw_manifest, receipt):
    from optimized_spatial_jepa_v13 import spatial_from_raw, feature_view_spatial, SPATIAL_FIELDS
    from optimized_spatial_jepa_v13r import feature_view_spatial_repaired, masked_video_statistics
    from optimized_feature_blocks_v10 import feature_view_blocks
    A.compare(list(SPATIAL_FIELDS), list(FIELDS), 'frozen spatial field order')
    proofs, common_names = [], None
    for i, (record, entry) in enumerate(zip(records, raw_manifest['videos'])):
        for k, value in (('name', record['name']), ('source_sha256', record['sha256']),
                         ('frames', record['frames']), ('fps', record['fps'])):
            A.compare(entry[k], value, 'raw row/' + str(i) + '/' + k)
        rel = RAW / entry['raw_directory'] / 'signals.npz'
        data = e.read(rel, receipt['raw_spatial']['files'][rel.as_posix()])
        arrays = A.read_archive(data, rel.as_posix())
        ref, ref_support, ref_names, anchor_proof = reference_raw_spatial(arrays, record['frames'], record['fps'])
        spatial, support, names = spatial_from_raw(BytesIO(data), record['frames'], record['fps'])
        close_float32(spatial, ref, 'all raw descriptor/interpolation semantics/' + str(i))
        bitwise_equal(support, ref_support, 'real support/' + str(i))
        A.compare(names, ref_names, 'spatial name order/' + str(i))
        record.update(spatial=spatial, spatial_support=support, spatial_names=names)
        clean = {k: v for k, v in record.items() if k not in FORBIDDEN}
        old_frame, old_video, old_names = feature_view_spatial(RECIPE, clean, None)
        frame, video, fnames = feature_view_spatial_repaired(clean)
        base_frame, base_video, base_names = feature_view_blocks(CONTROL, clean, None)
        bitwise_equal(frame, old_frame, 'frozen frame feature/' + str(i))
        A.compare(fnames, old_names, 'frozen frame feature order/' + str(i))
        bitwise_equal(frame[:, :base_frame.shape[1]], base_frame, 'control frame prefix/' + str(i))
        bitwise_equal(video[:len(base_video)], base_video, 'control video prefix/' + str(i))
        expected_extra = reference_video_statistics(spatial, support)
        close_float32(masked_video_statistics(spatial, support), expected_extra, 'observed-only statistics/' + str(i))
        expected_support = reference_video_statistics(support.astype(np.float32), np.ones_like(support))
        A.require(video.shape == old_video.shape == (len(base_video) + 8 * len(names),), 'video feature dimension differs')
        close_float32(video[len(base_video):len(base_video) + 4 * len(names)], expected_extra, 'real masked video/' + str(i))
        close_float32(video[-4 * len(names):], expected_support, 'unchanged support summary/' + str(i))
        A.require(not any(n.startswith('rgb/') for n in fnames + base_names), 'unexpected RGB feature block')
        if common_names is None:
            common_names = fnames
        A.compare(fnames, common_names, 'all77 feature name consistency')
        changed = {**clean, 'spatial': spatial.copy()}
        changed['spatial'][~support] += np.float32(10000.)
        altered = feature_view_spatial_repaired(changed)
        bitwise_equal(altered[0], frame, 'unsupported value/frame invariance/' + str(i))
        bitwise_equal(altered[1], video, 'unsupported value/video invariance/' + str(i))
        A.compare(altered[2], fnames, 'unsupported value/name invariance')
        proofs.append({'index': i, 'frames': record['frames'], 'fps': record['fps'], 'anchors': anchor_proof,
                       'unsupported_cells': int((~support).sum()), 'frame_features': frame.shape[1],
                       'video_features': len(video), 'control_frame_features': base_frame.shape[1],
                       'control_video_features': len(base_video),
                       'old_video_changed_channels': int(np.count_nonzero(video != old_video)),
                       'frame_bitwise_equal_to_V13': True, 'unsupported_frame_and_video_invariance': True,
                       'feature_inputs_without_labels_identity_RGB': True})
    for alias in aliases:
        first = records[alias['indices'][0]]
        for i in alias['indices'][1:]:
            bitwise_equal(records[i]['spatial'], first['spatial'], 'same-SHA spatial alias')
            bitwise_equal(records[i]['spatial_support'], first['spatial_support'], 'same-SHA support alias')
    # Empty support is not present in the real cache; validate the contract separately, not as a real case.
    empty = np.zeros((3, 2), bool)
    bitwise_equal(masked_video_statistics(np.ones((3, 2), np.float32), empty), np.zeros(8, np.float32),
                  'synthetic empty support = zero')
    return proofs, common_names


def repaired_transform(names):
    return {'feature_view': 'spatial-jepa-mask-repair-v13r', 'pca': None, 'frame_feature_names': names,
            'video_statistics': 'spatial_support_masked_mean_std_p10_p90_empty_zero_and_support_shape'}


def verify_repair_metadata(row, scope, records, outer, receipt, parent_meta, names):
    signed_object(row, 'signature', 'repair cache metadata')
    final = scope == 5
    val = [] if final else outer[scope]
    train = [i for i in range(len(records)) if i not in set(val)]
    parts = A.expected_partitions(records, train, scope)
    for key, value in (('scope', scope), ('final_selection', final), ('recipe', RECIPE), ('train', train),
                       ('validation', val), ('inner_partitions', parts),
                       ('training_receipt_sha256', receipt['receipt_sha256']),
                       ('parent_frame_npz_sha256', parent_meta['npz_sha256']),
                       ('frame_probabilities_reused_unchanged', True)):
        A.compare(row.get(key), value, 'repair/' + str(scope) + '/' + key)
    for key in ('scope', 'final_selection', 'recipe', 'train', 'validation', 'inner_partitions'):
        A.compare(row[key], parent_meta[key], 'repair/parent/' + key)
    A.require(A.is_hash(row.get('npz_sha256')), 'missing repair NPZ digest')
    expected = parts + ([] if final else [{'fit': train, 'validation': val, 'seed': SEED + scope * 53 + 99}])
    A.require(isinstance(row.get('fit_evidence'), list) and len(row['fit_evidence']) == len(expected),
              'repair fit evidence count differs')
    proofs = []
    for k, (proof, part) in enumerate(zip(row['fit_evidence'], expected)):
        universe = train if k < 3 else list(range(len(records)))
        A.check_partition(records, part['fit'], part['validation'], universe, 'repair fit/' + str(scope) + '/' + str(k))
        for key, value in (('partition', k if k < 3 else 'outer'), ('seed', part['seed']),
                           ('fit_content_sha256', sorted({records[i]['sha256'] for i in part['fit']})),
                           ('transform_sha256', A.value_hash(repaired_transform(names)))):
            A.compare(proof.get(key), value, 'repair proof/' + str(scope) + '/' + str(k) + '/' + key)
        A.require(A.is_hash(proof.get('video_model_sha256')), 'invalid recorded model-state digest')
        err = proof.get('max_portable_error')
        A.require(type(err) in (int, float) and math.isfinite(err) and 0 <= err <= 2e-6,
                  'invalid recorded portable error')
        proofs.append({'scope': scope, 'final_selection': final, 'partition': proof['partition'],
                       'fit': part['fit'], 'validation': part['validation'], 'seed': part['seed'],
                       'estimator_random_state': part['seed'] + 1100,
                       'fit_content_sha256': proof['fit_content_sha256'],
                       'transform_sha256': proof['transform_sha256'], 'video_model_sha256': proof['video_model_sha256'],
                       'recorded_max_portable_error': err, 'model_state_status': 'recorded_digest_only_not_replayed'})
    return proofs


def verify_repair_cache(e, scope, records, outer, receipt, parent_maps, parent_meta, names):
    path = artifact_path(OUT, scope)
    row = e.json(path.with_suffix('.json'))
    proofs = verify_repair_metadata(row, scope, records, outer, receipt, parent_meta, names)
    inner, held = A.probability_maps(e.read(path, row['npz_sha256']), row, records, path.as_posix())
    counts = {'inner': 0, 'outer': 0, 'final_inner': 0, 'frame_scalars': 0}
    for label, actual, original in zip(('inner', 'outer'), (inner, held), parent_maps):
        A.require(set(actual.frame) == set(original.frame), 'parent/repair frame coverage differs')
        for i, x in actual.frame.items():
            A.require(x.dtype.kind == 'f', 'frame probability dtype is not floating point')
            bitwise_equal(x, original.frame[i], 'frame probability/' + str(scope) + '/' + label + '/' + str(i))
            counts['final_inner' if scope == 5 else label] += 1
            counts['frame_scalars'] += len(x)
    return inner, held, row, proofs, counts


def rank_spatial(rows):
    A.compare([r['candidate'] for r in rows], [RECIPE, 'control'], 'candidate order/coverage')
    return max(rows, key=lambda r: (r['stable_utility'], *r['key'][1:],
                                    -[RECIPE, 'control'].index(r['candidate'])))


def independent_rows(records, ids, primary, control, scope, decoders):
    rows = []
    for candidate, probabilities in ((RECIPE, primary), ('control', control)):
        choice = A.independent_choice(records, ids, probabilities, SEED + scope, decoders)
        choice['candidate'] = candidate
        rows.append(choice)
    return rank_spatial(rows), rows


def verify_nested(e, report, records, outer, repaired, controls, baseline, receipt, protocol, decoders):
    for key, value in (('schema_version', 'spatial-jepa-mask-repair-v13r'), ('status', 'complete'),
                       ('role', protocol['role']), ('protocol_sha256', receipt['protocol_sha256']),
                       ('training_receipt_sha256', receipt['receipt_sha256']), ('grouped_v1_baseline', baseline)):
        A.compare(report.get(key), value, 'nested/' + key)
    A.require(len(report.get('folds', [])) == 5 and all(s in repaired for s in range(5)),
              'complete nested report lacks five verified outer caches')
    selected, fixed = {}, {n: {} for n in (RECIPE, 'control')}
    decisions = []
    for scope, val in enumerate(outer):
        ip, op, meta = repaired[scope]
        ci, co, cm = controls[scope]
        train = [i for i in range(len(records)) if i not in set(val)]
        A.compare(meta['inner_partitions'], cm['inner_partitions'], 'same-protocol control partitions')
        chosen, rows = independent_rows(records, train, ip, ci, scope, decoders)
        saved = report['folds'][scope]
        for key, value in (('fold', scope), ('train_indices', train), ('validation_indices', val),
                           ('inner_partitions', meta['inner_partitions']), ('primary', chosen),
                           ('all_inner_selections', rows)):
            A.compare(saved.get(key), value, 'outer/' + str(scope) + '/' + key)
        pred = A.decode(records, val, op if chosen['candidate'] == RECIPE else co, chosen['config'], decoders)
        A.compare(saved['primary_predictions'], {str(i): pred[i] for i in val}, 'outer prediction serialization')
        A.require(not set(selected) & set(pred), 'outer OOF rows repeated')
        selected.update(pred)
        for row in rows:
            fixed[row['candidate']].update(A.decode(records, val, op if row['candidate'] == RECIPE else co,
                                                   row['config'], decoders))
        decisions.append({'fold': scope, 'candidate': chosen['candidate'], 'config': chosen['config'],
                          'selection_ids': train, 'evaluation_ids': val, 'selection_seed': SEED + scope})
    primary = report['primary_v13r_nested']
    metrics = A.grouped_metrics(records, selected)
    A.compare(primary['metrics'], metrics, 'nested metrics')
    A.compare(primary['predictions'], A.prediction_rows(records, selected), 'nested predictions')
    A.compare(primary['summary'], A.prediction_summary(records, selected), 'nested summary')
    A.require(set(report['fixed_candidate_diagnostics_not_for_promotion']) == set(fixed), 'fixed candidate coverage')
    for candidate, pred in fixed.items():
        saved = report['fixed_candidate_diagnostics_not_for_promotion'][candidate]
        A.compare(saved['role'], 'fixed_recipe_outer_diagnostic_not_deployment_selection', 'fixed diagnostic role')
        A.compare(saved['metrics'], A.grouped_metrics(records, pred), 'fixed candidate metrics')
        A.compare(saved['predictions'], A.prediction_rows(records, pred), 'fixed candidate predictions')
    bp = A.read_prediction_rows(baseline['predictions'], records, 'grouped v1 baseline')
    bm = A.grouped_metrics(records, bp)
    A.compare(baseline['metrics'], bm, 'recomputed baseline metrics')
    guards = A.promotion_guards(metrics, bm)
    A.compare(report['statistical_promotion_checks'], guards, 'nested guards')
    A.compare(report['statistical_promotion_passed'], all(guards.values()), 'nested guard conjunction')
    uncertainty = A.paired_uncertainty(records, bp, selected)
    A.compare(report['paired_content_bootstrap'], uncertainty, 'paired content bootstrap (2000)')
    return {'status': 'passed', 'outer_decisions': decisions, 'metrics': metrics, 'summary': primary['summary'],
            'statistical_guards': guards, 'statistical_promotion_passed': all(guards.values()),
            'paired_content_bootstrap': uncertainty,
            'guard_note': 'Guard reproduction passes even when guards are false; no deployment approval.'}


def verify_final(e, saved, records, repaired, controls, receipt, decoders):
    A.require(5 in repaired, 'complete final selection lacks verified final inner OOF cache')
    ip, op, meta = repaired[5]
    ci, co, cm = controls[5]
    A.require(not op.frame and not co.frame, 'final selection cache contains outer/fullfit predictions')
    A.compare(meta['inner_partitions'], cm['inner_partitions'], 'final/control partitions')
    ids = list(range(len(records)))
    chosen, rows = independent_rows(records, ids, ip, ci, 5, decoders)
    for key, value in (('status', 'complete'), ('role', 'inner3_only_deployment_choice_not_validation'),
                       ('chosen', chosen), ('all_inner_selections', rows), ('inner_partitions', meta['inner_partitions']),
                       ('uses_outer_probabilities', False), ('uses_outer_metrics_for_selection', False),
                       ('coverage_per_row', 1), ('training_receipt_sha256', receipt['receipt_sha256'])):
        A.compare(saved.get(key), value, 'final/' + key)
    predictions = A.decode(records, ids, ip if chosen['candidate'] == RECIPE else ci, chosen['config'], decoders)
    return {'status': 'passed', 'candidate': chosen['candidate'], 'config': chosen['config'],
            'selection_seed': SEED + 5, 'coverage_per_row': 1, 'uses_outer_metrics_or_probabilities': False,
            'inner_only_diagnostic_metrics_not_validation': A.grouped_metrics(records, predictions)}


def verify_training_report(row, final):
    A.compare(row.get('status'), 'complete', 'training report/status')
    A.compare(row.get('new_video_fits'), 3 if final else 20, 'training report/video fits')
    A.compare(row.get('new_frame_fits'), 0, 'training report/frame fits')
    expected = {(5, True)} if final else {(s, False) for s in range(5)}
    results = row.get('results')
    A.require(isinstance(results, list) and len(results) == len(expected), 'training job count differs')
    jobs = []
    for item in results:
        A.require(type(item.get('scope')) is int and type(item.get('final_selection')) is bool,
                  'training job identity type differs')
        jobs.append((item['scope'], item['final_selection']))
        seconds = item.get('seconds')
        A.require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'invalid training time')
    A.require(len(set(jobs)) == len(jobs) and set(jobs) == expected, 'training job coverage differs')
    return {'status': 'passed', 'recorded_new_video_fits': 3 if final else 20, 'recorded_new_frame_fits': 0,
            'note': 'Job report metadata, not independent execution proof.'}


def pending_result(missing):
    return {'status': 'pending', 'missing_artifacts_at_start': sorted(str(x) for x in missing),
            'note': 'Not checked/passed; rerun after producer finishes using a NEW --output file.'}


def run_audit(root=ROOT):
    start = time.perf_counter()
    e = A.Evidence(root)
    # A finite one-shot snapshot; no polling or training synchronization.
    expected = [artifact_path(OUT, s).with_suffix(ext) for s in range(6) for ext in ('.json', '.npz')]
    expected += [OUT / f for f in ('training_report.json', 'final_training_report.json',
                                   'report.json', 'deployment_selection.json')]
    available = {}
    for rel in expected:
        target = A.safe_path(e.root, rel)
        A.require(not target.exists() or target.is_file(), 'artifact is not a file: ' + str(rel))
        available[rel] = target.is_file()
    missing = [p.as_posix() for p, present in available.items() if not present]
    observed_at = datetime.now(timezone.utc).isoformat()
    receipt, parent, baseline, raw, protocol = verify_lineage(e)
    source = source_contract(e)
    records, outer, aliases = A.load_records(e)
    real, names = verify_real_features(e, records, aliases, raw, receipt)
    decoders = A.decoder_functions()
    transforms = A.TransformEvidence(records)
    repaired, controls, all_fits, coverage = {}, {}, [], Counter()
    parent_fit_count = 0
    cache_states = []
    for scope in range(6):
        parent_ip, parent_op, parent_meta = P.verify_spatial_cache(e, scope, records, outer, parent, final=scope == 5)
        original_path = artifact_path(PARENT, scope)
        # Also verify dtype/range and strict parent NPZ binding on the consumed bytes.
        original_maps = A.probability_maps(e.read(original_path, parent_meta['npz_sha256']), parent_meta,
                                          records, original_path.as_posix())
        parent_fit_count += len(parent_meta['fit_evidence'])
        controls[scope] = A.load_cache(e, CONTROL, scope, records, outer, transforms, final=scope == 5)
        path = artifact_path(OUT, scope)
        pair = (path.with_suffix('.json'), path)
        absent = [p for p in pair if not available[p]]
        if absent:
            # Existing incomplete metadata is still fail-closed on its structural bindings.
            if available[pair[0]]:
                partial = e.json(pair[0])
                verify_repair_metadata(partial, scope, records, outer, receipt, parent_meta, names)
            if available[path]:
                A.read_archive(e.read(path), path.as_posix())
            cache_states.append({'scope': scope, **pending_result(absent)})
            continue
        ip, op, meta, proofs, counts = verify_repair_cache(e, scope, records, outer, receipt,
                                                         original_maps, parent_meta, names)
        A.compare(meta['inner_partitions'], controls[scope][2]['inner_partitions'], 'all scopes/control partitions')
        repaired[scope] = (ip, op, meta)
        all_fits.extend(proofs); coverage.update(counts)
        cache_states.append({'scope': scope, 'status': 'passed', 'video_fits': len(proofs),
                             'parent_frame_npz_sha256': parent_meta['npz_sha256'], 'bitwise_frame_coverage': counts})
    expected_counts = {'inner': sum(len(records) - len(v) for v in outer), 'outer': len(records),
                       'final_inner': len(records)}
    for key, expected_count in expected_counts.items():
        if (key == 'final_inner' and 5 in repaired) or (key != 'final_inner' and all(s in repaired for s in range(5))):
            A.compare(coverage[key], expected_count, 'frame tensor total/' + key)
    training_checks = {}
    for final, filename in ((False, 'training_report.json'), (True, 'final_training_report.json')):
        rel = OUT / filename
        training_checks[filename] = verify_training_report(e.json(rel), final) if available[rel] else pending_result([rel])
    rel = OUT / 'report.json'
    nested = verify_nested(e, e.json(rel), records, outer, repaired, controls, baseline, receipt, protocol, decoders) \
        if available[rel] else pending_result([rel])
    rel = OUT / 'deployment_selection.json'
    final = verify_final(e, e.json(rel), records, repaired, controls, receipt, decoders) \
        if available[rel] else pending_result([rel])
    e.read('code/audit_spatial_mask_repair_v13r.py')
    e.recheck()
    complete = not missing
    A.require(not complete or len(all_fits) == 23, 'complete audit does not cover all23 video fits')
    return {'schema_version': 'independent-spatial-mask-repair-v13r-audit-v1',
            'status': 'passed' if complete else 'pending', 'deployment_approved': False,
            'observed_at_utc': observed_at, 'snapshot_policy': 'one_shot_presence_at_start_no_polling',
            'missing_artifacts_at_start': missing, 'root': str(e.root),
            'receipt_sha256': receipt['receipt_sha256'], 'parent_receipt_sha256': parent['receipt_sha256'],
            'trust_anchor_timing': 'V13R/auditor pins observed at audit review, not independently pre-training',
            'rows': len(records), 'content_groups': len(A.content_groups(records, list(range(len(records))))),
            'aliases_preserved': aliases, 'source_contract': source,
            'real_mask_semantics': {'status': 'passed', 'rows': real, 'verified_rows': len(real),
                                    'float32_reference_tolerance': {'rtol': 2e-6, 'atol': 2e-6},
                                    'synthetic_empty_support': 'passed_not_a_real_cache_case'},
            'cache_scopes': cache_states, 'video_fit_proofs': all_fits,
            'video_fit_count': len(all_fits), 'expected_video_fits': 23,
            'transform_digests': {'repair_verified': len(all_fits), 'original_V13_verified': parent_fit_count,
                                  'control_verified': len(transforms.proofs),
                                  'repair_expected': A.value_hash(repaired_transform(names)), 'PCA': None},
            'frame_probability_bitwise': {'status': 'passed' if all(s in repaired for s in range(6)) else 'pending',
                                         'verified': dict(coverage), 'expected': expected_counts,
                                         'nested_contexts': coverage['inner'] + coverage['outer'],
                                         'note': '385 nested = 308 inner +77 outer; final inner =77, total462 tensors.'},
            'training_reports': training_checks, 'nested_replay': nested, 'final_replay': final,
            'model_states': 'recorded_hash_only_not_verified_or_fit_replayed',
            'portable_parity': 'recorded_errors_bounded_not_independently_replayed',
            'differing_consumed_files': 0, 'consumed_artifact_sha256': dict(sorted(e.hashes.items())),
            'consumed_artifact_absolute_paths': {p: str(A.safe_path(e.root, p)) for p in sorted(e.hashes)},
            'limitations': LIMITATIONS, 'elapsed_seconds': time.perf_counter() - start}


def write_exclusive(output, result):
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(output).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='NEW audit JSON file; parent must exist')
    parser.add_argument('--require-complete', action='store_true', help='exit2 on pending, while still writing evidence')
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('refuse audit overwrite: ' + str(args.output))
    if not args.output.parent.is_dir():
        raise FileNotFoundError('audit output parent must already exist')
    result = run_audit()
    write_exclusive(args.output, result)
    print(json.dumps({'status': result['status'], 'rows': result['rows'], 'video_fits': result['video_fit_count'],
                      'frame_bitwise': result['frame_probability_bitwise'],
                      'missing_artifacts': result['missing_artifacts_at_start'], 'deployment_approved': False}))
    return 2 if args.require_complete and result['status'] == 'pending' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (A.AuditError, KeyError, OSError) as exc:
        print('AUDIT FAILED (no success output): ' + str(exc), file=sys.stderr)
        sys.exit(1)
