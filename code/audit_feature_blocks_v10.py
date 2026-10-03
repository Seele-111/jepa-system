#!/usr/bin/env python3
"""Bounded, read-only audit of frozen feature-block ablation v10.

No fitting, extraction, services, polling, production selectors or metrics. Only
verified feature transforms, fixed decoders and boundary refinement are audited dependencies.
Matching, weighting, bootstrap, ranking, metrics and guards are independent.
Modes v10/v11/v12 require complete report.json and deployment_selection.json.
Use --output NEW.json; --experiment is an alias for --mode.
Every missing/inconsistent artifact fails; the output is created exclusively.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import types
from unittest.mock import patch
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('output/algorithm-opt-v10-blocks')
PARENT = Path('output/algorithm-opt-2026-10-02-v8')
OLD = Path('output/algorithm-opt-2026-10-02')
GROUPED = Path('output/algorithm-opt-2026-10-02-v2')
BASELINE = GROUPED / 'selection_audited/report.json'
SEED = 20261002
RECIPES = ('blocks_motion_et', 'blocks_rgb_motion_et',
           'blocks_corrected_motion_et', 'blocks_rgb_corrected_motion_et')
RICH = RECIPES[-1]
PARENT_RICH = 'event_compact_rgb_corrected_motion_et'
# Observed from the freeze BEFORE v10 report/deployment selection existed.
RECEIPT_PIN = '63087ac921d47fefeb8e0cebbe9914390b09a134a69a32a2b67e99d4bab35dcb'
PROTOCOL_PIN = 'ac9fc0e896bfe2736bfb4d0fc140b78689ad44b3ccce1232bdb273bdd8c3596d'
PARENT_PIN = '3fd139ebcab3cee2dd66a86f1aa211aad711e2459c1cd37720b871d650576643'
BASELINE_PIN = 'bd659ea747a6c662ced8fdeae41878c470930be93289dee317e0db26668b7202'
INPUTS_PIN = '013b8637f6d073261d438956def803e10beb8fb9d704657ff1f543ed334fd0ca'
SOURCES = (
    'run_feature_blocks_v10.py', 'optimized_feature_blocks_v10.py',
    'optimized_grouped_training.py', 'optimized_training_provenance.py',
    'run_event_compact_experiment_v8.py', 'optimized_event_compact_v8.py',
    'run_compact_experiment_v4.py', 'optimized_compact_features_v4.py',
    'optimized_compact_model_v4.py', 'optimized_duration_decoder_v4.py',
    'optimized_feature_view_v2.py', 'optimized_feature_view.py', 'optimized_locator.py',
    'optimized_recall_decoder.py', 'optimized_video_statistics.py',
    'run_recall_selection.py', 'run_optimization_selection.py',
    'run_algorithm_optimization.py', 'build_optimization_dataset.py',
    'optimized_event_training_v5.py', 'run_video_gate_experiment_v6.py',
    'optimized_video_gate_v6.py',
)
GUARD_SPEC = {'f1_03_no_worse': 0, 'f1_05_delta': .02, 'frame_delta': -.005,
              'normal_fp_delta': 0, 'positive_empty_delta': -3, 'parity_max': 2e-6}
ATOL = 1e-12
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
LIMITATIONS = [
    'Repeatedly inspected development data; not a blind or confirmatory result.',
    'SHA grouping proves exact-content isolation, not near-duplicate/source/scene isolation.',
    'Original alias labels, including conflicts, are preserved; only 14 normal rows.',
    'Historical raw RGB/JEPA feature origin is not recovered by audit-time hashing.',
    'Fit caches contain content lists/transform digests, not per-fit estimators or PCA states. '
    'Reconstructed covariance/digest consistency and static source review do not prove '
    'every historical estimator actually executed that code or recover omitted fit logs.',
    'PCA digests require the original numerical runtime; a mismatch fails, not silently passes.',
    'New fit metadata signatures are unkeyed consistency hashes, not authentication or '
    'pre-selection commitments. Coordinated rewriting of valid probabilities, metadata and '
    'reports cannot be excluded without independently recorded training-time artifact pins.',
    'Production transforms and decoders are verified audited dependencies, not independent '
    'implementations. Statistical replay, matching and ranking are independent.',
    'No portable/fresh-pixel parity or latency audit here; statistical guard success '
    'is not deployment approval. The protocol parity_max guard remains unverified.',
]


class AuditError(ValueError):
    pass


class MissingArtifact(AuditError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def value_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8'))


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def strict_json(data, label):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + label + '/' + key)
            result[key] = value
        return result
    def invalid(token):
        raise AuditError('nonfinite JSON constant: ' + label + '/' + token)
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError('invalid JSON: ' + label) from exc


def safe_path(root, relative):
    text = relative.as_posix() if isinstance(relative, Path) else str(relative)
    p, w = PurePosixPath(text), PureWindowsPath(text)
    require(text and not p.is_absolute() and not w.drive and not w.is_absolute()
            and '\\' not in text and ':' not in text and '..' not in p.parts,
            'unsafe artifact path: ' + text)
    target = (Path(root) / text).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'artifact escapes root: ' + text)
    return target


class Evidence:
    """Hash the consumed bytes, deny escaping paths, recheck every read at end."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def read(self, relative, expected=None):
        path = safe_path(self.root, relative)
        key = path.relative_to(self.root).as_posix()
        try:
            with path.open('rb') as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise MissingArtifact('missing/unreadable artifact: ' + key) from exc
        require(len(data) <= MAX_FILE_BYTES, 'file bound exceeded: ' + key)
        actual = sha(data)
        if expected is not None:
            require(is_hash(expected) and actual == expected, 'SHA256 mismatch: ' + key)
        if key in self.hashes:
            require(self.hashes[key] == actual, 'artifact changed during audit: ' + key)
        self.hashes[key] = actual
        return data

    def json(self, relative, expected=None):
        return strict_json(self.read(relative, expected), str(relative))

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.read(path, digest)


def read_archive(data, label):
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            files = archive.infolist()
            names = [entry.filename for entry in files]
            require(0 < len(files) <= 512 and len(set(names)) == len(names),
                    'duplicate/excess archive members: ' + label)
            require(sum(entry.file_size for entry in files) <= MAX_ARCHIVE_BYTES,
                    'archive expansion bound exceeded: ' + label)
            require(all(n.endswith('.npy') and '/' not in n and '\\' not in n for n in names),
                    'invalid archive member: ' + label)
            for entry in files:
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    require(version in ((1, 0), (2, 0)), 'unsupported NPY header: ' + label)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(stream, max_header_size=10000)
                    require(not dtype.hasobject and all(type(x) is int and x >= 0 for x in shape),
                            'object/invalid NPY shape: ' + label)
                    count = math.prod(shape)
                    require(count <= MAX_ARCHIVE_BYTES and count * dtype.itemsize <= MAX_ARCHIVE_BYTES
                            and count * dtype.itemsize == entry.file_size - stream.tell(),
                            'NPY allocation/payload bound differs: ' + label)
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        require(all(x.dtype.kind != 'O' for x in arrays.values()), 'object archive: ' + label)
        return arrays
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError('invalid NPZ: ' + label) from exc


def compare(actual, expected, label):
    """Strict structure/type equality, finite floating values within absolute 1e-12."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'keys differ: ' + label)
        for key in expected:
            compare(actual[key], expected[key], label + '/' + str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), 'length differs: ' + label)
        for j, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, label + '/' + str(j))
    elif isinstance(expected, (float, np.floating)):
        require(type(actual) in (int, float) and math.isfinite(actual) and
                math.isfinite(expected) and abs(actual - expected) <= ATOL, 'number differs: ' + label)
    elif isinstance(expected, (int, np.integer)) and not isinstance(expected, (bool, np.bool_)):
        require(type(actual) is int and actual == expected, 'integer differs: ' + label)
    else:
        require(type(actual) is type(expected) and actual == expected, 'value differs: ' + label)


def configurations():
    result = []
    for threshold in (.3, .4, .5, .6):
        for penalty in (.03, .10, .25):
            for vt in (0, .7):
                result.append({'kind': 'duration-logit-v4', 'threshold': threshold,
                    'transition_seconds': penalty, 'min_seconds': .12, 'video_threshold': vt})
        for ratio in (1., .7):
            for vt in (0, .7):
                result.append({'kind': 'recall-stable-v2', 'threshold': threshold, 'low_ratio': ratio,
                    'smooth_seconds': .15, 'gap_seconds': .12, 'min_seconds': .12,
                    'seed_seconds': .05, 'video_threshold': vt, 'video_strength': 0, 'video_floor': 0})
    return result


def label_spans(labels):
    result, start = [], None
    for position, positive in enumerate(np.asarray(labels).reshape(-1)):
        if positive and start is None:
            start = position
        elif not positive and start is not None:
            result.append((start, position - 1)); start = None
    if start is not None:
        result.append((start, len(labels) - 1))
    return result


def segments(value, length, label):
    require(isinstance(value, (list, tuple)), 'invalid segments: ' + label)
    result = []
    for pair in value:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2 and
                all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in pair),
                'invalid segment endpoints: ' + label)
        a, b = map(int, pair)
        require(0 <= a <= b < length, 'out-of-bounds segment: ' + label)
        result.append((a, b))
    return result


def row_statistics(prediction, labels):
    """12 sufficient stats: greedy inclusive matching; frame coverage is a UNION.

    Prediction order is significant. Equal-IoU matches choose the first unused
    GT event. This is the frozen legacy metric, not maximum bipartite matching.
    """
    labels = np.asarray(labels, dtype=bool)
    pred = segments(prediction, len(labels), 'statistics')
    truth = label_spans(labels)
    stats = []
    for cutoff in (.3, .5):
        claimed = set()
        for left, right in pred:
            overlaps = []
            for j, (a, b) in enumerate(truth):
                if j not in claimed:
                    shared = max(0, min(right, b) - max(left, a) + 1)
                    overlaps.append((shared / (right - left + b - a + 2 - shared), j))
            if overlaps:
                overlap, j = max(overlaps, key=lambda pair: (pair[0], -pair[1]))
                if overlap >= cutoff:
                    claimed.add(j)
        stats.extend((len(claimed), len(pred) - len(claimed), len(truth) - len(claimed)))
    covered = np.zeros(len(labels), dtype=bool)
    for a, b in pred:
        covered[a:b + 1] = True
    stats.extend((int(np.count_nonzero(covered & labels)), int(np.count_nonzero(covered & ~labels)),
                  int(np.count_nonzero(~covered & labels)), int(not truth and bool(pred)),
                  int(bool(truth) and not pred), len(pred)))
    return np.asarray(stats, dtype=np.int64)


def content_groups(records, ids):
    groups = {}
    for i in ids:
        groups.setdefault(records[i]['sha256'], []).append(i)
    return list(groups.values())


def content_weights(records, ids):
    counts = Counter(records[i]['sha256'] for i in ids)
    return np.asarray([1. / counts[records[i]['sha256']] for i in ids], dtype=np.float64)


def bootstrap_counts(records, ids, seed, replicates=32):
    require(0 < replicates <= 2000, 'bootstrap bound exceeded')
    grouped_positions = {}
    for position, i in enumerate(ids):
        grouped_positions.setdefault(records[i]['sha256'], []).append(position)
    strata = {}
    for positions in grouped_positions.values():
        event_class = min(2, int(records[ids[positions[0]]]['event_count']))
        strata.setdefault(event_class, []).append(positions)
    rng = np.random.default_rng(seed)
    result = np.zeros((replicates, len(ids)), dtype=np.int64)
    # Preserve first-occurrence stratum order: it is part of the frozen RNG protocol.
    for groups in strata.values():
        for row in range(replicates):
            sampled = rng.integers(0, len(groups), size=len(groups))
            multiplicities = np.bincount(sampled, minlength=len(groups))
            for group, count in zip(groups, multiplicities):
                result[row, group] = count
    return result


def utility(stats, normals, positives):
    stats = np.asarray(stats, dtype=np.float64)
    rates = []
    for start in (0, 3, 6):
        tp, fp, fn = (stats[..., start + k] for k in range(3))
        rates.append(2 * tp / np.maximum(1e-12, 2 * tp + fp + fn))
    fp_rate = np.divide(stats[..., 9], normals, out=np.zeros_like(stats[..., 9]),
                        where=np.asarray(normals) > 0)
    empty_rate = np.divide(stats[..., 10], positives, out=np.zeros_like(stats[..., 10]),
                           where=np.asarray(positives) > 0)
    return .35 * rates[0] + .65 * rates[1] + .10 * rates[2] - .30 * fp_rate - .30 * empty_rate


@dataclass(frozen=True)
class Probabilities:
    frame: dict
    video: dict


class RestrictedRecords:
    """Make an outer-record read during inner selection an explicit audit error."""
    def __init__(self, records, ids):
        self.records, self.allowed = records, frozenset(ids)

    def __getitem__(self, i):
        require(i in self.allowed, 'selection attempted held-out record access: ' + str(i))
        return self.records[i]


def decoder_functions():
    # Imported only AFTER pins, actual sources and dependency snapshots are checked.
    from optimized_duration_decoder_v4 import decode_duration
    from optimized_recall_decoder import decode_v2
    return decode_duration, decode_v2


def decode(records, ids, p, cfg, decoders):
    decoder = decoders[0] if cfg['kind'] == 'duration-logit-v4' else decoders[1]
    return {i: segments(decoder(p.frame[i], records[i]['fps'], p.video[i], cfg),
                        records[i]['frames'], 'decoder/' + str(i)) for i in ids}


def independent_choice(records, ids, p, seed, decoders, configs=None):
    require(ids == sorted(set(ids)) and ids, 'invalid selection ids')
    require(set(p.frame) == set(p.video) == set(ids), 'selection probability coverage differs')
    restricted = RestrictedRecords(records, ids)
    weights = content_weights(restricted, ids)
    normals = np.asarray([not np.any(restricted[i]['labels']) for i in ids], dtype=np.float64)
    positives = 1 - normals
    bootstrap = bootstrap_counts(restricted, ids, seed) * weights[None, :]
    normal_mass, positive_mass = weights @ normals, weights @ positives
    bnormals, bpositives = bootstrap @ normals, bootstrap @ positives
    grid = configurations() if configs is None else configs
    require(0 < len(grid) <= 40, 'decoder configuration bound exceeded')
    seen, rows = set(), []
    for ordinal, cfg in enumerate(grid):
        predictions = decode(restricted, ids, p, cfg, decoders)
        fingerprint = tuple(tuple(predictions[i]) for i in ids)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        matrix = np.stack([row_statistics(predictions[i], restricted[i]['labels']) for i in ids])
        total = weights @ matrix
        pooled = float(utility(total, normal_mass, positive_mass))
        f05 = float(2 * total[3] / max(1e-12, 2 * total[3] + total[4] + total[5]))
        rows.append({'config': deepcopy(cfg), 'stats': total.tolist(), 'pooled_utility': pooled,
                     'key': [pooled, f05, -total[10], -total[9], -total[11], -ordinal], 'matrix': matrix})
    rows = sorted(rows, key=lambda row: row['key'], reverse=True)[:8]
    require(rows, 'empty decoder shortlist')
    for row in rows:
        q20 = float(np.quantile(utility(bootstrap @ row.pop('matrix'), bnormals, bpositives),
                                 .2, method='linear'))
        row.update(bootstrap_q20=q20, stable_utility=.75 * row['pooled_utility'] + .25 * q20)
    chosen = deepcopy(max(rows, key=lambda row: (row['stable_utility'], *row['key'][1:])))
    chosen['shortlist'] = rows
    return chosen


def rank_candidates(rows):
    require([row['candidate'] for row in rows] == list(RECIPES), 'candidate order/coverage differs')
    return max(rows, key=lambda row: (row['stable_utility'], *row['key'][1:],
                                      -RECIPES.index(row['candidate'])))


def select_inner(records, ids, probabilities, scope, decoders):
    require(set(probabilities) == set(RECIPES), 'inner recipe coverage differs')
    rows = []
    for recipe in RECIPES:
        choice = independent_choice(records, ids, probabilities[recipe], SEED + scope, decoders)
        choice['candidate'] = recipe; rows.append(choice)
    return rank_candidates(rows), rows


def verify_receipts(e):
    receipt = e.json(OUT / 'training_receipt.json')
    require(receipt.get('receipt_sha256') == RECEIPT_PIN and
            value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) == RECEIPT_PIN,
            'v10 receipt differs from pre-report trust anchor')
    parent = e.json(PARENT / 'training_receipt.json')
    require(parent.get('receipt_sha256') == PARENT_PIN and
            value_hash({k: v for k, v in parent.items() if k != 'receipt_sha256'}) == PARENT_PIN,
            'v8 receipt differs from trust anchor')
    require(receipt['parent_receipt_sha256'] == PARENT_PIN, 'parent receipt lineage differs')
    for section, parent_section in (('input_hashes', 'input_hashes'),
                                    ('input_sources', 'input_source_hashes')):
        require(receipt[section] == parent[parent_section], 'input lineage differs: ' + section)
        for path, digest in receipt[section].items():
            e.read(path, digest)
    require(len(receipt['input_hashes']) == 479 and len(receipt['input_sources']) == 11 and
            receipt['inputs_sha256'] == INPUTS_PIN == parent['inputs_sha256'] ==
            value_hash(receipt['input_hashes']), 'input inventory count/digest differs')
    require(set(receipt['sources']) == set(SOURCES), 'v10 source inventory differs')
    for name, digest in receipt['sources'].items():
        e.read(OUT / 'sources' / name, digest)
        e.read(Path('code') / name, digest)
    for name, digest in parent['sources'].items():
        require(receipt['sources'].get(name) == digest, 'parent source lineage differs: ' + name)
        e.read(PARENT / 'sources' / name, digest)
    artifacts = {(PARENT / 'training' / f'{scope}_{PARENT_RICH}.{ext}').as_posix()
                 for scope in range(5) for ext in ('json', 'npz')}
    require(set(receipt['parent_artifacts']) == artifacts, 'five reused v8 caches coverage differs')
    for path, digest in receipt['parent_artifacts'].items():
        e.read(path, digest)
    require(receipt['protocol_sha256'] == PROTOCOL_PIN, 'v10 protocol anchor differs')
    protocol = e.json(OUT / 'protocol.json', PROTOCOL_PIN)
    require(protocol.get('schema_version') == 'feature-block-ablation-v10' and
            protocol.get('role') == 'repeated_development_not_blind_or_confirmatory', 'protocol identity differs')
    compare(protocol['recipes'], list(RECIPES), 'protocol/recipes')
    compare(protocol['decoder_configs'], configurations(), 'protocol/decoder_configs')
    compare(protocol['promotion_guards'], GUARD_SPEC, 'protocol/promotion_guards')
    for key, expected in (('rows', 77), ('content_groups', 76), ('nested_new_member_fits', 60),
                          ('reused_rich_member_fits', 20), ('final_selection_new_member_fits', 12),
                          ('reference_recipe', PARENT_RICH)):
        compare(protocol[key], expected, 'protocol/' + key)
    compare(protocol['seeds'], {'inner_split': '20261002+scope*31',
        'inner_fit': '20261002+scope*53+k', 'outer_fit': '20261002+scope*53+99',
        'selection': '20261002+scope', 'final_scope': 5}, 'protocol/seeds')
    require(receipt['baseline_sha256'] == BASELINE_PIN == parent['baseline_report_sha256'],
            'baseline lineage differs')
    baseline = e.json(BASELINE, BASELINE_PIN)['grouped_v1_baseline']
    require(set(receipt['default_models']) == {'optimized_locator_v1.json', 'optimized_motion_locator_v1.json'},
            'default model inventory differs')
    for name, digest in receipt['default_models'].items():
        e.read(Path('models') / name, digest)
    return receipt, baseline


def source_review(e):
    def tree(name):
        return ast.parse(e.read(Path('code') / name).decode('utf-8-sig'))
    new = tree('optimized_feature_blocks_v10.py')
    old = tree('optimized_compact_features_v4.py')
    parent = tree('optimized_event_compact_v8.py')
    def function(module, name):
        matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(matches) == 1, 'source function missing/duplicate: ' + name)
        return deepcopy(matches[0])
    fresh = function(new, 'feature_view_blocks')
    legacy = function(old, 'feature_view_v4')
    fresh.name = legacy.name
    class ViewAdapter(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'BASE_RECIPES':
                require(isinstance(node.slice, ast.Name) and node.slice.id == 'recipe', 'view adapter differs')
                return ast.copy_location(ast.Name(id='recipe', ctx=ast.Load()), node)
            return self.generic_visit(node)
    fresh = ViewAdapter().visit(fresh)
    # The rejected-recipe error wording is not a numerical transform difference.
    for node in ast.walk(fresh):
        if isinstance(node, ast.Constant) and node.value == 'unknown feature-block recipe':
            node.value = 'unknown compact recipe'
    require(ast.dump(fresh, include_attributes=False) == ast.dump(legacy, include_attributes=False),
            'v10 transform is not exactly v4 plus the declared recipe adapter')
    fitter = function(new, 'fit_block_member')
    old_fitter = function(parent, 'fit_event_compact_member')
    def calls(node, name):
        return [x for x in ast.walk(node) if isinstance(x, ast.Call) and
                isinstance(x.func, ast.Name) and x.func.id == name]
    require(sorted(ast.dump(c, include_attributes=False) for c in calls(fitter, 'ExtraTreesClassifier')) ==
            sorted(ast.dump(c, include_attributes=False) for c in calls(old_fitter, 'ExtraTreesClassifier')),
            'ET frame/video parameters differ from v8 rich branch')
    require(len(calls(fitter, 'event_weights')) == 1 and
            ast.unparse(calls(fitter, 'event_weights')[0]) == 'event_weights(records, train)',
            'frame objective is not unchanged v8/v5 train-only event weighting')
    assignments = {n.targets[0].id: n.value for n in fitter.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    require(ast.unparse(assignments['pca']) == "fit_rgb_pca(records, train) if 'rgb' in recipe else None",
            'RGB PCA is not train-only and block-conditional')
    require(ast.unparse(assignments['vw']) ==
            'np.asarray([cw[i] * (2 if vy[j] == 0 else 1) for j, i in enumerate(train)])',
            'normal video mass differs from v8')
    fits = [ast.unparse(n) for n in ast.walk(fitter) if isinstance(n, ast.Call) and
            isinstance(n.func, ast.Attribute) and n.func.attr == 'fit']
    require(fits == ['model.fit(x, y, sample_weight=weights)', 'vm.fit(video, vy, sample_weight=vw)'],
            'unexpected estimator fitting inputs')
    pca_code = function(old, 'fit_rgb_pca')
    require(not any(isinstance(n, ast.Constant) and n.value == 'labels' for n in ast.walk(pca_code)),
            'PCA source accesses labels')
    runner = tree('run_feature_blocks_v10.py')
    final = function(runner, 'final_select')
    direct_calls = [n for n in ast.walk(final) if isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Name) and n.func.id == 'select_inner']
    require(len(direct_calls) == 1 and ast.unparse(direct_calls[0]) ==
            'select_inner(records, ids, probabilities, 5)', 'final selection does not use direct inner3 scope5')
    # This is static compatibility review, not a training-time execution attestation.
    return {'transform_AST_equal_after_recipe_adapter_and_error_wording': True, 'ET_parameters_equal_to_v8': True,
            'frame_event_weights_unchanged': True, 'normal_video_mass': 2,
            'PCA_source_train_only': True, 'final_selector_direct_inner3': True}


def seeded_folds(records, groups, count, seed):
    strata = {}
    for position, group in enumerate(groups):
        r = records[group[0]]
        event = 'normal' if r['event_count'] == 0 else 'multi' if r['event_count'] > 1 else 'single'
        strata.setdefault((r['generator'], event), []).append(position)
    rng, folds, offset = np.random.default_rng(seed), [[] for _ in range(count)], 0
    for key in sorted(strata):
        positions = rng.permutation(strata[key])
        for j, position in enumerate(positions):
            folds[(offset + j) % count].extend(groups[int(position)])
        offset = (offset + len(positions)) % count
    return [sorted(fold) for fold in folds]


def check_partition(records, fit, validation, universe, label):
    for ids in (fit, validation, universe):
        require(isinstance(ids, list) and ids and all(type(i) is int and 0 <= i < len(records) for i in ids)
                and ids == sorted(set(ids)), 'invalid/duplicate partition ids: ' + label)
    require(set(fit).isdisjoint(validation) and set(fit) | set(validation) == set(universe),
            'partition coverage/overlap differs: ' + label)
    require(not {records[i]['sha256'] for i in fit} & {records[i]['sha256'] for i in validation},
            'SHA content leakage / alias split: ' + label)


def expected_partitions(records, train, scope):
    groups = content_groups(records, train)
    folds = seeded_folds(records, groups, 3, SEED + scope * 31)
    require(Counter(i for fold in folds for i in fold) == Counter(train), 'inner OOF coverage differs')
    result = []
    for k, val in enumerate(folds):
        fit = [i for i in train if i not in set(val)]
        check_partition(records, fit, val, train, 'inner/' + str(scope) + '/' + str(k))
        result.append({'fit': fit, 'validation': val, 'seed': SEED + scope * 53 + k})
    return result


def load_records(e):
    grouped = e.json(GROUPED / 'dataset_manifest.json')
    old = e.json(OLD / 'dataset_manifest.json')
    rows = grouped.get('rows')
    require(grouped.get('schema_version') == 'algorithm-opt-grouped-dataset-v2' and
            isinstance(rows, list) and len(rows) == 77 and rows == old['rows'], 'dataset rows/lineage differ')
    require(grouped['v1_manifest_sha256'] == e.hashes[(OLD / 'dataset_manifest.json').as_posix()],
            'original/grouped manifest binding differs')
    for key, value in (('outer_folds', 5), ('inner_folds', 3), ('seed', SEED)):
        compare(grouped['protocol'][key], value, 'dataset/protocol/' + key)
    arrays = read_archive(e.read(OLD / 'dataset.npz'), 'base dataset')
    require(set(arrays) == {f'{key}_{i}' for i in range(77) for key in ('labels', 'jepa', 'rank_motion')},
            'base dataset key coverage differs')
    records = []
    for i, row in enumerate(rows):
        y = arrays[f'labels_{i}']
        require(is_hash(row.get('sha256')) and type(row.get('frames')) is int and row['frames'] > 0
                and isinstance(row.get('name'), str) and isinstance(row.get('generator'), str)
                and type(row.get('fps')) in (int, float) and math.isfinite(row['fps']) and row['fps'] > 0,
                'invalid dataset row metadata: ' + str(i))
        require(y.shape == (row['frames'],) and y.dtype.kind in 'biuf' and np.isin(y, [0, 1]).all(),
                'invalid labels: ' + str(i))
        compare(row['event_count'], len(label_spans(y)), 'dataset/event_count/' + str(i))
        records.append({**row, 'labels': y.astype(bool)})
    groups = content_groups(records, list(range(77)))
    require(len(groups) == 76 and grouped['content_groups'] == groups and
            len({r['name'] for r in records}) == 77, 'content groups/name coverage differs')
    aliases = []
    for group in groups:
        if len(group) > 1:
            require(len({(records[i]['frames'], records[i]['fps']) for i in group}) == 1,
                    'content aliases have inconsistent frames/FPS')
            first = records[group[0]]['labels']
            conflict = np.logical_or.reduce([records[i]['labels'] != first for i in group])
            aliases.append({'indices': group, 'sha256': records[group[0]]['sha256'],
                            'conflicting_label_frames': np.flatnonzero(conflict).tolist()})
    original = seeded_folds(records, [[i] for i in range(77)], 5, SEED)
    require(old['outer_folds'] == original, 'original seeded row folds do not replay')
    owner = {i: f for f, fold in enumerate(original) for i in fold}
    outer = [[] for _ in range(5)]
    for group in groups:
        outer[owner[group[0]]].extend(group)
    outer = [sorted(fold) for fold in outer]
    compare(grouped['outer_folds'], outer, 'SHA-consolidated outer folds')
    require(Counter(i for fold in outer for i in fold) == Counter(range(77)), 'outer OOF coverage differs')
    for scope, val in enumerate(outer):
        train = [i for i in range(77) if i not in set(val)]
        check_partition(records, train, val, list(range(77)), 'outer/' + str(scope))
        expected_partitions(records, train, scope)
    folders = [('motion', OLD / 'motion_features'), ('rgb', OLD / 'rgb_features'),
               ('corrected', OLD / 'corrected_jepa_features_v2'), ('local', GROUPED / 'local_motion')]
    for key, folder in folders:
        manifest = e.json(folder / 'manifest.json')
        entries = manifest['videos']
        require(isinstance(entries, list) and len(entries) == (79 if key == 'rgb' else 77),
                'feature inventory coverage differs: ' + key)
        if key == 'rgb':
            # The frozen RGB card retains two explicitly missing, unselected rows.
            require(manifest.get('summary', {}).get('run_status') == 'completed_selected_inventory',
                    'RGB selected inventory incomplete')
            require(len({entry.get('name') for entry in entries}) == 79, 'duplicate RGB inventory names')
            omitted = [entry for entry in entries if entry.get('selected') is not True]
            require(len(omitted) == 2 and all(entry.get('selected') is False and
                    entry.get('status') == 'missing' for entry in omitted),
                    'RGB exclusion evidence differs')
            entries = [entry for entry in entries if entry.get('selected') is True]
        if key == 'local':
            require([entry['row_id'] for entry in entries] == list(range(77)), 'local row order differs')
            by_name = dict(zip([r['name'] for r in records], entries))
        else:
            by_name = {entry.get('name', entry.get('video_name')): entry for entry in entries}
        require(set(by_name) == {r['name'] for r in records}, 'feature name coverage differs: ' + key)
        names = None if key == 'rgb' else e.json(folder / 'feature_names.json')
        for i, record in enumerate(records):
            entry = by_name[record['name']]
            require(entry.get('status') in ('ok', 'done', 'success', 'complete'), 'incomplete feature: ' + key)
            if key != 'rgb':
                require(entry['source_sha256'] == record['sha256'], 'feature source SHA differs: ' + key)
            file = folder / entry.get('file', entry.get('npz_path'))
            array = read_archive(e.read(file, entry.get('feature_sha256')), file.as_posix())
            signal = 'features' if key == 'rgb' else 'signals'
            x = np.asarray(array[signal], dtype=np.float32)
            require(x.ndim == 2 and x.shape[0] == record['frames'] and x.shape[1] > 0 and
                    np.isfinite(x).all(), 'invalid/misaligned raw feature: ' + key + '/' + str(i))
            require(array['fps'].shape == () and float(array['fps']) == record['fps'], 'feature FPS differs')
            if names is not None:
                require(isinstance(names, list) and len(names) == x.shape[1] and
                        all(isinstance(n, str) for n in names), 'raw feature names differ: ' + key)
            if key == 'local':
                mask = array['feature_valid']
                require(mask.shape == x.shape == (record['frames'], 142) and
                        np.isin(mask, [0, 1]).all() and array['feature_names'].tolist() == names,
                        'local observation mask/schema differs')
                record[key] = np.concatenate([x, mask.astype(np.float32)], axis=1)
                record[key + '_names'] = ['local/' + n for n in names] + [
                    'local/' + n + '_support_flag' for n in names]
            else:
                record[key] = x
                if names is not None:
                    record[key + '_names'] = names
    require(sum(r['frames'] for r in records) == 6133 and sum(r['event_count'] for r in records) == 85 and
            sum(not r['labels'].any() for r in records) == 14, 'dataset aggregate identity differs')
    return records, outer, aliases


def reconstruct_pca(records, fit):
    """Reconstruct ONLY bound train-content covariance (not a classifier fit).

    The arithmetic order preserves the frozen PCA serialization contract. Full
    eigenstate digests, rather than a self-declared role string, are compared.
    """
    restricted = RestrictedRecords(records, fit)
    weights = content_weights(restricted, fit)
    width = restricted[fit[0]]['rgb'].shape[1]
    mean, second = np.zeros(width, np.float64), np.zeros((width, width), np.float64)
    mass = sum(float(w) for w in weights)
    for i, weight in zip(fit, weights):
        x = np.asarray(restricted[i]['rgb'], dtype=np.float64)
        require(x.ndim == 2 and len(x) > 0 and x.shape[1] == width and np.isfinite(x).all(),
                'invalid PCA training input')
        w = weight / (mass * len(x))
        mean += w * x.sum(axis=0)
        second += w * x.T @ x
    covariance = second - np.outer(mean, mean)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) * .5)
    directions = vectors[:, np.argsort(eigenvalues)[::-1][:min(16, width)]].T
    for direction in directions:
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
    return {'mean': mean.astype(np.float32).tolist(), 'components': directions.astype(np.float32).tolist(),
            'fit_role': 'training_content_only_equal_content_covariance'}


class TransformEvidence:
    def __init__(self, records):
        # Import only verified transform modules; never import either experiment runner.
        from optimized_feature_blocks_v10 import feature_view_blocks
        from optimized_compact_features_v4 import feature_view_v4
        self.records, self.fresh, self.legacy = records, feature_view_blocks, feature_view_v4
        self.pcas, self.proofs = {}, []

    def verify(self, recipe, fit, evidence, *, parent=False):
        key = tuple(fit)
        rgb = 'rgb' in recipe
        if rgb and key not in self.pcas:
            self.pcas[key] = reconstruct_pca(self.records, fit)
        pca = self.pcas.get(key) if rgb else None
        first = min(fit)
        view_recipe = 'compact_rgb_corrected_motion_et' if parent and recipe == RICH else recipe
        x, video, names = (self.legacy if parent else self.fresh)(view_recipe, self.records[first], pca)
        transform = {'feature_view': 'compact-fps-context-v4' if parent else 'compact-feature-blocks-v10',
                     'pca': pca, 'frame_feature_names': names}
        require(is_hash(evidence.get('transform_sha256')) and
                value_hash(transform) == evidence['transform_sha256'],
                'reconstructed transform/PCA digest differs: ' + recipe + '/' + str(evidence['partition']))
        if not parent:
            compare(evidence['frame_features'], x.shape[1], 'fit/frame_features')
            compare(evidence['video_features'], len(video), 'fit/video_features')
        require(not any(n.startswith('rgb/') for n in names) or rgb, 'omitted RGB block still present')
        require(not any(n.startswith('corrected/') for n in names) or 'corrected' in recipe,
                'omitted JEPA block still present')
        self.proofs.append({'recipe': recipe, 'partition': evidence['partition'],
            'fit_content_sha256': sorted({self.records[i]['sha256'] for i in fit}),
            'PCA': 'train_covariance_digest_reconstructed' if rgb else 'absent',
            'transform_sha256': value_hash(transform), 'frame_features': int(x.shape[1]),
            'video_features': int(len(video))})

    def compatibility(self, fit):
        pca = self.pcas.get(tuple(fit))
        require(pca is not None, 'rich compatibility lacks reconstructed PCA')
        for i, record in enumerate(self.records):
            fresh = self.fresh(RICH, record, pca)
            legacy = self.legacy('compact_rgb_corrected_motion_et', record, pca)
            require(fresh[2] == legacy[2] and np.array_equal(fresh[0], legacy[0]) and
                    np.array_equal(fresh[1], legacy[1]), 'rich v10/v8 transform parity differs: ' + str(i))
        return {'rows': len(self.records), 'frame_and_video_features_bitwise_equal': True,
                'feature_names_exact_equal': True, 'context': 'first outer inner-fit covariance'}


def verify_training_metadata(metadata, recipe, scope, records, outer, *, parent=False, final=False, parent_recipe=None, parent_receipt=None):
    require(isinstance(metadata, dict), 'fit metadata is not an object')
    signature = metadata.get('signature')
    require(is_hash(signature) and value_hash({k: v for k, v in metadata.items() if k != 'signature'}) == signature,
            'fit metadata signature differs')
    compare(metadata.get('recipe'), (parent_recipe or PARENT_RICH) if parent else recipe, 'fit/recipe')
    compare(metadata.get('fold' if parent else 'scope'), scope, 'fit/scope')
    if not parent:
        compare(metadata.get('final_selection'), final, 'fit/final_selection')
    val = [] if final else outer[scope]
    train = [i for i in range(len(records)) if i not in set(val)]
    expected = expected_partitions(records, train, scope)
    compare(metadata.get('train'), train, 'fit/train')
    compare(metadata.get('validation'), val, 'fit/validation')
    compare(metadata.get('inner_partitions'), expected, 'fit/inner_partitions')
    if not final:
        check_partition(records, train, val, list(range(len(records))), 'cached outer/' + str(scope))
    compare(metadata.get('outer_seed'), None if final else SEED + scope * 53 + 99, 'fit/outer_seed')
    compare(metadata.get('training_receipt_sha256'), (parent_receipt or PARENT_PIN) if parent else RECEIPT_PIN, 'fit/receipt')
    proofs = metadata.get('fit_evidence')
    parts = expected + ([] if final else [{'fit': train, 'validation': val}])
    require(isinstance(proofs, list) and len(proofs) == len(parts), 'fit evidence coverage differs')
    for k, (proof, part) in enumerate(zip(proofs, parts)):
        compare(proof.get('partition'), k if k < 3 else 'outer', 'fit/evidence partition')
        a, b = ({records[i]['sha256'] for i in part[side]} for side in ('fit', 'validation'))
        require(not a & b, 'fit evidence SHA leakage')
        compare(proof.get('fit_content_sha256'), sorted(a), 'fit/content/PCA training set')
        require(is_hash(proof.get('transform_sha256')), 'missing/invalid transform hash')
    return train, val, parts


def probability_maps(data, metadata, records, label):
    require(is_hash(metadata.get('npz_sha256')) and sha(data) == metadata['npz_sha256'],
            'fit NPZ SHA256 differs: ' + label)
    arrays = read_archive(data, label)
    train, val = metadata['train'], metadata['validation']
    expected = {f'{kind}_{head}_{i}' for kind, ids in (('inner', train), ('outer', val))
                for i in ids for head in ('frame', 'video')}
    require(set(arrays) == expected, 'probability key / OOF coverage differs: ' + label)
    result = []
    for kind, ids in (('inner', train), ('outer', val)):
        frame, video = {}, {}
        for i in ids:
            for head, shape in (('frame', (records[i]['frames'],)), ('video', ())):
                array = arrays[f'{kind}_{head}_{i}']
                require(array.dtype.kind in 'iuf' and array.shape == shape and np.isfinite(array).all()
                        and np.all((array >= 0) & (array <= 1)),
                        'invalid probability tensor: ' + label + '/' + kind + '/' + head + '/' + str(i))
                if head == 'frame':
                    frame[i] = array
                else:
                    video[i] = float(array)
        result.append(Probabilities(frame, video))
    return tuple(result)


def load_cache(e, recipe, scope, records, outer, transforms, *, final=False):
    parent = recipe == RICH and not final
    folder = PARENT if parent else OUT
    prefix = 'final_selection' if final else 'training'
    identifier = PARENT_RICH if parent else recipe
    path = folder / prefix / f'{scope}_{identifier}.npz'
    metadata = e.json(path.with_suffix('.json'))
    train, val, parts = verify_training_metadata(metadata, recipe, scope, records, outer, parent=parent, final=final)
    data = e.read(path)
    inner, held = probability_maps(data, metadata, records, path.as_posix())
    for proof, part in zip(metadata['fit_evidence'], parts):
        transforms.verify(recipe, part['fit'], proof, parent=parent)
    require(set(inner.frame) == set(inner.video) == set(train) and
            set(held.frame) == set(held.video) == set(val), 'cache head coverage differs')
    return inner, held, metadata


def metric_dict(stats, normals):
    result = {}
    for key, start in (('iou_0.3', 0), ('iou_0.5', 3)):
        tp, fp, fn = map(int, stats[start:start + 3])
        result[key] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(1, tp + fp),
                       'recall': tp / max(1, tp + fn), 'f1': 2 * tp / max(1, 2 * tp + fp + fn)}
    tp, fp, fn = map(int, stats[6:9])
    result['normal'] = {'videos': normals, 'false_positive_videos': int(stats[9]),
                        'false_positive_rate': int(stats[9]) / max(1, normals)}
    result['frame'] = {'f1': float(2 * tp / max(1, 2 * tp + fp + fn)), 'tp': tp, 'fp': fp, 'fn': fn}
    result['positive_videos_without_candidate'] = int(stats[10])
    result['predicted_segments'] = int(stats[11])
    return result


def grouped_metrics(records, predictions):
    require(set(predictions) == set(range(len(records))), 'prediction OOF coverage differs')
    result = {}
    filters = [('all', lambda r: True), ('normal', lambda r: r['event_count'] == 0),
               ('single_event', lambda r: r['event_count'] == 1), ('multi_event', lambda r: r['event_count'] > 1)]
    for name, predicate in filters:
        ids = [i for i, record in enumerate(records) if predicate(record)]
        stats = sum((row_statistics(predictions[i], records[i]['labels']) for i in ids), np.zeros(12, np.int64))
        result[name] = metric_dict(stats, sum(not records[i]['labels'].any() for i in ids))
    return result


def prediction_rows(records, predictions):
    require(set(predictions) == set(range(len(records))), 'serialized prediction coverage differs')
    return [{'index': i, 'name': r['name'], 'segments': [list(pair) for pair in predictions[i]],
             'has_candidates': bool(predictions[i]), 'is_normal': not bool(np.any(r['labels']))}
            for i, r in enumerate(records)]


def read_prediction_rows(rows, records, label):
    require(isinstance(rows, list) and len(rows) == len(records), 'prediction rows missing: ' + label)
    result = {}
    for i, (row, record) in enumerate(zip(rows, records)):
        require(isinstance(row, dict), 'invalid prediction row: ' + label)
        compare(row.get('index'), i, label + '/index')
        compare(row.get('name'), record['name'], label + '/name')
        result[i] = segments(row.get('segments'), record['frames'], label + '/' + str(i))
        compare(row.get('has_candidates'), bool(result[i]), label + '/has_candidates')
        compare(row.get('is_normal'), not bool(np.any(record['labels'])), label + '/is_normal')
    return result


def prediction_summary(records, predictions):
    normal = grouped_metrics(records, predictions)['all']['normal']
    empty = [i for i in range(len(records)) if not predictions[i]]
    return {'normal': {**normal,
        'false_positive_names': [r['name'] for i, r in enumerate(records) if not r['labels'].any() and predictions[i]],
        'correctly_empty_names': [r['name'] for i, r in enumerate(records) if not r['labels'].any() and not predictions[i]]},
        'no_candidates': {'videos': len(empty), 'names': [records[i]['name'] for i in empty],
            'normal_videos': sum(not bool(records[i]['labels'].any()) for i in empty),
            'positive_videos': sum(bool(records[i]['labels'].any()) for i in empty)}}


def promotion_guards(metrics, baseline):
    x, y = metrics['all'], baseline['all']
    return {'event_f1_03_no_worse': x['iou_0.3']['f1'] >= y['iou_0.3']['f1'],
        'event_f1_05_improved_02': x['iou_0.5']['f1'] >= y['iou_0.5']['f1'] + .02,
        'frame_f1_drop_at_most_005': x['frame']['f1'] >= y['frame']['f1'] - .005,
        'normal_fp_no_worse': x['normal']['false_positive_videos'] <= y['normal']['false_positive_videos'],
        'positive_empty_improved_3': x['positive_videos_without_candidate'] <= y['positive_videos_without_candidate'] - 3}


def paired_uncertainty(records, baseline, predictions):
    ids = list(range(len(records)))
    counts = bootstrap_counts(records, ids, SEED + 991, 2000)
    before = counts @ np.stack([row_statistics(baseline[i], records[i]['labels']) for i in ids])
    after = counts @ np.stack([row_statistics(predictions[i], records[i]['labels']) for i in ids])
    result = {}
    for key, offset in (('event_f1_03', 0), ('event_f1_05', 3), ('frame_f1', 6)):
        def f1(stats):
            tp, fp, fn = (stats[:, offset + k] for k in range(3))
            return 2 * tp / np.maximum(1, 2 * tp + fp + fn)
        diff = f1(after) - f1(before)
        result[key] = {'mean_difference': float(diff.mean()),
            'percentile_95_interval': np.quantile(diff, [.025, .975], method='linear').tolist(),
            'bootstrap_probability_positive': float(np.mean(diff > 0)), 'not_confirmatory_p_value': True}
    return result


def verify_training_reports(e):
    for final, filename in ((False, 'training_report.json'), (True, 'final_training_report.json')):
        report = e.json(OUT / filename)
        compare(report.get('status'), 'complete', filename + '/status')
        compare(report.get('member_fits'), 12 if final else 60, filename + '/member_fits')
        expected = {(5, recipe, True) for recipe in RECIPES} if final else {
            (scope, recipe, False) for scope in range(5) for recipe in RECIPES[:-1]}
        results = report.get('results')
        require(isinstance(results, list) and len(results) == len(expected), 'training job coverage differs: ' + filename)
        actual = []
        for row in results:
            require(type(row.get('scope')) is int and type(row.get('final_selection')) is bool and
                    isinstance(row.get('recipe'), str), 'invalid training job identity')
            actual.append((row['scope'], row['recipe'], row['final_selection']))
        require(len(set(actual)) == len(actual) and set(actual) == expected, 'training jobs duplicate/wrong: ' + filename)


def verify_deployment_selection(saved, chosen, rows, parts, outer_choices):
    compare(saved.get('status'), 'complete', 'deployment/status')
    compare(saved.get('role'), 'inner3_only_deployment_choice_not_validation', 'deployment/role')
    compare(saved.get('uses_outer_probabilities'), False, 'deployment/uses_outer_probabilities')
    compare(saved.get('uses_outer_metrics_for_selection'), False, 'deployment/uses_outer_metrics_for_selection')
    compare(saved.get('selection_function'), 'run_feature_blocks_v10.select_inner', 'deployment/function')
    compare(saved.get('coverage_per_row'), 1, 'deployment/OOF coverage')
    compare(saved.get('inner_partitions'), parts, 'deployment/inner_partitions')
    compare(saved.get('chosen'), chosen, 'deployment/chosen')
    compare(saved.get('all_inner_selections'), rows, 'deployment/all_inner_selections')
    compare(saved.get('outer_choices_diagnostic_only'), outer_choices, 'deployment/outer choices')
    compare(saved.get('same_choice_all_outer'), all(r == chosen['candidate'] for r in outer_choices),
            'deployment/same choice flag')
    compare(saved.get('training_receipt_sha256'), RECEIPT_PIN, 'deployment/receipt')


def run_audit(root=ROOT):
    e = Evidence(root)
    # Fail once on missing final artifacts. Never wait, poll or claim partial pass.
    report = e.json(OUT / 'report.json')
    deployment = e.json(OUT / 'deployment_selection.json')
    receipt, baseline = verify_receipts(e)
    static = source_review(e)
    verify_training_reports(e)
    records, outer, aliases = load_records(e)
    decoders = decoder_functions()
    transforms = TransformEvidence(records)
    compare(report.get('schema_version'), 'feature-block-ablation-v10', 'report/schema')
    compare(report.get('status'), 'complete', 'report/status')
    compare(report.get('role'), 'repeated_development_not_blind_or_confirmatory', 'report/role')
    compare(report.get('protocol_sha256'), PROTOCOL_PIN, 'report/protocol')
    compare(report.get('training_receipt_sha256'), RECEIPT_PIN, 'report/receipt')
    compare(report.get('grouped_v1_baseline'), baseline, 'report/baseline lineage')
    folds = report.get('folds')
    require(isinstance(folds, list) and len(folds) == 5, 'report fold coverage differs')
    selected, fixed = {}, {recipe: {} for recipe in RECIPES}
    coverage = {recipe: {'inner': Counter(), 'outer': Counter(), 'final': Counter()} for recipe in RECIPES}
    recomputed = []
    for scope, val in enumerate(outer):
        train = [i for i in range(len(records)) if i not in set(val)]
        parts = expected_partitions(records, train, scope)
        inner, held = {}, {}
        for recipe in RECIPES:
            inner[recipe], held[recipe], _ = load_cache(e, recipe, scope, records, outer, transforms)
            coverage[recipe]['inner'].update(inner[recipe].frame.keys())
            coverage[recipe]['outer'].update(held[recipe].frame.keys())
        chosen, rows = select_inner(records, train, inner, scope, decoders)
        prediction = decode(records, val, held[chosen['candidate']], chosen['config'], decoders)
        require(set(selected).isdisjoint(prediction), 'outer predictions repeated')
        selected.update(prediction)
        for row in rows:
            fixed[row['candidate']].update(decode(records, val, held[row['candidate']], row['config'], decoders))
        expected = {'fold': scope, 'train_indices': train, 'validation_indices': val, 'inner_partitions': parts,
            'primary': chosen, 'all_inner_selections': rows,
            'primary_predictions': {str(i): [list(pair) for pair in prediction[i]] for i in val}}
        compare(folds[scope], expected, 'report/fold/' + str(scope))
        recomputed.append({'scope': scope, 'chosen': chosen['candidate'], 'config': chosen['config']})
    baseline_pred = read_prediction_rows(baseline['predictions'], records, 'baseline predictions')
    baseline_metrics = grouped_metrics(records, baseline_pred)
    compare(baseline['metrics'], baseline_metrics, 'baseline independent metrics')
    metrics = grouped_metrics(records, selected)
    compare(report.get('primary_v10_nested'), {'metrics': metrics, 'predictions': prediction_rows(records, selected),
        'summary': prediction_summary(records, selected)}, 'report/primary')
    diagnostics = {r: {'role': 'fixed_recipe_outer_diagnostic_not_deployment_selection',
                      'metrics': grouped_metrics(records, p), 'predictions': prediction_rows(records, p)}
                   for r, p in fixed.items()}
    compare(report.get('fixed_candidate_diagnostics_not_for_promotion'), diagnostics, 'report/fixed diagnostics')
    guards = promotion_guards(metrics, baseline_metrics)
    compare(report.get('statistical_promotion_checks'), guards, 'report/guards')
    compare(report.get('statistical_promotion_passed'), all(guards.values()), 'report/guard outcome')
    compare(report.get('paired_content_bootstrap'), paired_uncertainty(records, baseline_pred, selected),
            'report/paired content bootstrap')
    probabilities = {}
    for recipe in RECIPES:
        probabilities[recipe], held, _ = load_cache(e, recipe, 5, records, outer, transforms, final=True)
        require(not held.frame and not held.video, 'final selection has outer predictions')
        coverage[recipe]['final'].update(probabilities[recipe].frame.keys())
    ids = list(range(len(records)))
    chosen, rows = select_inner(records, ids, probabilities, 5, decoders)
    parts = expected_partitions(records, ids, 5)
    verify_deployment_selection(deployment, chosen, rows, parts, [r['chosen'] for r in recomputed])
    for recipe in RECIPES:
        for phase, times in (('inner', 4), ('outer', 1), ('final', 1)):
            require(coverage[recipe][phase] == Counter({i: times for i in ids}),
                    'per-row OOF coverage differs: ' + recipe + '/' + phase)
    compatibility = transforms.compatibility(expected_partitions(records, [i for i in ids if i not in set(outer[0])], 0)[0]['fit'])
    require(len(transforms.proofs) == 92, '80 nested + 12 final fit evidence coverage differs')
    e.recheck()
    return {'status': 'passed', 'scope': 'frozen_v10_evidence_and_independent_selection_replay',
        'receipt_sha256': RECEIPT_PIN, 'trust_anchor_timing': 'observed_before_v10_reports',
        'source_review': static, 'rich_v8_compatibility': compatibility,
        'rows': len(records), 'content_groups': 76, 'aliases_retained': aliases,
        'OOF_coverage': {recipe: {'inner_per_row': 4, 'outer_per_row': 1, 'final_inner_per_row': 1}
                         for recipe in RECIPES},
        'fit_evidence': transforms.proofs, 'PCA_covariance_contexts': len(transforms.pcas),
        'outer_choices': recomputed, 'final_direct_inner3_choice': chosen,
        'primary_metrics': metrics, 'statistical_promotion_checks': guards,
        'statistical_promotion_passed': all(guards.values()), 'deployment_approved': False,
        'portable_fresh_parity': {'status': 'not_audited', 'required_max': 2e-6},
        'consumed_artifact_sha256': dict(sorted(e.hashes.items())), 'limitations': LIMITATIONS}



V11 = Path('output/algorithm-opt-v11-proposal-review')
V4 = Path('output/algorithm-opt-2026-10-02-v4')
V11_RECEIPT_PIN = '0292063f55a65db7681eb7b2fd7585d9ee32e0ffc25737a58f60594851cba2d6'
V11_PROTOCOL_PIN = 'd4117b7ab0cc2292b6a281de68f62f49a95aa0260067f1d66289cd1423a50136'
V4_RECEIPT_PIN = '3ef89888ae2560607e40770032879edfcd5e0f6ef6379efd74678bbb3f5635f4'
REVIEWER = 'compact_rgb_corrected_motion_hgb'
PRIOR_DETECTOR = Path('output/algorithm-opt-2026-10-02-v9/final_runtime_sources/optimized_detector.py')
PRIOR_DETECTOR_PIN = '9663a3b9080f1be3a99ec71ede043969a223268aeaa9c11daea1c7c704184c0b'
V11_DETECTOR_PIN = '0c52c372588584fdc863e13b855394586b4ae6e2f0f516f1d6740524892ad7c3'
# Preserve the export-time pin. Current-runtime repairs are separate reviewed pins;
# neither the receipt nor the frozen source snapshot is re-anchored to the repair.
V11_FROZEN_DETECTOR = V11 / 'deployment_diagnostic/sources/optimized_detector.py'
V11_HARDENED_PINS = {
    '899e3b5128e687a9be7e27702db91487ef3319154b713fdf594331cc2a0c9ea7': 'verifier_guard_and_cache_only_rejection_v13r',
    '45a0d689c60ba89fe1ec2579ac8e7f3bd07cd16a3ad51901212be06537b3af9e': 'verifier_guard_only',
    '067d348c39aa8ea1f1396cd8d548128b4d4d350f9064cefdc52d14f946296ba8': 'verifier_guard_and_cache_only_rejection',
}


def verify_v11_detector_revision(frozen, current):
    """Pin both revisions and forbid changes outside the explicitly reviewed guards."""
    require(sha(frozen) == V11_DETECTOR_PIN, 'frozen export detector SHA256 differs')
    digest = sha(current)
    if digest == V11_DETECTOR_PIN:
        return {'revision': 'frozen_unhardened', 'P2_guard_revision_reviewed': False,
                'frozen_export_detector_sha256': V11_DETECTOR_PIN, 'current_detector_sha256': digest,
                'changed_functions': []}
    require(digest in V11_HARDENED_PINS, 'unreviewed current detector SHA256')
    before, after = (ast.parse(b.decode('utf-8-sig')) for b in (frozen, current))
    def function(tree, name):
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(nodes) == 1, 'detector function missing/duplicate: ' + name)
        return nodes[0]
    old_guard, new_guard = (function(t, '_validated_proposal_verifier') for t in (before, after))
    require(ast.dump(old_guard, include_attributes=False) != ast.dump(new_guard, include_attributes=False),
            'hardened detector guard did not change')
    # Keep the new signature/decorators, replacing only the reviewed body. A signature
    # or unrelated module edit is not covered by the repair, even with a new SHA pin.
    new_guard.body = deepcopy(old_guard.body)
    changes = ['_validated_proposal_verifier']
    if V11_HARDENED_PINS[digest] in ['verifier_guard_and_cache_only_rejection','verifier_guard_and_cache_only_rejection_v13r']:
        literal = ("""if bundle.get('transform',{}).get('feature_view')=='spatial-jepa-v13':
    raise ValueError('experimental spatial JEPA cache has no verified live extraction profile; not deployable')
""")
        if V11_HARDENED_PINS[digest]=='verifier_guard_and_cache_only_rejection_v13r':
            literal=literal.replace("=='spatial-jepa-v13'", " in ['spatial-jepa-v13','spatial-jepa-mask-repair-v13r']")
        deny = ast.parse(literal).body[0]
        base = function(after, '_required_channels_base')
        positions = [i for i, n in enumerate(base.body) if ast.dump(n, include_attributes=False) ==
                     ast.dump(deny, include_attributes=False)]
        require(len(positions) == 1, 'cache-only rejection differs from reviewed literal guard')
        del base.body[positions[0]]
        changes.append('_required_channels_base:cache_only_rejection')
    require(ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False),
            'detector changed outside reviewed guard bodies/signatures')
    return {'revision': V11_HARDENED_PINS[digest], 'P2_guard_revision_reviewed': True,
            'frozen_export_detector_sha256': V11_DETECTOR_PIN, 'current_detector_sha256': digest,
            'changed_functions': changes, 'module_AST_equal_excluding_reviewed_guard_bodies': True,
            'trust_anchor_timing': 'current runtime pin observed during repair review, not at experiment freeze',
            'scope': 'V11 repair/default compatibility only; cache-only rejection is not a V13 algorithm audit'}


def read_v11_detector(e):
    frozen = e.read(V11_FROZEN_DETECTOR, V11_DETECTOR_PIN)
    current = e.read(Path('code/optimized_detector.py'))
    return current, verify_v11_detector_revision(frozen, current)


def v11_cached_score_bundle(primary, proposal, review):
    """Valid portable-schema fixtures; only cached OOF scores drive branch replay.

    The rich v10/v4 transform equivalence is checked separately in source_review.
    Exported dimensions/PCA/names are retained, but no fullfit model is used to
    produce the injected OOF evidence. This is NOT a portable reviewer parity test.
    """
    leaf = {k: deepcopy(primary[k]) for k in ('recipe', 'transform')}
    leaf.update(weight=1., frame_model={'kind': 'constant',
        'n_features': primary['frame_model']['n_features'], 'probability': .7},
        video_model={'kind': 'constant', 'n_features': primary['video_model']['n_features'], 'probability': .8})
    reviewer = deepcopy(leaf)
    reviewer['recipe'] = REVIEWER
    reviewer['transform']['feature_view'] = 'compact-fps-context-v4'
    reviewer['frame_model']['probability'] = .3
    return {'recipe': leaf['recipe'], 'members': [leaf], 'calibration': None,
            'decoder': deepcopy(proposal), 'proposal_verifier': {'schema_version': 'proposal-verifier-v11',
                'member': reviewer, 'review': deepcopy(review)}}


def v11_verifier_risk_cases(valid):
    """Independent, bounded mutations of the P2 structural/composition contract."""
    cases = []
    def change(name, path, value=None, *, remove=False):
        row = deepcopy(valid); target = row
        for key in path[:-1]:
            target = target[key]
        if remove:
            del target[path[-1]]
        else:
            target[path[-1]] = deepcopy(value)
        cases.append((name, row))
    member = ('proposal_verifier', 'member')
    for field in ('members', 'decoder', 'proposal_verifier'):
        change('reviewer_nonleaf_' + field, member + (field,), [] if field == 'members' else {})
    for field in ('transform', 'frame_model', 'video_model'):
        change('reviewer_missing_' + field, member + (field,), remove=True)
        change('reviewer_nondict_' + field, member + (field,), [])
    change('reviewer_unverified_feature_view', member + ('transform', 'feature_view'), 'compact-feature-blocks-v10')
    change('reviewer_unknown_recipe', member + ('recipe',), 'not-a-v4-recipe')
    for field in ('calibration', 'video_gate', 'boundary_models'):
        change('base_' + field, (field,), {})
        change('reviewer_' + field, member + (field,), {})
        change('primary_' + field, ('members', 0, field), {})
    change('primary_empty_ensemble', ('members',), [])
    change('primary_multiple_members', ('members',), valid['members'] * 2)
    change('primary_nonunit_weight', ('members', 0, 'weight'), .5)
    change('primary_nonfinite_weight', ('members', 0, 'weight'), float('nan'))
    change('decoder_unknown_kind', ('decoder', 'kind'), 'unverified-decoder')
    change('decoder_boundary_enabled', ('decoder', 'boundary_seconds'), .2)
    change('decoder_nondict', ('decoder',), [])
    change('verifier_unknown_schema', ('proposal_verifier', 'schema_version'), 'unknown-verifier')
    change('verifier_extra_schema_field', ('proposal_verifier', 'extra'), True)
    change('review_unknown_kind', ('proposal_verifier', 'review', 'kind'), 'unknown-review')
    change('review_extra_field', ('proposal_verifier', 'review', 'extra'), True)
    change('review_missing_minimum', ('proposal_verifier', 'review', 'minimum'), remove=True)
    change('review_nondict', ('proposal_verifier', 'review'), [])
    for name, value in (('negative', -.01), ('over_one', 1.01), ('nonfinite', float('nan'))):
        change('review_minimum_' + name, ('proposal_verifier', 'review', 'minimum'), value)
        change('review_rescue_' + name, ('proposal_verifier', 'review', 'rescue_peak'), value)
    return cases


def audit_v11_verifier_contract(detector, valid):
    # Positive controls prevent an always-raise implementation from passing rejection tests.
    channels = detector.required_channels(valid)
    require(channels == {'motion', 'local', 'corrected', 'rgb'}, 'valid verifier channels differ')
    for kind in ('duration-logit-v4', 'recall-stable-v2'):
        control = deepcopy(valid); control['decoder']['kind'] = kind
        require(detector._validated_proposal_verifier(control) is control['proposal_verifier'],
                'supported verifier/decoder rejected or rewritten')
    disabled = deepcopy(valid); disabled['proposal_verifier'] = None
    require(detector._validated_proposal_verifier(disabled) is None, 'disabled verifier contract changed')
    rejected = []
    for name, row in v11_verifier_risk_cases(valid):
        for entry in ('_validated_proposal_verifier', 'required_channels'):
            try:
                getattr(detector, entry)(row)
            except ValueError:
                pass
            else:
                raise AuditError('P2 fail-closed mutation accepted by ' + entry + ': ' + name)
        rejected.append(name)
    return {'status': 'passed', 'rejected_cases': rejected, 'rejected_case_count': len(rejected),
            'entry_points': ['_validated_proposal_verifier', 'required_channels'],
            'supported_decoder_positive_controls': 2, 'disabled_positive_control': True,
            'scope': 'structural/schema/composition/cutoff rejection, not full portable model/PCA validation',
            'ordering_limit': 'predict_record computes primary evidence before calling the verifier guard; '
                              'this audit does not claim pre-primary-inference schema validation'}




def review_configurations():
    return [{'kind': 'proposal-local-q75-v11', 'minimum': 0., 'rescue_peak': None}] + [
        {'kind': 'proposal-local-q75-v11', 'minimum': cutoff, 'rescue_peak': rescue}
        for cutoff in (.35, .45, .55) for rescue in (None, .8)]


def independent_review(proposals, primary, reviewer, config):
    require(isinstance(config, dict) and type(config.get('minimum')) in (int, float) and
            (config.get('rescue_peak') is None or type(config.get('rescue_peak')) in (int, float)) and
            config in review_configurations(), 'review is outside frozen seven-policy grid')
    primary, reviewer = np.asarray(primary, np.float64), np.asarray(reviewer, np.float64)
    require(primary.ndim == 1 and primary.shape == reviewer.shape and
            np.isfinite(primary).all() and np.isfinite(reviewer).all() and
            np.all((primary >= 0) & (primary <= 1)) and np.all((reviewer >= 0) & (reviewer <= 1)),
            'invalid proposal-review probabilities')
    accepted, trace = [], []
    for start, end in segments(proposals, len(primary), 'review'):
        q75 = float(np.quantile(reviewer[start:end + 1], .75, method='linear'))
        peak = float(np.max(primary[start:end + 1]))
        passed = config['minimum'] == 0 or q75 >= config['minimum']
        rescued = not passed and config['rescue_peak'] is not None and peak >= config['rescue_peak']
        keep = passed or rescued
        if keep:
            accepted.append((start, end))
        trace.append({'span': [start, end], 'review_q75': q75, 'primary_peak': peak,
                      'accepted': bool(keep), 'rescued': bool(rescued)})
    return accepted, trace


def review_predictions(records, ids, primary, reviewer, config, decoders):
    compare(config.get('kind'), 'proposal-review-v11', 'review decoder kind')
    proposals = decode(records, ids, primary, config['proposal'], decoders)
    prediction, trace = {}, {}
    for i in ids:
        prediction[i], trace[i] = independent_review(proposals[i], primary.frame[i], reviewer.frame[i], config['review'])
    return prediction, trace, proposals


def independent_review_choice(records, ids, primary, reviewer, scope, decoders):
    require(set(reviewer.frame) == set(reviewer.video) == set(ids), 'reviewer inner coverage differs')
    restricted = RestrictedRecords(records, ids)
    base = independent_choice(restricted, ids, primary, SEED + scope, decoders)
    weights = content_weights(restricted, ids)
    normals = np.asarray([not np.any(restricted[i]['labels']) for i in ids], np.float64)
    positives = 1 - normals
    boot = bootstrap_counts(restricted, ids, SEED + scope) * weights[None, :]
    nm, pm, bn, bp = weights @ normals, weights @ positives, boot @ normals, boot @ positives
    rows, seen = [], set()
    for ordinal, policy in enumerate(review_configurations()):
        config = {'kind': 'proposal-review-v11', 'proposal': deepcopy(base['config']), 'review': policy}
        prediction, _, _ = review_predictions(restricted, ids, primary, reviewer, config, decoders)
        fingerprint = tuple(tuple(prediction[i]) for i in ids)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        matrix = np.stack([row_statistics(prediction[i], restricted[i]['labels']) for i in ids])
        total = weights @ matrix
        pooled = float(utility(total, nm, pm))
        f05 = float(2 * total[3] / max(1e-12, 2 * total[3] + total[4] + total[5]))
        q20 = float(np.quantile(utility(boot @ matrix, bn, bp), .2, method='linear'))
        rows.append({'config': config, 'stats': total.tolist(), 'pooled_utility': pooled,
            'key': [pooled, f05, -total[10], -total[9], -total[11], -ordinal],
            'bootstrap_q20': q20, 'stable_utility': .75 * pooled + .25 * q20})
    # Stage B has at most seven UNIQUE policies; preserves declared ordinal order.
    chosen = deepcopy(max(rows, key=lambda row: (row['stable_utility'], *row['key'][1:])))
    chosen.update(shortlist=rows, base_selection=base,
        candidate='et_control' if chosen['config']['review']['minimum'] == 0 else 'et_proposals_hgb_local_review')
    return chosen


def verify_v11_receipt(e, v10_receipt):
    receipt = e.json(V11 / 'training_receipt.json')
    require(receipt.get('receipt_sha256') == V11_RECEIPT_PIN and
            value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) == V11_RECEIPT_PIN,
            'v11 receipt differs from observed post-report anchor')
    compare(receipt['parent_receipts'], {'v8': PARENT_PIN, 'v4': V4_RECEIPT_PIN, 'v10': RECEIPT_PIN},
            'v11/parent receipts')
    parent = e.json(V4 / 'training_receipt.json')
    require(parent['receipt_sha256'] == V4_RECEIPT_PIN and
            value_hash({k: v for k, v in parent.items() if k != 'receipt_sha256'}) == V4_RECEIPT_PIN,
            'v4 reviewer parent receipt differs')
    require(parent['input_hashes'] == v10_receipt['input_hashes'] and parent['inputs_sha256'] == INPUTS_PIN,
            'v11 reviewer input lineage differs')
    for name, digest in parent['sources'].items():
        require(v10_receipt['sources'].get(name) == digest, 'v4 source lineage differs')
        e.read(V4 / 'sources' / name, digest)
    names = ('run_proposal_review_v11.py', 'optimized_proposal_review_v11.py') + SOURCES
    require(set(receipt['sources']) == set(names), 'v11 source coverage differs')
    for name, digest in receipt['sources'].items():
        if name in v10_receipt['sources']:
            require(v10_receipt['sources'][name] == digest, 'v11/v10 source lineage differs')
        e.read(Path('code') / name, digest)
        e.read(V11 / 'sources' / name, digest)
    expected = {(folder / 'training' / f'{scope}_{recipe}.{ext}').as_posix()
        for folder, recipe in ((PARENT, PARENT_RICH), (V4, REVIEWER))
        for scope in range(5) for ext in ('npz', 'json')}
    expected |= {(OUT / 'final_selection' / f'5_{RICH}.{ext}').as_posix() for ext in ('npz', 'json')}
    require(set(receipt['consumed_artifacts']) == expected, 'v11 consumed cache coverage differs')
    for path, digest in receipt['consumed_artifacts'].items():
        e.read(path, digest)
    require(receipt['protocol_sha256'] == V11_PROTOCOL_PIN and receipt['baseline_sha256'] == BASELINE_PIN,
            'v11 protocol/baseline pins differ')
    compare(receipt['default_models'], v10_receipt['default_models'], 'v11/default models')
    protocol = e.json(V11 / 'protocol.json', V11_PROTOCOL_PIN)
    compare(protocol['schema_version'], 'proposal-review-v11', 'v11/protocol schema')
    compare(protocol['review_policies'], review_configurations(), 'v11/seven review policies')
    compare(protocol['promotion_guards'], GUARD_SPEC, 'v11/protocol guards')
    compare(protocol['new_outer_model_fits'], 0, 'v11/new outer fits')
    compare(protocol['new_final_selection_model_fits'], 3, 'v11/new final fits')
    return receipt


def load_reviewer_cache(e, scope, records, outer, transforms, *, final=False):
    path = V11 / 'final_selection/reviewer.npz' if final else V4 / 'training' / f'{scope}_{REVIEWER}.npz'
    metadata = e.json(path.with_suffix('.json'))
    if final:
        require(metadata.get('signature') == value_hash({k: v for k, v in metadata.items() if k != 'signature'}),
                'v11 final reviewer signature differs')
        ids = list(range(len(records)))
        parts = expected_partitions(records, ids, 5)
        for key, value in (('scope', 5), ('recipe', REVIEWER), ('train', ids), ('validation', []),
                           ('inner_partitions', parts), ('member_fits', 3),
                           ('training_receipt_sha256', V11_RECEIPT_PIN)):
            compare(metadata.get(key), value, 'v11 final reviewer/' + key)
        proofs = metadata.get('fit_evidence')
        require(isinstance(proofs, list) and len(proofs) == 3, 'v11 final fit proofs coverage differs')
        for k, (part, proof) in enumerate(zip(parts, proofs)):
            compare(proof['partition'], k, 'v11 reviewer fit partition')
            compare(proof['fit_content_sha256'], sorted({records[i]['sha256'] for i in part['fit']}),
                    'v11 reviewer fit content')
        parity = metadata.get('parity')
        require(isinstance(parity, list) and len(parity) == 3, 'v11 reviewer parity attestation missing')
        for k, row in enumerate(parity):
            compare(row['partition'], k, 'v11 reviewer parity partition')
            require(all(type(row[key]) in (float, int) and math.isfinite(row[key]) and 0 <= row[key] <= 2e-6
                        for key in ('max_frame_error', 'max_video_error')), 'v11 final parity attestation fails')
    else:
        _, _, parts = verify_training_metadata(metadata, REVIEWER, scope, records, outer, parent=True,
                                               parent_recipe=REVIEWER, parent_receipt=V4_RECEIPT_PIN)
    for part, proof in zip(parts, metadata['fit_evidence']):
        transforms.verify(REVIEWER, part['fit'], proof, parent=True)
    inner, held = probability_maps(e.read(path), metadata, records, path.as_posix())
    return inner, held, metadata




def module_from_bytes(data, path, name):
    """Execute only hash-verified, consumed source bytes; never re-open a snapshot."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data.decode('utf-8-sig'), str(path), 'exec'), module.__dict__)
    return module


def audit_detector(e, records, choices, outer_probabilities, final_choice):
    current_path = Path('code/optimized_detector.py')
    current_bytes, revision = read_v11_detector(e)
    require(revision['P2_guard_revision_reviewed'], 'V11 hardened audit requires the reviewed P2 repair')
    current = module_from_bytes(current_bytes, e.root / current_path, '_audited_detector_current')
    old = module_from_bytes(e.read(PRIOR_DETECTOR, PRIOR_DETECTOR_PIN), e.root / PRIOR_DETECTOR,
                            '_audited_detector_prior')
    # This dependency is outside the v10 training-source inventory but used by both
    # detector versions. It is the frozen v9 snapshot's explicit calibration pin.
    e.read(Path('code/optimized_calibration_state.py'),
           'c5fe62c028f7f211a7bc1e192aada89abc67c81bb4992d8bb5ff8c23b59df895')
    pairs = 0
    for name in ('optimized_locator_v1.json', 'optimized_motion_locator_v1.json'):
        bundle = e.json(Path('models') / name)
        require('proposal_verifier' not in bundle, 'default bundle unexpectedly enables review')
        require(current.required_channels(bundle) == old.required_channels(bundle), 'default required channels changed')
        for record in records:
            before, after = old.predict_record(bundle, record), current.predict_record(bundle, record)
            require(before.keys() == after.keys(), 'default output contract changed')
            for key in before:
                if isinstance(before[key], np.ndarray):
                    require(np.array_equal(before[key], after[key]), 'default output changed: ' + key)
                else:
                    require(before[key] == after[key], 'default scalar/interval changed: ' + key)
            pairs += 1
    # Exercise the active branch even when the selected/exported final model is control.
    # Fixed cached frame scores are injected in this audit process only. No fitting,
    # network calls, feature extraction or mutation of the server/production code.
    fixture_report = e.json(V11 / 'deployment_diagnostic/report.json')
    fixture_export = e.json(V11 / 'deployment_diagnostic/locator_bundle.json', fixture_report['bundle_sha256'])
    fixture_primary = fixture_export['members'][0]
    valid = v11_cached_score_bundle(fixture_primary, final_choice['config']['proposal'], review_configurations()[2])
    contract = audit_v11_verifier_contract(current, valid)
    branch_rows = 0
    branch_counts = Counter()
    for scope, ids in enumerate(outer_probabilities):
        primary, reviewer, val = outer_probabilities[scope]
        cfg = choices[scope]['config']
        # Use a nonzero policy to prove actual filtering/rescue wiring, not just identity.
        active_cfg = {**cfg, 'review': review_configurations()[2]}
        expected, traces, _ = review_predictions(records, val, primary, reviewer, active_cfg, decoder_functions())
        bundle = v11_cached_score_bundle(fixture_primary, active_cfg['proposal'], active_cfg['review'])
        compare(sorted(current.required_channels(bundle)), ['corrected', 'local', 'motion', 'rgb'],
                'active verifier required channels')
        for i in val:
            def cached(member, _record):
                probability = reviewer if member['recipe'] == REVIEWER else primary
                return probability.frame[i], probability.video[i], None
            with patch.object(current, 'predict_member', side_effect=cached) as injected:
                unreviewed = current.predict_record({k: v for k, v in bundle.items()
                                                    if k != 'proposal_verifier'}, records[i])
                result = current.predict_record(bundle, records[i])
                require(injected.call_count == 3, 'cached-score replay did not consume exactly primary/primary/reviewer')
            compare(result['intervals'], expected[i], 'active verifier intervals/' + str(i))
            compare(result['proposal_review_trace'], traces[i], 'active verifier trace/' + str(i))
            require(np.array_equal(result['frame_probabilities'], primary.frame[i]),
                    'review branch altered primary frame probabilities')
            require(np.array_equal(result['score'], unreviewed['score']) and
                    result['video_probability'] == unreviewed['video_probability'],
                    'review branch altered primary score/video evidence')
            for trace in traces[i]:
                branch_counts['accepted' if trace['accepted'] else 'rejected'] += 1
                branch_counts['rescued'] += int(trace['rescued'])
            branch_rows += 1
    folder = V11 / 'deployment_diagnostic'
    diagnostic = e.json(folder / 'report.json')
    fullfit = e.json(folder / 'fullfit_receipt.json')
    require(fullfit['receipt_sha256'] == value_hash({k: v for k, v in fullfit.items() if k != 'receipt_sha256'})
            and diagnostic['fullfit_receipt_sha256'] == fullfit['receipt_sha256'], 'fullfit receipt binding differs')
    compare(fullfit['training_receipt_sha256'], V11_RECEIPT_PIN, 'diagnostic training receipt')
    require(fullfit['inputs_sha256'] == INPUTS_PIN == value_hash(fullfit['input_hashes']), 'fullfit input digest differs')
    base_receipt = e.json(OUT / 'training_receipt.json')
    compare(fullfit['input_hashes'], base_receipt['input_hashes'], 'diagnostic fit inputs')
    require(fullfit['selection_sha256'] == e.hashes[(V11 / 'deployment_selection.json').as_posix()],
            'export did not bind actual final inner3 choice')
    for name, digest in fullfit['sources'].items():
        # The export receipt binds the ORIGINAL detector snapshot. Its current guarded
        # revision was pinned/reviewed above; every other current source must remain frozen.
        if name != 'optimized_detector.py':
            e.read(Path('code') / name, digest)
        e.read(folder / 'sources' / name, digest)
    require(fullfit['sources']['optimized_detector.py'] == V11_DETECTOR_PIN, 'diagnostic detector source changed')
    compare(diagnostic['status'], 'complete', 'diagnostic status')
    compare(diagnostic['default_promoted'], False, 'diagnostic promoted flag')
    compare(diagnostic['default_models_unchanged'], True, 'diagnostic default model flag')
    compare(diagnostic['candidate'], final_choice['candidate'], 'diagnostic candidate')
    compare(diagnostic['config'], final_choice['config'], 'diagnostic chosen config')
    bundle = e.json(folder / 'locator_bundle.json', diagnostic['bundle_sha256'])
    compare(bundle['experiment_id'], 'v11', 'export experiment identity')
    compare(bundle['protocol_sha256'], V11_PROTOCOL_PIN, 'export protocol binding')
    compare(bundle['comparison_report_sha256'], e.hashes[(V11 / 'report.json').as_posix()], 'export report binding')
    compare(bundle['statistical_promotion_checks'], e.json(V11 / 'report.json')['statistical_promotion_checks'],
            'export promotion guards')
    compare(bundle['default_deployment_acceptance'], 'rejected', 'export default acceptance')
    compare(bundle['deployment_selection'], {'role': 'inner3_only_deployment_choice_not_validation',
        'candidate': final_choice['candidate'], 'uses_outer_OOF_probabilities': False}, 'export selection metadata')
    compare(bundle['training_video_names'], [r['name'] for r in records], 'export training row names')
    compare(bundle['training_video_sha256'], [r['sha256'] for r in records], 'export training row SHAs')
    compare(bundle['decoder'], final_choice['config']['proposal'], 'export Stage A decoder')
    require(bundle.get('calibration') is None and bundle.get('video_gate') is None and
            not bundle['decoder'].get('boundary_seconds', 0), 'export adds an unvalidated calibration/gate/boundary')
    require(len(bundle.get('members', [])) == 1 and bundle['members'][0]['recipe'] == RICH,
            'export primary recipe is not the frozen rich ET')
    content = sorted({r['sha256'] for r in records})
    compare(bundle['members'][0]['fit_content_sha256'], content, 'export primary fit content')
    if final_choice['candidate'] == 'et_control':
        require(bundle.get('proposal_verifier') is None, 'control export unexpectedly enables verifier')
    else:
        verifier = bundle.get('proposal_verifier')
        require(isinstance(verifier, dict) and verifier.get('schema_version') == 'proposal-verifier-v11',
                'enabled export missing verifier schema')
        compare(verifier['review'], final_choice['config']['review'], 'export Stage B policy')
        compare(verifier['member']['recipe'], REVIEWER, 'export reviewer recipe')
        compare(verifier['member']['fit_content_sha256'], content, 'export reviewer fit content')
    require(bundle.get('diagnostic_only') is True and diagnostic['diagnostic_only'] is True,
            'failed guards must remain diagnostic-only')
    for record in records:
        frame, video, _ = current.predict_member(bundle['members'][0], record)
        expected = decoder_functions()[0 if bundle['decoder']['kind'] == 'duration-logit-v4' else 1](
            frame, record['fps'], video, bundle['decoder'])
        if bundle.get('proposal_verifier'):
            review, _, _ = current.predict_member(bundle['proposal_verifier']['member'], record)
            expected, _ = independent_review(expected, frame, review, bundle['proposal_verifier']['review'])
        result = current.predict_record(bundle, record)
        compare(result['intervals'], expected, 'diagnostic portable wiring')
    e.recheck()
    return {'default_snapshot_reference_sha256': PRIOR_DETECTOR_PIN,
        'current_detector_sha256': revision['current_detector_sha256'],
        'frozen_export_detector_sha256': V11_DETECTOR_PIN, 'hardening_source_review': revision,
        'P2_fail_closed_contract': contract, 'default_pairs_bitwise_identical': pairs,
        'active_branch_fixed_score_rows': branch_rows, 'active_proposal_counts': dict(branch_counts),
        'diagnostic_portable_wiring_rows': len(records),
        'final_verifier_enabled': bundle.get('proposal_verifier') is not None,
        'note': 'Active-branch cached-score replay is wiring evidence, not fullfit/fresh-pixel or sklearn parity.'}


def run_v11_audit(root=ROOT):
    e = Evidence(root)
    report, deployment = e.json(V11 / 'report.json'), e.json(V11 / 'deployment_selection.json')
    v10_receipt, baseline = verify_receipts(e)
    receipt = verify_v11_receipt(e, v10_receipt)
    source_review(e)
    records, outer, aliases = load_records(e)
    transforms, decoders = TransformEvidence(records), decoder_functions()
    for key, value in (('schema_version', 'proposal-review-v11'), ('status', 'complete'),
                       ('role', 'repeated_development_not_blind_or_confirmatory'),
                       ('protocol_sha256', V11_PROTOCOL_PIN), ('training_receipt_sha256', V11_RECEIPT_PIN)):
        compare(report.get(key), value, 'v11 report/' + key)
    compare(report.get('grouped_v1_baseline'), baseline, 'v11 baseline lineage')
    require(isinstance(report.get('folds'), list) and len(report['folds']) == 5, 'v11 outer fold coverage differs')
    primary, control, choices, held_cache = {}, {}, [], []
    coverage = {who: {phase: Counter() for phase in ('inner', 'outer', 'final')} for who in ('primary', 'reviewer')}
    for scope, val in enumerate(outer):
        train = [i for i in range(len(records)) if i not in set(val)]
        ip, op, _ = load_cache(e, RICH, scope, records, outer, transforms)
        iq, oq, _ = load_reviewer_cache(e, scope, records, outer, transforms)
        for who, inner, held in (('primary', ip, op), ('reviewer', iq, oq)):
            coverage[who]['inner'].update(inner.frame.keys()); coverage[who]['outer'].update(held.frame.keys())
        chosen = independent_review_choice(records, train, ip, iq, scope, decoders)
        prediction, trace, proposals = review_predictions(records, val, op, oq, chosen['config'], decoders)
        primary.update(prediction); control.update(proposals); choices.append(chosen); held_cache.append((op, oq, val))
        compare(report['folds'][scope], {'fold': scope, 'train_indices': train, 'validation_indices': val,
            'inner_partitions': expected_partitions(records, train, scope), 'primary': chosen,
            'primary_predictions': {str(i): [list(pair) for pair in prediction[i]] for i in val},
            'outer_review_trace': {str(i): trace[i] for i in val}}, 'v11/fold/' + str(scope))
    baseline_pred = read_prediction_rows(baseline['predictions'], records, 'v11/baseline')
    bm, metrics = grouped_metrics(records, baseline_pred), grouped_metrics(records, primary)
    compare(baseline['metrics'], bm, 'v11 baseline independent metrics')
    compare(report['primary_v11_nested'], {'metrics': metrics, 'predictions': prediction_rows(records, primary),
        'summary': prediction_summary(records, primary)}, 'v11 primary')
    compare(report['verifier_disabled_control'], {'metrics': grouped_metrics(records, control),
        'predictions': prediction_rows(records, control)}, 'v11 disabled control')
    guards = promotion_guards(metrics, bm)
    compare(report['statistical_promotion_checks'], guards, 'v11 guards')
    compare(report['statistical_promotion_passed'], all(guards.values()), 'v11 guard result')
    compare(report['paired_content_bootstrap'], paired_uncertainty(records, baseline_pred, primary),
            'v11 paired bootstrap')
    ip, held, _ = load_cache(e, RICH, 5, records, outer, transforms, final=True)
    iq, reviewer_held, _ = load_reviewer_cache(e, 5, records, outer, transforms, final=True)
    require(not held.frame and not held.video and not reviewer_held.frame and not reviewer_held.video,
            'v11 final consumes outer probabilities')
    ids = list(range(len(records)))
    coverage['primary']['final'].update(ip.frame.keys()); coverage['reviewer']['final'].update(iq.frame.keys())
    final = independent_review_choice(records, ids, ip, iq, 5, decoders)
    compare(deployment, {'status': 'complete', 'role': 'inner3_only_deployment_choice_not_validation',
        'chosen': final, 'selection_function': 'optimized_proposal_review_v11.choose_review',
        'inner_partitions': expected_partitions(records, ids, 5), 'uses_outer_probabilities': False,
        'uses_outer_metrics_for_selection': False, 'coverage_per_row': 1,
        'training_receipt_sha256': V11_RECEIPT_PIN}, 'v11 final selection')
    for who in coverage:
        for phase, times in (('inner', 4), ('outer', 1), ('final', 1)):
            require(coverage[who][phase] == Counter({i: times for i in ids}), 'v11 OOF coverage differs')
    require(len(transforms.proofs) == 46, 'v11 nested40/final6 fit evidence coverage differs')
    runtime = audit_detector(e, records, choices, held_cache, final)
    e.recheck()
    findings = [
        {'priority': 'P1', 'kind': 'promotion_blocker_not_audit_mismatch',
         'detail': 'Formal normal-FP and positive-empty guards failed; do not replace default models.'},
        {'priority': 'P2', 'kind': 'deployment_evidence_limit',
         'detail': 'Final inner3 chooses et_control; diagnostic export omits review. Enabled-branch '
                   'wiring is separately checked with fixed OOF scores, not learned/fullfit review validation.'},
        {'priority': 'P3', 'kind': 'locality_claim_limit',
         'detail': 'q75 only aggregates in-span frame scores, but those scores use the frozen offline '
                   'centered/future/global transform; not causal or strictly in-span raw-pixel evidence.'},
    ]
    return {'status': 'passed', 'scope': 'v11_independent_StageA_StageB_and_detector_read_only_review',
        'receipt_sha256': receipt['receipt_sha256'], 'trust_anchor_timing': 'observed_after_v11_reports',
        'outer_and_final_selection_recomputed': True, 'rows': len(records), 'aliases_retained': aliases,
        'nested_fit_evidence': 40, 'final_fit_evidence': 6, 'PCA_covariance_contexts': len(transforms.pcas),
        'fit_evidence': transforms.proofs, 'outer_choices': [c['config'] for c in choices],
        'final_direct_inner3_choice': final, 'primary_metrics': metrics,
        'statistical_promotion_checks': guards, 'statistical_promotion_passed': all(guards.values()),
        'deployment_approved': False, 'runtime_compatibility': runtime, 'findings': findings,
        'resolved_findings': [{'priority': 'P2', 'kind': 'generic_detector_contract_risk',
            'status': 'resolved_for_reviewed_structural_and_composition_contract',
            'detail': 'Current runtime rejects nonleaf/incomplete/unverified v4 reviewer and policy schemas, '
                      'base/reviewer/primary calibration-gate-boundary additions, nonsingle primary and '
                      'unsupported decoder/boundary combinations. Original export evidence remains frozen. '
                      'This is not full portable-tree/PCA parameter validation or a deployment promotion.'}],
        'consumed_artifact_sha256': dict(sorted(e.hashes.items())),
        'limitations': LIMITATIONS + ['V11 pins were observed after reports, not a pre-selection independent commitment.',
            'Historical per-fit sklearn/portable parity is only attested in signed metadata; no estimator states '
            'are available to replay those references. No training or fresh-pixel latency checks were run.']}



V12 = Path('output/algorithm-opt-v12-compact-boundary')
V12_RECEIPT_PIN = 'ef7c92b1453c2f4d483daf4aa87db12a84926edd10152e5e611b47c1d9c80338'
V12_PROTOCOL_PIN = '995a6ba2b336b17a25da7b3ea207174dedd8af0bebdc111e3341e877b126e2a8'
V12_BOUNDARY_SOURCE_PIN = 'f49de1549eb6faddb4cc56073ce156315fe96884f00aa2d7d09c1dabd0318f8a'
V12_BOUNDARY_MODEL_RECIPE = RICH


def v12_policies():
    return [{'boundary_seconds': 0., 'minimum_boundary_evidence': 0.}] + [
        {'boundary_seconds': seconds, 'minimum_boundary_evidence': minimum}
        for seconds in (.2, .4, .6) for minimum in (0., .3, .5)]


def independent_boundary_apply(intervals, heads, fps, policy, refiner):
    require(policy in v12_policies(), 'boundary policy is outside frozen ten-policy grid')
    start, end = np.asarray(heads[0]), np.asarray(heads[1])
    require(start.ndim == end.ndim == 1 and start.shape == end.shape and
            start.dtype.kind in 'iuf' and end.dtype.kind in 'iuf' and
            np.isfinite(start).all() and np.isfinite(end).all() and
            np.all((start >= 0) & (start <= 1)) and np.all((end >= 0) & (end <= 1)),
            'invalid boundary head probabilities')
    refined = refiner(intervals, start, end, fps, policy)
    accepted, trace = [], []
    minimum = policy['minimum_boundary_evidence']
    for left, right in refined:
        start_evidence, end_evidence = float(start[left]), float(end[right])
        keep = minimum == 0 or min(start_evidence, end_evidence) >= minimum
        if keep:
            accepted.append((left, right))
        trace.append({'span': [left, right], 'start_evidence': start_evidence,
                      'end_evidence': end_evidence, 'accepted': bool(keep)})
    return accepted, trace


def independent_boundary_predictions(records, ids, primary, heads, config, decoders, refiner):
    require(isinstance(config, dict) and config.get('kind') == 'compact-boundary-v12' and
            set(config) == {'kind', 'proposal', 'boundary'}, 'invalid V12 decoder schema')
    proposals = decode(records, ids, primary, config['proposal'], decoders)
    prediction, trace = {}, {}
    for i in ids:
        prediction[i], trace[i] = independent_boundary_apply(
            proposals[i], heads[i], records[i]['fps'], config['boundary'], refiner)
    return prediction, trace, proposals


def independent_boundary_choice(records, ids, primary, heads, scope, decoders, refiner):
    require(set(primary.frame) == set(primary.video) == set(ids) and set(heads) == set(ids),
            'V12 boundary choice coverage differs')
    restricted = RestrictedRecords(records, ids)
    base = independent_choice(restricted, ids, primary, SEED + scope, decoders)
    weights = content_weights(restricted, ids)
    truths = [restricted[i]['labels'] for i in ids]
    normals = np.asarray([not np.any(y) for y in truths], dtype=np.float64)
    positives = 1 - normals
    boot = bootstrap_counts(restricted, ids, SEED + scope) * weights[None, :]
    nm, pm, bn, bp = weights @ normals, weights @ positives, boot @ normals, boot @ positives
    rows, seen = [], set()
    for ordinal, policy in enumerate(v12_policies()):
        config = {'kind': 'compact-boundary-v12', 'proposal': deepcopy(base['config']),
                  'boundary': deepcopy(policy)}
        prediction, _, _ = independent_boundary_predictions(
            restricted, ids, primary, heads, config, decoders, refiner)
        fingerprint = tuple(tuple(prediction[i]) for i in ids)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        matrix = np.stack([row_statistics(prediction[i], restricted[i]['labels']) for i in ids])
        total = weights @ matrix
        pooled = float(utility(total, nm, pm))
        f05 = float(2 * total[3] / max(1e-12, 2 * total[3] + total[4] + total[5]))
        q20 = float(np.quantile(utility(boot @ matrix, bn, bp), .2, method='linear'))
        rows.append({'config': config, 'stats': total.tolist(), 'pooled_utility': pooled,
            'key': [pooled, f05, -total[10], -total[9], -total[11], -ordinal],
            'bootstrap_q20': q20, 'stable_utility': .75 * pooled + .25 * q20})
    require(rows, 'V12 boundary shortlist is empty')
    chosen = deepcopy(max(rows, key=lambda row: (row['stable_utility'], *row['key'][1:])))
    chosen.update(shortlist=rows, base_selection=base,
        candidate='boundary_control' if chosen['config']['boundary']['boundary_seconds'] == 0
        else 'compact_boundary_refine_accept')
    return chosen


def v12_source_review(e, v12_receipt):
    def parse(name):
        return ast.parse(e.read(Path('code') / name).decode('utf-8-sig'), filename=name)
    boundary = parse('optimized_compact_boundary_v12.py')
    runner = parse('run_compact_boundary_v12.py')
    head = ast.parse(e.read(Path('code/optimized_boundary_head.py'), V12_BOUNDARY_SOURCE_PIN).decode('utf-8-sig'),
                     filename='optimized_boundary_head.py')
    def fn(module, name):
        found = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(found) == 1, 'V12 source function missing/duplicate: ' + name)
        return found[0]
    fit = fn(boundary, 'fit_content_boundary')
    names = {node.id for node in ast.walk(fit) if isinstance(node, ast.Name)}
    require('primary' not in names and 'oof' not in {n.lower() for n in names},
            'boundary head fit reads frame predictions/OOF meta features')
    calls = [node for node in ast.walk(fit) if isinstance(node, ast.Call)]
    text = {ast.unparse(node) for node in calls}
    require('fit_rgb_pca(records, train)' in text and
            'feature_view_blocks(RECIPE, records[i], pca)' in text and
            'content_weights(records, train)' in text,
            'V12 head does not use train-only rich transform/content weights')
    require('targets = [_boundary_targets(records[i][\'labels\'], records[i][\'fps\']) for i in train]' in
            ast.unparse(fit), 'V12 targets are not train-label boundary targets')
    require('base = np.concatenate([np.full(records[i][\'frames\'], weights[i] / records[i][\'frames\']) for i in train])' in
            ast.unparse(fit), 'V12 base frame weights differ')
    require("models = {'start_model': _fit_head(x, starts, base, seed), 'end_model': _fit_head(x, ends, base, seed + 1)}" in
            ast.unparse(fit), 'V12 start/end seed separation differs')
    require('full_fit' in names and "set(train) != set(predict)" in ast.unparse(fit),
            'V12 full-fit/content-isolation guard missing')
    apply = fn(boundary, 'apply_boundary')
    require('refine_intervals(intervals, *heads, fps, cfg)' in ast.unparse(apply) and
            'min(start, end) >= minimum' in ast.unparse(apply),
            'V12 boundary refinement/evidence rule differs')
    choose = fn(boundary, 'choose_boundary')
    require('V8.choose(records, ids, primary, seed)' in ast.unparse(choose),
            'V12 StageA is not the declared V8 chooser')
    require('grouped_bootstrap(records, ids, seed, 32)' in ast.unparse(choose) and
            'for ordinal, policy in enumerate(configurations())' in ast.unparse(choose),
            'V12 StageB bootstrap/policy grid differs')
    # Existing head: exact fixed ET budget, balanced weights, and no labels at prediction.
    fit_head = fn(head, '_fit_head')
    et_calls = [node for node in ast.walk(fit_head) if isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and node.func.id == 'ExtraTreesClassifier']
    require(len(et_calls) == 1, 'boundary head estimator count differs')
    fixed_kwargs = {'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'n_jobs', 'class_weight'}
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in et_calls[0].keywords
              if kw.arg in fixed_kwargs}
    compare(kwargs, {'n_estimators': 128, 'max_depth': 7, 'min_samples_leaf': 5,
                    'max_features': .5, 'n_jobs': 2, 'class_weight': None}, 'boundary ET budget')
    require('random_state=seed' in ast.unparse(et_calls[0]) and
            'sample_weight=_balanced_weights(target, base)' in ast.unparse(fit_head),
            'boundary head seed/weights differ')
    predict_head = fn(head, 'predict_boundary_heads')
    pred_names = {node.id for node in ast.walk(predict_head) if isinstance(node, ast.Name)}
    require('labels' not in pred_names and 'target' not in pred_names,
            'boundary prediction path reads labels/targets')
    # Runner: 5 outer jobs, final scope 5, three inner pairs, seed +2200, no frame fits.
    run_text = ast.unparse(runner)
    require("part['seed'] + 2200" in run_text and 'fit_content_boundary' in run_text and
            "jobs = [(5, True)] if final else [(f, False) for f in range(5)]" in run_text,
            'V12 fit job/seed protocol differs')
    require('V10.load_fold(5, V10.RICH, records, V10.check_receipt(), final=True)' in run_text and
            'choose_boundary(records, ids, p, heads, 5)' in run_text,
            'V12 final selection is not direct scope5 inner3')
    require('new_frame_model_fits' in ast.unparse(fn(runner, 'protocol')),
            'V12 protocol lacks zero new frame fits')
    return {'boundary_train_only_raw_features': True, 'start_end_heads_fixed_budget': True,
            'head_estimator': kwargs, 'head_prediction_label_free': True,
            'stage_a_v8_choice_declared': True, 'stage_b_ten_policies_declared': True,
            'seed_offset_2200_declared': True, 'final_scope5_inner3_declared': True,
            'no_new_frame_fits_declared': True}


def verify_v12_receipt(e, v10_receipt):
    receipt = e.json(V12 / 'training_receipt.json')
    require(receipt.get('receipt_sha256') == V12_RECEIPT_PIN and
            value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) == V12_RECEIPT_PIN,
            'V12 receipt differs from observed anchor')
    compare(receipt['parent_receipts'], {'v8': PARENT_PIN, 'v10': RECEIPT_PIN}, 'V12/parent receipts')
    require(receipt['input_hashes'] == v10_receipt['input_hashes'] and
            receipt['inputs_sha256'] == INPUTS_PIN, 'V12 input lineage differs')
    for name, digest in v10_receipt['sources'].items():
        require(receipt['sources'].get(name) == digest, 'V12/V10 source lineage differs: ' + name)
    names = tuple(dict.fromkeys(('run_compact_boundary_v12.py', 'optimized_compact_boundary_v12.py',
        'optimized_boundary_head.py') + SOURCES))
    require(set(receipt['sources']) == set(names), 'V12 source inventory differs')
    for name, digest in receipt['sources'].items():
        e.read(Path('code') / name, digest)
        e.read(V12 / 'sources' / name, digest)
    expected = {(PARENT / 'training' / f'{scope}_{PARENT_RICH}.{ext}').as_posix()
        for scope in range(5) for ext in ('npz', 'json')}
    expected |= {(OUT / 'final_selection' / f'5_{RICH}.{ext}').as_posix() for ext in ('npz', 'json')}
    require(set(receipt['consumed_artifacts']) == expected, 'V12 parent cache coverage differs')
    for path, digest in receipt['consumed_artifacts'].items():
        e.read(path, digest)
    require(receipt['protocol_sha256'] == V12_PROTOCOL_PIN and receipt['baseline_sha256'] == BASELINE_PIN,
            'V12 protocol/baseline pins differ')
    compare(receipt['default_models'], v10_receipt['default_models'], 'V12/default models')
    protocol = e.json(V12 / 'protocol.json', V12_PROTOCOL_PIN)
    compare(protocol['schema_version'], 'compact-boundary-v12', 'V12/protocol schema')
    compare(protocol['boundary_policies'], v12_policies(), 'V12 ten boundary policies')
    compare(protocol['promotion_guards'], GUARD_SPEC, 'V12/protocol guards')
    for key, value in (('new_nested_boundary_pair_fits', 20),
                       ('new_final_boundary_pair_fits', 3), ('new_frame_model_fits', 0)):
        compare(protocol[key], value, 'V12/' + key)
    compare(protocol['partitions'], 'Frozen SHA grouped outer5/inner3; final scope5 direct inner3, no averaging other experiments OOF.',
            'V12/partitions')
    return receipt


def verify_v12_transform(records, fit, evidence, transforms):
    key = tuple(fit)
    if key not in transforms.pcas:
        transforms.pcas[key] = reconstruct_pca(records, fit)
    pca = transforms.pcas[key]
    first = min(fit)
    x, video, names = transforms.fresh(RICH, records[first], pca)
    transform = {'feature_view': 'compact-feature-blocks-v10', 'pca': pca,
                 'frame_feature_names': names}
    require(is_hash(evidence.get('transform_sha256')) and value_hash(transform) == evidence['transform_sha256'],
            'V12 PCA/transform digest differs')
    require(is_hash(evidence.get('models_sha256')), 'V12 head model digest missing/invalid')
    require(type(evidence.get('base_content_mass')) in (int, float) and
            math.isfinite(evidence['base_content_mass']) and
            evidence['base_content_mass'] == len({records[i]['sha256'] for i in fit}),
            'V12 base content mass differs')
    proof = {'recipe': RICH, 'partition': evidence.get('partition'),
             'fit_content_sha256': sorted({records[i]['sha256'] for i in fit}),
             'PCA': 'train_covariance_digest_reconstructed',
             'transform_sha256': evidence['transform_sha256'], 'models_sha256': evidence['models_sha256'],
             'frame_features': int(x.shape[1]), 'video_features': int(len(video))}
    transforms.proofs.append(proof)
    return {'fit_content_sha256': sorted({records[i]['sha256'] for i in fit}),
            'transform_sha256': evidence['transform_sha256'], 'models_sha256': evidence['models_sha256'],
            'frame_features': int(x.shape[1]), 'video_features': int(len(video))}


def v12_boundary_arrays(data, metadata, records, label):
    require(is_hash(metadata.get('npz_sha256')) and sha(data) == metadata['npz_sha256'],
            'V12 boundary NPZ SHA differs: ' + label)
    arrays = read_archive(data, label)
    expected = {f'{kind}_{head}_{i}' for kind, ids in (('inner', metadata['train']),
        ('outer', metadata['validation'])) for head in ('start', 'end') for i in ids}
    require(set(arrays) == expected, 'V12 boundary key/OOF coverage differs: ' + label)
    result = {}
    for kind, ids in (('inner', metadata['train']), ('outer', metadata['validation'])):
        for i in ids:
            heads = []
            for head in ('start', 'end'):
                value = arrays[f'{kind}_{head}_{i}']
                require(value.dtype == np.dtype('float32') and value.shape == (records[i]['frames'],)
                        and np.isfinite(value).all() and np.all((value >= 0) & (value <= 1)),
                        'V12 boundary tensor invalid: ' + label + '/' + kind + '/' + head)
                heads.append(value)
            result.setdefault(kind, {})[i] = (heads[0], heads[1])
    return result.get('inner', {}), result.get('outer', {})



def verify_v12_head_metadata(metadata, scope, records, outer, receipt, *, final=False):
    require(isinstance(metadata, dict) and is_hash(metadata.get('signature')) and
            value_hash({k: v for k, v in metadata.items() if k != 'signature'}) == metadata['signature'],
            'V12 boundary metadata signature differs')
    train = list(range(len(records))) if final else [i for i in range(len(records)) if i not in set(outer[scope])]
    val = [] if final else outer[scope]
    parts = expected_partitions(records, train, scope)
    for key, value in (('scope', scope), ('final_selection', final), ('train', train),
                       ('validation', val), ('inner_partitions', parts),
                       ('training_receipt_sha256', receipt['receipt_sha256'])):
        compare(metadata.get(key), value, 'V12 cache/' + key)
    expected = parts + ([] if final else [{'fit': train, 'validation': val,
                                            'seed': SEED + scope * 53 + 99}])
    proofs = metadata.get('fit_evidence')
    require(isinstance(proofs, list) and len(proofs) == len(expected), 'V12 head fit evidence coverage differs')
    for ordinal, (proof, part) in enumerate(zip(proofs, expected)):
        compare(proof.get('partition'), ordinal if ordinal < 3 else 'outer', 'V12 fit partition')
        a = {records[i]['sha256'] for i in part['fit']}
        b = {records[i]['sha256'] for i in part['validation']}
        require(not a & b and proof.get('fit_content_sha256') == sorted(a), 'V12 fit content leakage/proof differs')
        compare(proof.get('seed'), part['seed'] + 2200, 'V12 fit seed')
        require(is_hash(proof.get('transform_sha256')) and is_hash(proof.get('models_sha256')),
                'V12 transform/head model hash missing or invalid')
        compare(proof.get('base_content_mass'), float(len(a)), 'V12 fit base content mass')
    return expected


def load_v12_heads(e, scope, records, outer, receipt, transforms, *, final=False):
    path = V12 / ('final_selection' if final else 'training') / f'{scope}_boundary.npz'
    metadata = e.json(path.with_suffix('.json'))
    require(is_hash(metadata.get('signature')) and
            value_hash({k: v for k, v in metadata.items() if k != 'signature'}) == metadata['signature'],
            'V12 boundary metadata signature differs')
    expected = verify_v12_head_metadata(metadata, scope, records, outer, receipt, final=final)
    for proof, part in zip(metadata['fit_evidence'], expected):
        verify_v12_transform(records, part['fit'], proof, transforms)
    data = e.read(path)
    return (*v12_boundary_arrays(data, metadata, records, path.as_posix()), metadata)


def audit_v12_detector_contract(e, final_choice, receipt):
    detector_bytes, revision = read_v11_detector(e)
    text = detector_bytes.decode('utf-8-sig')
    require('compact-boundary-v12' not in text and 'minimum_boundary_evidence' not in text,
            'detector unexpectedly claims an unreviewed V12 adapter')
    for name, digest in receipt['default_models'].items():
        e.read(Path('models') / name, digest)
    # There is deliberately no V12 full-fit deployment bundle. The current generic
    # detector accepts duration/recall decoders, not the nested V12 decoder schema.
    return {'detector_supports_v12_schema': False, 'default_models_unchanged': True,
            'detector_revision': revision,
            'final_choice_is_not_deployable_without_adapter': True,
            'final_candidate': final_choice['candidate']}



def verify_v12_training_reports(e):
    summaries = []
    for final, name in ((False, 'training_report.json'), (True, 'final_training_report.json')):
        report = e.json(V12 / name)
        compare(report.get('status'), 'complete', 'V12/' + name + '/status')
        compare(report.get('boundary_pair_fits'), 3 if final else 20, 'V12/' + name + '/fit count')
        expected = {(5, True)} if final else {(scope, False) for scope in range(5)}
        results = report.get('results')
        require(isinstance(results, list) and len(results) == len(expected),
                'V12 training result coverage differs: ' + name)
        identities = []
        for row in results:
            require(isinstance(row, dict) and type(row.get('scope')) is int and
                    type(row.get('final_selection')) is bool, 'V12 training result identity invalid')
            elapsed = row.get('seconds')
            require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                    'V12 training elapsed value invalid')
            identities.append((row['scope'], row['final_selection']))
        require(len(set(identities)) == len(identities) and set(identities) == expected,
                'V12 duplicate/missing training scope or incorrect final flag')
        summaries.append({'file': name, 'jobs': len(results),
                          'boundary_pair_fits': report['boundary_pair_fits'], 'complete': True})
    return summaries


def run_v12_audit(root=ROOT):
    e = Evidence(root)
    report = e.json(V12 / 'report.json')
    deployment = e.json(V12 / 'deployment_selection.json')
    v10_receipt, baseline = verify_receipts(e)
    receipt = verify_v12_receipt(e, v10_receipt)
    static = v12_source_review(e, receipt)
    training_summaries = verify_v12_training_reports(e)
    records, outer, aliases = load_records(e)
    transforms, decoders = TransformEvidence(records), decoder_functions()
    boundary_head = __import__('optimized_boundary_head').refine_intervals
    for key, value in (('schema_version', 'compact-boundary-v12'), ('status', 'complete'),
                       ('role', 'repeated_development_not_blind_or_confirmatory'),
                       ('protocol_sha256', V12_PROTOCOL_PIN), ('training_receipt_sha256', V12_RECEIPT_PIN)):
        compare(report.get(key), value, 'V12 report/' + key)
    compare(report.get('grouped_v1_baseline'), baseline, 'V12 baseline lineage')
    require(isinstance(report.get('folds'), list) and len(report['folds']) == 5, 'V12 outer fold coverage differs')
    primary, control, choices = {}, {}, []
    coverage = {'primary': {phase: Counter() for phase in ('inner', 'outer', 'final')},
                'boundary': {phase: Counter() for phase in ('inner', 'outer', 'final')}}
    for scope, val in enumerate(outer):
        train = [i for i in range(len(records)) if i not in set(val)]
        ip, op, _ = load_cache(e, RICH, scope, records, outer, transforms)
        ih, oh, _ = load_v12_heads(e, scope, records, outer, receipt, transforms)
        coverage['primary']['inner'].update(ip.frame.keys()); coverage['primary']['outer'].update(op.frame.keys())
        coverage['boundary']['inner'].update(ih.keys()); coverage['boundary']['outer'].update(oh.keys())
        chosen = independent_boundary_choice(records, train, ip, ih, scope, decoders, boundary_head)
        prediction, trace, proposals = independent_boundary_predictions(
            records, val, op, oh, chosen['config'], decoders, boundary_head)
        primary.update(prediction); control.update(proposals); choices.append(chosen)
        compare(report['folds'][scope], {'fold': scope, 'train_indices': train,
            'validation_indices': val, 'inner_partitions': expected_partitions(records, train, scope),
            'primary': chosen, 'primary_predictions': {str(i): [list(pair) for pair in prediction[i]] for i in val},
            'outer_boundary_trace': {str(i): trace[i] for i in val}}, 'V12/fold/' + str(scope))
    baseline_pred = read_prediction_rows(baseline['predictions'], records, 'V12/baseline')
    bm, metrics = grouped_metrics(records, baseline_pred), grouped_metrics(records, primary)
    compare(baseline['metrics'], bm, 'V12 baseline independent metrics')
    compare(report['primary_v12_nested'], {'metrics': metrics, 'predictions': prediction_rows(records, primary),
        'summary': prediction_summary(records, primary)}, 'V12 primary')
    compare(report['boundary_disabled_control'], {'metrics': grouped_metrics(records, control),
        'predictions': prediction_rows(records, control)}, 'V12 disabled control')
    guards = promotion_guards(metrics, bm)
    compare(report['statistical_promotion_checks'], guards, 'V12 guards')
    compare(report['statistical_promotion_passed'], all(guards.values()), 'V12 guard result')
    compare(report['paired_content_bootstrap'], paired_uncertainty(records, baseline_pred, primary),
            'V12 paired bootstrap')
    ip, _, _ = load_cache(e, RICH, 5, records, outer, transforms, final=True)
    ih, oh, _ = load_v12_heads(e, 5, records, outer, receipt, transforms, final=True)
    require(not oh, 'V12 final boundary cache contains outer probabilities')
    ids = list(range(len(records)))
    coverage['primary']['final'].update(ip.frame.keys()); coverage['boundary']['final'].update(ih.keys())
    final = independent_boundary_choice(records, ids, ip, ih, 5, decoders, boundary_head)
    compare(deployment, {'status': 'complete', 'role': 'inner3_only_deployment_choice_not_validation',
        'chosen': final, 'inner_partitions': expected_partitions(records, ids, 5),
        'uses_outer_probabilities': False, 'uses_outer_metrics_for_selection': False,
        'coverage_per_row': 1, 'selection_function': 'optimized_compact_boundary_v12.choose_boundary',
        'training_receipt_sha256': V12_RECEIPT_PIN}, 'V12 final selection')
    for who in coverage:
        for phase, times in (('inner', 4), ('outer', 1), ('final', 1)):
            require(coverage[who][phase] == Counter({i: times for i in ids}),
                    'V12 per-row OOF coverage differs: ' + who + '/' + phase)
    require(len(transforms.proofs) == 46, 'V12 20 parent + 3 final + 23 boundary transform proofs expected')
    detector = audit_v12_detector_contract(e, final, receipt)
    e.recheck()
    findings = [
        {'priority': 'P1', 'kind': 'promotion_blocker_not_audit_mismatch',
         'detail': 'Formal normal-FP guard failed; positive-empty and event/frame guards passed. Do not promote V12.'},
        {'priority': 'P1', 'kind': 'deployment_adapter_missing',
         'detail': 'Current optimized_detector.py has no compact-boundary-v12/minimum_boundary_evidence branch or V12 export. '
                   'The final control choice is not a detector-compatible deployment bundle by itself.'},
        {'priority': 'P2', 'kind': 'head_model_replay_limit',
         'detail': 'Audit verifies head model hash fields, train-content/PCA transforms, seeds, and cached start/end tensors. '
                   'No estimator state is stored, so historical _fit_head execution cannot be independently refit/replayed without training.'},
        {'priority': 'P3', 'kind': 'boundary_scope_limit',
         'detail': 'Boundary head can refine existing ET proposals only; it cannot recover missing events or split merged proposals. '
                   'The final inner3 chose boundary_control, so no boundary refinement is selected for deployment.'},
    ]
    return {'status': 'passed', 'scope': 'v12_independent_StageA_StageB_boundary_cache_and_deployment_read_only_review',
        'receipt_sha256': receipt['receipt_sha256'], 'trust_anchor_timing': 'observed_after_v12_reports',
        'outer_and_final_selection_recomputed': True, 'rows': len(records), 'aliases_retained': aliases,
        'nested_boundary_pair_fits': 20, 'final_boundary_pair_fits': 3,
        'PCA_covariance_contexts': len(transforms.pcas), 'fit_evidence': transforms.proofs,
        'training_reports': training_summaries,
        'head_model_state_replay': {'status': 'unavailable',
            'model_hash_fields_valid': True, 'model_hash_recomputed_from_saved_states': False,
            'reason': 'per-fit start/end estimator states were not saved'},
        'outer_choices': [c['config'] for c in choices], 'final_direct_inner3_choice': final,
        'primary_metrics': metrics, 'statistical_promotion_checks': guards,
        'statistical_promotion_passed': all(guards.values()), 'deployment_approved': False,
        'source_review': static, 'detector_compatibility': detector, 'findings': findings,
        'consumed_artifact_sha256': dict(sorted(e.hashes.items())),
        'limitations': LIMITATIONS + ['V12 pins were observed after reports, not pre-registered in an external immutable log.',
            'Boundary model hashes are unkeyed consistency digests and no estimator state is available for replay.',
            'No new V12 production bundle/fresh-pixel/latency test exists; detector support is explicitly absent.']}

def unsupported_json(value):
    raise TypeError('unsupported audit output type: ' + type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', '--experiment', dest='mode', choices=('v10', 'v11', 'v12'), default='v10', help='frozen experiment to audit')
    parser.add_argument('--output', type=Path, required=True, help='NEW audit JSON; exclusive creation only')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        require(output.suffix.lower() == '.json', 'output must be a new .json file')
        require(output.name not in {'report.json', 'deployment_selection.json', 'training_receipt.json',
            'protocol.json', 'training_report.json', 'final_training_report.json', 'dataset_manifest.json'},
            'output cannot impersonate an audited input')
        if output.is_relative_to(ROOT.resolve()):
            parts = output.relative_to(ROOT.resolve()).parts
            require(parts[0] not in {'code', 'models', 'data'} and
                    not set(parts) & {'sources', 'training', 'final_selection'},
                    'output cannot modify code/models/data or frozen fit directories')
    except AuditError as exc:
        print('FAIL: ' + str(exc), file=sys.stderr)
        return 2
    # No mkdir, overwrite, recursive operations, or default output in the frozen namespace.
    # Reserve the only output before expensive replay; a racing writer cannot overwrite it.
    try:
        stream = output.open('x', encoding='utf-8')
    except OSError as exc:
        print('FAIL: output must be new and its parent must exist: ' + str(exc), file=sys.stderr)
        return 2
    start = time.perf_counter()
    with stream:
        try:
            result = run_audit() if args.mode == 'v10' else run_v11_audit() if args.mode == 'v11' else run_v12_audit()
            code = 0
        except MissingArtifact as exc:
            result = {'status': 'failed', 'error_kind': 'missing_artifact', 'error': str(exc),
                      'deployment_approved': False, 'limitations': LIMITATIONS}
            code = 2
        except (AuditError, OSError, KeyError, TypeError, ImportError, IndexError) as exc:
            result = {'status': 'failed', 'error_kind': 'inconsistent_evidence_or_runtime',
                      'error': str(exc), 'deployment_approved': False, 'limitations': LIMITATIONS}
            code = 1
        except Exception as exc:
            # Unexpected decoder/NumPy errors must also produce an explicit failure.
            result = {'status': 'failed', 'error_kind': 'unexpected_audit_failure',
                      'error': type(exc).__name__ + ': ' + str(exc), 'deployment_approved': False}
            code = 1
        result.update(schema_version='independent-feature-block-audit-' + args.mode,
            finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start,
            auditor_source_sha256=sha(Path(__file__).read_bytes()))
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=lambda value: value.item() if isinstance(value, np.generic) else unsupported_json(value))
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(output), 'exit_code': code}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
