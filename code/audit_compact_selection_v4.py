#!/usr/bin/env python3
"""Independent, read-only replay of frozen 2026-10-02 v4/v5 selection.

Run only after the main worker announces the chosen report.json completion:
    python -B code/audit_compact_selection_v4.py --reports-complete
    python -B code/audit_compact_selection_v4.py --experiment v5 --reports-complete

No polling, waiting, fitting, extraction, model imports, or outer-best selection.
Only the two validated decoders are reused, lazily after source verification.
Matching, frame statistics, weighting, grouping, bootstrap, utility, shortlist,
ranking, metrics, summaries and guards are independently implemented here.
Output is created exclusively. Missing/unfinished inputs exit 2 without output;
a completed inconsistent experiment produces a failed audit (exit 1).
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
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1 = Path('output/algorithm-opt-2026-10-02')
V2 = Path('output/algorithm-opt-2026-10-02-v2')
V4 = Path('output/algorithm-opt-2026-10-02-v4')
SEED = 20261002
OUTPUT_NAME = 'independent_selection_audit.json'
ATOL = 1e-12
# Observed before report.json existed: never regenerate from a supplied receipt.
RECEIPT_PIN = '3ef89888ae2560607e40770032879edfcd5e0f6ef6379efd74678bbb3f5635f4'
PROTOCOL_PIN = '831235be821641d31a8a56ab2dac37dc6f90d5ccfd28ac6aec83837cc79cfbe1'
BASELINE_PIN = 'bd659ea747a6c662ced8fdeae41878c470930be93289dee317e0db26668b7202'
RECIPE_NAMES = ('compact_corrected_motion_et', 'compact_rgb_corrected_motion_et',
                'compact_rgb_corrected_motion_hgb', 'compact_motion_hgb')
CANDIDATES = (
    ('compact_corrected_motion_et', (('compact_corrected_motion_et', 1.),)),
    ('compact_rgb_corrected_motion_et', (('compact_rgb_corrected_motion_et', 1.),)),
    ('compact_rgb_corrected_motion_hgb', (('compact_rgb_corrected_motion_hgb', 1.),)),
    ('corrected_motion_et_duration_control', (('corrected_motion_et', 1.),)),
    ('compact_rgb_boost_old_70_30', (('compact_rgb_corrected_motion_hgb', .7),
                                   ('corrected_motion_et', .3))),
)
SOURCE_NAMES = (
    'run_compact_experiment_v4.py', 'optimized_compact_features_v4.py',
    'optimized_compact_model_v4.py', 'optimized_duration_decoder_v4.py',
    'optimized_grouped_training.py', 'optimized_feature_view_v2.py',
    'optimized_feature_view.py', 'optimized_locator.py', 'optimized_recall_decoder.py',
    'optimized_video_statistics.py', 'optimized_training_provenance.py',
    'run_recall_selection.py', 'run_optimization_selection.py',
    'run_algorithm_optimization.py', 'build_optimization_dataset.py',
)
LEGACY_SOURCE_NAMES = (
    'optimized_grouped_training.py', 'run_recall_training.py',
    'run_algorithm_optimization.py', 'optimized_feature_view.py',
    'optimized_feature_view_v2.py', 'optimized_locator.py',
    'optimized_temporal_head.py', 'optimized_boundary_head.py',
)
GUARD_LIMITS = {
    'event_f1_03_no_worse': 0., 'event_f1_05_improved_02': .02,
    'frame_f1_drop_at_most_005': -.005, 'normal_fp_no_worse': 0,
    'positive_empty_improved_3': -3, 'portable_fresh_parity_max': 2e-6,
}
STAT_NAMES = (
    'event03_tp', 'event03_fp', 'event03_fn', 'event05_tp', 'event05_fp',
    'event05_fn', 'frame_tp', 'frame_fp', 'frame_fn', 'normal_fp_videos',
    'positive_empty_videos', 'predicted_segments',
)


@dataclass(frozen=True)
class Experiment:
    version: str
    folder: Path
    schema: str
    recipes: tuple
    candidates: tuple
    sources: tuple
    runner: str
    chooser: str
    receipt_pin: str
    protocol_pin: str
    empty_penalty: float
    fast_recipe: str | None
    trust_anchor_timing: str


V4_EXPERIMENT = Experiment(
    'v4', V4, 'compact-experiment-v4', RECIPE_NAMES, CANDIDATES, SOURCE_NAMES,
    'run_compact_experiment_v4.py', 'choose_v4', RECEIPT_PIN, PROTOCOL_PIN,
    .15, 'compact_motion_hgb', 'observed_before_selection_report')
V5_RECIPES = ('event_corrected_motion_et', 'event_rgb_corrected_motion_et',
              'event_rgb_corrected_motion_hgb')
V5_CANDIDATES = tuple((r, ((r, 1.),)) for r in V5_RECIPES) + (
    ('event_old_et_50_50', (('event_corrected_motion_et', .5), ('corrected_motion_et', .5))),
    ('old_et_recall_control', (('corrected_motion_et', 1.),)))
V5_EXPERIMENT = Experiment(
    'v5', Path('output/algorithm-opt-2026-10-02-v5'), 'event-experiment-v5',
    V5_RECIPES, V5_CANDIDATES,
    ('run_event_experiment_v5.py', 'optimized_event_training_v5.py') + SOURCE_NAMES[1:],
    'run_event_experiment_v5.py', 'choose_v5',
    '9e8f609487ed5f364ea7793c8ce35ea2d68debaee14fc25655bff6a9cc8692b8',
    'b5c2b36e7acf64be04706bf6a6d2e94a2ddbdca9694d92bff8c7bf462352ee3c',
    .30, None, 'observed_after_main_worker_completion_notice')
EXPERIMENTS = {'v4': V4_EXPERIMENT, 'v5': V5_EXPERIMENT}


class AuditError(ValueError):
    """A completed artifact failed an integrity/contract check."""


class IncompleteArtifacts(AuditError):
    """Not ready: no waiting, retries, partial audit or output file."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def value_hash(value):
    return sha(canonical(value))


def is_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def strict_json(data, label):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + label + '/' + key)
            result[key] = value
        return result

    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON number: ' + label)
        return value

    def invalid(text):
        raise AuditError('nonfinite JSON constant: ' + label)

    try:
        return json.loads(data, object_pairs_hook=unique, parse_float=finite_float,
                          parse_constant=invalid)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise AuditError('invalid JSON: ' + label + ': ' + str(exc)) from exc


def safe_path(root, relative):
    text = str(relative)
    posix, windows = PurePosixPath(text), PureWindowsPath(text)
    require(text and not posix.is_absolute() and not windows.drive
            and not windows.is_absolute() and '\\' not in text and ':' not in text
            and '..' not in posix.parts, 'unsafe artifact path: ' + text)
    path = (root / text).resolve()
    require(path.is_relative_to(root.resolve()), 'artifact escapes workspace: ' + text)
    return path


class Evidence:
    """Consume the same bytes we hash; recheck consumed files at the end."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def read(self, relative, expected=None):
        key = relative.as_posix() if isinstance(relative, Path) else str(PurePosixPath(relative))
        path = safe_path(self.root, key)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise AuditError('missing/unreadable artifact: ' + key) from exc
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
        for relative, expected in list(self.hashes.items()):
            self.read(relative, expected)


def configurations():
    """Independent literal replay, not a protocol-supplied arbitrary grid."""
    result = []
    for threshold in (.3, .4, .5, .6):
        for transition in (.03, .10, .25):
            for gate in (0, .7):
                result.append({'kind': 'duration-logit-v4', 'threshold': threshold,
                               'transition_seconds': transition, 'min_seconds': .12,
                               'video_threshold': gate})
        for ratio in (1., .7):
            for gate in (0, .7):
                result.append({'kind': 'recall-stable-v2', 'threshold': threshold,
                               'low_ratio': ratio, 'smooth_seconds': .15,
                               'gap_seconds': .12, 'min_seconds': .12,
                               'seed_seconds': .05, 'video_threshold': gate,
                               'video_strength': 0, 'video_floor': 0})
    return result


def complete_inputs(root, acknowledged, experiment=V4_EXPERIMENT):
    """One-shot readiness gate, before decoder imports or data replay."""
    if not acknowledged:
        raise IncompleteArtifacts('requires --reports-complete after the main worker notification')
    root = Path(root).resolve()
    report, training = root / experiment.folder / 'report.json', root / experiment.folder / 'training_report.json'
    needed = [report, training, root / experiment.folder / 'protocol.json',
              root / experiment.folder / 'training_receipt.json', root / V1 / 'dataset_manifest.json',
              root / V1 / 'dataset.npz', root / V2 / 'dataset_manifest.json',
              root / V2 / 'selection_audited/report.json']
    needed.extend(root / experiment.folder / 'sources' / name for name in experiment.sources)
    needed.extend(root / V2 / 'grouped_training/sources' / name for name in LEGACY_SOURCE_NAMES)
    for fold in range(5):
        for recipe in experiment.recipes:
            needed.extend(root / experiment.folder / 'training' / f'{fold}_{recipe}.{suffix}'
                          for suffix in ('json', 'npz'))
        needed.extend(root / V2 / 'grouped_training/folds' /
                      f'{fold}_corrected_motion_et.{suffix}' for suffix in ('json', 'npz'))
    missing = [str(p.relative_to(root)) for p in needed if not p.is_file()]
    if missing:
        raise IncompleteArtifacts('incomplete artifacts (no wait): ' + ', '.join(missing))
    for path in (report, training):
        try:
            value = strict_json(path.read_bytes(), str(path.relative_to(root)))
        except (AuditError, OSError) as exc:
            raise IncompleteArtifacts('report not readable/complete: ' + path.name) from exc
        if not isinstance(value, dict) or value.get('status') != 'complete':
            raise IncompleteArtifacts('report status is not complete: ' + path.name)
    receipt = strict_json((root / experiment.folder / 'training_receipt.json').read_bytes(), 'receipt readiness')
    for section in ('input_hashes', 'input_source_hashes'):
        require(isinstance(receipt.get(section), dict), 'invalid receipt readiness inventory')
        missing = [name for name in receipt[section] if not safe_path(root, name).is_file()]
        if missing:
            raise IncompleteArtifacts('missing receipt-bound inputs (no wait): ' + ', '.join(missing))
    if (root / experiment.folder / OUTPUT_NAME).exists():
        raise IncompleteArtifacts('audit output exists; refusing overwrite: ' + OUTPUT_NAME)


def frozen_inventory(root, experiment=V4_EXPERIMENT):
    folder = safe_path(root, experiment.folder.as_posix() + '/training')
    expected = {f'{f}_{r}.{ext}' for f in range(5) for r in experiment.recipes for ext in ('json', 'npz')}
    actual = {p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.json', '.npz')}
    require(actual == expected, f'training inventory must be exactly {5 * len(experiment.recipes)} NPZ + metadata')
    sources = safe_path(root, experiment.folder.as_posix() + '/sources')
    require({p.name for p in sources.iterdir() if p.is_file()} == set(experiment.sources),
            'frozen source inventory mismatch')


def verify_receipt(evidence, experiment=V4_EXPERIMENT):
    receipt = evidence.json(experiment.folder / 'training_receipt.json')
    require(isinstance(receipt, dict), 'invalid training receipt')
    core = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
    require(receipt.get('receipt_sha256') == experiment.receipt_pin and value_hash(core) == experiment.receipt_pin,
            'receipt differs from the pinned experiment trust anchor')
    require(receipt.get('protocol_sha256') == experiment.protocol_pin
            and receipt.get('baseline_report_sha256') == BASELINE_PIN, 'receipt trust anchors differ')
    protocol = evidence.json(experiment.folder / 'protocol.json', experiment.protocol_pin)
    require(protocol.get('schema_version') == experiment.schema
            and protocol.get('date') == '2026-10-02', 'unexpected frozen protocol')
    require(protocol.get('trained_recipes') == list(experiment.recipes), 'trained recipe order differs')
    require(protocol.get('primary_candidates') == [[n, [list(m) for m in members]] for n, members in experiment.candidates],
            'candidate/blend order differs')
    require(protocol.get('decoder_configs') == configurations(), '40-config decoder grid differs')
    require(protocol.get('promotion_guards') == GUARD_LIMITS, 'guard limits differ')
    require(set(receipt['sources']) == set(experiment.sources), 'source coverage differs')
    for name, digest in receipt['sources'].items():
        evidence.read(experiment.folder / 'sources' / name, digest)
        evidence.read(Path('code') / name, digest)
    require(len(receipt['input_source_hashes']) == 11, 'training input-source coverage differs')
    for name, digest in receipt['input_source_hashes'].items():
        evidence.read(name, digest)
    require(len(receipt['input_hashes']) == 479
            and value_hash(receipt['input_hashes']) == receipt['inputs_sha256'], 'input inventory/digest differs')
    for name, digest in receipt['input_hashes'].items():
        evidence.read(name, digest)
    evidence.read(V2 / 'selection_audited/report.json', BASELINE_PIN)
    return receipt, protocol


def source_review(evidence, experiment=V4_EXPERIMENT):
    """AST is inspected, never executed; pins bind the reviewed implementation."""
    name = experiment.folder / 'sources' / experiment.runner
    tree = ast.parse(evidence.read(name).decode('utf-8'))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    chooser = funcs[experiment.chooser]
    calls = {n.func.id for n in ast.walk(chooser)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    require(not calls & {'metrics', '_group_metrics', 'load_new_fold', 'load_fold'},
            'chooser invokes outer metrics or cache loading')
    fold_loop = next(n for n in funcs['select_all'].body if isinstance(n, ast.For))
    decision = next(n for n in fold_loop.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'decision' for t in n.targets))
    outer_decode = next(n for n in fold_loop.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'pred' for t in n.targets))
    require(decision.lineno < outer_decode.lineno, 'choice is not locked before outer decode')
    require(isinstance(decision.value, ast.Call) and isinstance(decision.value.func, ast.Name)
            and decision.value.func.id == 'max' and isinstance(decision.value.args[0], ast.Name)
            and decision.value.args[0].id == 'rows', 'primary decision is not inner-row ranking')
    return {'source': name.as_posix(), 'sha256': evidence.hashes[name.as_posix()],
            'chooser_lines': [chooser.lineno, chooser.end_lineno],
            'inner_candidate_lock_line': decision.lineno,
            'first_primary_outer_decode_line': outer_decode.lineno,
            'findings': [
                'Only inner train indices/probabilities enter configuration selection.',
                'Prediction fingerprints deduplicate; earliest configuration ordinal survives.',
                'Pooled top8 precedes bootstrap q20; excluded configurations cannot win.',
                'Candidate ties: stable utility, F1@.5, fewer empty/FP/segments, config then candidate order.',
                'Primary is locked before outer decoding; fixed candidates are diagnostics only.',
                'Choosers/statistics/grouping are independently replayed, not imported or executed.',
            ]}

def label_spans(labels):
    """Plain inclusive scan, without importing the locator span utility."""
    result, start = [], None
    for frame, positive in enumerate(labels):
        if positive and start is None:
            start = frame
        elif not positive and start is not None:
            result.append((start, frame - 1))
            start = None
    if start is not None:
        result.append((start, len(labels) - 1))
    return result


def row_statistics(predictions, labels):
    """Prediction-order greedy inclusive IoU + an independent dense frame mask.

    Never sort predictions before matching. Equal IoU keeps the FIRST unmatched
    GT (strict >); an overlap equal to the threshold is accepted. Frame counts
    clamp and union intervals independently, including overlapping predictions.
    """
    labels = np.asarray(labels, dtype=bool)
    require(labels.ndim == 1, 'labels must be one-dimensional')
    truth = label_spans(labels)
    pred = [(int(s), int(e)) for s, e in predictions]
    values = []
    for threshold in (.3, .5):
        matched, true_positives = set(), 0
        for start, end in pred:
            best_id, best_overlap = -1, 0.
            for j, (left, right) in enumerate(truth):
                if j in matched:
                    continue
                intersection = max(0, min(end, right) - max(start, left) + 1)
                union = max(1, end - start + 1 + right - left + 1 - intersection)
                overlap = intersection / union
                if overlap > best_overlap:
                    best_id, best_overlap = j, overlap
            if best_id != -1 and best_overlap >= threshold:
                matched.add(best_id)
                true_positives += 1
        values.extend((true_positives, len(pred) - true_positives, len(truth) - true_positives))
    covered = np.zeros(len(labels), dtype=bool)
    for start, end in pred:
        left, stop = max(0, start), min(len(labels), end + 1)
        if left < stop:
            covered[left:stop] = True
    values.extend((int(np.count_nonzero(covered & labels)),
                   int(np.count_nonzero(covered & ~labels)),
                   int(np.count_nonzero(~covered & labels)),
                   int(not truth and bool(pred)), int(bool(truth) and not pred), len(pred)))
    return np.asarray(values, dtype=np.int64)


def content_groups(records, indices):
    groups = {}
    for i in indices:
        groups.setdefault(records[i]['sha256'], []).append(i)
    return list(groups.values())


def content_weights(records, indices):
    counts = Counter(records[i]['sha256'] for i in indices)
    return np.asarray([1. / counts[records[i]['sha256']] for i in indices], dtype=np.float64)


def stratified_folds(records, groups, count, seed):
    """Given fixed groups, replay sorted generator/event strata and round robin."""
    strata = {}
    for position, group in enumerate(groups):
        row = records[group[0]]
        event = 'normal' if row['event_count'] == 0 else 'multi' if row['event_count'] > 1 else 'single'
        strata.setdefault((row['generator'], event), []).append(position)
    rng, folds, offset = np.random.default_rng(seed), [[] for _ in range(count)], 0
    for key in sorted(strata):
        permuted = rng.permutation(strata[key])
        for j, position in enumerate(permuted):
            folds[(offset + j) % count].extend(groups[int(position)])
        offset = (offset + len(permuted)) % count
    return [sorted(fold) for fold in folds]


def grouped_folds(records, indices, count, seed):
    return stratified_folds(records, content_groups(records, indices), count, seed)


def frozen_outer_replay(records):
    """Preserve the frozen v1 row-fold allocation, consolidate SHA aliases only.

    v2 did NOT regenerate outer folds using the later grouped inner splitter.
    The sole duplicate's later row moves to its first row's original fold. The
    old leaking fold allocation is reconstructed for provenance, never trained.
    """
    original = stratified_folds(records, [[i] for i in range(len(records))], 5, SEED)
    owner = {i: fold for fold, ids in enumerate(original) for i in ids}
    consolidated = [[] for _ in range(5)]
    for group in content_groups(records, range(len(records))):
        consolidated[owner[group[0]]].extend(group)
    return original, [sorted(ids) for ids in consolidated]


def check_isolation(records, fit, validation, universe, label):
    require(fit == sorted(set(fit)) and validation == sorted(set(validation))
            and fit and validation, 'invalid/empty partition: ' + label)
    require(set(fit).isdisjoint(validation) and set(fit) | set(validation) == set(universe),
            'row coverage/leakage: ' + label)
    require({records[i]['sha256'] for i in fit}.isdisjoint(records[i]['sha256'] for i in validation),
            'content leakage: ' + label)


def expected_partitions(records, outer_folds, fold):
    validation = outer_folds[fold]
    train = [i for i in range(len(records)) if i not in set(validation)]
    check_isolation(records, train, validation, range(len(records)), f'outer {fold}')
    inner = []
    validations = grouped_folds(records, train, 3, SEED + fold * 31)
    require(sorted(i for part in validations for i in part) == train, 'inner OOF coverage')
    for k, held in enumerate(validations):
        fit = [i for i in train if i not in set(held)]
        check_isolation(records, fit, held, train, f'outer {fold}/inner {k}')
        inner.append({'fit': fit, 'validation': held, 'seed': SEED + fold * 53 + k})
    return train, validation, inner


def read_archive(data, label):
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), 'duplicate NPZ keys: ' + label)
            return {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise AuditError('invalid NPZ: ' + label + ': ' + str(exc)) from exc


def load_records(evidence):
    manifest = evidence.json(V2 / 'dataset_manifest.json')
    old = evidence.json(V1 / 'dataset_manifest.json')
    rows = manifest.get('rows')
    require(manifest.get('schema_version') == 'algorithm-opt-grouped-dataset-v2'
            and isinstance(rows, list) and len(rows) == 77 and rows == old.get('rows'),
            '77 grouped/original manifest rows differ')
    require(manifest.get('v1_manifest_sha256') == evidence.hashes[(V1 / 'dataset_manifest.json').as_posix()],
            'grouped manifest original-manifest binding differs')
    protocol = manifest.get('protocol', {})
    require(all(type(protocol.get(k)) is int and protocol[k] == v
                for k, v in [('outer_folds', 5), ('inner_folds', 3), ('seed', SEED)]),
            'dataset outer5/inner3/seed differs')
    arrays = read_archive(evidence.read(V1 / 'dataset.npz'), 'base dataset')
    expected = {f'{key}_{i}' for i in range(77) for key in ('jepa', 'rank_motion', 'labels')}
    require(set(arrays) == expected, 'base dataset NPZ row coverage differs')
    records = []
    for i, row in enumerate(rows):
        require(is_hash(row.get('sha256')) and isinstance(row.get('name'), str)
                and isinstance(row.get('generator'), str) and type(row.get('frames')) is int
                and row['frames'] > 0 and type(row.get('event_count')) is int
                and row['event_count'] >= 0 and type(row.get('fps')) in (int, float)
                and math.isfinite(row['fps']) and row['fps'] > 0, 'invalid row metadata: ' + str(i))
        labels = arrays[f'labels_{i}']
        require(labels.shape == (row['frames'],) and labels.dtype.kind in 'biuf'
                and np.isin(labels, [0, 1]).all(), 'invalid binary labels: ' + str(i))
        require(len(label_spans(labels)) == row['event_count'], 'event metadata/labels disagree: ' + str(i))
        records.append({**row, 'labels': labels.astype(bool)})
    require(len({r['name'] for r in records}) == 77, 'duplicate row names')
    known = [r['prompt_id'] for r in records if r.get('prompt_id', 'unknown') != 'unknown']
    require(len(set(known)) == len(known), 'duplicate known prompt ids')
    groups = content_groups(records, range(77))
    require(len(groups) == 76 and manifest.get('content_groups') == groups, '76 content groups differ')
    duplicates = []
    for group in groups:
        if len(group) > 1:
            require(len({(records[i]['frames'], records[i]['fps']) for i in group}) == 1,
                    'duplicate content frame/FPS disagreement')
            stack = np.stack([records[i]['labels'] for i in group])
            duplicates.append({'indices': group, 'sha256': records[group[0]]['sha256'],
                               'label_disagreement_frames': np.flatnonzero(np.any(stack != stack[0], axis=0)).tolist()})
    folds = manifest.get('outer_folds')
    original_folds, consolidated_folds = frozen_outer_replay(records)
    require(old.get('outer_folds') == original_folds, 'original seeded row folds do not replay')
    require(folds == consolidated_folds, 'frozen alias-consolidated outer folds do not replay')
    require(sorted(i for val in folds for i in val) == list(range(77)), 'outer OOF coverage differs')
    partitions = [expected_partitions(records, folds, fold) for fold in range(5)]
    appearances = Counter(i for train, _, _ in partitions for i in train)
    require(appearances == Counter({i: 4 for i in range(77)}), 'inner OOF must cover each row four times')
    summary = {'rows': 77, 'unique_content_SHA256': 76, 'outer_fold_sizes': list(map(len, folds)),
               'normal_rows': sum(not r['labels'].any() for r in records),
               'outer_seed': SEED,
               'outer_replay': 'original seeded row folds, then SHA aliases consolidated to first-row fold',
               'old_leaking_folds_used_for_training': False,
               'duplicate_annotations_retained': duplicates, 'inner_oof_appearances_per_row': 4,
               'content_weights': content_weights(records, list(range(77))).tolist()}
    return records, partitions, summary


@dataclass(frozen=True)
class Probabilities:
    frame: dict[int, np.ndarray]
    video: dict[int, float]


def probability_maps(data, metadata, records, label, cast=False):
    arrays = read_archive(data, label)
    expected = {f'{kind}_{head}_{i}' for kind, ids in [('inner', metadata['train']),
                                                    ('outer', metadata['validation'])]
                for head in ('frame', 'video') for i in ids}
    require(set(arrays) == expected, 'exact probability/split coverage differs: ' + label)
    bundles = []
    for kind, ids in [('inner', metadata['train']), ('outer', metadata['validation'])]:
        frame, video = {}, {}
        for i in ids:
            for head, shape in [('frame', (records[i]['frames'],)), ('video', ())]:
                key = f'{kind}_{head}_{i}'
                values = arrays[key]
                require(values.dtype.kind in 'iuf' and values.shape == shape
                        and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                        'invalid probability shape/type/range: ' + label + '/' + key)
            frame[i] = arrays[f'{kind}_frame_{i}'].astype(np.float32) if cast else arrays[f'{kind}_frame_{i}']
            frame[i].setflags(write=False)
            video[i] = float(arrays[f'{kind}_video_{i}'])
        bundles.append(Probabilities(frame, video))
    return tuple(bundles)


def verify_training_metadata(metadata, recipe, fold, partition, records, receipt_sha):
    train, validation, inner = partition
    require(type(metadata.get('fold')) is int and metadata['fold'] == fold
            and metadata.get('recipe') == recipe, 'new cache identity differs')
    require(metadata.get('signature') == value_hash({k: v for k, v in metadata.items() if k != 'signature'}),
            'new cache metadata signature differs')
    require(metadata.get('training_receipt_sha256') == receipt_sha, 'new cache receipt differs')
    require(metadata.get('train') == train and metadata.get('validation') == validation
            and metadata.get('inner_partitions') == inner, 'new cache partition/seed differs')
    require(type(metadata.get('outer_seed')) is int and metadata['outer_seed'] == SEED + fold * 53 + 99,
            'new cache outer seed differs')
    fits = metadata.get('fit_evidence')
    require(isinstance(fits, list) and len(fits) == 4, 'new cache fit evidence coverage differs')
    for k, fit in enumerate(inner + [{'fit': train, 'validation': validation}]):
        expected_sha = sorted({records[i]['sha256'] for i in fit['fit']})
        require(fits[k].get('partition') == (k if k < 3 else 'outer')
                and fits[k].get('fit_content_sha256') == expected_sha,
                'training-only fit/PCA content differs')
        require(is_hash(fits[k].get('transform_sha256')), 'missing transform digest')
    require(type(metadata.get('seconds')) in (int, float) and math.isfinite(metadata['seconds'])
            and metadata['seconds'] >= 0, 'invalid training elapsed time')
    runtime = metadata.get('runtime', {})
    require(all(isinstance(runtime.get(k), str) and runtime[k] for k in ('numpy', 'sklearn')),
            'missing training runtime evidence')


def load_caches(evidence, records, partitions, receipt, experiment=V4_EXPERIMENT):
    training = evidence.json(experiment.folder / 'training_report.json')
    require(training.get('status') == 'complete' and training.get('before_after_evidence_equal') is True,
            'training completion/before-after evidence differs')
    results = training.get('results')
    require(isinstance(results, list) and len(results) == 5 * len(experiment.recipes), 'training report fit count differs')
    by_fit = {}
    for row in results:
        require(isinstance(row, dict) and type(row.get('fold')) is int, 'invalid training result')
        key = (row['fold'], row.get('recipe'))
        require(key not in by_fit, 'duplicate training result')
        by_fit[key] = row
    require(set(by_fit) == {(f, r) for f in range(5) for r in experiment.recipes}, 'training report fit coverage')
    caches, audited = [], []
    legacy_sources = {'/mnt/e/jepa-system/code/' + name: receipt['input_source_hashes']['code/' + name]
                      for name in LEGACY_SOURCE_NAMES}
    for name in LEGACY_SOURCE_NAMES:
        evidence.read(V2 / 'grouped_training/sources' / name, legacy_sources['/mnt/e/jepa-system/code/' + name])
    manifest_sha = evidence.hashes[(V2 / 'dataset_manifest.json').as_posix()]
    for fold, partition in enumerate(partitions):
        inner, outer = {}, {}
        for recipe in experiment.recipes:
            path = experiment.folder / 'training' / f'{fold}_{recipe}.npz'
            metadata = evidence.json(path.with_suffix('.json'))
            verify_training_metadata(metadata, recipe, fold, partition, records, receipt['receipt_sha256'])
            data = evidence.read(path, metadata.get('npz_sha256'))
            row = by_fit[fold, recipe]
            require(row.get('npz_sha256') == sha(data) and row.get('seconds') == metadata['seconds'],
                    'NPZ/metadata/training-report receipt disagreement')
            inner[recipe], outer[recipe] = probability_maps(data, metadata, records, path.as_posix())
            audited.append({'fold': fold, 'recipe': recipe, 'npz_sha256': sha(data),
                            'metadata_sha256': evidence.hashes[path.with_suffix('.json').as_posix()],
                            'metadata_signature': metadata['signature'], 'inner_rows': len(partition[0]),
                            'outer_rows': len(partition[1]), 'exact_keys': 154,
                            'outer_seed': metadata['outer_seed'],
                            'fit_content_and_transform_receipts': metadata['fit_evidence']})
        # Independently validate the fixed old control; never call legacy load_fold.
        path = V2 / 'grouped_training/folds' / f'{fold}_corrected_motion_et.npz'
        metadata = evidence.json(path.with_suffix('.json'))
        require(type(metadata.get('fold')) is int and metadata['fold'] == fold
                and metadata.get('recipe') == 'corrected_motion_et' and metadata.get('content_holdout') is True,
                'legacy control identity/content holdout differs')
        require(metadata.get('train') == partition[0] and metadata.get('validation') == partition[1]
                and metadata.get('inner_partitions') == partition[2], 'legacy partitions/seeds differ')
        require(metadata.get('sources') == legacy_sources
                and metadata.get('signature') == manifest_sha + json.dumps(legacy_sources, sort_keys=True),
                'legacy original snapshot signature differs')
        data = evidence.read(path, metadata.get('npz_sha256'))
        inner['corrected_motion_et'], outer['corrected_motion_et'] = probability_maps(
            data, metadata, records, path.as_posix(), cast=True)
        caches.append((inner, outer))
    return caches, audited

def bootstrap_counts(records, indices, seed, replicates=32):
    """Resample content, never aliases separately; frozen RNG order is preserved.

    Group/stratum order is first appearance, NOT sorted stratum order. Conflicting
    aliases share the first representative's event-count stratum. A content draw
    counts EVERY alias; inverse-multiplicity weights are applied afterward.
    """
    positions, strata = {}, {}
    for column, i in enumerate(indices):
        positions.setdefault(records[i]['sha256'], []).append(column)
    for group in positions.values():
        event = min(2, int(records[indices[group[0]]]['event_count']))
        strata.setdefault(event, []).append(group)
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(indices)), dtype=np.int64)
    for groups in strata.values():
        for replicate in range(replicates):
            for draw in rng.integers(0, len(groups), size=len(groups)):
                counts[replicate, groups[int(draw)]] += 1
    return counts


def score_components(stats, normal_mass, positive_mass, empty_penalty=.15):
    stats = np.asarray(stats, dtype=np.float64)
    require(stats.shape == (12,) and np.isfinite(stats).all() and np.all(stats >= 0), 'invalid sufficient statistics')
    require(math.isfinite(normal_mass) and normal_mass >= 0
            and math.isfinite(positive_mass) and positive_mass >= 0, 'invalid content mass')
    f = [float(2 * stats[k] / max(1e-12, 2 * stats[k] + stats[k + 1] + stats[k + 2]))
         for k in (0, 3, 6)]
    # Do NOT max(1, normal_mass): a duplicate alias can contribute only .5.
    normal_rate = float(stats[9] / normal_mass) if normal_mass > 0 else 0.
    empty_rate = float(stats[10] / positive_mass) if positive_mass > 0 else 0.
    require(empty_penalty in (.15, .30), 'unknown frozen empty penalty')
    utility = .35 * f[0] + .65 * f[1] + .10 * f[2] - .20 * normal_rate - empty_penalty * empty_rate
    return {'event_f1_03': f[0], 'event_f1_05': f[1], 'frame_f1': f[2],
            'normal_false_positive_rate': normal_rate, 'positive_empty_rate': empty_rate,
            'utility': float(utility)}


@dataclass(frozen=True)
class SelectionContext:
    indices: list[int]
    weights: np.ndarray
    normal: np.ndarray
    positive: np.ndarray
    normal_mass: float
    positive_mass: float
    counts: np.ndarray
    weighted_counts: np.ndarray
    seed: int
    empty_penalty: float = .15


def selection_context(records, indices, seed, empty_penalty=.15):
    indices = list(indices)
    require(indices and indices == sorted(set(indices)), 'selection indices must be sorted/unique/nonempty')
    weights = content_weights(records, indices)
    normal = np.asarray([not records[i]['labels'].any() for i in indices], dtype=np.float64)
    positive = 1. - normal
    counts = bootstrap_counts(records, indices, seed)
    require(math.isclose(float(weights.sum()), len(content_groups(records, indices)), abs_tol=ATOL),
            'content-normalized row mass differs')
    require(np.allclose(counts @ weights, weights.sum(), rtol=0, atol=ATOL), 'bootstrap content mass differs')
    return SelectionContext(indices, weights, normal, positive, float(weights @ normal),
                            float(weights @ positive), counts, counts * weights[None, :], seed, empty_penalty)


def blend(members, probabilities, indices):
    require(members and len({recipe for recipe, _ in members}) == len(members)
            and all(type(weight) in (int, float) and math.isfinite(weight) and weight > 0 for _, weight in members)
            and math.isclose(sum(w for _, w in members), 1., rel_tol=0, abs_tol=ATOL), 'invalid fixed blend')
    frame, video = {}, {}
    for i in indices:
        # Keep float32 per-member multiplication/addition order from the protocol.
        frame[i] = sum(weight * probabilities[recipe].frame[i]
                       for recipe, weight in members).astype(np.float32)
        video[i] = float(sum(weight * probabilities[recipe].video[i] for recipe, weight in members))
    return Probabilities(frame, video)


def decoder_functions():
    # ONLY reused computation: decode_v2 transitively uses locator rolling_mean
    # and spans, not its metrics. No selector/statistics/model module is imported.
    from optimized_duration_decoder_v4 import decode_duration
    from optimized_recall_decoder import decode_v2
    return {'duration-logit-v4': decode_duration, 'recall-stable-v2': decode_v2}


def decode(records, indices, probabilities, config, decoders):
    decoder = decoders[config['kind']]
    return {i: decoder(probabilities.frame[i], records[i]['fps'], probabilities.video[i], config) for i in indices}


def independent_choice(records, context, probabilities, decoders):
    """Recompute all40, then dedupe/top8/q20; never inspect unlisted/outer rows."""
    ids, weights = context.indices, context.weights
    bootstrap_normals = context.weighted_counts @ context.normal
    bootstrap_positives = context.weighted_counts @ context.positive
    seen, unique, all_configs = {}, [], []
    for ordinal, config in enumerate(configurations()):
        predictions = decode(records, ids, probabilities, config, decoders)
        matrix = np.stack([row_statistics(predictions[i], records[i]['labels']) for i in ids])
        total = weights @ matrix
        components = score_components(total, context.normal_mass, context.positive_mass, context.empty_penalty)
        pooled = components['utility']
        key = [pooled, components['event_f1_05'], -float(total[10]), -float(total[9]), -float(total[11]), -ordinal]
        fingerprint = tuple(tuple(tuple(pair) for pair in predictions[i]) for i in ids)
        duplicate_of = seen.get(fingerprint)
        if duplicate_of is None:
            seen[fingerprint] = ordinal
        bootstrap_stats = context.weighted_counts @ matrix
        utilities = np.asarray([score_components(s, float(n), float(p), context.empty_penalty)['utility']
                                for s, n, p in zip(bootstrap_stats, bootstrap_normals, bootstrap_positives)])
        quantile = float(np.quantile(utilities, .2, method='linear'))
        row = {'config': config, 'stats': total.tolist(), 'pooled_utility': pooled, 'key': key,
               'bootstrap_q20': quantile, 'stable_utility': .75 * pooled + .25 * quantile}
        if duplicate_of is None:
            unique.append(row)
        all_configs.append({'ordinal': ordinal, **deepcopy(row), 'components': components,
                            'duplicate_of_ordinal': duplicate_of,
                            'prediction_fingerprint_sha256': value_hash(fingerprint),
                            'row_statistics_sha256': value_hash(matrix.tolist()),
                            'bootstrap_utilities': utilities.tolist()})
    # Pooled top8 FIRST. Even a very stable 9th configuration is not admissible.
    # Keep shortlist in pooled order; stable sorting would change report evidence.
    short = sorted(unique, key=lambda row: row['key'], reverse=True)[:8]
    require(short, 'empty configuration shortlist')
    chosen = deepcopy(max(short, key=lambda row: (row['stable_utility'], *row['key'][1:])))
    chosen['shortlist'] = deepcopy(short)
    shortlisted = {-int(row['key'][-1]) for row in short}
    for row in all_configs:
        row['in_pooled_top8'] = row['ordinal'] in shortlisted
    return chosen, {'configurations_evaluated': 40, 'unique_prediction_sets': len(unique),
                    'bootstrap_seed': context.seed, 'bootstrap_replicates': 32, 'positive_empty_penalty': context.empty_penalty,
                    'quantile_method': 'linear', 'bootstrap_counts_sha256': value_hash(context.counts.tolist()),
                    'all_configurations': all_configs}


def rank_candidates(choices, candidates=CANDIDATES):
    require(len(choices) == len(candidates)
            and [row['candidate'] for row in choices] == [name for name, _ in candidates], 'candidate order differs')
    def key(pair):
        ordinal, row = pair
        return (row['stable_utility'], *row['key'][1:], -ordinal)
    ranked = sorted(enumerate(choices), key=key, reverse=True)
    primary = deepcopy(ranked[0][1])
    ranking = [{'candidate': row['candidate'], 'candidate_ordinal': ordinal,
                'selection_key': list(key((ordinal, row))), 'config': row['config']} for ordinal, row in ranked]
    return primary, ranking


class Comparisons:
    """Record mismatches without letting report choices drive independent replay."""
    def __init__(self):
        self.checked, self.mismatch_count, self.differences = 0, 0, []

    def mismatch(self, path, expected, actual):
        self.mismatch_count += 1
        if len(self.differences) < 100:
            self.differences.append({'path': path, 'expected': repr(expected)[:300], 'actual': repr(actual)[:300]})

    def compare(self, actual, expected, path):
        self.checked += 1
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return self.mismatch(path, expected, actual)
            if set(actual) != set(expected):
                self.mismatch(path + '/keys', sorted(expected), sorted(actual))
            for key in expected:
                if key in actual:
                    self.compare(actual[key], expected[key], path + '/' + str(key))
        elif isinstance(expected, (tuple, list)):
            if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
                return self.mismatch(path, expected, actual)
            for k, (a, e) in enumerate(zip(actual, expected)):
                self.compare(a, e, path + '/' + str(k))
        elif isinstance(expected, (float, np.floating)):
            if (type(actual) not in (int, float) or not math.isfinite(actual)
                    or not math.isclose(actual, float(expected), rel_tol=0, abs_tol=ATOL)):
                self.mismatch(path, expected, actual)
        elif type(actual) is not type(expected) or actual != expected:
            self.mismatch(path, expected, actual)


def prediction_rows(records, predictions):
    return [{'index': i, 'name': r['name'], 'segments': [list(pair) for pair in predictions[i]],
             'has_candidates': bool(predictions[i]), 'is_normal': not bool(r['labels'].any())}
            for i, r in enumerate(records)]


def read_prediction_rows(value, records, label):
    require(isinstance(value, list) and len(value) == len(records), 'prediction row coverage: ' + label)
    predictions = {}
    for i, (row, record) in enumerate(zip(value, records)):
        require(isinstance(row, dict) and type(row.get('index')) is int and row['index'] == i
                and row.get('name') == record['name'], 'prediction row identity/order: ' + label)
        segments = row.get('segments')
        require(isinstance(segments, list), 'prediction segments must be a list: ' + label)
        previous_end = -1
        for pair in segments:
            require(isinstance(pair, list) and len(pair) == 2 and all(type(x) is int for x in pair)
                    and 0 <= pair[0] <= pair[1] < record['frames'] and pair[0] > previous_end,
                    'invalid/out-of-order/overlapping reported segment: ' + label)
            previous_end = pair[1]
        predictions[i] = [tuple(pair) for pair in segments]
    return predictions


def aggregate_metrics(records, predictions, indices):
    indices, totals, normal_count = list(indices), np.zeros(12, dtype=np.int64), 0
    for i in indices:
        totals += row_statistics(predictions[i], records[i]['labels'])
        normal_count += int(not records[i]['labels'].any())
    output = {}
    for threshold, offset in [('0.3', 0), ('0.5', 3)]:
        tp, fp, fn = (int(x) for x in totals[offset:offset + 3])
        output['iou_' + threshold] = {'tp': tp, 'fp': fp, 'fn': fn,
                                     'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn),
                                     'f1': 2 * tp / max(1, 2 * tp + fp + fn)}
    tp, fp, fn = (int(x) for x in totals[6:9])
    output['frame'] = {'f1': 2 * tp / max(1, 2 * tp + fp + fn), 'tp': tp, 'fp': fp, 'fn': fn}
    output['normal'] = {'videos': normal_count, 'false_positive_videos': int(totals[9]),
                        'false_positive_rate': int(totals[9]) / max(1, normal_count)}
    output['positive_videos_without_candidate'] = int(totals[10])
    output['predicted_segments'] = int(totals[11])
    return output


def grouped_metrics(records, predictions):
    require(set(predictions) == set(range(len(records))), 'outer predictions must cover every row exactly once')
    indices = {'all': list(range(len(records))),
               'normal': [i for i, r in enumerate(records) if r['event_count'] == 0],
               'single_event': [i for i, r in enumerate(records) if r['event_count'] == 1],
               'multi_event': [i for i, r in enumerate(records) if r['event_count'] > 1]}
    return {group: aggregate_metrics(records, predictions, ids) for group, ids in indices.items()}


def prediction_summary(records, predictions):
    normal = aggregate_metrics(records, predictions, range(len(records)))['normal']
    empty = [i for i in range(len(records)) if not predictions[i]]
    return {'normal': {**normal, 'false_positive_names': [r['name'] for i, r in enumerate(records)
                                                        if not r['labels'].any() and predictions[i]],
                       'correctly_empty_names': [r['name'] for i, r in enumerate(records)
                                                 if not r['labels'].any() and not predictions[i]]},
            'no_candidates': {'videos': len(empty), 'names': [records[i]['name'] for i in empty],
                              'normal_videos': sum(not records[i]['labels'].any() for i in empty),
                              'positive_videos': sum(bool(records[i]['labels'].any()) for i in empty)}}


def promotion_guards(primary, baseline):
    # Gates apply to the LOCKED primary, never the best fixed outer diagnostic.
    return {
        'event_f1_03_no_worse': primary['iou_0.3']['f1'] >= baseline['iou_0.3']['f1'],
        'event_f1_05_improved_02': primary['iou_0.5']['f1'] >= baseline['iou_0.5']['f1'] + .02,
        'frame_f1_drop_at_most_005': primary['frame']['f1'] >= baseline['frame']['f1'] - .005,
        'normal_fp_no_worse': primary['normal']['false_positive_videos'] <= baseline['normal']['false_positive_videos'],
        'positive_empty_improved_3': primary['positive_videos_without_candidate'] <= baseline['positive_videos_without_candidate'] - 3,
    }


def paired_bootstrap(records, baseline, primary):
    # Preserve report's raw-row metric contract: aliases share draws, but these
    # statistics do NOT use selection's inverse-multiplicity weighting.
    ids = list(range(len(records)))
    counts = bootstrap_counts(records, ids, SEED + 991, replicates=2000)
    old = counts @ np.stack([row_statistics(baseline[i], records[i]['labels']) for i in ids])
    new = counts @ np.stack([row_statistics(primary[i], records[i]['labels']) for i in ids])
    result = {}
    for key, offset in [('event_f1_03', 0), ('event_f1_05', 3), ('frame_f1', 6)]:
        a = 2 * old[:, offset] / np.maximum(1, 2 * old[:, offset] + old[:, offset + 1] + old[:, offset + 2])
        b = 2 * new[:, offset] / np.maximum(1, 2 * new[:, offset] + new[:, offset + 1] + new[:, offset + 2])
        diff = b - a
        result[key] = {'mean_difference': float(diff.mean()),
                       'percentile_95_interval': np.quantile(diff, [.025, .975], method='linear').tolist(),
                       'bootstrap_probability_positive': float(np.mean(diff > 0)),
                       'not_confirmatory_p_value': True}
    return result

def replay(evidence, result, experiment=V4_EXPERIMENT):
    frozen_inventory(evidence.root, experiment)
    receipt, protocol = verify_receipt(evidence, experiment)
    result['receipt'] = {'receipt_sha256': experiment.receipt_pin, 'protocol_sha256': experiment.protocol_pin,
                         'baseline_report_sha256': BASELINE_PIN, 'bound_inputs': 479,
                         'input_sources': 11, 'frozen_and_live_sources': len(experiment.sources),
                         'trust_anchor_timing': experiment.trust_anchor_timing}
    result['selection_source_review'] = source_review(evidence, experiment)
    records, partitions, summary = load_records(evidence)
    result['dataset'] = summary
    caches, audited = load_caches(evidence, records, partitions, receipt, experiment)
    result['new_training_caches'] = audited
    result['legacy_control_cache_count'] = 5
    report = evidence.json(experiment.folder / 'report.json')
    require(isinstance(report, dict) and report.get('status') == 'complete'
            and report.get('schema_version') == experiment.schema, 'selection report schema/completion differs')
    comparisons = Comparisons()
    for field, expected in [('protocol_sha256', experiment.protocol_pin), ('training_receipt_sha256', experiment.receipt_pin),
                            ('baseline_report_sha256', BASELINE_PIN), ('role', protocol['role'])]:
        comparisons.compare(report.get(field), expected, 'report/' + field)
    report_folds = report.get('folds')
    require(isinstance(report_folds, list) and len(report_folds) == 5, 'selection report five-fold coverage')
    decoders = decoder_functions()
    primary_bundle = 'primary_' + experiment.version + '_nested'
    selected, fast, fixed = {}, {}, {name: {} for name, _ in experiment.candidates}
    result['folds'] = []
    for fold, ((train, validation, parts), (inner, outer)) in enumerate(zip(partitions, caches)):
        context = selection_context(records, train, SEED + fold, experiment.empty_penalty)
        choices, details = [], []
        for name, members in experiment.candidates:
            p = blend(members, inner, train)
            choice, detail = independent_choice(records, context, p, decoders)
            choice['candidate'] = name
            choices.append(choice)
            details.append({'candidate': name, **detail})
        primary, ranking = rank_candidates(choices, experiment.candidates)
        fast_choice = fast_details = None
        if experiment.fast_recipe:
            fast_choice, fast_details = independent_choice(records, context, inner[experiment.fast_recipe], decoders)
        # Independent choices are LOCKED. Neither report choices nor outer scores
        # have entered independent_choice/rank_candidates above.
        fold_report = report_folds[fold]
        require(isinstance(fold_report, dict), 'invalid selection fold row')
        expected = {'fold': fold, 'train_indices': train, 'validation_indices': validation,
                    'inner_partitions': parts, 'primary': primary, 'all_inner_selections': choices}
        if experiment.fast_recipe:
            expected['fast'] = fast_choice
        for field, value in expected.items():
            comparisons.compare(fold_report.get(field), value, f'report/folds/{fold}/{field}')
        members = dict(experiment.candidates)[primary['candidate']]
        predictions = decode(records, validation, blend(members, outer, validation), primary['config'], decoders)
        require(set(selected).isdisjoint(predictions), 'duplicate nested outer coverage')
        selected.update(predictions)
        comparisons.compare(fold_report.get('primary_predictions'),
                            {str(i): [list(pair) for pair in predictions[i]] for i in validation},
                            f'report/folds/{fold}/primary_predictions')
        if experiment.fast_recipe:
            fast.update(decode(records, validation, outer[experiment.fast_recipe], fast_choice['config'], decoders))
        for choice in choices:
            name = choice['candidate']
            fixed[name].update(decode(records, validation, blend(dict(experiment.candidates)[name], outer, validation),
                                      choice['config'], decoders))
        strata = [{'representative_index': group[0], 'aliases': group,
                   'event_stratum': min(2, records[group[0]]['event_count'])} for group in content_groups(records, train)]
        result['folds'].append({**expected, 'candidate_ranking': ranking,
                                'content_weights': {str(i): float(w) for i, w in zip(train, context.weights)},
                                'normal_content_mass': context.normal_mass, 'positive_content_mass': context.positive_mass,
                                'bootstrap': {'seed': context.seed, 'replicates': 32,
                                              'stratum_representatives_in_first_appearance_order': strata,
                                              'counts': context.counts.tolist()},
                                'candidate_configuration_replays': details, 'fast_configuration_replay': fast_details,
                                'outer_primary_predictions': {str(i): [list(pair) for pair in predictions[i]] for i in validation}})
    baseline_report = evidence.json(V2 / 'selection_audited/report.json', BASELINE_PIN)
    require(baseline_report.get('status') == 'complete', 'audited fixed baseline report not complete')
    baseline = read_prediction_rows(baseline_report['grouped_v1_baseline']['predictions'], records, 'fixed baseline')
    baseline_metrics = grouped_metrics(records, baseline)
    comparisons.compare(baseline_report['grouped_v1_baseline']['metrics'], baseline_metrics, 'fixed_baseline/metrics')
    primary_metrics = grouped_metrics(records, selected)
    bundle_checks = [('grouped_v1_baseline', baseline, baseline_metrics),
                     (primary_bundle, selected, primary_metrics)]
    if experiment.fast_recipe:
        bundle_checks.append(('fast_v4_diagnostic', fast, grouped_metrics(records, fast)))
    result['outer_metrics'] = {}
    for name, predictions, metrics in bundle_checks:
        bundle = report.get(name, {})
        require(isinstance(bundle, dict), 'invalid prediction bundle: ' + name)
        comparisons.compare(bundle.get('predictions'), prediction_rows(records, predictions), 'report/' + name + '/predictions')
        comparisons.compare(bundle.get('metrics'), metrics, 'report/' + name + '/metrics')
        result['outer_metrics'][name] = metrics
    comparisons.compare(report[primary_bundle].get('summary'), prediction_summary(records, selected),
                        'report/' + primary_bundle + '/summary')
    diagnostics = report.get('fixed_candidate_diagnostics_not_for_promotion')
    require(isinstance(diagnostics, dict) and set(diagnostics) == set(fixed), 'fixed diagnostic coverage differs')
    result['fixed_candidate_diagnostics_not_for_promotion'] = {}
    for name, predictions in fixed.items():
        metrics = grouped_metrics(records, predictions)
        comparisons.compare(diagnostics[name].get('metrics'), metrics, 'report/fixed/' + name + '/metrics')
        comparisons.compare(diagnostics[name].get('predictions'), prediction_rows(records, predictions), 'report/fixed/' + name + '/predictions')
        result['fixed_candidate_diagnostics_not_for_promotion'][name] = metrics
    guards = promotion_guards(primary_metrics['all'], baseline_metrics['all'])
    comparisons.compare(report.get('statistical_promotion_checks'), guards, 'report/statistical_promotion_checks')
    comparisons.compare(report.get('statistical_promotion_passed'), all(guards.values()), 'report/statistical_promotion_passed')
    uncertainty = paired_bootstrap(records, baseline, selected)
    comparisons.compare(report.get('paired_content_bootstrap'), uncertainty, 'report/paired_content_bootstrap')
    result['paired_content_bootstrap'] = uncertainty
    result['deployment_guards'] = {'statistical_checks': guards, 'statistical_passed': all(guards.values()),
                                   'parity_max_required': 2e-6,
                                   'portable_full_fit_and_fresh_pixels': 'pending_separate_worker_not_attested_here',
                                   'deployment_candidate_rule': 'inner-only OOF averages; four appearances/row; never outer-best',
                                   'deployment_choice_and_actual_promotion_not_audited': True,
                                   'promotion_authorized_by_this_audit': False}
    frozen_inventory(evidence.root, experiment)
    evidence.recheck()
    result['before_after_consumed_files_equal'] = True
    result['comparisons'] = {'checked_nodes': comparisons.checked, 'mismatch_count': comparisons.mismatch_count,
                             'differences': comparisons.differences, 'absolute_float_tolerance': ATOL}
    result['selection_integrity'] = {
        'report_matches_inner_locked_primary_choices_and_predictions': comparisons.mismatch_count == 0,
        'outer_best_substitution': 'not_detected' if comparisons.mismatch_count == 0 else 'unverified_due_to_mismatches',
        'fixed_outer_candidates_diagnostic_only': True,
        'independent_choice_uses_outer_labels_or_probabilities': False}
    result['independent_selection_verified'] = comparisons.mismatch_count == 0
    result['status'] = 'passed' if comparisons.mismatch_count == 0 else 'failed'


def audit(root, experiment=V4_EXPERIMENT):
    """Invoke only after complete_inputs; no writes until the caller saves."""
    evidence = Evidence(root)
    result = {'schema_version': 'independent-selection-audit-' + experiment.version, 'status': 'failed',
              'experiment': experiment.version, 'positive_empty_penalty': experiment.empty_penalty,
              'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'role': 'iterative_development_validation_not_new_blind_test',
              'independent_selection_verified': False, 'original_artifacts_modified': False,
              'statistics_columns': list(STAT_NAMES), 'reused_functions': ['decode_duration', 'decode_v2'],
              'prohibited_reuse': ['choose_v4', 'utility_v4', 'choose_v5', 'utility_v5', 'video_statistics', 'metrics',
                                   'grouped_folds', 'grouped_bootstrap', 'load_fold'],
              'limitations': [
                  '77 inspected development rows are not new blind generalization evidence.',
                  'Conflicting duplicate annotations are retained; selection divides content mass among aliases.',
                  'Digests bind files, not authenticated training or omitted historical pixel provenance.',
                  'Transform digest/content receipts are checked; absent transforms/models cannot be refitted here.',
                  'Portable full-fit parity, fresh extraction and deployment choice belong to the separate worker.',
              ]}
    started = time.perf_counter()
    try:
        result['verifier_sha256'] = sha(Path(__file__).read_bytes())
        replay(evidence, result, experiment)
    except (AuditError, OSError, KeyError, TypeError, IndexError, AttributeError, ValueError) as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        result['status'] = 'failed'
        result['independent_selection_verified'] = False
    result['consumed_file_sha256'] = dict(sorted(evidence.hashes.items()))
    result['elapsed_seconds'] = time.perf_counter() - started
    return result


def write_report(root, result, experiment=V4_EXPERIMENT):
    target = Path(root).resolve() / experiment.folder / OUTPUT_NAME
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    # Experiment parent exists. No mkdir/overwrite/migration/auxiliary output.
    with target.open('x', encoding='utf-8') as stream:
        stream.write(data + '\n')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), default='v4',
                        help='frozen namespace; v4 default, optional event-balanced v5')
    parser.add_argument('--reports-complete', action='store_true',
                        help='main worker notified that all training/selection artifacts are complete')
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    experiment = EXPERIMENTS[args.experiment]
    try:
        complete_inputs(ROOT, args.reports_complete, experiment)
        result = audit(ROOT, experiment)
        target = write_report(ROOT, result, experiment)
    except (AuditError, OSError) as exc:
        print('AUDIT NOT RUN: ' + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({'status': result['status'], 'report': str(target),
                      'independent_selection_verified': result['independent_selection_verified'],
                      'mismatch_count': result.get('comparisons', {}).get('mismatch_count'),
                      'error': result.get('error')}, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
