#!/usr/bin/env python3
"""Independent read-only audit of frozen multiview/error-budget v7.

    python -B code/audit_multiview_v7.py --reports-complete

One-shot complete-artifact gate; no polling, fitting, extraction or outer-best.
Only previously independent v4/v5 helpers and the validated decoders are reused.
Never import/execute v7/v6/parent runners, their utility, choose or statistics.
Output is a new, exclusive v7 independent_selection_audit.json. Missing inputs
or an existing output refuse execution. Statistical guard failure is NOT an
audit failure and never authorizes promotion. Source snapshots are AST-read.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import audit_compact_selection_v4 as independent

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('output/algorithm-opt-2026-10-02-v7')
SCHEMA = 'multiview-error-budget-v7'
RUNNER = 'run_multiview_experiment_v7.py'
COMPACT = 'compact_rgb_corrected_motion_hgb'
EVENT = 'event_rgb_corrected_motion_hgb'
CANDIDATES = ('compact_control', 'event_control', 'equal_multiview')
NORMAL_BUDGET, EMPTY_BUDGET = .30, .20
NORMAL_PENALTY, EMPTY_PENALTY = .30, .30
FEASIBILITY_EPSILON = 1e-12
# Receipt supplied directly by the main worker; do not regenerate trust pins.
RECEIPT_PIN = 'fe948cb0d7428d523c1d8f5eab037ff56fdca15b33db41729a67f73f3a423786'
PROTOCOL_PIN = '55d190201a9dc8dfbf8e947319516f4302ddcf4456dbbb19e03939652b044404'
PARENTS = ((independent.V4_EXPERIMENT, COMPACT), (independent.V5_EXPERIMENT, EVENT))
SOURCES = (RUNNER,) + tuple(dict.fromkeys(
    independent.V4_EXPERIMENT.sources + independent.V5_EXPERIMENT.sources +
    ('run_video_gate_experiment_v6.py', 'optimized_video_gate_v6.py')))
OUTPUT_NAME = independent.OUTPUT_NAME
require = independent.require
AuditError = independent.AuditError
IncompleteArtifacts = independent.IncompleteArtifacts


def expected_inputs():
    return {path.as_posix() for parent, recipe in PARENTS
            for path in ([parent.folder / OUTPUT_NAME] +
                         [parent.folder / 'training' / f'{fold}_{recipe}.{ext}'
                          for fold in range(5) for ext in ('npz', 'json')])}


def complete_inputs(root, acknowledged):
    if not acknowledged:
        raise IncompleteArtifacts('requires --reports-complete after the main worker completion notice')
    root = Path(root).resolve()
    needed = {str(OUT / name) for name in ('protocol.json', 'training_receipt.json', 'report.json')}
    needed.update(expected_inputs())
    for parent, _ in PARENTS:
        needed.update((parent.folder / name).as_posix()
                      for name in ('training_receipt.json', 'training_report.json'))
    needed.update((OUT / 'sources' / name).as_posix() for name in SOURCES)
    # Normalize concrete Windows Paths before applying the relative-path guard.
    needed = {Path(name).as_posix() for name in needed}
    missing = sorted(name for name in needed if not independent.safe_path(root, name).is_file())
    if missing:
        raise IncompleteArtifacts('incomplete v7 evidence; no wait: ' + ', '.join(missing))
    receipt = independent.strict_json((root / OUT / 'training_receipt.json').read_bytes(), 'v7 readiness receipt')
    require(isinstance(receipt, dict) and isinstance(receipt.get('inputs'), dict)
            and isinstance(receipt.get('sources'), dict), 'invalid v7 readiness inventory')
    missing = [name for name in receipt['inputs'] if not independent.safe_path(root, name).is_file()]
    missing += ['code/' + name for name in receipt['sources']
                if not independent.safe_path(root, 'code/' + name).is_file()]
    for parent, _ in PARENTS:
        value = independent.strict_json((root / parent.folder / 'training_receipt.json').read_bytes(), 'parent readiness')
        for key in ('input_hashes', 'input_source_hashes'):
            require(isinstance(value.get(key), dict), 'invalid parent input inventory')
            missing += [name for name in value[key] if not independent.safe_path(root, name).is_file()]
    if missing:
        raise IncompleteArtifacts('missing bound evidence; no wait: ' + ', '.join(missing))
    try:
        report = independent.strict_json((root / OUT / 'report.json').read_bytes(), 'v7 readiness report')
    except (AuditError, OSError) as exc:
        raise IncompleteArtifacts('v7 report is not readable/complete') from exc
    if not isinstance(report, dict) or report.get('status') != 'complete':
        raise IncompleteArtifacts('v7 report status is not complete')
    if (root / OUT / OUTPUT_NAME).exists():
        raise IncompleteArtifacts('existing audit output; refusing overwrite')


def verify_v7_receipt(evidence):
    receipt = evidence.json(OUT / 'training_receipt.json')
    core = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
    require(receipt.get('receipt_sha256') == RECEIPT_PIN
            and independent.value_hash(core) == RECEIPT_PIN, 'v7 receipt differs from human-supplied pin')
    require(receipt.get('protocol_sha256') == PROTOCOL_PIN
            and receipt.get('baseline_report_sha256') == independent.BASELINE_PIN, 'v7 receipt trust anchors differ')
    require(receipt.get('parent_receipt_sha256') == {p.version: p.receipt_pin for p, _ in PARENTS},
            'parent training receipt pins differ')
    require(set(receipt.get('inputs', {})) == expected_inputs(), 'v7 must bind exactly both 5-fold pairs and parent audits')
    require(set(receipt.get('sources', {})) == set(SOURCES), 'v7 source coverage differs')
    folder = independent.safe_path(evidence.root, (OUT / 'sources').as_posix())
    require({p.name for p in folder.iterdir() if p.is_file()} == set(SOURCES), 'v7 snapshot inventory differs')
    for name, digest in receipt['sources'].items():
        evidence.read(OUT / 'sources' / name, digest)
        evidence.read(Path('code') / name, digest)
    for name, digest in receipt['inputs'].items():
        evidence.read(name, digest)
    protocol = evidence.json(OUT / 'protocol.json', PROTOCOL_PIN)
    expected = {'schema_version': SCHEMA, 'date': '2026-10-02',
                'role': 'iterative_development_validation_not_new_blind_test',
                'compact_member': COMPACT, 'event_member': EVENT, 'blend_weights': [.5, .5],
                'primary_candidates': list(CANDIDATES), 'decoder_configs': independent.configurations(),
                'utility': '.35F1.3+.65F1.5+.10FrameF1-.30normalFPR-.30positiveEmpty',
                'promotion_guards': independent.GUARD_LIMITS}
    for key, value in expected.items():
        require(protocol.get(key) == value, 'v7 protocol contract differs: ' + key)
    budget = protocol.get('inner_budget', {})
    require(budget.get('normal_false_positive_rate') == NORMAL_BUDGET
            and budget.get('positive_empty_rate') == EMPTY_BUDGET, 'v7 budget contract differs')
    return receipt, protocol


def source_review_text(source):
    """Static AST contract check; never execute the runner."""
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'COMPACT', 'EVENT', 'CANDIDATES', 'NORMAL_BUDGET', 'EMPTY_BUDGET'}:
                constants[name] = ast.literal_eval(node.value)
    require(constants == {'COMPACT': COMPACT, 'EVENT': EVENT, 'CANDIDATES': CANDIDATES,
                          'NORMAL_BUDGET': NORMAL_BUDGET, 'EMPTY_BUDGET': EMPTY_BUDGET},
            'runner fixed member/candidate/budget literals differ')
    chooser = functions['choose']
    calls = {n.func.id for n in ast.walk(chooser) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    require(not calls & {'metrics', '_group_metrics', 'load_fold', 'select'}, 'chooser invokes outer metrics/loading')
    outer_names = {n.id for n in ast.walk(chooser) if isinstance(n, ast.Name)}
    require('outer' not in outer_names and 'selected' not in outer_names, 'outer data referenced in chooser')
    selector = functions['select']
    fold_loop = next(n for n in selector.body if isinstance(n, ast.For))
    chosen = next(n for n in fold_loop.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'chosen' for t in n.targets))
    predicted = next(n for n in fold_loop.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'pred' for t in n.targets))
    require(chosen.lineno < predicted.lineno, 'outer decode occurs before primary lock')
    require(isinstance(chosen.value, ast.Call) and isinstance(chosen.value.func, ast.Name)
            and chosen.value.func.id == 'max' and ast.unparse(chosen.value.args[0]) == 'rows',
            'primary is not selected from inner decision rows')
    choose_calls = [n for n in ast.walk(fold_loop) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'choose']
    require(len(choose_calls) == 1 and [ast.unparse(x) for x in choose_calls[0].args] ==
            ['records', "meta['train']", 'inner[c]', '20261002 + fold'],
            'candidate choice inputs/seed are not inner-only')
    require(isinstance(predicted.value, ast.Call) and isinstance(predicted.value.func, ast.Name)
            and predicted.value.func.id == 'decode_predictions'
            and [ast.unparse(x) for x in predicted.value.args] ==
            ['records', "meta['validation']", "outer[chosen['candidate']]", "chosen['config']"],
            'primary outer decode does not use the locked candidate/config')
    outer_decodes = [n for n in ast.walk(fold_loop) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'decode_predictions']
    require(all(n.lineno > chosen.lineno for n in outer_decodes), 'outer decode precedes candidate lock')
    load_calls = {n.func.id for n in ast.walk(functions['load_fold'])
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    require(not load_calls & {'decode_predictions', 'metrics', '_group_metrics'}, 'loader evaluates outer scores')
    return {'freeze_blockers_detected': [], 'chooser_lines': [chooser.lineno, chooser.end_lineno],
            'primary_lock_line': chosen.lineno, 'first_primary_outer_decode_line': predicted.lineno,
            'all_outer_decode_lines': sorted(n.lineno for n in outer_decodes),
            'outer_probability_prefusion': 'label-free preload only; no decoding/metrics before selection lock',
            'findings': [
                'Three fixed candidates; both means use exactly .5+.5, no learned gate/calibration.',
                'All choices use inner train rows/probabilities and the fixed seed schedule.',
                'Budget feasibility then normalized violation precede utility at both shortlist and final rank.',
                'Top8 is frozen before bootstrap ranking; candidate order breaks the final tie.',
                'Primary is locked before any outer decode; fixed outer diagnostics never replace it.',
            ]}


def verify_static_sources(evidence):
    source = evidence.read(OUT / 'sources' / RUNNER).decode('utf-8')
    review = source_review_text(source)
    utility_source = evidence.read(OUT / 'sources/run_video_gate_experiment_v6.py').decode('utf-8')
    tree = ast.parse(utility_source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'utility')
    expression = ast.parse('.35*f(s[...,:3])+.65*f(s[...,3:6])+.10*f(s[...,6:9])-.30*nf-.30*pe', mode='eval').body
    require(isinstance(function.body[-1], ast.Return)
            and ast.dump(function.body[-1].value) == ast.dump(expression), 'frozen v6 utility coefficients differ')
    review['frozen_runner_sha256'] = evidence.hashes[(OUT / 'sources' / RUNNER).as_posix()]
    review['utility_source_sha256'] = evidence.hashes[(OUT / 'sources/run_video_gate_experiment_v6.py').as_posix()]
    review['utility_read_not_imported'] = True
    return review


def verify_parent_pair(compact_meta, event_meta):
    """Match all shared fit evidence; distinct-view transform digests are allowed."""
    for field in ('fold', 'train', 'validation', 'inner_partitions', 'outer_seed'):
        require(compact_meta.get(field) == event_meta.get(field), 'parent partition/seed mismatch: ' + field)
    a, b = compact_meta.get('fit_evidence'), event_meta.get('fit_evidence')
    require(isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 4,
            'both parent fit proofs must cover exactly three inner fits plus outer fit')
    for k, (left, right) in enumerate(zip(a, b)):
        require(left.get('partition') == right.get('partition')
                and left.get('fit_content_sha256') == right.get('fit_content_sha256'),
                'parent fit-content/partition proof mismatch: ' + str(k))
        require(independent.is_hash(left.get('transform_sha256')) and independent.is_hash(right.get('transform_sha256')),
                'missing per-view transform receipt')


def load_parent_pairs(evidence, records, partitions):
    parent_receipts, parent_audits, training_reports = {}, {}, {}
    for parent, recipe in PARENTS:
        receipt, _ = independent.verify_receipt(evidence, parent)
        parent_receipts[parent.version] = receipt
        audit = evidence.json(parent.folder / OUTPUT_NAME)
        require(audit.get('status') == 'passed' and audit.get('independent_selection_verified') is True
                and audit.get('before_after_consumed_files_equal') is True
                and audit.get('comparisons', {}).get('mismatch_count') == 0,
                'parent independent audit is not verified: ' + parent.version)
        require(audit.get('receipt', {}).get('receipt_sha256') == parent.receipt_pin, 'parent audit receipt mismatch')
        parent_audits[parent.version] = audit
        training = evidence.json(parent.folder / 'training_report.json')
        require(training.get('status') == 'complete' and training.get('before_after_evidence_equal') is True,
                'parent training report not complete')
        rows = [row for row in training.get('results', []) if row.get('recipe') == recipe]
        require(len(rows) == 5 and {row.get('fold') for row in rows} == set(range(5)), 'parent member training coverage')
        training_reports[parent.version] = {row['fold']: row for row in rows}
    left, right = parent_receipts['v4'], parent_receipts['v5']
    require(left['input_hashes'] == right['input_hashes'] and left['inputs_sha256'] == right['inputs_sha256']
            and left['input_source_hashes'] == right['input_source_hashes'], 'parents do not bind identical content/labels/features')
    pairs, proofs = [], []
    for fold, partition in enumerate(partitions):
        bundles, metadata, cache_proofs = [], [], []
        for parent, recipe in PARENTS:
            path = parent.folder / 'training' / f'{fold}_{recipe}.npz'
            meta = evidence.json(path.with_suffix('.json'))
            independent.verify_training_metadata(meta, recipe, fold, partition, records, parent.receipt_pin)
            data = evidence.read(path, meta.get('npz_sha256'))
            training = training_reports[parent.version][fold]
            require(training.get('npz_sha256') == independent.sha(data) and training.get('seconds') == meta['seconds'],
                    'parent training/metadata/NPZ binding mismatch')
            old_proofs = [row for row in parent_audits[parent.version]['new_training_caches']
                          if row['fold'] == fold and row['recipe'] == recipe]
            require(len(old_proofs) == 1 and old_proofs[0]['npz_sha256'] == independent.sha(data)
                    and old_proofs[0]['metadata_signature'] == meta['signature']
                    and old_proofs[0]['metadata_sha256'] == evidence.hashes[path.with_suffix('.json').as_posix()],
                    'cache differs from parent independent audit evidence')
            bundles.append(independent.probability_maps(data, meta, records, path.as_posix()))
            metadata.append(meta)
            cache_proofs.append({'parent': parent.version, 'recipe': recipe, 'npz_sha256': independent.sha(data),
                                 'metadata_signature': meta['signature'], 'fit_evidence': meta['fit_evidence']})
        verify_parent_pair(*metadata)
        pairs.append((bundles[0], bundles[1]))
        proofs.append({'fold': fold, 'train_indices': partition[0], 'validation_indices': partition[1],
                       'inner_partitions': partition[2], 'outer_seed': metadata[0]['outer_seed'],
                       'shared_fit_content_proofs_equal': True, 'per_view_transform_receipts_valid': True,
                       'transform_digests_may_differ_between_views': True, 'caches': cache_proofs})
    return pairs, proofs


def fusion(compact, event, indices):
    """Fixed arithmetic fusion; no utility/label dependence and no parent mutation."""
    indices = list(indices)
    require(indices and len(indices) == len(set(indices)), 'fusion indices must be unique/nonempty')
    for probability in (compact, event):
        require(set(probability.frame) == set(probability.video) and set(indices) <= set(probability.frame),
                'unaligned fusion frame/video/indices')
    frame, video = {}, {}
    for i in indices:
        a, b = np.asarray(compact.frame[i], np.float32), np.asarray(event.frame[i], np.float32)
        va, vb = compact.video[i], event.video[i]
        require(a.ndim == 1 and a.shape == b.shape and len(a)
                and np.isfinite(a).all() and np.isfinite(b).all()
                and np.all((a >= 0) & (a <= 1)) and np.all((b >= 0) & (b <= 1))
                and type(va) in (int, float) and type(vb) in (int, float)
                and np.isfinite([va, vb]).all() and 0 <= va <= 1 and 0 <= vb <= 1,
                'invalid fusion probability evidence')
        frame[i] = (.5 * a + .5 * b).astype(np.float32)
        video[i] = float(.5 * va + .5 * vb)
    return independent.Probabilities(frame, video)


def components(stats, normal_mass, positive_mass):
    """Independent v6-equivalent utility, including fractional and zero mass."""
    stats = np.asarray(stats, dtype=np.float64)
    require(stats.shape == (12,) and np.isfinite(stats).all() and np.all(stats >= 0), 'invalid statistics')
    require(np.isfinite([normal_mass, positive_mass]).all() and min(normal_mass, positive_mass) >= 0, 'invalid content mass')
    f = [float(2 * stats[k] / max(1e-12, 2 * stats[k] + stats[k + 1] + stats[k + 2])) for k in (0, 3, 6)]
    normal = float(stats[9] / normal_mass) if normal_mass > 0 else 0.
    empty = float(stats[10] / positive_mass) if positive_mass > 0 else 0.
    utility = .35 * f[0] + .65 * f[1] + .10 * f[2] - NORMAL_PENALTY * normal - EMPTY_PENALTY * empty
    return {'event_f1_03': f[0], 'event_f1_05': f[1], 'frame_f1': f[2],
            'normal_false_positive_rate': normal, 'positive_empty_rate': empty, 'utility': float(utility)}


def error_budget(stats, normal_mass, positive_mass):
    values = components(stats, normal_mass, positive_mass)
    normal, empty = values['normal_false_positive_rate'], values['positive_empty_rate']
    violation = max(0., normal - NORMAL_BUDGET) / NORMAL_BUDGET + max(0., empty - EMPTY_BUDGET) / EMPTY_BUDGET
    return {'normal_false_positive_rate': normal, 'positive_empty_rate': empty,
            'feasible': bool(violation <= FEASIBILITY_EPSILON), 'normalized_violation': float(violation)}


def stable_key(row):
    return (int(row['error_budget']['feasible']), -row['error_budget']['normalized_violation'],
            row['stable_utility'], *row['key'][3:])


def shortlist_and_select(rows):
    require(rows, 'empty decoder candidate set')
    shortlisted = sorted(rows, key=lambda row: row['key'], reverse=True)[:8]
    chosen = deepcopy(max(shortlisted, key=stable_key))
    chosen['shortlist'] = deepcopy(shortlisted)
    return chosen


def choose_independently(records, context, probabilities, decoders, grid=None):
    """Evaluate all40 independently; reports and outer rows never drive choices."""
    grid = independent.configurations() if grid is None else grid  # custom grid is test-only; CLI never accepts it
    indices = context.indices
    boot_normal = context.weighted_counts @ context.normal
    boot_positive = context.weighted_counts @ context.positive
    seen, unique, details = {}, [], []
    for ordinal, config in enumerate(grid):
        predictions = independent.decode(records, indices, probabilities, config, decoders)
        matrix = np.stack([independent.row_statistics(predictions[i], records[i]['labels']) for i in indices])
        totals = context.weights @ matrix
        score = components(totals, context.normal_mass, context.positive_mass)
        budget = error_budget(totals, context.normal_mass, context.positive_mass)
        key = [int(budget['feasible']), -budget['normalized_violation'], score['utility'], score['event_f1_05'],
               -float(totals[10]), -float(totals[9]), -float(totals[11]), -ordinal]
        fingerprint = tuple(tuple(tuple(pair) for pair in predictions[i]) for i in indices)
        duplicate_of = seen.get(fingerprint)
        if duplicate_of is None:
            seen[fingerprint] = ordinal
        sampled = context.weighted_counts @ matrix
        utilities = np.asarray([components(s, float(n), float(p))['utility']
                                for s, n, p in zip(sampled, boot_normal, boot_positive)])
        q20 = float(np.quantile(utilities, .2, method='linear'))
        row = {'config': config, 'stats': totals.tolist(), 'pooled_utility': score['utility'],
               'error_budget': budget, 'key': key, 'bootstrap_q20': q20,
               'stable_utility': .75 * score['utility'] + .25 * q20}
        if duplicate_of is None:
            unique.append(row)
        details.append({'ordinal': ordinal, **deepcopy(row), 'components': score,
                        'duplicate_of_ordinal': duplicate_of,
                        'prediction_fingerprint_sha256': independent.value_hash(fingerprint),
                        'row_statistics_sha256': independent.value_hash(matrix.tolist()),
                        'bootstrap_utilities': utilities.tolist()})
    chosen = shortlist_and_select(unique)
    shortlist_ordinals = {-int(row['key'][-1]) for row in chosen['shortlist']}
    for row in details:
        row['in_budget_pooled_top8'] = row['ordinal'] in shortlist_ordinals
    feasible_count = sum(row['error_budget']['feasible'] for row in unique)
    return chosen, {'configurations_evaluated': len(grid), 'unique_prediction_sets': len(unique),
                    'feasible_unique_configurations': feasible_count,
                    'explicit_fallback_used': feasible_count == 0,
                    'bootstrap_seed': context.seed, 'bootstrap_replicates': 32,
                    'bootstrap_counts_sha256': independent.value_hash(context.counts.tolist()),
                    'all_configurations': details}


def rank_candidates(rows):
    require(len(rows) == 3 and tuple(row.get('candidate') for row in rows) == CANDIDATES, 'candidate coverage/order differs')
    ranked = sorted(enumerate(rows), key=lambda item: (*stable_key(item[1]), -item[0]), reverse=True)
    return deepcopy(ranked[0][1]), [{'candidate': row['candidate'], 'key': [*stable_key(row), -ordinal]}
                                  for ordinal, row in ranked]

def replay(evidence, result):
    require(Path(independent.__file__).resolve() == (evidence.root / 'code/audit_compact_selection_v4.py').resolve(),
            'independent helper module is not the workspace helper')
    helper = evidence.read(Path('code/audit_compact_selection_v4.py'))
    result['independent_helper_sha256'] = independent.sha(helper)
    receipt, protocol = verify_v7_receipt(evidence)
    result['receipt'] = {'receipt_sha256': RECEIPT_PIN, 'protocol_sha256': PROTOCOL_PIN,
                         'source_snapshots_and_live_sources': len(SOURCES), 'bound_reuse_artifacts': 22,
                         'parent_receipts': receipt['parent_receipt_sha256']}
    result['static_review'] = verify_static_sources(evidence)
    records, partitions, dataset = independent.load_records(evidence)
    result['dataset'] = dataset
    pairs, proofs = load_parent_pairs(evidence, records, partitions)
    result['parent_cache_pair_proofs'] = proofs
    result['cache_count'] = 10
    result['base_fits'] = 0
    report = evidence.json(OUT / 'report.json')
    require(isinstance(report, dict) and report.get('status') == 'complete' and report.get('schema_version') == SCHEMA,
            'v7 report completion/schema differs')
    folds = report.get('folds')
    require(isinstance(folds, list) and len(folds) == 5, 'v7 report five-fold coverage differs')
    comparisons = independent.Comparisons()
    for field, expected in [('protocol_sha256', PROTOCOL_PIN), ('training_receipt_sha256', RECEIPT_PIN),
                            ('role', protocol['role']), ('base_fits', 0)]:
        comparisons.compare(report.get(field), expected, 'report/' + field)
    decoders = independent.decoder_functions()
    selected, fixed = {}, {name: {} for name in CANDIDATES}
    result['folds'] = []
    fallback_folds = []
    for fold, ((train, validation, inner_parts), ((ci, co), (ei, eo))) in enumerate(zip(partitions, pairs)):
        # Controls and fusion use ONLY parent inner OOF. No outer fusion/decode yet.
        inner = {'compact_control': ci, 'event_control': ei, 'equal_multiview': fusion(ci, ei, train)}
        context = independent.selection_context(records, train, independent.SEED + fold, EMPTY_PENALTY)
        choices, details = [], []
        for candidate in CANDIDATES:
            choice, detail = choose_independently(records, context, inner[candidate], decoders)
            choice['candidate'] = candidate
            choices.append(choice)
            details.append({'candidate': candidate, **detail})
        primary, ranking = rank_candidates(choices)
        any_feasible = any(row['feasible_unique_configurations'] > 0 for row in details)
        require(primary['error_budget']['feasible'] == any_feasible, 'feasible candidate not chosen when available')
        minimum_violation = min(row['error_budget']['normalized_violation']
                                for detail in details for row in detail['all_configurations']
                                if row['duplicate_of_ordinal'] is None)
        require(abs(primary['error_budget']['normalized_violation'] - minimum_violation) <= independent.ATOL,
                'primary did not minimize normalized budget violation')
        if not any_feasible:
            fallback_folds.append(fold)
        # Primary is now LOCKED. Only after this point may outer evidence decode.
        expected = {'fold': fold, 'train_indices': train, 'validation_indices': validation,
                    'primary': primary, 'all_inner_selections': choices}
        fold_report = folds[fold]
        require(isinstance(fold_report, dict), 'invalid v7 fold row')
        for field, value in expected.items():
            comparisons.compare(fold_report.get(field), value, f'report/folds/{fold}/{field}')
        outer = {'compact_control': co, 'event_control': eo, 'equal_multiview': fusion(co, eo, validation)}
        pred = independent.decode(records, validation, outer[primary['candidate']], primary['config'], decoders)
        require(set(selected).isdisjoint(pred), 'duplicate outer coverage')
        selected.update(pred)
        outer_rows = {str(i): [list(pair) for pair in pred[i]] for i in validation}
        comparisons.compare(fold_report.get('primary_predictions'), outer_rows, f'report/folds/{fold}/primary_predictions')
        for choice in choices:
            candidate = choice['candidate']
            fixed[candidate].update(independent.decode(records, validation, outer[candidate], choice['config'], decoders))
        result['folds'].append({**expected, 'inner_partitions': inner_parts, 'candidate_ranking': ranking,
                                'budget_outcome': {'any_candidate_feasible': any_feasible,
                                                   'explicit_infeasible_fallback': not any_feasible,
                                                   'minimum_normalized_violation': minimum_violation,
                                                   'feasible_unique_by_candidate': {
                                                       d['candidate']: d['feasible_unique_configurations'] for d in details}},
                                'content_weights': {str(i): float(w) for i, w in zip(train, context.weights)},
                                'normal_content_mass': context.normal_mass, 'positive_content_mass': context.positive_mass,
                                'bootstrap': {'seed': context.seed, 'replicates': 32, 'counts': context.counts.tolist()},
                                'candidate_configuration_replays': details, 'primary_predictions': outer_rows})
    baseline_report = evidence.json(independent.V2 / 'selection_audited/report.json', independent.BASELINE_PIN)
    baseline = independent.read_prediction_rows(baseline_report['grouped_v1_baseline']['predictions'], records, 'fixed baseline')
    baseline_metrics = independent.grouped_metrics(records, baseline)
    comparisons.compare(baseline_report['grouped_v1_baseline']['metrics'], baseline_metrics, 'fixed_baseline/metrics')
    primary_metrics = independent.grouped_metrics(records, selected)
    for name, predictions, metrics in [('grouped_v1_baseline', baseline, baseline_metrics),
                                       ('primary_v7_nested', selected, primary_metrics)]:
        bundle = report.get(name, {})
        require(isinstance(bundle, dict), 'invalid v7 report prediction bundle')
        comparisons.compare(bundle.get('predictions'), independent.prediction_rows(records, predictions), 'report/' + name + '/predictions')
        comparisons.compare(bundle.get('metrics'), metrics, 'report/' + name + '/metrics')
    comparisons.compare(report['primary_v7_nested'].get('summary'), independent.prediction_summary(records, selected),
                        'report/primary_v7_nested/summary')
    diagnostic = report.get('fixed_candidate_diagnostics_not_for_promotion')
    require(isinstance(diagnostic, dict) and set(diagnostic) == set(CANDIDATES), 'fixed diagnostic coverage differs')
    diagnostic_metrics = {}
    for candidate in CANDIDATES:
        metrics = independent.grouped_metrics(records, fixed[candidate])
        diagnostic_metrics[candidate] = metrics
        comparisons.compare(diagnostic[candidate].get('metrics'), metrics, 'report/fixed/' + candidate + '/metrics')
        comparisons.compare(diagnostic[candidate].get('predictions'), independent.prediction_rows(records, fixed[candidate]),
                            'report/fixed/' + candidate + '/predictions')
    guards = independent.promotion_guards(primary_metrics['all'], baseline_metrics['all'])
    comparisons.compare(report.get('statistical_promotion_checks'), guards, 'report/statistical_promotion_checks')
    comparisons.compare(report.get('statistical_promotion_passed'), all(guards.values()), 'report/statistical_promotion_passed')
    uncertainty = independent.paired_bootstrap(records, baseline, selected)
    comparisons.compare(report.get('paired_content_bootstrap'), uncertainty, 'report/paired_content_bootstrap')
    result['outer_metrics'] = {'grouped_v1_baseline': baseline_metrics, 'primary_v7_nested': primary_metrics}
    result['fixed_candidate_diagnostics_not_for_promotion'] = diagnostic_metrics
    result['paired_content_bootstrap'] = uncertainty
    result['inner_budget_summary'] = {'normal_budget': NORMAL_BUDGET, 'empty_budget': EMPTY_BUDGET,
                                      'feasibility_epsilon': FEASIBILITY_EPSILON,
                                      'infeasible_fallback_folds': fallback_folds,
                                      'infeasible_fallback_fold_count': len(fallback_folds),
                                      'fallback_never_claims_feasibility': True}
    result['deployment_guards'] = {'statistical_checks': guards, 'statistical_passed': all(guards.values()),
                                   'portable_fresh_parity_max_required': 2e-6,
                                   'portable_fullfit_and_fresh_pixels': 'separate_worker_not_attested_here',
                                   'inner_budget_targets_do_not_replace_outer_promotion_guards': True,
                                   'promotion_authorized_by_this_audit': False}
    evidence.recheck()
    result['before_after_consumed_files_equal'] = True
    result['comparisons'] = {'checked_nodes': comparisons.checked, 'mismatch_count': comparisons.mismatch_count,
                             'differences': comparisons.differences, 'absolute_float_tolerance': independent.ATOL}
    result['selection_integrity'] = {
        'report_matches_inner_locked_choices_and_predictions': comparisons.mismatch_count == 0,
        'outer_best_substitution': 'not_detected' if comparisons.mismatch_count == 0 else 'unverified_due_to_mismatches',
        'independent_choice_uses_outer_labels_or_probabilities': False,
        'fixed_outer_candidates_diagnostic_only': True}
    result['independent_selection_verified'] = comparisons.mismatch_count == 0
    result['status'] = 'passed' if comparisons.mismatch_count == 0 else 'failed'


def audit(root):
    evidence = independent.Evidence(root)
    result = {'schema_version': 'independent-selection-audit-v7', 'status': 'failed',
              'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'role': 'iterative_development_validation_not_new_blind_test',
              'independent_selection_verified': False, 'original_artifacts_modified': False,
              'reused_computation': ['previously independent row_statistics/content grouping/bootstrap/metrics/guards',
                                     'decode_duration', 'decode_v2'],
              'prohibited_imports': ['run_multiview_experiment_v7', 'run_video_gate_experiment_v6',
                                     'run_compact_experiment_v4', 'run_event_experiment_v5', 'optimized_video_statistics'],
              'limitations': [
                  'This is an adaptive inspected development experiment, not a new blind benchmark.',
                  'Shared fit proof means identical training-content ids, partitions and seeds; per-view transforms differ by design.',
                  'Transform digests are separately bound by metadata/receipts; no model/transform refit is performed.',
                  'Hashes attest current files/receipts, not omitted historical pixel-to-probability provenance.',
                  'Fullfit/fresh deployment and actual promotion are not authorized or attested by this audit.',
              ]}
    started = time.perf_counter()
    try:
        result['verifier_sha256'] = independent.sha(Path(__file__).read_bytes())
        replay(evidence, result)
    except (AuditError, OSError, KeyError, TypeError, IndexError, AttributeError, ValueError, StopIteration, SyntaxError) as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    result['consumed_file_sha256'] = dict(sorted(evidence.hashes.items()))
    result['elapsed_seconds'] = time.perf_counter() - started
    return result


def write_report(root, result):
    target = Path(root).resolve() / OUT / OUTPUT_NAME
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    with target.open('x', encoding='utf-8') as stream:
        stream.write(data + '\n')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-complete', action='store_true', help='main worker announced complete frozen v7 report')
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    try:
        complete_inputs(ROOT, args.reports_complete)
        result = audit(ROOT)
        target = write_report(ROOT, result)
    except (AuditError, OSError) as exc:
        print('AUDIT NOT RUN: ' + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({'status': result['status'], 'report': str(target),
                      'comparisons': result.get('comparisons', {}).get('checked_nodes'),
                      'mismatch_count': result.get('comparisons', {}).get('mismatch_count'),
                      'error': result.get('error')}, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
