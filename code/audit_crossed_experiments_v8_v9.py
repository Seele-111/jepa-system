"""Bounded independent v8/v9 asset, objective and nested-selection replay.

Run once after both reports are complete:
  python -B code/audit_crossed_experiments_v8_v9.py --reports-complete
Only new independent_selection_audit.json files are written, exclusively.
No training, portable/fresh-pixel replay, grids or production selectors/metrics.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import time

import numpy as np
import audit_compact_selection_v4 as B

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = 'independent_selection_audit.json'
V8_RECIPES = ('event_compact_rgb_corrected_motion_et', 'event_compact_rgb_corrected_motion_hgb')
V9_RECIPES = ('normalcost_event_compact_rgb_corrected_motion_et',)
V8_SOURCES = ('run_event_compact_experiment_v8.py', 'optimized_event_compact_v8.py') + tuple(
    dict.fromkeys(B.SOURCE_NAMES + ('optimized_event_training_v5.py', 'run_video_gate_experiment_v6.py',
                                  'optimized_video_gate_v6.py')))


@dataclass(frozen=True)
class Crossed:
    version: str
    schema: str
    recipes: tuple
    candidates: tuple
    receipt_pin: str
    parent_version: str
    parent_pin: str
    control: str
    sources: tuple
    runner: str
    training_jobs: int

    @property
    def folder(self):
        return Path('output') / ('algorithm-opt-2026-10-02-' + self.version)


V8 = Crossed('v8', 'event-compact-crossed-v8', V8_RECIPES,
             tuple((r, ((r, 1.),)) for r in V8_RECIPES) +
             (('compact_hgb_control', (('compact_rgb_corrected_motion_hgb', 1.),)),),
             '3fd139ebcab3cee2dd66a86f1aa211aad711e2459c1cd37720b871d650576643',
             'v4', B.RECEIPT_PIN, 'compact_rgb_corrected_motion_hgb', V8_SOURCES,
             'run_event_compact_experiment_v8.py', 40)
V9 = Crossed('v9', 'normal-negative-cost-v9', V9_RECIPES,
             ((V9_RECIPES[0], ((V9_RECIPES[0], 1.),)),
              ('event_compact_normal2_control', ((V8_RECIPES[0], 1.),))),
             'fb8892315ae0a15e3b0f6c3598ab8e6d0deacfffb9028c7595d13d9bcb51501e',
             'v8', V8.receipt_pin, V8_RECIPES[0],
             ('run_normal_cost_experiment_v9.py', 'optimized_normal_cost_v9.py') + V8_SOURCES,
             'run_normal_cost_experiment_v9.py', 20)
EXPERIMENTS = (V8, V9)


class TrainRecords:
    """Deny *all* held-out record access in objective and chooser replay."""
    def __init__(self, records, allowed):
        self.records, self.allowed = records, frozenset(allowed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        B.require(index in self.allowed, 'attempted held-out record access: ' + str(index))
        return self.records[index]


def receipt_verify(e, spec, parent):
    B.frozen_inventory(e.root, spec)
    receipt = e.json(spec.folder / 'training_receipt.json')
    B.require(receipt.get('receipt_sha256') == spec.receipt_pin and
              B.value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) == spec.receipt_pin,
              spec.version + ' receipt differs from user-supplied trust anchor')
    B.require(receipt['control_receipt_sha256'] == spec.parent_pin == parent['receipt_sha256'],
              'parent receipt lineage differs')
    for key in ('input_hashes', 'inputs_sha256', 'input_source_hashes', 'baseline_report_sha256'):
        B.require(receipt[key] == parent[key], 'parent/new input inventory differs: ' + key)
    B.require(len(receipt['input_hashes']) == 479 and len(receipt['input_source_hashes']) == 11 and
              B.value_hash(receipt['input_hashes']) == receipt['inputs_sha256'], 'input coverage/digest differs')
    B.require(receipt['baseline_report_sha256'] == B.BASELINE_PIN, 'baseline pin differs')
    for section in ('input_hashes', 'input_source_hashes'):
        for path, digest in receipt[section].items():
            e.read(path, digest)
    B.require(set(receipt['sources']) == set(spec.sources), 'source inventory differs')
    for name, digest in receipt['sources'].items():
        e.read(spec.folder / 'sources' / name, digest)
        e.read(Path('code') / name, digest)
    parent_folder = Path('output') / ('algorithm-opt-2026-10-02-' + spec.parent_version)
    controls = {str(parent_folder / 'training' / f'{f}_{spec.control}.{ext}').replace('\\', '/')
                for f in range(5) for ext in ('json', 'npz')}
    B.require(set(receipt['control_artifacts']) == controls, 'five parent controls coverage differs')
    for path, digest in receipt['control_artifacts'].items():
        e.read(path, digest)
    protocol = e.json(spec.folder / 'protocol.json', receipt['protocol_sha256'])
    B.require(protocol.get('schema_version') == spec.schema and protocol.get('date') == '2026-10-02',
              'protocol identity differs')
    B.require(protocol['primary_candidates'] == [[n, [list(m) for m in members]] for n, members in spec.candidates],
              'frozen candidate/order/blend differs')
    B.require(protocol['decoder_configs'] == B.configurations() and
              protocol['promotion_guards'] == B.GUARD_LIMITS and protocol['training_jobs'] == spec.training_jobs,
              'frozen 40 decoders/guards/fit count differs')
    if spec.version == 'v9':
        B.require(protocol['normal_cost_candidates'] == [2, 4] and protocol['new_recipe'] == list(spec.recipes),
                  'normal cost scope differs')
    return receipt, protocol


def functions(e, folder, name):
    path = folder / 'sources' / name
    tree = ast.parse(e.read(path).decode('utf-8-sig'), filename=str(path))
    return tree, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def same_ast(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def expression(text):
    return ast.parse(text, mode='eval').body


def assignment_name(node):
    return (node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) else None)


def normalized_fit(function, version):
    """Remove ONLY the reviewed objective block and RGB-recipe metadata adapters."""
    node = deepcopy(function)
    node.name = 'fit_member'
    start = next(k for k, n in enumerate(node.body) if assignment_name(n) == 'weights')
    stop = next(k for k in range(start + 1, len(node.body)) if isinstance(node.body[k], ast.If)
                and same_ast(node.body[k].test, expression("recipe.endswith('_hgb')")))
    if version == 'v4':
        expected = ast.parse('''weights = np.concatenate([np.full(len(records[i]['labels']), content[i] * (2 if not np.any(records[i]['labels']) else 1) / len(records[i]['labels'])) for i in train])
positive, negative = weights[y > 0].sum(), weights[y == 0].sum()
if min(positive, negative) <= 0:
    raise ValueError('training requires both frame classes')
weights[y > 0] *= negative / positive
weights *= len(weights) / weights.sum()
''').body
    else:
        weight_name = 'event_weights' if version == 'v8' else 'normal_cost_weights'
        expected = ast.parse(f'weights = {weight_name}(records, train)').body
    B.require(len(expected) == stop - start and all(same_ast(a, b) for a, b in zip(node.body[start:stop], expected)),
              version + ' has unexpected changes in objective block')
    node.body[start:stop] = ast.parse('weights = OBJECTIVE(records, train)').body
    if version == 'v4':
        pca = next(n for n in node.body if assignment_name(n) == 'pca')
        B.require(same_ast(pca.value, expression("fit_rgb_pca(records, train) if 'rgb' in recipe else None")),
                  'v4 RGB PCA flow differs')
        pca.value = pca.value.body
    else:
        adapters = [n for n in node.body if assignment_name(n) == 'view_recipe']
        B.require(len(adapters) == 1 and same_ast(adapters[0].value, expression('VIEW_RECIPES[recipe]')),
                  'unexpected view adapter')
        node.body.remove(adapters[0])

    class Adapters(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id == 'view_recipe':
                n.id = 'recipe'
            return n

        def visit_Dict(self, n):
            n = self.generic_visit(n)
            pairs = [(k, v) for k, v in zip(n.keys, n.values)
                     if not isinstance(k, ast.Constant) or k.value != 'training_recipe_id']
            n.keys, n.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return n

    return Adapters().visit(node)


def source_review(e, spec):
    base_folder = B.V4
    _, base = functions(e, base_folder, 'optimized_compact_model_v4.py')
    _, event = functions(e, V8.folder, 'optimized_event_compact_v8.py')
    _, cost = functions(e, V9.folder, 'optimized_normal_cost_v9.py')
    bf, ef, cf = base['fit_compact_member'], event['fit_event_compact_member'], cost['fit_normal_cost_member']
    B.require(same_ast(normalized_fit(bf, 'v4'), normalized_fit(ef, 'v8')),
              'v8 non-objective model/feature/video-head source changed')
    B.require(same_ast(normalized_fit(ef, 'v8'), normalized_fit(cf, 'v9')),
              'v9 changes beyond frame weights and recipe metadata')
    for function in (ef, cf):
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'fit_rgb_pca']
        B.require(len(calls) == 1 and same_ast(calls[0], expression('fit_rgb_pca(records, train)')),
                  'PCA is not fitted only on train')
        for n in ast.walk(function):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == 'labels':
                owners = [c for c in ast.walk(function) if isinstance(c, (ast.ListComp, ast.SetComp, ast.GeneratorExp))
                          and any(child is n for child in ast.walk(c))]
                B.require(owners and any(any(isinstance(g.target, ast.Name) and g.target.id == 'i'
                                            and same_ast(g.iter, expression('train')) for g in c.generators)
                                         for c in owners), 'labels read outside train comprehension')
    _, feature = functions(e, spec.folder, 'optimized_compact_features_v4.py')
    for name in ('fit_rgb_pca', '_raw', 'feature_view_v4'):
        B.require(not any(isinstance(n, ast.Constant) and n.value in ('labels', 'event_count', 'name', 'generator')
                          for n in ast.walk(feature[name])), 'non-label-free feature source: ' + name)
    _, v8_runner = functions(e, V8.folder, V8.runner)
    _, runner = functions(e, spec.folder, spec.runner)
    B.require(same_ast(v8_runner['choose'], runner['choose']) and
              same_ast(v8_runner['fit_task'], runner['fit_task']), 'crossed source chooser/fit partition flow differs')
    chooser = runner['choose']
    calls = [n for n in ast.walk(runner['select_all']) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'choose']
    B.require(len(calls) == 1 and same_ast(calls[0], expression('choose(records, train, blended(members, inner, train), 20261002 + fold)')),
              'chooser receives held-out labels/probabilities or different seeds')
    B.require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id in ('metrics', '_group_metrics', 'load_new_fold', 'load_fold')
                      for n in ast.walk(chooser)), 'chooser uses metrics/cache loading')
    loop = next(n for n in runner['select_all'].body if isinstance(n, ast.For))
    lock = next(n for n in loop.body if assignment_name(n) == 'chosen')
    outer = next(n for n in loop.body if assignment_name(n) == 'pred')
    B.require(lock.lineno < outer.lineno and isinstance(lock.value, ast.Call)
              and same_ast(lock.value.args[0], expression('rows')), 'primary not locked before outer decoding')
    _, utility = functions(e, spec.folder, 'run_video_gate_experiment_v6.py')
    ret = next(n for n in utility['utility'].body if isinstance(n, ast.Return))
    B.require(same_ast(ret.value, expression('.35*f(s[...,:3])+.65*f(s[...,3:6])+.10*f(s[...,6:9])-.30*nf-.30*pe')),
              'unconstrained .30/.30 utility differs')
    return {'status': 'passed', 'review_method': 'frozen AST inspection; fit/choose/utility never executed',
            'v8_only_frame_objective_changed': True, 'v9_only_normal_frame_cost_changed': True,
            'video_head_remains_normal2': True, 'train_only_PCA_and_labels': True,
            'heldout_labels_or_scores_enter_chooser': False,
            'inner_lock_line': lock.lineno, 'first_primary_outer_decode_line': outer.lineno,
            'normalized_training_ast_sha256': B.value_hash(ast.dump(normalized_fit(ef, 'v8'), include_attributes=False))}


def training_results(e, folder, recipes):
    training = e.json(folder / 'training_report.json')
    B.require(training.get('status') == 'complete' and training.get('before_after_evidence_equal') is True,
              'training report is incomplete or changed its evidence')
    rows = training.get('results')
    B.require(isinstance(rows, list) and len(rows) == 5 * len(recipes), 'training job receipt coverage differs')
    by_key = {}
    for row in rows:
        key = row['fold'], row['recipe']
        B.require(key not in by_key, 'duplicate training job receipt')
        by_key[key] = row
    B.require(set(by_key) == {(f, r) for f in range(5) for r in recipes}, 'training report fold/recipe coverage differs')
    return by_key


def load_recipe(e, folder, recipe, fold, partition, records, receipt, results):
    path = folder / 'training' / f'{fold}_{recipe}.npz'
    meta = e.json(path.with_suffix('.json'))
    B.verify_training_metadata(meta, recipe, fold, partition, records, receipt['receipt_sha256'])
    data = e.read(path, meta['npz_sha256'])
    row = results[fold, recipe]
    B.require(row['npz_sha256'] == B.sha(data) and row['seconds'] == meta['seconds'],
              'cache/metadata/training report disagreement')
    inner, outer = B.probability_maps(data, meta, records, path.as_posix())
    B.require(all(x.dtype == np.float32 for p in (inner, outer) for x in p.frame.values()),
              'frame probabilities are not the declared float32 inference outputs')
    return inner, outer, meta


def control_comparison(records, new, control):
    for key in ('train', 'validation', 'inner_partitions', 'outer_seed', 'fit_evidence'):
        B.require(new[2][key] == control[2][key], 'control partition/seed/fit-content/transform mismatch: ' + key)
    maximum = max(abs(p.video[i] - q.video[i]) for p, q in zip(new[:2], control[:2]) for i in p.video)
    B.require(maximum <= B.ATOL, 'weight-only experiment changed unchanged video head probabilities')
    return {'all_partitions_seeds_fit_content_and_transform_proofs_equal': True,
            'unchanged_video_head_probability_max_abs_difference': maximum}


def independent_event_weights(records, train, normal_mass):
    """Raw normal2/normal4 construction, BEFORE global class rebalance."""
    weights_by_row = B.content_weights(records, train)
    rows, label_rows = [], []
    for i, content_weight in zip(train, weights_by_row):
        labels = np.asarray(records[i]['labels'], dtype=bool)
        events = B.label_spans(labels)
        weights = np.zeros(len(labels), dtype=np.float64)
        background = np.flatnonzero(~labels)
        if len(background):
            weights[background] = content_weight * (normal_mass if not events else 1.) / len(background)
        for begin, end in events:
            weights[begin:end + 1] = content_weight / (len(events) * (end - begin + 1))
        rows.append(weights)
        label_rows.append(labels)
    weights, labels = np.concatenate(rows), np.concatenate(label_rows)
    negative, positive = float(weights[~labels].sum()), float(weights[labels].sum())
    B.require(min(positive, negative) > 0, 'objective needs both classes')
    weights[labels] *= negative / positive
    return weights * (len(weights) / weights.sum())


def objective_functions(e):
    # Execute ONLY pure frozen weight functions; never a fit or model import.
    _, event = functions(e, V8.folder, 'optimized_event_training_v5.py')
    _, cost = functions(e, V9.folder, 'optimized_normal_cost_v9.py')
    ns = {'np': np, 'spans': B.label_spans,
          'content_weights': lambda records, train: dict(zip(train, B.content_weights(records, train)))}
    for name, function in (('event_weights', event['event_weights']), ('normal_cost_weights', cost['normal_cost_weights'])):
        module = ast.fix_missing_locations(ast.Module(body=[deepcopy(function)], type_ignores=[]))
        exec(compile(module, '<frozen-weight-only-' + name + '>', 'exec'), ns)
    return ns['event_weights'], ns['normal_cost_weights']


def objective_replay(e, records, partitions):
    event_weights, normal_cost_weights = objective_functions(e)
    results, maximum2, maximum4 = [], 0., 0.
    for fold, (train, validation, inner) in enumerate(partitions):
        for k, fit in enumerate(inner + [{'fit': train, 'validation': validation}]):
            ids = fit['fit']
            guarded = TrainRecords(records, ids)
            old, new = event_weights(guarded, ids), normal_cost_weights(guarded, ids)
            reference2 = independent_event_weights(guarded, ids, 2.)
            reference4 = independent_event_weights(guarded, ids, 4.)
            delta2, delta4 = float(np.max(np.abs(old - reference2))), float(np.max(np.abs(new - reference4)))
            B.require(np.allclose(old, reference2, rtol=2e-13, atol=2e-12) and
                      np.allclose(new, reference4, rtol=2e-13, atol=2e-12), 'normal2/normal4 objective inequivalence')
            maximum2, maximum4 = max(maximum2, delta2), max(maximum4, delta4)
            labels = np.concatenate([records[i]['labels'] for i in ids]).astype(bool)
            masses = [float(new[~labels].sum()), float(new[labels].sum())]
            B.require(math.isclose(masses[0], masses[1], rel_tol=2e-13, abs_tol=2e-10) and
                      math.isclose(float(new.mean()), 1., rel_tol=2e-13), 'weight class balance/mean1 differs')
            results.append({'fold': fold, 'partition': k if k < 3 else 'outer', 'fit_rows': len(ids),
                            'fit_content_sha256': sorted({records[i]['sha256'] for i in ids}),
                            'frames': len(new), 'normal2_max_abs_difference': delta2,
                            'normal4_max_abs_difference': delta4, 'normal4_negative_positive_mass': masses,
                            'normal4_mean': float(new.mean())})
    return {'status': 'passed', 'fit_partitions': len(results), 'heldout_record_access_denied': True,
            'v8_event_normal2_max_abs_difference': maximum2, 'v9_raw_normal4_equivalence_max_abs_difference': maximum4,
            'positive_negative_balanced_and_mean1': True,
            'normal4_changes_frame_normal_negatives_only_before_rebalance': True,
            'video_head_stays_normal2': True, 'partitions': results}


def components(stats, normal_mass, positive_mass):
    """Independent v8/v9 utility; Bacon's .20 normal penalty is NOT reused."""
    s = np.asarray(stats, dtype=np.float64)
    B.require(s.shape == (12,) and np.isfinite(s).all() and np.all(s >= 0), 'invalid sufficient statistics')
    f = [float(2 * s[k] / max(1e-12, 2 * s[k] + s[k + 1] + s[k + 2])) for k in (0, 3, 6)]
    normal_rate = float(s[9] / normal_mass) if normal_mass > 0 else 0.
    empty_rate = float(s[10] / positive_mass) if positive_mass > 0 else 0.
    return {'event_f1_03': f[0], 'event_f1_05': f[1], 'frame_f1': f[2],
            'normal_false_positive_rate': normal_rate, 'positive_empty_rate': empty_rate,
            'utility': .35 * f[0] + .65 * f[1] + .10 * f[2] - .30 * normal_rate - .30 * empty_rate}


def choose_independently(records, context, probabilities, decoders, deadline=None):
    ids = context.indices
    B.require(set(probabilities.frame) == set(ids) == set(probabilities.video), 'chooser probability scope differs')
    seen, unique, matrices, trace = {}, [], {}, []
    for ordinal, config in enumerate(B.configurations()):
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError('bounded chooser time budget exhausted')
        pred = B.decode(records, ids, probabilities, config, decoders)
        fingerprint = tuple(tuple(tuple(pair) for pair in pred[i]) for i in ids)
        duplicate = seen.get(fingerprint)
        item = {'ordinal': ordinal, 'config': config, 'duplicate_of_ordinal': duplicate,
                'prediction_fingerprint_sha256': B.value_hash(fingerprint)}
        if duplicate is None:
            seen[fingerprint] = ordinal
            matrix = np.stack([B.row_statistics(pred[i], records[i]['labels']) for i in ids])
            totals = context.weights @ matrix
            part = components(totals, context.normal_mass, context.positive_mass)
            utility = part['utility']
            key = [utility, part['event_f1_05'], -float(totals[10]), -float(totals[9]), -float(totals[11]), -ordinal]
            row = {'config': config, 'stats': totals.tolist(), 'pooled_utility': utility, 'key': key}
            unique.append(row)
            matrices[ordinal] = matrix
            item.update(components=part, key=key, stats=totals.tolist(), row_statistics_sha256=B.value_hash(matrix.tolist()))
        trace.append(item)
    # Dedupe (first ordinal survives), pooled top8 FIRST, then bootstrap q20.
    shortlist = sorted(unique, key=lambda row: row['key'], reverse=True)[:8]
    B.require(shortlist, 'empty independent shortlist')
    bn, bp = context.weighted_counts @ context.normal, context.weighted_counts @ context.positive
    for row in shortlist:
        ordinal = -int(row['key'][-1])
        boot = context.weighted_counts @ matrices[ordinal]
        utilities = [components(s, float(n), float(p))['utility'] for s, n, p in zip(boot, bn, bp)]
        row['bootstrap_q20'] = float(np.quantile(utilities, .2, method='linear'))
        row['stable_utility'] = .75 * row['pooled_utility'] + .25 * row['bootstrap_q20']
        trace[ordinal].update(in_pooled_top8=True, bootstrap_utilities=utilities,
                              bootstrap_q20=row['bootstrap_q20'], stable_utility=row['stable_utility'])
    chosen = deepcopy(max(shortlist, key=lambda row: (row['stable_utility'], *row['key'][1:])))
    chosen['shortlist'] = deepcopy(shortlist)
    detail = {'configurations_evaluated': 40, 'unique_prediction_sets': len(unique), 'pooled_top_k': 8,
              'bootstrap_replicates': 32, 'bootstrap_seed': context.seed, 'bootstrap_quantile': .20,
              'quantile_method': 'linear', 'normal_fpr_penalty': .30, 'positive_empty_penalty': .30,
              'bootstrap_counts_sha256': B.value_hash(context.counts.tolist()), 'all_configurations': trace}
    return chosen, detail


def replay_selection(e, spec, records, partitions, caches, receipt, protocol, result, budget):
    report = e.json(spec.folder / 'report.json')
    B.require(report.get('schema_version') == spec.schema and report.get('status') == 'complete', 'report not complete')
    compare = B.Comparisons()
    for key, expected in (('protocol_sha256', receipt['protocol_sha256']),
                          ('training_receipt_sha256', spec.receipt_pin), ('role', protocol['role'])):
        compare.compare(report.get(key), expected, 'report/' + key)
    report_folds = report.get('folds')
    B.require(isinstance(report_folds, list) and len(report_folds) == 5, 'report fold coverage differs')
    decoders = B.decoder_functions()
    selected, fixed, details = {}, {n: {} for n, _ in spec.candidates}, []
    deadline = time.perf_counter() + budget if budget else None
    complete, reason = True, None
    for fold, (train, validation, parts) in enumerate(partitions):
        inner, outer = caches[spec.version][fold]
        guarded = TrainRecords(records, train)
        context = B.selection_context(guarded, train, B.SEED + fold, empty_penalty=.30)
        choices, traces = [], []
        if complete:
            try:
                for name, members in spec.candidates:
                    choice, trace = choose_independently(guarded, context, B.blend(members, inner, train), decoders, deadline)
                    choice['candidate'] = name
                    choices.append(choice)
                    traces.append({'candidate': name, **trace})
            except TimeoutError as exc:
                complete, reason = False, str(exc)
        fold_report = report_folds[fold]
        for key, expected in (('fold', fold), ('train_indices', train), ('validation_indices', validation), ('inner_partitions', parts)):
            compare.compare(fold_report.get(key), expected, f'report/folds/{fold}/{key}')
        if complete:
            primary, ranking = B.rank_candidates(choices, spec.candidates)
            compare.compare(fold_report.get('primary'), primary, f'report/folds/{fold}/primary')
            compare.compare(fold_report.get('all_inner_selections'), choices, f'report/folds/{fold}/all_inner_selections')
        else:
            # Honest fallback: saved choice, NOT independently attested selection.
            primary, ranking = fold_report['primary'], None
            B.require(primary['candidate'] in dict(spec.candidates) and primary['config'] in B.configurations(),
                      'saved primary is outside frozen candidates/configurations')
        prediction = B.decode(records, validation, B.blend(dict(spec.candidates)[primary['candidate']], outer, validation), primary['config'], decoders)
        B.require(not set(selected) & set(prediction), 'primary duplicated outer rows')
        selected.update(prediction)
        compare.compare(fold_report.get('primary_predictions'), {str(i): prediction[i] for i in validation},
                        f'report/folds/{fold}/primary_predictions')
        if complete:
            for choice in choices:
                p = B.decode(records, validation, B.blend(dict(spec.candidates)[choice['candidate']], outer, validation), choice['config'], decoders)
                fixed[choice['candidate']].update(p)
        details.append({'fold': fold, 'selection_recomputed': complete, 'primary': primary,
                        'candidate_ranking': ranking, 'candidate_replays': traces,
                        'primary_predictions': {str(i): prediction[i] for i in validation}})
        print(json.dumps({'experiment': spec.version, 'fold': fold, 'selection_recomputed': complete,
                          'candidate': primary['candidate']}, ensure_ascii=False), flush=True)
    baseline_report = e.json(B.V2 / 'selection_audited/report.json', B.BASELINE_PIN)
    baseline = B.read_prediction_rows(baseline_report['grouped_v1_baseline']['predictions'], records, 'baseline')
    baseline_metrics, primary_metrics = B.grouped_metrics(records, baseline), B.grouped_metrics(records, selected)
    compare.compare(baseline_report['grouped_v1_baseline']['metrics'], baseline_metrics, 'frozen_baseline/metrics')
    compare.compare(report['grouped_v1_baseline'], {'metrics': baseline_metrics, 'predictions': B.prediction_rows(records, baseline)},
                    'report/grouped_v1_baseline')
    primary_bundle = {'metrics': primary_metrics, 'predictions': B.prediction_rows(records, selected),
                      'summary': B.prediction_summary(records, selected)}
    compare.compare(report.get('primary_' + spec.version + '_nested'), primary_bundle, 'report/primary_nested')
    guards = B.promotion_guards(primary_metrics['all'], baseline_metrics['all'])
    compare.compare(report.get('statistical_promotion_checks'), guards, 'report/statistical_promotion_checks')
    compare.compare(report.get('statistical_promotion_passed'), all(guards.values()), 'report/statistical_promotion_passed')
    uncertainty = B.paired_bootstrap(records, baseline, selected)
    compare.compare(report.get('paired_content_bootstrap'), uncertainty, 'report/paired_content_bootstrap')
    if complete:
        expected_fixed = {c: {'metrics': B.grouped_metrics(records, p), 'predictions': B.prediction_rows(records, p)}
                          for c, p in fixed.items()}
        compare.compare(report.get('fixed_candidate_diagnostics_not_for_promotion'), expected_fixed, 'report/fixed_diagnostics')
    result.update(folds=details, primary=primary_bundle, baseline_metrics=baseline_metrics,
                  paired_content_bootstrap=uncertainty,
                  guards={'statistical_checks': guards, 'statistical_passed': all(guards.values()),
                          'portable_fresh_parity': 'separate_main_worker_not_attested_here',
                          'promotion_authorized_by_this_audit': False},
                  chooser={'fully_recomputed': complete, 'reason_if_incomplete': reason,
                           'candidate_fold_replays_completed': sum(len(f['candidate_replays']) for f in details),
                           'normal_fpr_penalty': .30, 'positive_empty_penalty': .30,
                           'time_budget_seconds': budget, 'heldout_record_access_denied': True},
                  comparisons={'checked_nodes': compare.checked, 'mismatch_count': compare.mismatch_count,
                               'differences': compare.differences, 'absolute_float_tolerance': B.ATOL})
    result['independent_selection_verified'] = complete and compare.mismatch_count == 0
    result['primary_predictions_metrics_guards_verified'] = compare.mismatch_count == 0
    result['status'] = ('failed' if compare.mismatch_count else 'passed' if complete else 'partial_selection_not_recomputed')


def readiness(root, acknowledged):
    B.require(acknowledged, 'requires --reports-complete; no polling/waiting')
    for spec in EXPERIMENTS:
        output = root / spec.folder / OUTPUT_NAME
        B.require(not output.exists(), 'refusing to overwrite independent audit: ' + str(output))
        for name in ('protocol.json', 'training_receipt.json', 'training_report.json', 'report.json'):
            path = root / spec.folder / name
            B.require(path.is_file(), 'missing artifact (no wait): ' + str(path))
            if name in ('training_report.json', 'report.json'):
                B.require(B.strict_json(path.read_bytes(), str(path)).get('status') == 'complete',
                          'report not complete (no wait): ' + str(path))


def initial_result(spec):
    return {'schema_version': 'independent-crossed-selection-audit-' + spec.version,
            'experiment': spec.version, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'failed', 'role': 'iterative_development_validation_not_new_blind_test',
            'receipt_trust_anchor': spec.receipt_pin, 'receipt_anchor_source': 'human_supplied_completed_training_receipt',
            'independent_selection_verified': False, 'primary_predictions_metrics_guards_verified': False,
            'original_artifacts_modified': False, 'no_training_or_fresh_portable_replay': True,
            'reused_independent_helper': 'code/audit_compact_selection_v4.py',
            'reused_production_computation': ['decode_duration', 'decode_v2'],
            'not_called': ['main_choose', 'main_utility', 'video_statistics', 'main_metrics', 'model.fit', 'fresh_extraction'],
            'limitations': [
                'Repeated development validation on these 77 rows is not a blind test or confirmatory evidence.',
                'Hashes bind bytes, not authenticated execution or omitted historical pixel provenance.',
                'Fit-content/transform proofs are cache/source attestations; model/PCA fitting is not repeated.',
                'Portable/full-fit/fresh-pixel parity and deployment promotion belong to the main worker.',
                'Normal4 doubles raw normal frame negatives; rebalance/mean1 rescales final weights, so final normal weights are not simply 2x.',
                'Primary metrics retain the report raw-row contract; inner selection shares duplicate content mass.',
            ]}


def run_audits(root, chooser_budget=180.):
    e, started = B.Evidence(root), time.perf_counter()
    results = {spec.version: initial_result(spec) for spec in EXPERIMENTS}
    try:
        script_sha = B.sha(e.read(Path('code') / Path(__file__).name))
        helper_sha = B.sha(e.read(Path('code') / 'audit_compact_selection_v4.py'))
        B.frozen_inventory(e.root, B.V4_EXPERIMENT)
        parent_receipt, _ = B.verify_receipt(e, B.V4_EXPERIMENT)
        r8, p8 = receipt_verify(e, V8, parent_receipt)
        r9, p9 = receipt_verify(e, V9, r8)
        receipts, protocols = {'v4': parent_receipt, 'v8': r8, 'v9': r9}, {'v8': p8, 'v9': p9}
        records, partitions, dataset = B.load_records(e)
        weights = objective_replay(e, records, partitions)
        tr = {'v4': training_results(e, B.V4, B.RECIPE_NAMES),
              'v8': training_results(e, V8.folder, V8.recipes),
              'v9': training_results(e, V9.folder, V9.recipes)}
        raw, caches = {'v4': [], 'v8': [], 'v9': []}, {'v8': [], 'v9': []}
        for fold, partition in enumerate(partitions):
            parent = load_recipe(e, B.V4, V8.control, fold, partition, records, parent_receipt, tr['v4'])
            raw['v4'].append({V8.control: parent})
            for spec in EXPERIMENTS:
                control = raw[spec.parent_version][fold][spec.control]
                members, asset_details = {spec.control: control}, []
                for recipe in spec.recipes:
                    new = load_recipe(e, spec.folder, recipe, fold, partition, records, receipts[spec.version], tr[spec.version])
                    proof = control_comparison(records, new, control)
                    members[recipe] = new
                    asset_details.append({'fold': fold, 'recipe': recipe, 'npz_sha256': new[2]['npz_sha256'],
                                          'metadata_signature': new[2]['signature'], 'exact_NPZ_keys': 154,
                                          'fit_evidence': new[2]['fit_evidence'], 'parent_control': spec.control, **proof})
                raw[spec.version].append(members)
                caches[spec.version].append(({r: a[0] for r, a in members.items()}, {r: a[1] for r, a in members.items()}))
                results[spec.version].setdefault('new_training_caches', []).extend(asset_details)
        for spec in EXPERIMENTS:
            result = results[spec.version]
            result.update(verifier_sha256=script_sha, independent_helper_sha256=helper_sha, dataset=dataset,
                          receipt={'receipt_sha256': spec.receipt_pin, 'protocol_sha256': receipts[spec.version]['protocol_sha256'],
                                   'parent_receipt_sha256': spec.parent_pin, 'baseline_report_sha256': B.BASELINE_PIN,
                                   'inputs_sha256': receipts[spec.version]['inputs_sha256'], 'bound_inputs': 479,
                                   'input_sources': 11, 'frozen_and_live_sources': len(spec.sources),
                                   'parent_control_artifacts': 10, 'new_fit_proofs': spec.training_jobs},
                          assets={'status': 'passed', 'new_NPZ_caches': 5 * len(spec.recipes),
                                  'parent_control_NPZ_caches': 5, 'new_inner_fits': 15 * len(spec.recipes),
                                  'new_outer_fits': 5 * len(spec.recipes), 'fit_proofs': spec.training_jobs,
                                  'identical_parent_partitions_seeds_content_and_transform_proofs': True,
                                  'exact_rows_and_content_isolation': True},
                          objective_equivalence=weights, source_flow=source_review(e, spec))
            try:
                replay_selection(e, spec, records, partitions, caches, receipts[spec.version], protocols[spec.version], result, chooser_budget)
            except (B.AuditError, OSError, KeyError, TypeError, IndexError, AttributeError, ValueError) as exc:
                result.update(status='failed', independent_selection_verified=False,
                              primary_predictions_metrics_guards_verified=False, error={'type': type(exc).__name__, 'message': str(exc)})
        e.recheck()
        for result in results.values():
            result['before_after_consumed_files_equal'] = True
    except (B.AuditError, OSError, KeyError, TypeError, IndexError, AttributeError, ValueError, SyntaxError) as exc:
        for result in results.values():
            result.update(status='failed', independent_selection_verified=False,
                          primary_predictions_metrics_guards_verified=False, error={'type': type(exc).__name__, 'message': str(exc)})
    for result in results.values():
        result['consumed_file_sha256'] = dict(sorted(e.hashes.items()))
        result['watched_assets'] = len(e.hashes)
        result['elapsed_seconds_both_experiments'] = time.perf_counter() - started
    return results


def write_report(root, spec, result):
    path = root / spec.folder / OUTPUT_NAME
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data + '\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reports-complete', action='store_true', help='human-confirmed complete reports; no polling')
    parser.add_argument('--chooser-budget-seconds', type=float, default=180., help='bounded chooser budget per experiment')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not math.isfinite(args.chooser_budget_seconds) or args.chooser_budget_seconds <= 0:
        parser.error('chooser time budget must be finite and positive')
    try:
        readiness(root, args.reports_complete)
    except (B.AuditError, OSError) as exc:
        parser.error(str(exc))
    results = run_audits(root, args.chooser_budget_seconds)
    for spec in EXPERIMENTS:
        result = results[spec.version]
        path = write_report(root, spec, result)
        print(json.dumps({'experiment': spec.version, 'status': result['status'],
                          'independent_selection_verified': result['independent_selection_verified'],
                          'comparisons': result.get('comparisons', {}), 'error': result.get('error'),
                          'audit_path': str(path)}, ensure_ascii=False), flush=True)
    return 1 if any(r['status'] == 'failed' for r in results.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
