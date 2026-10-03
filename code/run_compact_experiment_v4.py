#!/usr/bin/env python3
"""Frozen, content-grouped development experiment v4; no outer-best promotion."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from optimized_compact_features_v4 import RECIPES, content_weights
from optimized_compact_model_v4 import fit_compact_member, export_compact_member, predict_compact_member
from optimized_grouped_training import ROOT, NEW, load_grouped, grouped_folds, digest, fit_member
from optimized_duration_decoder_v4 import decode_duration
from optimized_locator import metrics, SCHEMA
from optimized_recall_decoder import decode_v2
from optimized_training_provenance import collect_new_fit_inputs
from optimized_video_statistics import VideoTruth, video_statistics
from run_algorithm_optimization import _group_metrics
from run_recall_selection import load_fold, grouped_bootstrap, paired_uncertainty
from run_optimization_selection import Probabilities, predictions_json, prediction_summary

OUT = ROOT / 'output' / 'algorithm-opt-2026-10-02-v4'
BASELINE = NEW / 'selection_audited/report.json'
PRIMARY = ('compact_corrected_motion_et', 'compact_rgb_corrected_motion_et',
           'compact_rgb_corrected_motion_hgb')
CANDIDATES = tuple((r, ((r, 1.),)) for r in PRIMARY) + (
    ('corrected_motion_et_duration_control', (('corrected_motion_et', 1.),)),
    ('compact_rgb_boost_old_70_30', (('compact_rgb_corrected_motion_hgb', .7), ('corrected_motion_et', .3))))
SOURCES = ('run_compact_experiment_v4.py', 'optimized_compact_features_v4.py',
           'optimized_compact_model_v4.py', 'optimized_duration_decoder_v4.py',
           'optimized_grouped_training.py', 'optimized_feature_view_v2.py',
           'optimized_feature_view.py', 'optimized_locator.py', 'optimized_recall_decoder.py',
           'optimized_video_statistics.py', 'optimized_training_provenance.py',
           'run_recall_selection.py', 'run_optimization_selection.py', 'run_algorithm_optimization.py',
           'build_optimization_dataset.py')
_STATE = None


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda x: x.item() if isinstance(x, np.generic) else None)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def configurations():
    for threshold in [.3, .4, .5, .6]:
        for penalty in [.03, .10, .25]:
            for vt in [0, .7]:
                yield {'kind': 'duration-logit-v4', 'threshold': threshold,
                       'transition_seconds': penalty, 'min_seconds': .12, 'video_threshold': vt}
        for ratio in [1., .7]:
            for vt in [0, .7]:
                yield {'kind': 'recall-stable-v2', 'threshold': threshold, 'low_ratio': ratio,
                       'smooth_seconds': .15, 'gap_seconds': .12, 'min_seconds': .12,
                       'seed_seconds': .05, 'video_threshold': vt, 'video_strength': 0, 'video_floor': 0}


def protocol():
    return {'schema_version': 'compact-experiment-v4', 'date': '2026-10-02',
        'role': 'iterative_development_validation_not_new_blind_test',
        'data': '77 annotation rows, 76 content SHA groups; conflicting aliases retained and grouped',
        'outer': 'same frozen content-grouped 5 folds as v2; each outer train gets inner group-held-out 3-fold predictions',
        'trained_recipes': list(RECIPES), 'primary_candidates': [[n, [list(item) for item in m]] for n, m in CANDIDATES],
        'feature_view': 'compact-fps-context-v4; first32 local signals + masked permutation-invariant 3x3 tile pools; absolute/pair-supported delta/absdelta/masked robust-z/0.2s+0.6s masked means and contrasts/0.3s masked past-future contrast; validity never differentiated; no filename/hash/generator/duration-ratio input',
        'RGB_PCA': '16 training-content-only weighted covariance components; held-out frames and labels never fit PCA',
        'frame_sample_weight': 'inverse content multiplicity/frame count; normal videos x2; then class positive/negative mass balance and mean weight one',
        'ET': {'trees': 192, 'depth': 8, 'leaf': 16, 'max_features': .5},
        'HGB': {'iterations': 180, 'rate': .06, 'leaves': 7, 'depth': 3, 'leaf': 24, 'L2': 6, 'bins': 64, 'max_features': .8, 'early_stopping': False},
        'video_head': 'ExtraTrees128/depth4/leaf2/max_features.75/class_weight balanced; inverse duplicate mass and normal x2; optional single hard gate, no multiplication',
        'decoder_configs': list(configurations()),
        'inner_ranking': 'content-normalized row statistics; utility=.35 F1@.3+.65 F1@.5+.10 frameF1-.20 normalFPR-.15 positiveEmpty; top8 pooled shortlist; rank .75 pooled + .25 content-stratified bootstrap q20 (32 replicates); same rule across candidates; deterministic tie on F1@.5, fewer empty/FP/segments, order',
        'baseline': 'fixed SHA-grouped v1 nested predictions from audited v2 comparison, not old leaky folds',
        'outer_analysis': 'single primary nested pipeline; fixed candidates are diagnostics only and never substitute for primary',
        'promotion_guards': {'event_f1_03_no_worse': 0., 'event_f1_05_improved_02': .02,
                             'frame_f1_drop_at_most_005': -.005, 'normal_fp_no_worse': 0,
                             'positive_empty_improved_3': -3, 'portable_fresh_parity_max': 2e-6},
        'deployment_choice': 'only inner OOF averages (4 appearances per row); never outer-best; new candidate must pass unchanged guards + independent audit + fresh pixel extraction; failed candidate exported only explicitly diagnostic',
        'experiment_limit': 'one frozen round; no decoder/candidate change after outer scores are opened',
        'compatibility': 'keep public API, defaults, previous models/reports and original data unchanged unless all acceptance checks pass',
        'limitations': ['small inspected development set, repeated CV is not blind generalization evidence',
                       'normal examples only14 and duplicate content has annotation conflict',
                       'legacy cached features have original provenance limitations documented in v2',
                       'offline future context does not imply causal streaming suitability']}


def current_evidence():
    evidence = collect_new_fit_inputs()
    sources = {n: digest(ROOT / 'code' / n) for n in SOURCES}
    return {'input_hashes': dict(evidence.input_hashes), 'inputs_sha256': evidence.inputs_sha256,
            'input_source_hashes': dict(evidence.sources), 'sources': sources,
            'protocol_sha256': digest(OUT / 'protocol.json'), 'baseline_report_sha256': digest(BASELINE)}


def freeze():
    write_new(OUT / 'protocol.json', protocol())
    evidence = current_evidence()
    for name in SOURCES:
        target = OUT / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / 'code' / name).read_bytes())
    evidence['receipt_sha256'] = object_hash(evidence)
    write_new(OUT / 'training_receipt.json', evidence)
    print('FROZEN', evidence['receipt_sha256'], flush=True)


def check_receipt():
    receipt = json.loads((OUT / 'training_receipt.json').read_text('utf-8'))
    core = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
    if object_hash(core) != receipt['receipt_sha256'] or current_evidence() != core:
        raise ValueError('frozen v4 input/source/protocol receipt mismatch')
    if json.loads((OUT / 'protocol.json').read_text('utf-8')) != protocol():
        raise ValueError('protocol differs from frozen implementation')
    for name, expected in receipt['sources'].items():
        if digest(OUT / 'sources' / name) != expected:
            raise ValueError('frozen v4 source snapshot mismatch: ' + name)
    return receipt


def init_worker():
    global _STATE
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    receipt = check_receipt()
    _STATE = (*load_grouped(), receipt)


def fit_task(args):
    fold, recipe = args
    manifest, records, profiles, receipt = _STATE
    file = OUT / 'training' / f'{fold}_{recipe}.npz'
    meta = file.with_suffix('.json')
    if file.exists() or meta.exists():
        raise ValueError('existing training artifact; never overwrite/refit in a frozen namespace')
    start = time.perf_counter()
    val = manifest['outer_folds'][fold]
    train = [i for i in range(len(records)) if i not in set(val)]
    splits = grouped_folds(records, train, 3, 20261002 + fold * 31)
    arrays, partitions, fits = {}, [], []
    for k, inner_val in enumerate(splits):
        inner_train = [i for i in train if i not in set(inner_val)]
        seed = 20261002 + fold * 53 + k
        fp, vp, state = fit_compact_member(recipe, records, inner_train, inner_val, seed)
        for i in inner_val:
            arrays[f'inner_frame_{i}'] = fp[i]; arrays[f'inner_video_{i}'] = np.asarray(vp[i])
        partitions.append({'fit': inner_train, 'validation': inner_val, 'seed': seed})
        fits.append({'partition': k, 'fit_content_sha256': state['fit_content_sha256'],
                     'transform_sha256': object_hash(state['transform'])})
    seed = 20261002 + fold * 53 + 99
    fp, vp, state = fit_compact_member(recipe, records, train, val, seed)
    for i in val:
        arrays[f'outer_frame_{i}'] = fp[i]; arrays[f'outer_video_{i}'] = np.asarray(vp[i])
    fits.append({'partition': 'outer', 'fit_content_sha256': state['fit_content_sha256'],
                 'transform_sha256': object_hash(state['transform'])})
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    row = {'fold': fold, 'recipe': recipe, 'train': train, 'validation': val,
           'inner_partitions': partitions, 'outer_seed': seed, 'fit_evidence': fits,
           'training_receipt_sha256': receipt['receipt_sha256'], 'npz_sha256': digest(file),
           'seconds': time.perf_counter() - start, 'runtime': {'numpy': np.__version__}}
    import sklearn
    row['runtime']['sklearn'] = sklearn.__version__
    row['signature'] = object_hash(row)
    write_new(meta, row)
    return {'fold': fold, 'recipe': recipe, 'seconds': row['seconds'], 'npz_sha256': row['npz_sha256']}


def load_new_fold(fold, recipe, records, receipt):
    if recipe not in RECIPES:
        raise ValueError('unknown cached recipe')
    file = OUT / 'training' / f'{fold}_{recipe}.npz'
    row = json.loads(file.with_suffix('.json').read_text('utf-8'))
    core = {k: v for k, v in row.items() if k != 'signature'}
    if object_hash(core) != row['signature'] or row['training_receipt_sha256'] != receipt['receipt_sha256']:
        raise ValueError('fit metadata signature mismatch')
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != row['npz_sha256']:
        raise ValueError('fit NPZ mismatch')
    import io
    manifest = json.loads((NEW / 'dataset_manifest.json').read_text('utf-8'))
    val = manifest['outer_folds'][fold]; train = [i for i in range(len(records)) if i not in set(val)]
    if row['recipe'] != recipe or row['fold'] != fold or row['train'] != train or row['validation'] != val or row['outer_seed'] != 20261002 + fold * 53 + 99:
        raise ValueError('fit outer partition mismatch')
    splits = grouped_folds(records, train, 3, 20261002 + fold * 31)
    expected = [{'fit': [i for i in train if i not in set(v)], 'validation': v,
                 'seed': 20261002 + fold * 53 + k} for k, v in enumerate(splits)]
    if row['inner_partitions'] != expected:
        raise ValueError('fit inner partition mismatch')
    for k, part in enumerate(expected + [{'fit': train, 'validation': val}]):
        if {records[i]['sha256'] for i in part['fit']} & {records[i]['sha256'] for i in part['validation']}:
            raise ValueError('cache content leakage')
        fitids = sorted({records[i]['sha256'] for i in part['fit']})
        if row['fit_evidence'][k]['fit_content_sha256'] != fitids:
            raise ValueError('cache PCA/fit training content mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        expected_keys = {f'{kind}_{head}_{i}' for kind, ids in [('inner', train), ('outer', val)] for i in ids for head in ['frame', 'video']}
        if set(archive.files) != expected_keys:
            raise ValueError('cache probability coverage mismatch')
        a = {k: archive[k].copy() for k in archive.files}
    for kind, ids in [('inner', train), ('outer', val)]:
        for i in ids:
            for head, shape in [('frame', (records[i]['frames'],)), ('video', ())]:
                x = a[f'{kind}_{head}_{i}']
                if x.shape != shape or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
                    raise ValueError('invalid cached probabilities')
    def read(kind, ids):
        return Probabilities({i: a[f'{kind}_frame_{i}'] for i in ids},
                             {i: float(a[f'{kind}_video_{i}']) for i in ids}, None)
    return read('inner', train), read('outer', val), row


def blended(members, probabilities, indices):
    if not members or not np.isclose(sum(w for r, w in members), 1):
        raise ValueError('invalid fixed blend')
    return Probabilities({i: sum(w * probabilities[r].frame[i] for r, w in members).astype(np.float32) for i in indices},
                         {i: float(sum(w * probabilities[r].video[i] for r, w in members)) for i in indices}, None)


def decode_predictions(records, indices, p, cfg):
    decoder = decode_duration if cfg['kind'] == 'duration-logit-v4' else decode_v2
    return {i: decoder(p.frame[i], records[i]['fps'], p.video[i], cfg) for i in indices}


def utility_v4(stats, normals, positives):
    s = np.asarray(stats)
    def f(v):
        return 2 * v[..., 0] / np.maximum(1e-12, 2 * v[..., 0] + v[..., 1] + v[..., 2])
    nfpr = np.divide(s[..., 9], normals, out=np.zeros_like(s[..., 9], dtype=float), where=np.asarray(normals) > 0)
    empty = np.divide(s[..., 10], positives, out=np.zeros_like(s[..., 10], dtype=float), where=np.asarray(positives) > 0)
    return .35 * f(s[..., :3]) + .65 * f(s[..., 3:6]) + .10 * f(s[..., 6:9]) - .20 * nfpr - .15 * empty


def choose_v4(records, indices, p, seed):
    weights = np.asarray([content_weights(records, indices)[i] for i in indices])
    truths = [VideoTruth(records[i]['labels']) for i in indices]
    n = np.asarray([t.normal for t in truths], float); positive = 1 - n
    normal_mass = weights @ n; positive_mass = weights @ positive
    boot = grouped_bootstrap(records, indices, seed, 32) * weights[None, :]
    bn, bp = boot @ n, boot @ positive
    short, seen = [], set()
    for ordinal, cfg in enumerate(configurations()):
        pred = decode_predictions(records, indices, p, cfg)
        matrix = np.stack([video_statistics(pred[i], t) for i, t in zip(indices, truths)])
        fingerprint = json.dumps([pred[i] for i in indices])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        total = weights @ matrix
        pooled = float(utility_v4(total, normal_mass, positive_mass))
        f05 = float(2 * total[3] / max(1e-12, 2 * total[3] + total[4] + total[5]))
        row = {'config': cfg, 'stats': total.tolist(), 'pooled_utility': pooled,
               'key': [pooled, f05, -total[10], -total[9], -total[11], -ordinal], 'matrix': matrix}
        short.append(row); short.sort(key=lambda z: z['key'], reverse=True); del short[8:]
    for row in short:
        b = utility_v4(boot @ row['matrix'], bn, bp)
        row['bootstrap_q20'] = float(np.quantile(b, .2))
        row['stable_utility'] = .75 * row['pooled_utility'] + .25 * row['bootstrap_q20']
        del row['matrix']
    chosen = deepcopy(max(short, key=lambda z: (z['stable_utility'], *z['key'][1:])))
    chosen['shortlist'] = short
    return chosen


def train_all(workers):
    started = time.perf_counter(); check_receipt()
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        futures = [pool.submit(fit_task, (f, r)) for f in range(5) for r in RECIPES]
        for future in as_completed(futures):
            row = future.result(); results.append(row); print(json.dumps(row), flush=True)
    check_receipt()
    write_new(OUT / 'training_report.json', {'status': 'complete', 'results': results,
              'before_after_evidence_equal': True, 'elapsed_seconds': time.perf_counter() - started})


def select_all():
    start = time.perf_counter(); receipt = check_receipt(); manifest, records, profiles = load_grouped()
    selected, fast, folds = {}, {}, []
    fixed = {n: {} for n, members in CANDIDATES}
    for fold, val in enumerate(manifest['outer_folds']):
        train = [i for i in range(len(records)) if i not in set(val)]
        inner, outer = {}, {}; meta = None
        for recipe in RECIPES:
            inner[recipe], outer[recipe], meta = load_new_fold(fold, recipe, records, receipt)
        inner['corrected_motion_et'], outer['corrected_motion_et'], _ = load_fold(fold, 'corrected_motion_et', records)
        rows = []
        for name, members in CANDIDATES:
            p = blended(members, inner, train)
            decision = choose_v4(records, train, p, 20261002 + fold)
            decision['candidate'] = name; rows.append(decision)
        decision = max(rows, key=lambda z: (z['stable_utility'], *z['key'][1:], -[n for n, _ in CANDIDATES].index(z['candidate'])))
        members = dict(CANDIDATES)[decision['candidate']]
        # Choices are now locked. Only below this line do outer scores get decoded/evaluated.
        pred = decode_predictions(records, val, blended(members, outer, val), decision['config'])
        selected.update(pred)
        for row in rows:
            fp = decode_predictions(records, val, blended(dict(CANDIDATES)[row['candidate']], outer, val), row['config'])
            fixed[row['candidate']].update(fp)
        fd = choose_v4(records, train, inner['compact_motion_hgb'], 20261002 + fold)
        fastpred = decode_predictions(records, val, outer['compact_motion_hgb'], fd['config']); fast.update(fastpred)
        folds.append({'fold': fold, 'train_indices': train, 'validation_indices': val,
                      'inner_partitions': meta['inner_partitions'], 'primary': decision,
                      'all_inner_selections': rows, 'fast': fd,
                      'primary_predictions': {str(i): pred[i] for i in val}})
        print(json.dumps({'fold': fold, 'chosen': decision['candidate'], 'config': decision['config']}), flush=True)
    baseline = json.loads(BASELINE.read_text('utf-8'))
    bp = {row['index']: [tuple(v) for v in row['segments']] for row in baseline['grouped_v1_baseline']['predictions']}
    a = _group_metrics(records, selected); b = _group_metrics(records, bp)
    if b != baseline['grouped_v1_baseline']['metrics']:
        raise ValueError('baseline re-evaluation differs')
    x, y = a['all'], b['all']
    guards = {'event_f1_03_no_worse': x['iou_0.3']['f1'] >= y['iou_0.3']['f1'],
              'event_f1_05_improved_02': x['iou_0.5']['f1'] >= y['iou_0.5']['f1'] + .02,
              'frame_f1_drop_at_most_005': x['frame']['f1'] >= y['frame']['f1'] - .005,
              'normal_fp_no_worse': x['normal']['false_positive_videos'] <= y['normal']['false_positive_videos'],
              'positive_empty_improved_3': x['positive_videos_without_candidate'] <= y['positive_videos_without_candidate'] - 3}
    report = {'schema_version': 'compact-experiment-v4', 'status': 'complete',
              'role': 'iterative_development_validation_not_new_blind_test',
              'protocol_sha256': digest(OUT / 'protocol.json'), 'training_receipt_sha256': receipt['receipt_sha256'],
              'baseline_report_sha256': digest(BASELINE), 'folds': folds,
              'grouped_v1_baseline': {'metrics': b, 'predictions': predictions_json(records, bp)},
              'primary_v4_nested': {'metrics': a, 'predictions': predictions_json(records, selected),
                                    'summary': prediction_summary(records, selected)},
              'fast_v4_diagnostic': {'metrics': _group_metrics(records, fast), 'predictions': predictions_json(records, fast)},
              'fixed_candidate_diagnostics_not_for_promotion': {n: {'metrics': _group_metrics(records, p), 'predictions': predictions_json(records, p)} for n, p in fixed.items()},
              'statistical_promotion_checks': guards, 'statistical_promotion_passed': all(guards.values()),
              'paired_content_bootstrap': paired_uncertainty(records, bp, selected),
              'elapsed_seconds': time.perf_counter() - start}
    check_receipt(); write_new(OUT / 'report.json', report)
    print(json.dumps({'primary': x, 'guards': guards, 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['freeze', 'train', 'select'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'freeze': freeze()
    elif args.phase == 'train': train_all(args.workers)
    else: select_all()

if __name__ == '__main__':
    main()
