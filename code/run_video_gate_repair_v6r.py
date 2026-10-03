#!/usr/bin/env python3
"""Frozen precision-only replay of v6; no refits or outer-informed tuning."""
from __future__ import annotations
import argparse, io, json, time
from copy import deepcopy
import numpy as np
import run_video_gate_experiment_v6 as PARENT
from optimized_video_gate_v6r import apply_gate, repaired_state, NUMERIC_REVISION, GATE_KINDS
from optimized_grouped_training import ROOT, NEW, load_grouped, digest
from run_compact_experiment_v4 import write_new, object_hash, decode_predictions
from run_optimization_selection import Probabilities, predictions_json, prediction_summary
from run_algorithm_optimization import _group_metrics
from run_recall_selection import paired_uncertainty

OUT = ROOT / 'output/algorithm-opt-2026-10-02-v6r'
MEMBERS, CANDIDATES, BASELINE = PARENT.MEMBERS, PARENT.CANDIDATES, PARENT.BASELINE
choose = PARENT.choose
SOURCES = ('run_video_gate_repair_v6r.py', 'optimized_video_gate_v6r.py') + PARENT.SOURCES


def protocol():
    return {'schema_version': 'strict-video-gate-precision-replay-v6r',
            'date': '2026-10-02', 'role': 'iterative_development_validation_not_new_blind_test',
            'parent_protocol_sha256': digest(PARENT.OUT/'protocol.json'),
            'numerical_change': NUMERIC_REVISION,
            'changes': 'ONLY logistic inference standardization retains fitted float64 mean/scale then clips and casts to float32 as fit did. Fitted coefficients, folds, features, base scores, grids, selection utility, bootstrap seeds and promotion guards unchanged.',
            'training': 'Reuse all parent frozen fits; no refit of base or gate for nested evaluation.',
            'selection': 'Exactly parent inner-only choose; no outer-informed edits.',
            'decoder_configs': list(PARENT.configurations()),
            'promotion_guards': PARENT.protocol()['promotion_guards']}


def evidence():
    receipt = PARENT.check_receipt()
    files = sorted((PARENT.OUT/'training').glob('*')) + [PARENT.OUT/'report.json']
    return {'parent_receipt_sha256': receipt['receipt_sha256'],
            'parent_artifacts': {p.relative_to(ROOT).as_posix(): digest(p) for p in files if p.is_file()},
            'sources': {n: digest(ROOT/'code'/n) for n in SOURCES},
            'protocol_sha256': digest(OUT/'protocol.json'),
            'baseline_report_sha256': digest(BASELINE)}


def freeze():
    write_new(OUT/'protocol.json', protocol())
    row = evidence(); row['receipt_sha256'] = object_hash(row)
    write_new(OUT/'training_receipt.json', row)
    for n in SOURCES:
        target = OUT/'sources'/n; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f: f.write((ROOT/'code'/n).read_bytes())
    print('FROZEN', row['receipt_sha256'], flush=True)


def check_receipt():
    row = json.loads((OUT/'training_receipt.json').read_text('utf-8'))
    core = {k:v for k,v in row.items() if k!='receipt_sha256'}
    if object_hash(core)!=row['receipt_sha256'] or core!=evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):
        raise ValueError('v6r frozen receipt mismatch')
    for n,h in row['sources'].items():
        if digest(OUT/'sources'/n)!=h: raise ValueError('v6r source snapshot mismatch')
    return row


def replay():
    start = time.perf_counter(); receipt = check_receipt(); parent = PARENT.check_receipt()
    _, records, _ = load_grouped(); results = []
    for fold in range(5):
        inner, outer, oldrow, arrays = PARENT.load_fold(fold, records, parent)
        row = deepcopy(oldrow); changes = []
        pairs = [('inner', p['validation'], p['gate_states']) for p in row['inner_partitions']]
        pairs.append(('outer', row['validation'], row['outer_gate_states']))
        for side, ids, states in pairs:
            for c in GATE_KINDS:
                states[c] = repaired_state(states[c])
                for i in ids:
                    key = f'{side}_video_{c}_{i}'; old = float(arrays[key])
                    source = inner if side=='inner' else outer
                    new = apply_gate(states[c], source['ungated_control'].frame[i], source['ungated_control'].video[i], records[i])
                    arrays[key] = np.asarray(new)
                    changes.append({'side':side, 'index':i, 'candidate':c, 'old':old, 'repaired':new, 'absolute_difference':abs(new-old)})
        path = OUT/'training'/f'{fold}_gates.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f: np.savez_compressed(f, **arrays)
        row.pop('signature'); row.update(training_receipt_sha256=receipt['receipt_sha256'], npz_sha256=digest(path),
            parent_npz_sha256=oldrow['npz_sha256'], parent_metadata_sha256=digest(PARENT.OUT/'training'/f'{fold}_gates.json'),
            numeric_revision=NUMERIC_REVISION)
        row['signature'] = object_hash(row); write_new(path.with_suffix('.json'), row)
        result = {'fold':fold, 'max_probability_difference':max(x['absolute_difference'] for x in changes), 'changes':changes}
        results.append(result); print(json.dumps({k:v for k,v in result.items() if k!='changes'}), flush=True)
    check_receipt()
    write_new(OUT/'training_report.json', {'status':'complete', 'base_fits':0, 'gate_fits':0,
        'before_after_evidence_equal':True, 'precision_replay':results, 'elapsed_seconds':time.perf_counter()-start})


def load_fold(fold, records, receipt):
    path = OUT/'training'/f'{fold}_gates.npz'; row = json.loads(path.with_suffix('.json').read_text('utf-8'))
    if (row['fold']!=fold or row['training_receipt_sha256']!=receipt['receipt_sha256']
            or object_hash({k:v for k,v in row.items() if k!='signature'})!=row['signature'] or digest(path)!=row['npz_sha256']):
        raise ValueError('repaired gate artifact mismatch')
    pi, po, parent, original = PARENT.load_fold(fold, records, PARENT.check_receipt())
    if row['parent_npz_sha256']!=parent['npz_sha256'] or row['train']!=parent['train'] or row['validation']!=parent['validation']:
        raise ValueError('repaired gate partition/source mismatch')
    with np.load(path, allow_pickle=False) as z: a = {k:z[k].copy() for k in z.files}
    if set(a)!=set(original): raise ValueError('repaired array coverage mismatch')
    for key,x in a.items():
        if x.shape!=original[key].shape or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
            raise ValueError('repaired probability shape/range mismatch')
        if '_video_ridge_logistic_v6_' not in key and not np.array_equal(x, original[key]):
            raise ValueError('precision replay changed nonlogistic evidence')
    for side,ids,states,source in [('inner',p['validation'],p['gate_states'],pi) for p in row['inner_partitions']] + [('outer',row['validation'],row['outer_gate_states'],po)]:
        for c in GATE_KINDS:
            for i in ids:
                expected = apply_gate(states[c],source['ungated_control'].frame[i],source['ungated_control'].video[i],records[i])
                if expected!=float(a[f'{side}_video_{c}_{i}']): raise ValueError('repair inference not reproducible')
    def prob(side, ids, c):
        return Probabilities({i:a[f'{side}_frame_{i}'] for i in ids}, {i:float(a[f'{side}_video_{c}_{i}']) for i in ids}, None)
    return {c:prob('inner',row['train'],c) for c in CANDIDATES}, {c:prob('outer',row['validation'],c) for c in CANDIDATES}, row, a


def select():
    start = time.perf_counter(); receipt = check_receipt(); _, records, _ = load_grouped()
    selected = {}; folds = []; fixed = {c:{} for c in CANDIDATES}
    for fold in range(5):
        inner, outer, meta, _ = load_fold(fold, records, receipt)
        rows = []
        for c in CANDIDATES:
            d = choose(records,meta['train'],inner[c],20261002+fold); d['candidate'] = c; rows.append(d)
        chosen = max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:],-CANDIDATES.index(z['candidate'])))
        pred = decode_predictions(records,meta['validation'],outer[chosen['candidate']],chosen['config']); selected.update(pred)
        for d in rows: fixed[d['candidate']].update(decode_predictions(records,meta['validation'],outer[d['candidate']],d['config']))
        folds.append({'fold':fold,'train_indices':meta['train'],'validation_indices':meta['validation'],
                      'primary':chosen,'all_inner_selections':rows,'primary_predictions':{str(i):pred[i] for i in meta['validation']}})
    baseline = json.loads(BASELINE.read_text('utf-8')); bp = {x['index']:[tuple(v) for v in x['segments']] for x in baseline['grouped_v1_baseline']['predictions']}
    bm = _group_metrics(records,bp); m = _group_metrics(records,selected); x,y = m['all'],bm['all']
    if bm!=baseline['grouped_v1_baseline']['metrics']: raise ValueError('baseline reconstruction mismatch')
    guards = {'event_f1_03_no_worse':x['iou_0.3']['f1']>=y['iou_0.3']['f1'],
              'event_f1_05_improved_02':x['iou_0.5']['f1']>=y['iou_0.5']['f1']+.02,
              'frame_f1_drop_at_most_005':x['frame']['f1']>=y['frame']['f1']-.005,
              'normal_fp_no_worse':x['normal']['false_positive_videos']<=y['normal']['false_positive_videos'],
              'positive_empty_improved_3':x['positive_videos_without_candidate']<=y['positive_videos_without_candidate']-3}
    parent_report = json.loads((PARENT.OUT/'report.json').read_text('utf-8'))
    parent_pred = {p['index']:p['segments'] for p in parent_report['primary_v6_nested']['predictions']}
    changed = [i for i in range(len(records)) if [list(v) for v in selected[i]]!=parent_pred[i]]
    report = {'schema_version':'strict-video-gate-precision-replay-v6r','status':'complete',
              'role':'iterative_development_validation_not_new_blind_test','protocol_sha256':digest(OUT/'protocol.json'),
              'training_receipt_sha256':receipt['receipt_sha256'],'folds':folds,
              'grouped_v1_baseline':{'metrics':bm,'predictions':predictions_json(records,bp)},
              'primary_v6r_nested':{'metrics':m,'predictions':predictions_json(records,selected),'summary':prediction_summary(records,selected)},
              'fixed_candidate_diagnostics_not_for_promotion':{c:{'metrics':_group_metrics(records,p),'predictions':predictions_json(records,p)} for c,p in fixed.items()},
              'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),
              'paired_content_bootstrap':paired_uncertainty(records,bp,selected),
              'parent_primary_changed_indices':changed,'elapsed_seconds':time.perf_counter()-start}
    check_receipt(); write_new(OUT/'report.json',report); print(json.dumps({'primary':x,'guards':guards,'changed_indices':changed}),flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['freeze','replay','select'], required=True); args = p.parse_args()
    {'freeze':freeze,'replay':replay,'select':select}[args.phase]()

if __name__=='__main__': main()
