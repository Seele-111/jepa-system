#!/usr/bin/env python3
"""Prospective v7: fixed two-view blend with explicit inner error budgets."""
from __future__ import annotations
import argparse, json, time
from copy import deepcopy
import numpy as np
import run_compact_experiment_v4 as V4
import run_event_experiment_v5 as V5
from optimized_grouped_training import ROOT, load_grouped, digest
from optimized_compact_features_v4 import content_weights
from optimized_video_statistics import VideoTruth, video_statistics
from run_compact_experiment_v4 import write_new, object_hash, decode_predictions
from run_video_gate_experiment_v6 import utility, BASELINE
from run_recall_selection import grouped_bootstrap, paired_uncertainty
from run_algorithm_optimization import _group_metrics
from run_optimization_selection import Probabilities, predictions_json, prediction_summary

OUT=ROOT/'output/algorithm-opt-2026-10-02-v7'
COMPACT='compact_rgb_corrected_motion_hgb'
EVENT='event_rgb_corrected_motion_hgb'
CANDIDATES=('compact_control','event_control','equal_multiview')
NORMAL_BUDGET=.30
EMPTY_BUDGET=.20
SOURCES=('run_multiview_experiment_v7.py',)+tuple(dict.fromkeys(V4.SOURCES+V5.SOURCES+('run_video_gate_experiment_v6.py','optimized_video_gate_v6.py')))


def configurations():
    # Exactly the existing v4/v5 40 candidates; do not expand an outer-informed grid.
    return list(V5.configurations())


def error_budget(stats, normal_mass, positive_mass):
    stats=np.asarray(stats,np.float64)
    nfpr=float(stats[9]/normal_mass) if normal_mass else 0.
    empty=float(stats[10]/positive_mass) if positive_mass else 0.
    violation=max(0.,nfpr-NORMAL_BUDGET)/NORMAL_BUDGET + max(0.,empty-EMPTY_BUDGET)/EMPTY_BUDGET
    return {'normal_false_positive_rate':nfpr,'positive_empty_rate':empty,
            'feasible':bool(violation<=1e-12),'normalized_violation':float(violation)}


def choose(records, indices, p, seed):
    weights=np.asarray([content_weights(records,indices)[i] for i in indices])
    truths=[VideoTruth(records[i]['labels']) for i in indices]
    n=np.asarray([t.normal for t in truths],float);pos=1-n;nm,pm=weights@n,weights@pos
    boot=grouped_bootstrap(records,indices,seed,32)*weights[None,:];bn,bp=boot@n,boot@pos
    rows=[];seen=set()
    for ordinal,cfg in enumerate(configurations()):
        pred=decode_predictions(records,indices,p,cfg);fingerprint=json.dumps([pred[i] for i in indices])
        if fingerprint in seen:continue
        seen.add(fingerprint)
        mat=np.stack([video_statistics(pred[i],t) for i,t in zip(indices,truths)]);total=weights@mat
        budget=error_budget(total,nm,pm);u=float(utility(total,nm,pm));f05=float(2*total[3]/max(1e-12,2*total[3]+total[4]+total[5]))
        key=[int(budget['feasible']),-budget['normalized_violation'],u,f05,-total[10],-total[9],-total[11],-ordinal]
        row={'config':cfg,'stats':total.tolist(),'pooled_utility':u,'error_budget':budget,'key':key,'matrix':mat}
        rows.append(row);rows.sort(key=lambda x:x['key'],reverse=True);del rows[8:]
    if not rows:raise ValueError('no decoder candidates')
    for row in rows:
        row['bootstrap_q20']=float(np.quantile(utility(boot@row['matrix'],bn,bp),.2))
        row['stable_utility']=.75*row['pooled_utility']+.25*row['bootstrap_q20'];del row['matrix']
    best=deepcopy(max(rows,key=selection_key));best['shortlist']=rows
    return best


def selection_key(row):
    return (int(row['error_budget']['feasible']),-row['error_budget']['normalized_violation'],row['stable_utility'],*row['key'][3:])


def fuse(compact,event,indices):
    if any(set(p.frame)!=set(p.video) or not set(indices)<=set(p.frame) for p in [compact,event]):raise ValueError('unaligned fusion heads')
    frame={};video={}
    for i in indices:
        a,b=np.asarray(compact.frame[i],np.float32),np.asarray(event.frame[i],np.float32)
        va,vb=compact.video[i],event.video[i]
        if a.shape!=b.shape or a.ndim!=1 or not len(a) or not all(np.isfinite(x).all() and np.all((x>=0)&(x<=1)) for x in [a,b,np.asarray([va,vb])]):raise ValueError('invalid fusion evidence')
        frame[i]=(.5*a+.5*b).astype(np.float32);video[i]=float(.5*va+.5*vb)
    return Probabilities(frame,video,None)


def protocol():
    return {'schema_version':'multiview-error-budget-v7','date':'2026-10-02',
            'role':'iterative_development_validation_not_new_blind_test',
            'rationale':'v4 compact evidence rejects normal but misses positives; v5 event evidence recovers positives but false alarms. Test fixed arithmetic fusion of complementary representations; improve inner decision policy with explicit simultaneous error budgets, not an expanded threshold search.',
            'selection_of_members':'Both HGB members were independently chosen by previously frozen deployment inner aggregate, not fixed outer-best. Architecture is still adaptive on this inspected development data, not an unbiased new benchmark.',
            'data':'same77 annotation rows/76 SHA; unchanged outer5/inner3 seeds, duplicate annotations preserved',
            'compact_member':COMPACT,'event_member':EVENT,'blend_weights':[.5,.5],
            'primary_candidates':list(CANDIDATES),'decoder_configs':configurations(),
            'inner_budget':{'normal_false_positive_rate':NORMAL_BUDGET,'positive_empty_rate':EMPTY_BUDGET,
                'scope':'content-weighted pooled inner OOF; these are selection targets, not changed outer promotion guards',
                'fallback':'If none feasible, minimize sum of normalized excess; never suppress output or claim feasibility'},
            'inner_selection':'Deduplicate predictions; top8 ordered by feasibility,negative budget violation,pooled utility,F1.5,fewer empty/normal/segments,configuration ordinal. Rank shortlist by feasibility,negative violation,.75pooled+.25q20 of32 same seeded content-stratified bootstrap,then same remaining tie-breaks. Candidate index breaks final ties.',
            'utility':'.35F1.3+.65F1.5+.10FrameF1-.30normalFPR-.30positiveEmpty',
            'probability_rules':'No trained stack/gate; fixed linear score/video mean; never filename/hash/GT/duration feature; no new fit, calibration, temperature or normalization.',
            'cache_requirement':'For each outer, both members must have byte-identical inner/outer index partitions and fit-content proofs; outer score may decode only after candidate/config lock.',
            'promotion_guards':{'event_f1_03_no_worse':0.,'event_f1_05_improved_02':.02,
                'frame_f1_drop_at_most_005':-.005,'normal_fp_no_worse':0,'positive_empty_improved_3':-3,'portable_fresh_parity_max':2e-6},
            'deployment_selection':'Only row-wise four-inner-OOF averages; never fixed outer-best; no default promotion by this script'}


def evidence():
    receipts={'v4':V4.check_receipt(),'v5':V5.check_receipt()}
    files=[]
    for mod,recipe in [(V4,COMPACT),(V5,EVENT)]:
        for fold in range(5):files += [mod.OUT/'training'/f'{fold}_{recipe}{ext}' for ext in ['.npz','.json']]
        files.append(mod.OUT/'independent_selection_audit.json')
    for mod in [V4,V5]:
        if json.loads((mod.OUT/'independent_selection_audit.json').read_text('utf-8'))['status']!='passed':raise ValueError('parent independent audit failed')
    return {'parent_receipt_sha256':{k:r['receipt_sha256'] for k,r in receipts.items()},
            'inputs':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
            'sources':{n:digest(ROOT/'code'/n) for n in SOURCES},'protocol_sha256':digest(OUT/'protocol.json'),
            'baseline_report_sha256':digest(BASELINE)}


def freeze():
    write_new(OUT/'protocol.json',protocol());row=evidence();row['receipt_sha256']=object_hash(row);write_new(OUT/'training_receipt.json',row)
    for n in SOURCES:
        target=OUT/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    print('FROZEN',row['receipt_sha256'],flush=True)


def check_receipt():
    row=json.loads((OUT/'training_receipt.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='receipt_sha256'}
    if object_hash(core)!=row['receipt_sha256'] or core!=evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):raise ValueError('v7 frozen receipt mismatch')
    if any(digest(OUT/'sources'/n)!=h for n,h in row['sources'].items()):raise ValueError('v7 snapshot mismatch')
    return row


def load_fold(fold,records):
    ci,co,cm=V4.load_new_fold(fold,COMPACT,records,V4.check_receipt())
    ei,eo,em=V5.load_new_fold(fold,EVENT,records,V5.check_receipt())
    for key in ['train','validation','inner_partitions']:
        if cm[key]!=em[key]:raise ValueError('fusion base partition mismatch')
    for a,b in zip(cm['fit_evidence'],em['fit_evidence']):
        if a['fit_content_sha256']!=b['fit_content_sha256']:raise ValueError('fusion fit content mismatch')
    return {'compact_control':ci,'event_control':ei,'equal_multiview':fuse(ci,ei,cm['train'])}, {'compact_control':co,'event_control':eo,'equal_multiview':fuse(co,eo,cm['validation'])}, cm


def select():
    start=time.perf_counter();receipt=check_receipt();_,records,_=load_grouped();selected={};folds=[];fixed={c:{} for c in CANDIDATES}
    for fold in range(5):
        inner,outer,meta=load_fold(fold,records);rows=[]
        for c in CANDIDATES:
            row=choose(records,meta['train'],inner[c],20261002+fold);row['candidate']=c;rows.append(row)
        chosen=max(rows,key=lambda r:(*selection_key(r),-CANDIDATES.index(r['candidate'])))
        pred=decode_predictions(records,meta['validation'],outer[chosen['candidate']],chosen['config']);selected.update(pred)
        for row in rows:fixed[row['candidate']].update(decode_predictions(records,meta['validation'],outer[row['candidate']],row['config']))
        folds.append({'fold':fold,'train_indices':meta['train'],'validation_indices':meta['validation'],'primary':chosen,'all_inner_selections':rows,'primary_predictions':{str(i):pred[i] for i in meta['validation']}})
        print(json.dumps({'fold':fold,'candidate':chosen['candidate'],'budget':chosen['error_budget'],'config':chosen['config']}),flush=True)
    baseline=json.loads(BASELINE.read_text('utf-8'));bp={p['index']:[tuple(x) for x in p['segments']] for p in baseline['grouped_v1_baseline']['predictions']}
    bm=_group_metrics(records,bp);m=_group_metrics(records,selected);x,y=m['all'],bm['all']
    if bm!=baseline['grouped_v1_baseline']['metrics']:raise ValueError('baseline reconstruction mismatch')
    guards={'event_f1_03_no_worse':x['iou_0.3']['f1']>=y['iou_0.3']['f1'],'event_f1_05_improved_02':x['iou_0.5']['f1']>=y['iou_0.5']['f1']+.02,
            'frame_f1_drop_at_most_005':x['frame']['f1']>=y['frame']['f1']-.005,'normal_fp_no_worse':x['normal']['false_positive_videos']<=y['normal']['false_positive_videos'],
            'positive_empty_improved_3':x['positive_videos_without_candidate']<=y['positive_videos_without_candidate']-3}
    report={'schema_version':'multiview-error-budget-v7','status':'complete','role':'iterative_development_validation_not_new_blind_test',
            'protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],'folds':folds,
            'grouped_v1_baseline':{'metrics':bm,'predictions':predictions_json(records,bp)},
            'primary_v7_nested':{'metrics':m,'predictions':predictions_json(records,selected),'summary':prediction_summary(records,selected)},
            'fixed_candidate_diagnostics_not_for_promotion':{c:{'metrics':_group_metrics(records,p),'predictions':predictions_json(records,p)} for c,p in fixed.items()},
            'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),
            'paired_content_bootstrap':paired_uncertainty(records,bp,selected),'base_fits':0,'elapsed_seconds':time.perf_counter()-start}
    check_receipt();write_new(OUT/'report.json',report);print(json.dumps({'primary':x,'guards':guards}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','select'],required=True);args=p.parse_args()
    {'freeze':freeze,'select':select}[args.phase]()

if __name__=='__main__':main()
