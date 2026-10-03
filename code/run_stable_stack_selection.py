#!/usr/bin/env python3
"""Finite v3: fixed complementary branches, monotone shrinkage and small risk grid.

No outer-label or outer-family selection. This is iterative development CV, not
an unseen confirmation set. Protocol/sources are frozen before evaluating outer
predictions. v2 failure is retained; base probabilities are explicitly reused.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import itertools
import json
from datetime import datetime,timezone
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from optimized_grouped_training import ROOT,OLD,NEW,load_grouped,digest
from optimized_stack_v3 import BRANCHES,FAST_BRANCHES,fit_stack,apply_stack
from optimized_video_statistics import VideoTruth,video_statistics,utility
from run_optimization_selection import Probabilities,predictions_json,prediction_summary
from run_algorithm_optimization import _group_metrics
from run_recall_selection import load_fold,grouped_bootstrap,evaluate,paired_uncertainty

OUT=ROOT/'output/algorithm-opt-2026-10-02-v3'
SOURCE_NAMES=['run_stable_stack_selection.py','optimized_stack_v3.py','run_recall_selection.py','optimized_recall_decoder.py','optimized_video_statistics.py','optimized_calibration_state.py','optimized_training_provenance.py','optimized_grouped_training.py','optimized_locator.py','run_optimization_selection.py','run_algorithm_optimization.py','build_optimization_dataset.py','optimized_feature_view.py','optimized_feature_view_v2.py','optimized_probability_calibration.py','optimized_joint_calibration.py','optimized_boundary_head.py','optimized_temporal_head.py']
_STATE=None


def recipes_for(branches):return tuple(sorted({r for b in branches for r,w in b}))


def member_inputs(pack,indices):
    return [{r:p.frame[i] for r,p in pack.items()} for i in indices],[{r:p.video[i] for r,p in pack.items()} for i in indices]


def fit_meta(records,indices,pack,branches):
    f,v=member_inputs(pack,indices);groups=Counter(records[i]['sha256'] for i in indices)
    return fit_stack(f,v,[records[i]['labels'] for i in indices],branches,
                     content_mass=[1/groups[records[i]['sha256']] for i in indices])


def apply_meta(indices,pack,state):
    f,v=member_inputs(pack,indices);fp={};vp={}
    for i,ff,vv in zip(indices,f,v):fp[i],vp[i]=apply_stack(ff,vv,state)
    return Probabilities(fp,vp,None)


def cross_meta(records,train,pack,partitions,branches):
    fp={};vp={};fitted=[]
    for partition in partitions:
        val=partition['validation'];fit=[i for i in train if i not in set(val)]
        if set(fit)!=set(partition['fit']):raise ValueError('unexpected meta partition')
        if {records[i]['sha256'] for i in fit}&{records[i]['sha256'] for i in val}:raise ValueError('content leakage in meta split')
        state=fit_meta(records,fit,pack,branches);q=apply_meta(val,pack,state)
        fp.update(q.frame);vp.update(q.video);fitted.append({'fit':fit,'validation':val,'state':state})
    if set(fp)!=set(train):raise ValueError('meta validation coverage mismatch')
    return Probabilities(fp,vp,None),fit_meta(records,train,pack,branches),fitted


def config_grid():
    for threshold,ratio,minimum in itertools.product([.25,.35,.45,.55,.65],[.65,.8,1.],[.12,.25]):
        yield {'kind':'recall-stable-v2','threshold':threshold,'low_ratio':ratio,
               'smooth_seconds':.15,'gap_seconds':.12,'min_seconds':minimum,
               'seed_seconds':.1,'video_strength':0.,'video_floor':0.,
               'video_threshold':0.,'boundary_seconds':0.}


def select_risk_config(records,indices,p,seed):
    truths=[VideoTruth(records[i]['labels']) for i in indices]
    group_sizes=Counter(records[i]['sha256'] for i in indices)
    content_weight=np.array([1/group_sizes[records[i]['sha256']] for i in indices])
    normal_mass=content_weight*np.array([t.normal for t in truths])
    positive_mass=content_weight-normal_mass
    normal=float(normal_mass.sum());positive=float(positive_mass.sum())
    if not normal or not positive:raise ValueError('risk selection needs both strata')
    boot=grouped_bootstrap(records,indices,seed);bn=boot@normal_mass;bp=boot@positive_mass
    rows=[]
    for ordinal,cfg in enumerate(config_grid()):
        pred=evaluate(records,indices,p,cfg);raw_matrix=np.stack([video_statistics(pred[i],t) for i,t in zip(indices,truths)])
        matrix=raw_matrix*content_weight[:,None];s=matrix.sum(0)
        nfpr=float(s[9]/normal);empty=float(s[10]/positive);pooled=float(utility(s,normal,positive))
        # Frozen soft-risk ceilings are INNER constraints, not confidence bounds
        # and not assurances for previously unseen normal videos.
        violation=max(0.,nfpr/.20-1.,empty/.25-1.);feasible=violation<1e-12
        totals=boot@matrix;us=np.array([utility(q,n,pn) for q,n,pn in zip(totals,bn,bp)])
        q20=float(np.percentile(us,20));stable=.75*pooled+.25*q20
        f05=float(2*s[3]/max(1,2*s[3]+s[4]+s[5]))
        key=(int(feasible),-violation,stable,f05,-float(s[9]),-float(s[10]),-float(s[11]),-ordinal)
        rows.append({'config':cfg,'key':key,'stats':raw_matrix.sum(0).tolist(),'content_stats':s.tolist(),'inner_risk_unit':'unique_content_mean_annotation','inner_normal_fpr':nfpr,'inner_positive_empty_rate':empty,
                     'risk_ceiling_feasible':feasible,'risk_ceiling_relative_violation':violation,
                     'pooled_utility':pooled,'bootstrap_q20':q20,'stable_utility':stable})
    chosen=max(rows,key=lambda r:r['key'])
    return chosen,rows


def init_worker():
    global _STATE
    _STATE=load_grouped()


def fold_task(fold):
    manifest,records,_=_STATE;val=manifest['outer_folds'][fold];train=[i for i in range(len(records)) if i not in set(val)]
    all_recipes=set(recipes_for(BRANCHES))|set(recipes_for(FAST_BRANCHES));inner={};outer={};partitions=None
    for recipe in sorted(all_recipes):
        inp,out,row=load_fold(fold,recipe,records);inner[recipe]=inp;outer[recipe]=out
        if row['train']!=train or row['validation']!=val:raise ValueError('unexpected outer split')
        if partitions is None:partitions=row['inner_partitions']
        elif partitions!=row['inner_partitions']:raise ValueError('unaligned member folds')
    result={'fold':fold,'train_indices':train,'validation_indices':val,'inner_partitions':partitions}
    started=time.perf_counter()
    for name,branches in [('primary',BRANCHES),('fast',FAST_BRANCHES)]:
        needed=recipes_for(branches);pack={r:inner[r] for r in needed}
        q,state,cross=cross_meta(records,train,pack,partitions,branches)
        decision,decisions=select_risk_config(records,train,q,20261002+fold)
        # These decisions and the full meta fit are FINAL before any outer label
        # or outer prediction metric is read. No candidate-family winner exists.
        result[name]={'decision':decision,'calibration':state,'cross_meta_fits':cross,'all_decoder_decisions':decisions}
    # BOTH primary and fast decisions are fixed before reporting outer metrics.
    for name,branches in [('primary',BRANCHES),('fast',FAST_BRANCHES)]:
        needed=recipes_for(branches);state=result[name]['calibration'];decision=result[name]['decision']
        p=apply_meta(val,{r:outer[r] for r in needed},state);pred=evaluate(records,val,p,decision['config'])
        result[name].update(predictions={str(i):pred[i] for i in val},
                            validation=_group_metrics([records[i] for i in val],{j:pred[i] for j,i in enumerate(val)}))
    result['seconds']=time.perf_counter()-started;p=OUT/'folds'/f'fold{fold}.json';p.parent.mkdir(exist_ok=True)
    with p.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return {'fold':fold,'seconds':result['seconds'],'primary_inner_feasible':result['primary']['decision']['risk_ceiling_feasible']}


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('v3 already frozen; no retroactive protocol changes')
    sources={};(OUT/'sources').mkdir(exist_ok=True)
    for name in SOURCE_NAMES:
        p=ROOT/'code'/name;sources[name]=digest(p);(OUT/'sources'/name).write_bytes(p.read_bytes())
    cache_inventory={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sorted((NEW/'grouped_training/folds').iterdir()) if p.suffix in ('.json','.npz')}
    if len(cache_inventory)!=130:raise ValueError('expected 65 cache NPZ/metadata pairs')
    protocol={'schema_version':'frozen-stable-stack-v3','created_at':datetime.now(timezone.utc).isoformat(),'role':'iterative_development_not_new_blind_test',
      'predecessor':'v2 failed normal-FP and positive-empty deployment guards; development continues without hiding failure',
      'dataset_manifest_sha256':digest(NEW/'dataset_manifest.json'),'sources':sources,
      'base_cache':'v2 65 audited SHA-grouped CPU fits; no reuse of pre-grouping probabilities',
      'cache_inventory':cache_inventory,'provenance_note':'Audit-time binding and verified fixed training snapshots; original legacy signature omitted feature hashes, original RGB NPZ hashes were absent. This is NOT retroactive training-time provenance.',
      'primary_branches':BRANCHES,'fast_branches':FAST_BRANCHES,'family_selection':'none, fixed complementary branches',
      'meta_head':'5 params: three nonnegative frame-logit weights, nonnegative mean-video-logit slope, intercept; ridge=1 anchored at equal branches and no video veto',
      'fitting_weight':'per-content mean frame BCE; aliases share one unit mass; no positive rebalancing',
      'decoder_configs':list(config_grid()),'inner_risk_unit':'unique content; aliases share one unit mass in risk, objective and bootstrap; outer report retains 77 rows for paired baseline compatibility','inner_risk_ceilings':{'normal_fpr':.20,'positive_empty_rate':.25},
      'selection':'feasible first; if none minimize maximum relative risk-ceiling violation; then .75 pooled utility+.25 content-stratified-bootstrap q20',
      'pooled_utility':'.35*F1.3+.65*F1.5-.20*normalFPR-.15*positive_empty_rate','bootstrap_replicates':32,
      'holdout':'same SHA grouped outer5/inner3 as v2; decoder on leave-inner-partition-out meta predictions, refit meta on full inner OOF for outer',
      'baseline':'exact original13 decoder/family tie behavior on same corrected folds, selection_audited/report.json',
      'promotion_minima':{'event_f1_03_no_worse':True,'event_f1_05_improvement':.02,'frame_f1_drop_at_most':.005,'normal_fp_no_worse':True,'positive_empty_reduce_at_least':3},
      'deployment_selection':'same fixed branch profile; decoder and meta from averaged inner OOF only, never outer OOF',
      'limitations':['already inspected development data','meta cross-fitting retains base OOF cross-training dependence','14 normal samples and conflicting alias annotation are not enough for reliable industrial normal specificity','no independent evidence of probability calibration'],
      'stopping_rule':'one finite v3 primary; no outer-best alternative substitution; if it fails keep v1 default and document no promotion',
      'safeguards':['no per-video rules','no file name/hash/generator features','no downloads/dependencies','no original-data/model/report overwrite']}
    with (OUT/'protocol.json').open('x',encoding='utf-8') as stream:json.dump(protocol,stream,indent=2,ensure_ascii=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.freeze:freeze();return
    protocol=json.loads((OUT/'protocol.json').read_text('utf-8'))
    if protocol['dataset_manifest_sha256']!=digest(NEW/'dataset_manifest.json'):raise ValueError('changed manifest after freeze')
    for name,sha in protocol['sources'].items():
        if digest(ROOT/'code'/name)!=sha:raise ValueError('changed source after freeze: '+name)
    for relative,sha in protocol['cache_inventory'].items():
        if digest(ROOT/relative)!=sha:raise ValueError('changed probability artifact after freeze: '+relative)
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as executor:
        futures=[executor.submit(fold_task,f) for f in range(5)]
        for future in as_completed(futures):results.append(future.result());print(json.dumps(results[-1]),flush=True)
    _,records,_=load_grouped();folds=[json.loads((OUT/'folds'/f'fold{f}.json').read_text('utf-8')) for f in range(5)]
    pred={name:{} for name in ['primary','fast']}
    for fold in folds:
        for name in pred:pred[name].update({int(i):[tuple(x) for x in p] for i,p in fold[name]['predictions'].items()})
    baseline_path=NEW/'selection_audited/report.json';baseline=json.loads(baseline_path.read_text('utf-8'))['grouped_v1_baseline']
    old={int(row['index']):[tuple(v) for v in row['segments']] for row in baseline['predictions']}
    a=baseline['metrics']['all'];b=_group_metrics(records,pred['primary'])['all']
    checks={'event_f1_03_no_worse':b['iou_0.3']['f1']>=a['iou_0.3']['f1'],'event_f1_05_improved_02':b['iou_0.5']['f1']>=a['iou_0.5']['f1']+.02,
      'frame_f1_drop_at_most_005':b['frame']['f1']>=a['frame']['f1']-.005,'normal_fp_no_worse':b['normal']['false_positive_videos']<=a['normal']['false_positive_videos'],
      'positive_empty_improved_3':b['positive_videos_without_candidate']<=a['positive_videos_without_candidate']-3}
    report={'schema_version':'stable-stack-grouped-nested-v3','status':'complete','role':'iterative_content_grouped_development_not_blind',
      'protocol_sha256':digest(OUT/'protocol.json'),'baseline_report_sha256':digest(baseline_path),'dataset_manifest_sha256':digest(NEW/'dataset_manifest.json'),
      'folds':folds,'grouped_v1_baseline':baseline,
      'primary_v3_nested':{'metrics':_group_metrics(records,pred['primary']),'predictions':predictions_json(records,pred['primary']),**prediction_summary(records,pred['primary'])},
      'fast_v3_diagnostic':{'metrics':_group_metrics(records,pred['fast']),'predictions':predictions_json(records,pred['fast'])},
      'paired_content_bootstrap':paired_uncertainty(records,old,pred['primary']),'statistical_promotion_checks':checks,
      'statistical_promotion_passed':all(checks.values()),'elapsed_seconds':time.perf_counter()-started}
    with (OUT/'report.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'phase':'complete','baseline':a,'v3':b,'checks':checks,'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
