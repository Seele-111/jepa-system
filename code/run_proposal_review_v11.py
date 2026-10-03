#!/usr/bin/env python3
"""Frozen ET-proposal/HGB-local-review experiment, reusing matched grouped fits."""
from __future__ import annotations
import argparse, io, json, time
import numpy as np
import run_event_compact_experiment_v8 as V8
import run_compact_experiment_v4 as V4
import run_feature_blocks_v10 as V10
from optimized_grouped_training import ROOT, load_grouped, digest
from optimized_compact_model_v4 import fit_compact_member, export_compact_member, predict_compact_member
from optimized_proposal_review_v11 import choose_review, decode_review, review_configurations
from run_compact_experiment_v4 import write_new, object_hash, decode_predictions
from run_optimization_selection import Probabilities, predictions_json, prediction_summary
from run_algorithm_optimization import _group_metrics
from run_recall_selection import paired_uncertainty

OUT=ROOT/'output/algorithm-opt-v11-proposal-review'
REVIEWER='compact_rgb_corrected_motion_hgb'
SOURCES=tuple(dict.fromkeys(('run_proposal_review_v11.py','optimized_proposal_review_v11.py')+V10.SOURCES))


def protocol():
    return {'schema_version':'proposal-review-v11','role':'repeated_development_not_blind_or_confirmatory',
        'hypothesis':'Locally verify high-recall ET proposals with the more precision-oriented compact HGB; do not average/multiply frame scores or globally veto a video.',
        'primary_source':'v8 event_compact_rgb_corrected_motion_et; identical to v10 rich block',
        'reviewer_source':REVIEWER,'partitions':'identical frozen outer5/inner3 and final direct inner3; no labels refit or changed',
        'selection':'StageA: unchanged V8.choose selects ET proposal decoder from40configs. StageB: select7 bounded local-review policies with same inner content utility/bootstrap/stable ranking. Identical stages at deployment; no new learned meta-head.',
        'review_policies':list(review_configurations()),'evidence':'HGB raw-frame q75 strictly within each ET candidate; optional fixed ET peak>=.8 rescue; accepted spans remain bitwise unchanged.',
        'new_outer_model_fits':0,'new_final_selection_model_fits':3,
        'promotion_guards':V10.protocol()['promotion_guards'],
        'limitations':V10.protocol()['limitations']+['Review can only remove proposals; no rescue of events absent from ET proposals.', 'Same inspected development evidence motivated this experiment.']}


def current_evidence():
    a=V8.check_receipt();b=V4.check_receipt();c=V10.check_receipt()
    files=[parent.OUT/'training'/f'{f}_{recipe}{ext}' for parent,recipe in [(V8,V10.PARENT_RICH),(V4,REVIEWER)] for f in range(5) for ext in ('.npz','.json')]
    files += [V10.OUT/'final_selection'/f'5_{V10.RICH}{ext}' for ext in ('.npz','.json')]
    return {'parent_receipts':{'v8':a['receipt_sha256'],'v4':b['receipt_sha256'],'v10':c['receipt_sha256']},
        'sources':{n:digest(ROOT/'code'/n) for n in SOURCES},
        'consumed_artifacts':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
        'protocol_sha256':digest(OUT/'protocol.json'),'baseline_sha256':digest(V8.BASELINE),
        'default_models':c['default_models']}


def freeze():
    write_new(OUT/'protocol.json',protocol());row=current_evidence();row['receipt_sha256']=object_hash(row)
    write_new(OUT/'training_receipt.json',row)
    for n in SOURCES:
        target=OUT/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    print('FROZEN',row['receipt_sha256'],flush=True)


def check_receipt():
    row=json.loads((OUT/'training_receipt.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='receipt_sha256'}
    if object_hash(core)!=row['receipt_sha256'] or core!=current_evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):
        raise ValueError('v11 frozen evidence differs')
    if any(digest(OUT/'sources'/n)!=h for n,h in row['sources'].items()):raise ValueError('v11 source snapshot differs')
    return row


def select_all():
    start=time.perf_counter();receipt=check_receipt();manifest,records,_=load_grouped();primary={};control={};folds=[]
    for scope,val in enumerate(manifest['outer_folds']):
        train=[i for i in range(len(records)) if i not in set(val)]
        ip,op,meta=V8.load_new_fold(scope,V10.PARENT_RICH,records,V8.check_receipt())
        iq,oq,review_meta=V4.load_new_fold(scope,REVIEWER,records,V4.check_receipt())
        if any(meta[k]!=review_meta[k] for k in ['train','validation','inner_partitions']):raise ValueError('verifier/base partitions differ')
        if [x['fit_content_sha256'] for x in meta['fit_evidence']] != [x['fit_content_sha256'] for x in review_meta['fit_evidence']]:raise ValueError('verifier/base fit content differs')
        chosen=choose_review(records,train,ip,iq,scope)
        pred,trace=decode_review(records,val,op,oq,chosen['config'],trace=True);primary.update(pred)
        before=decode_predictions(records,val,op,chosen['config']['proposal']);control.update(before)
        if any(not set(pred[i])<=set(before[i]) for i in val):raise ValueError('review added/changed a proposal')
        folds.append({'fold':scope,'train_indices':train,'validation_indices':val,'inner_partitions':meta['inner_partitions'],
            'primary':chosen,'primary_predictions':{str(i):pred[i] for i in val},'outer_review_trace':{str(i):trace[i] for i in val}})
        print(json.dumps({'fold':scope,'candidate':chosen['candidate'],'review':chosen['config']['review']}),flush=True)
    baseline=json.loads(V8.BASELINE.read_text('utf-8'))['grouped_v1_baseline'];bp={p['index']:[tuple(x) for x in p['segments']] for p in baseline['predictions']}
    bm=_group_metrics(records,bp);m=_group_metrics(records,primary);cm=_group_metrics(records,control)
    if bm!=baseline['metrics']:raise ValueError('baseline reconstruction differs')
    guards=V10.promotion_guards(m,bm)
    row={'schema_version':'proposal-review-v11','status':'complete','role':protocol()['role'],
        'protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],'folds':folds,
        'grouped_v1_baseline':baseline,'primary_v11_nested':{'metrics':m,'predictions':predictions_json(records,primary),'summary':prediction_summary(records,primary)},
        'verifier_disabled_control':{'metrics':cm,'predictions':predictions_json(records,control)},
        'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),
        'paired_content_bootstrap':paired_uncertainty(records,bp,primary),'elapsed_seconds':time.perf_counter()-start}
    check_receipt();write_new(OUT/'report.json',row);print(json.dumps({'primary':m['all'],'guards':guards}),flush=True)


def fit_final_reviewer():
    start=time.perf_counter();receipt=check_receipt();_,records,_=load_grouped();ids=list(range(len(records)))
    parts=V10.partitions(records,ids,5);arrays={};evidence=[];parity=[]
    for k,part in enumerate(parts):
        fp,vp,state=fit_compact_member(REVIEWER,records,part['fit'],part['validation'],part['seed'])
        member=export_compact_member(state);mf=mv=0.
        for i in part['validation']:
            f,v=predict_compact_member(member,records[i]);mf=max(mf,float(np.max(np.abs(f-fp[i]))));mv=max(mv,abs(v-vp[i]))
            arrays[f'inner_frame_{i}']=fp[i];arrays[f'inner_video_{i}']=np.asarray(vp[i])
        if max(mf,mv)>2e-6:raise ValueError('final reviewer portable parity differs')
        parity.append({'partition':k,'max_frame_error':mf,'max_video_error':mv})
        evidence.append({'partition':k,'fit_content_sha256':state['fit_content_sha256'],'transform_sha256':object_hash(state['transform'])})
    path=OUT/'final_selection/reviewer.npz';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    row={'scope':5,'recipe':REVIEWER,'train':ids,'validation':[],'inner_partitions':parts,'fit_evidence':evidence,
        'training_receipt_sha256':receipt['receipt_sha256'],'npz_sha256':digest(path),'member_fits':3,'parity':parity,'elapsed_seconds':time.perf_counter()-start}
    row['signature']=object_hash(row);check_receipt();write_new(path.with_suffix('.json'),row);print(json.dumps({'final_reviewer_fits':3,'parity':parity}),flush=True)


def load_final_reviewer(records,receipt):
    path=OUT/'final_selection/reviewer.npz';row=json.loads(path.with_suffix('.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='signature'}
    if object_hash(core)!=row['signature'] or row['training_receipt_sha256']!=receipt['receipt_sha256'] or digest(path)!=row['npz_sha256']:
        raise ValueError('final reviewer artifacts differ')
    ids=list(range(len(records)));expected=V10.partitions(records,ids,5)
    if row['inner_partitions']!=expected or row['train']!=ids or row['validation']!=[] or row['recipe']!=REVIEWER or row['scope']!=5 or len(row['fit_evidence'])!=3:raise ValueError('final reviewer partitions differ')
    for fit,part in zip(row['fit_evidence'],expected):
        a={records[i]['sha256'] for i in part['fit']};b={records[i]['sha256'] for i in part['validation']}
        if a&b or fit['fit_content_sha256']!=sorted(a):raise ValueError('final reviewer fit leakage')
    keys={f'inner_{head}_{i}' for head in ['frame','video'] for i in ids}
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!=keys:raise ValueError('reviewer coverage differs')
        arrays={k:a[k].copy() for k in a.files}
    for i in ids:
        for head,shape in [('frame',(records[i]['frames'],)),('video',())]:
            x=arrays[f'inner_{head}_{i}']
            if x.shape!=shape or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('reviewer probability invalid')
    return Probabilities({i:arrays[f'inner_frame_{i}'] for i in ids},{i:float(arrays[f'inner_video_{i}']) for i in ids},None),row


def final_select():
    receipt=check_receipt();_,records,_=load_grouped();ids=list(range(len(records)))
    ip,_,meta=V10.load_fold(5,V10.RICH,records,V10.check_receipt(),final=True);iq,qmeta=load_final_reviewer(records,receipt)
    if meta['inner_partitions']!=qmeta['inner_partitions']:raise ValueError('final model partitions differ')
    chosen=choose_review(records,ids,ip,iq,5)
    row={'status':'complete','role':'inner3_only_deployment_choice_not_validation','chosen':chosen,
        'selection_function':'optimized_proposal_review_v11.choose_review','inner_partitions':meta['inner_partitions'],
        'uses_outer_probabilities':False,'uses_outer_metrics_for_selection':False,'coverage_per_row':1,
        'training_receipt_sha256':receipt['receipt_sha256']}
    check_receipt();write_new(OUT/'deployment_selection.json',row);print(json.dumps({'final_candidate':chosen['candidate'],'config':chosen['config']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['freeze','select','final-train','final-select'],required=True);args=p.parse_args()
    if args.phase=='freeze':freeze()
    elif args.phase=='select':select_all()
    elif args.phase=='final-train':fit_final_reviewer()
    else:final_select()

if __name__=='__main__':main()
