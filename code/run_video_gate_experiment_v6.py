#!/usr/bin/env python3
"""Strictly nested temporal-evidence normal rejection experiment v6."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import io,json,multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from optimized_grouped_training import ROOT, NEW, load_grouped, grouped_folds, digest
from optimized_compact_features_v4 import content_weights
from optimized_event_training_v5 import fit_event_member,RECIPES
from optimized_video_gate_v6 import GATE_KINDS,fit_gate,apply_gate
from optimized_video_statistics import VideoTruth,video_statistics
from optimized_locator import metrics
from run_algorithm_optimization import _group_metrics
from run_recall_selection import grouped_bootstrap,paired_uncertainty
from run_optimization_selection import Probabilities,predictions_json,prediction_summary
from run_compact_experiment_v4 import object_hash,write_new,decode_predictions
import run_event_experiment_v5 as V5

OUT=ROOT/'output/algorithm-opt-2026-10-02-v6'
BASELINE=NEW/'selection_audited/report.json'
MEMBERS=(('event_rgb_corrected_motion_et',.5),('event_rgb_corrected_motion_hgb',.5))
CANDIDATES=('ungated_control',)+GATE_KINDS
SOURCES=('run_video_gate_experiment_v6.py','optimized_video_gate_v6.py','optimized_event_training_v5.py',
         'optimized_compact_features_v4.py','optimized_compact_model_v4.py','optimized_duration_decoder_v4.py',
         'run_compact_experiment_v4.py','run_event_experiment_v5.py','optimized_locator.py',
         'optimized_recall_decoder.py','optimized_video_statistics.py','run_recall_selection.py',
         'optimized_grouped_training.py','optimized_feature_view.py','optimized_training_provenance.py',
         'build_optimization_dataset.py','run_optimization_selection.py','run_algorithm_optimization.py')
_STATE=None


def configurations():
    for t in [.3,.4,.5,.6]:
        for penalty in [.03,.1,.25]:
            for vt in [0,.4,.6]:yield {'kind':'duration-logit-v4','threshold':t,'transition_seconds':penalty,'min_seconds':.12,'video_threshold':vt}
        for ratio in [1.,.7]:
            for vt in [0,.4,.6]:yield {'kind':'recall-stable-v2','threshold':t,'low_ratio':ratio,'smooth_seconds':.15,'gap_seconds':.12,'min_seconds':.12,'seed_seconds':.05,'video_threshold':vt,'video_strength':0,'video_floor':0}


def protocol():
    return {'schema_version':'strict-video-gate-experiment-v6','date':'2026-10-02',
            'role':'iterative_development_validation_not_new_blind_test',
            'data':'same77 rows/76 content,5 SHA-heldout outer,3 inner,duplicate conflict preserved',
            'base_ensemble':[[r,w] for r,w in MEMBERS],
            'new_stage':'learn normal video rejection from heldout temporal score morphology and absolute corrected JEPA magnitudes; frame scores not multiplied by gate',
            'candidates':list(CANDIDATES),'decoder_configs':list(configurations()),
            'strict_meta_train':'For each outer fold and each inner validation part, deeper3 base ensemble fits on inner fit only create gate train features. No inner validation labels/content may fit any base model producing gate training evidence. Gate validates on original v5 inner OOF from a model fit on full inner-fit. Outer gate uses v5 inner-OOF evidence across full outer-fit; predicts v5 outer base scores. This resolves cross-level contamination in naive OOF stacking.',
            'deep_seed':'20261006+fold*223+inner_part*31+deep_part; each member adds1000*member ordinal',
            'gate_seed':'20261006+fold*223+inner_part*31+5000; outer+6000',
            'gate_features':'41 dimensions: raw video probability; frame quantiles,contrast,top10mean,0.2/0.6mean max+p90,threshold .3/.5/.7 score mass,longest seconds,component count; 8 absolute corrected channels mean+p90. No filename/generator/hash/frame-index/GT-duration feature. Score-mass fractions are temporal evidence statistics, not a normalized frame position feature.',
            'gate_training':'inverse content multiplicity, class-positive mass balanced with negative, mean weight1; labels only video existence',
            'ridge_logistic':{'C':.3,'solver':'lbfgs','max_iter':1000,'standardization':'weighted train mean/std floor1e-6,clip+-8'},
            'forest':{'trees':160,'depth':3,'min_leaf':3,'max_features':.75},
            'selection':'content weighted same top8/32stratified q20; utility=.35F1.3+.65F1.5+.10frameF1-.30normalFPR-.30positiveEmpty; stable=.75pooled+.25q20; select only inner',
            'promotion_guards':{'event_f1_03_no_worse':0,'event_f1_05_improved_02':.02,'frame_f1_drop_at_most_005':-.005,'normal_fp_no_worse':0,'positive_empty_improved_3':-3,'portable_fresh_parity_max':2e-6},
            'baseline':'fixed audited grouped v1 nested pipeline',
            'deployment_choice':'only inner4-average morphology/gate decisions, gate full training from strictly heldout base evidence; all77 final fit for runtime; no outer-best',
            'experiment_limit':'one finite gate round; never change after outer score opening',
            'limitations':['all data repeatedly inspected: not blind or confirmatory','only14 normal labels','meta evidence models see fewer training content than final-fit runtime; shift must be reported','offline centered/global context is noncausal'],
            'compatibility':'defaults/API/original data and older experiment models/reports unchanged unless guards+independent/fresh checks all pass'}


def evidence():
    v5=V5.check_receipt()
    artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sorted((V5.OUT/'training').glob('*')) if p.suffix in ['.json','.npz']}
    return {'v5_training_receipt_sha256':v5['receipt_sha256'],'v5_inputs_sha256':v5['inputs_sha256'],
            'v5_artifacts':artifacts,'sources':{n:digest(ROOT/'code'/n) for n in SOURCES},
            'protocol_sha256':digest(OUT/'protocol.json'),'baseline_report_sha256':digest(BASELINE)}


def freeze():
    write_new(OUT/'protocol.json',protocol());row=evidence();row['receipt_sha256']=object_hash(row)
    write_new(OUT/'training_receipt.json',row)
    for n in SOURCES:
        target=OUT/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    print('FROZEN',row['receipt_sha256'],flush=True)


def check_receipt():
    row=json.loads((OUT/'training_receipt.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='receipt_sha256'}
    if object_hash(core)!=row['receipt_sha256'] or core!=evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):raise ValueError('v6 frozen receipt mismatch')
    for n,h in row['sources'].items():
        if digest(OUT/'sources'/n)!=h:raise ValueError('v6 frozen source snapshot mismatch')
    return row


def init_worker():
    global _STATE
    _STATE=(*load_grouped(),check_receipt(),V5.check_receipt())


def base_fold(fold,records,v5receipt):
    inner,outer={},{}
    for r,w in MEMBERS:
        inner[r],outer[r],_=V5.load_new_fold(fold,r,records,v5receipt)
    val=_STATE[0]['outer_folds'][fold] if _STATE is not None else json.loads((NEW/'dataset_manifest.json').read_text('utf-8'))['outer_folds'][fold]
    train=[i for i in range(len(records)) if i not in set(val)]
    return V5.blended(MEMBERS,inner,train),V5.blended(MEMBERS,outer,val)


def fit_fold(fold):
    manifest,records,profiles,receipt,v5receipt=_STATE
    start=time.perf_counter();val=manifest['outer_folds'][fold];train=[i for i in range(len(records)) if i not in set(val)]
    baseinner,baseouter=base_fold(fold,records,v5receipt)
    arrays={};parts=[];outer_states={};deep_receipts=[]
    for i in train:
        arrays[f'inner_frame_{i}']=baseinner.frame[i];arrays[f'inner_video_ungated_control_{i}']=np.asarray(baseinner.video[i])
    for i in val:
        arrays[f'outer_frame_{i}']=baseouter.frame[i];arrays[f'outer_video_ungated_control_{i}']=np.asarray(baseouter.video[i])
    for k,inner_val in enumerate(grouped_folds(records,train,3,20261002+fold*31)):
        inner_train=[i for i in train if i not in set(inner_val)];deepfp={};deepvp={};deep_parts=[]
        for d,deep_val in enumerate(grouped_folds(records,inner_train,3,20261006+fold*223+k*31)):
            deep_train=[i for i in inner_train if i not in set(deep_val)]
            memberp={};fitproof=[]
            for j,(r,w) in enumerate(MEMBERS):
                seed=20261006+fold*223+k*31+d+j*1000
                fp,vp,state=fit_event_member(r,records,deep_train,deep_val,seed)
                memberp[r]=Probabilities(fp,vp,None)
                fitproof.append({'recipe':r,'seed':seed,'fit_content_sha256':state['fit_content_sha256'],'transform_sha256':object_hash(state['transform'])})
            p=V5.blended(MEMBERS,memberp,deep_val);deepfp.update(p.frame);deepvp.update(p.video)
            deep_parts.append({'fit':deep_train,'validation':deep_val,'member_fit_proof':fitproof})
        if set(deepfp)!=set(inner_train) or {records[i]['sha256'] for i in inner_train}&{records[i]['sha256'] for i in inner_val}:raise ValueError('deep gate training leakage/coverage')
        gate_states={}
        for kind in GATE_KINDS:
            state=fit_gate(kind,records,inner_train,deepfp,deepvp,20261006+fold*223+k*31+5000)
            gate_states[kind]=state
            for i in inner_val:
                arrays[f'inner_video_{kind}_{i}']=np.asarray(apply_gate(state,baseinner.frame[i],baseinner.video[i],records[i]))
        for i in inner_train:
            arrays[f'deep{k}_frame_{i}']=deepfp[i];arrays[f'deep{k}_video_{i}']=np.asarray(deepvp[i])
        parts.append({'fit':inner_train,'validation':inner_val,'deep_partitions':deep_parts,'gate_states':gate_states})
        print(json.dumps({'fold':fold,'deep_meta_inner':k,'seconds':time.perf_counter()-start}),flush=True)
    for kind in GATE_KINDS:
        state=fit_gate(kind,records,train,baseinner.frame,baseinner.video,20261006+fold*223+6000);outer_states[kind]=state
        for i in val:arrays[f'outer_video_{kind}_{i}']=np.asarray(apply_gate(state,baseouter.frame[i],baseouter.video[i],records[i]))
    file=OUT/'training'/f'{fold}_gates.npz';file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('xb') as f:np.savez_compressed(f,**arrays)
    row={'fold':fold,'train':train,'validation':val,'inner_partitions':parts,'outer_gate_states':outer_states,
         'outer_gate_seed':20261006+fold*223+6000,'training_receipt_sha256':receipt['receipt_sha256'],
         'npz_sha256':digest(file),'seconds':time.perf_counter()-start,'runtime':{'numpy':np.__version__}}
    row['signature']=object_hash(row);write_new(file.with_suffix('.json'),row)
    return {'fold':fold,'seconds':row['seconds'],'npz_sha256':row['npz_sha256']}


def load_fold(fold,records,receipt):
    file=OUT/'training'/f'{fold}_gates.npz';row=json.loads(file.with_suffix('.json').read_text('utf-8'))
    if row['fold']!=fold or row['training_receipt_sha256']!=receipt['receipt_sha256'] or object_hash({k:v for k,v in row.items() if k!='signature'})!=row['signature']:raise ValueError('gate metadata mismatch')
    data=file.read_bytes()
    import hashlib
    if hashlib.sha256(data).hexdigest()!=row['npz_sha256']:raise ValueError('gate NPZ mismatch')
    manifest=json.loads((NEW/'dataset_manifest.json').read_text('utf-8'));val=manifest['outer_folds'][fold];train=[i for i in range(len(records)) if i not in set(val)]
    if row['train']!=train or row['validation']!=val:raise ValueError('gate outer partition mismatch')
    parts=grouped_folds(records,train,3,20261002+fold*31);expected=set()
    for kind,ids in [('inner',train),('outer',val)]:
        for i in ids:
            expected.add(f'{kind}_frame_{i}')
            expected.update(f'{kind}_video_{c}_{i}' for c in CANDIDATES)
    for k,iv in enumerate(parts):
        it=[i for i in train if i not in set(iv)];part=row['inner_partitions'][k]
        if part['fit']!=it or part['validation']!=iv:raise ValueError('gate inner partition mismatch')
        if {records[i]['sha256'] for i in it}&{records[i]['sha256'] for i in iv}:raise ValueError('inner leakage')
        ds=grouped_folds(records,it,3,20261006+fold*223+k*31)
        for d,dv in enumerate(ds):
            dt=[i for i in it if i not in set(dv)];dp=part['deep_partitions'][d]
            if dp['fit']!=dt or dp['validation']!=dv:raise ValueError('gate deep partition mismatch')
            if {records[i]['sha256'] for i in dt}&{records[i]['sha256'] for i in dv+iv}:raise ValueError('cross-level gate leakage')
            for j,proof in enumerate(dp['member_fit_proof']):
                if proof['recipe']!=MEMBERS[j][0] or proof['seed']!=20261006+fold*223+k*31+d+j*1000 or proof['fit_content_sha256']!=sorted({records[i]['sha256'] for i in dt}):raise ValueError('deep base provenance mismatch')
        for c in GATE_KINDS:
            if part['gate_states'][c]['fit_content_sha256']!=sorted({records[i]['sha256'] for i in it}):raise ValueError('inner gate content mismatch')
        for i in it:expected.update([f'deep{k}_frame_{i}',f'deep{k}_video_{i}'])
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        if set(z.files)!=expected:raise ValueError('gate probability coverage mismatch')
        a={k:z[k].copy() for k in z.files}
    for key,x in a.items():
        i=int(key.rsplit('_',1)[1]);shape=(records[i]['frames'],) if '_frame_' in key else ()
        if x.shape!=shape or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('gate probability shape/range mismatch')
    def probs(kind,ids,c):return Probabilities({i:a[f'{kind}_frame_{i}'] for i in ids},{i:float(a[f'{kind}_video_{c}_{i}']) for i in ids},None)
    return {c:probs('inner',train,c) for c in CANDIDATES},{c:probs('outer',val,c) for c in CANDIDATES},row,a


def utility(stats,normals,positives):
    s=np.asarray(stats)
    def f(a):return 2*a[...,0]/np.maximum(1e-12,2*a[...,0]+a[...,1]+a[...,2])
    nf=np.divide(s[...,9],normals,out=np.zeros_like(s[...,9],dtype=float),where=np.asarray(normals)>0)
    pe=np.divide(s[...,10],positives,out=np.zeros_like(s[...,10],dtype=float),where=np.asarray(positives)>0)
    return .35*f(s[...,:3])+.65*f(s[...,3:6])+.10*f(s[...,6:9])-.30*nf-.30*pe


def choose(records,ids,p,seed):
    cw=content_weights(records,ids);weights=np.array([cw[i] for i in ids]);truths=[VideoTruth(records[i]['labels']) for i in ids]
    normals=np.array([t.normal for t in truths],float);positives=1-normals
    nm,pm=weights@normals,weights@positives;boot=grouped_bootstrap(records,ids,seed,32)*weights[None,:];bn,bp=boot@normals,boot@positives
    short=[];seen=set()
    for ordinal,cfg in enumerate(configurations()):
        pred=decode_predictions(records,ids,p,cfg);fingerprint=json.dumps([pred[i] for i in ids])
        if fingerprint in seen:continue
        seen.add(fingerprint);mat=np.stack([video_statistics(pred[i],t) for i,t in zip(ids,truths)]);total=weights@mat;u=float(utility(total,nm,pm));f05=2*total[3]/max(1e-12,2*total[3]+total[4]+total[5])
        row={'config':cfg,'stats':total.tolist(),'pooled_utility':u,'key':[u,float(f05),-total[10],-total[9],-total[11],-ordinal],'matrix':mat}
        short.append(row);short.sort(key=lambda z:z['key'],reverse=True);del short[8:]
    for row in short:
        row['bootstrap_q20']=float(np.quantile(utility(boot@row['matrix'],bn,bp),.2));row['stable_utility']=.75*row['pooled_utility']+.25*row['bootstrap_q20'];del row['matrix']
    result=deepcopy(max(short,key=lambda z:(z['stable_utility'],*z['key'][1:])));result['shortlist']=short;return result


def train_all(workers):
    start=time.perf_counter();check_receipt();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        jobs=[pool.submit(fit_fold,f) for f in range(5)]
        for job in as_completed(jobs):row=job.result();results.append(row);print(json.dumps(row),flush=True)
    check_receipt();write_new(OUT/'training_report.json',{'status':'complete','results':results,'before_after_evidence_equal':True,'deeper_base_fits':90,'elapsed_seconds':time.perf_counter()-start})


def select_all():
    start=time.perf_counter();receipt=check_receipt();manifest,records,profiles=load_grouped();selected={};folds=[];fixed={c:{} for c in CANDIDATES}
    for fold,val in enumerate(manifest['outer_folds']):
        inner,outer,meta,a=load_fold(fold,records,receipt);train=meta['train'];rows=[]
        for c in CANDIDATES:
            row=choose(records,train,inner[c],20261002+fold);row['candidate']=c;rows.append(row)
        row=max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:],-CANDIDATES.index(z['candidate'])))
        pred=decode_predictions(records,val,outer[row['candidate']],row['config']);selected.update(pred)
        for d in rows:fixed[d['candidate']].update(decode_predictions(records,val,outer[d['candidate']],d['config']))
        folds.append({'fold':fold,'train_indices':train,'validation_indices':val,'primary':row,'all_inner_selections':rows,'primary_predictions':{str(i):pred[i] for i in val}})
        print(json.dumps({'fold':fold,'chosen':row['candidate'],'config':row['config']}),flush=True)
    baseline=json.loads(BASELINE.read_text('utf-8'));bp={x['index']:[tuple(v) for v in x['segments']] for x in baseline['grouped_v1_baseline']['predictions']}
    bm=_group_metrics(records,bp);m=_group_metrics(records,selected);x,y=m['all'],bm['all']
    if bm!=baseline['grouped_v1_baseline']['metrics']:raise ValueError('baseline reconstruction mismatch')
    guards={'event_f1_03_no_worse':x['iou_0.3']['f1']>=y['iou_0.3']['f1'],'event_f1_05_improved_02':x['iou_0.5']['f1']>=y['iou_0.5']['f1']+.02,
            'frame_f1_drop_at_most_005':x['frame']['f1']>=y['frame']['f1']-.005,'normal_fp_no_worse':x['normal']['false_positive_videos']<=y['normal']['false_positive_videos'],
            'positive_empty_improved_3':x['positive_videos_without_candidate']<=y['positive_videos_without_candidate']-3}
    report={'schema_version':'strict-video-gate-experiment-v6','status':'complete','role':'iterative_development_validation_not_new_blind_test','protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],
            'folds':folds,'grouped_v1_baseline':{'metrics':bm,'predictions':predictions_json(records,bp)},'primary_v6_nested':{'metrics':m,'predictions':predictions_json(records,selected),'summary':prediction_summary(records,selected)},
            'fixed_candidate_diagnostics_not_for_promotion':{c:{'metrics':_group_metrics(records,p),'predictions':predictions_json(records,p)} for c,p in fixed.items()},
            'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),'paired_content_bootstrap':paired_uncertainty(records,bp,selected),'elapsed_seconds':time.perf_counter()-start}
    check_receipt();write_new(OUT/'report.json',report);print(json.dumps({'primary':x,'guards':guards}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','train','select'],required=True);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if args.phase=='freeze':freeze()
    elif args.phase=='train':train_all(args.workers)
    else:select_all()

if __name__=='__main__':main()
