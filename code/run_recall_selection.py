#!/usr/bin/env python3
"""Content-grouped nested comparison. Outer labels never select any parameters.

Baseline re-runs all thirteen original families on corrected SHA-held-out folds.
New pipeline retains them plus four fixed frame views/two fixed blends. Selection
uses only inner OOF, monotone calibration, fixed decoder grid and stability rank.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from optimized_grouped_training import ROOT,OLD,NEW,NEW_RECIPES,RECIPES,load_grouped,digest
from optimized_training_provenance import audit_legacy_inputs,verify_fold_cache
from optimized_locator import decode,metrics,rolling_mean,spans
from optimized_recall_decoder import decode_v2,effective_score
from optimized_probability_calibration import fit_oof_calibration
from optimized_calibration_state import validate_calibration_state,apply_calibration_state
from optimized_video_statistics import VideoTruth,video_statistics,utility
from optimized_boundary_head import refine_intervals
from run_optimization_selection import CANDIDATES,Probabilities,blend_probabilities,predictions_json,prediction_summary
from run_algorithm_optimization import decoder_grid,_group_metrics

@dataclass(frozen=True)
class Candidate:
    name:str
    members:tuple
    def __post_init__(self):
        recipes=[r for r,w in self.members];weights=np.asarray([w for r,w in self.members])
        if not recipes or len(set(recipes))!=len(recipes) or not set(recipes)<=set(RECIPES) or not np.isfinite(weights).all() or np.any(weights<=0) or not np.isclose(weights.sum(),1):raise ValueError('invalid v2 fixed candidate')
    def as_dict(self):return {'name':self.name,'members':[{'recipe':r,'weight':w} for r,w in self.members]}

EXTRA=tuple(Candidate(r,((r,1.),)) for r in NEW_RECIPES)+(Candidate('local_noglobal_50_50',(('local_corrected_motion_et',.5),('noglobal_corrected_motion_et',.5))),Candidate('local_noglobal_rgb_60_20_20',(('local_corrected_motion_et',.6),('noglobal_corrected_motion_et',.2),('rgb_motion_tcn',.2))))
ALL_CANDIDATES=CANDIDATES+EXTRA
_STATE=None
_TRAINING_EVIDENCE=None
_SELECTION_OUTPUT=NEW/'selection'


def init_worker(output=None):
    global _STATE, _SELECTION_OUTPUT, _TRAINING_EVIDENCE
    _STATE=load_grouped()
    _TRAINING_EVIDENCE=audit_legacy_inputs()
    if output is not None:_SELECTION_OUTPUT=Path(output)


def load_fold(fold,recipe,records):
    global _TRAINING_EVIDENCE
    if _TRAINING_EVIDENCE is None:_TRAINING_EVIDENCE=audit_legacy_inputs()
    checked=verify_fold_cache(_TRAINING_EVIDENCE,fold,recipe,records)
    row=checked.metadata;arrays=checked.arrays
    def read(kind,indices):
        fp={i:np.asarray(arrays[f'{kind}_frame_{i}'],np.float32) for i in indices}
        vp={i:float(arrays[f'{kind}_video_{i}']) for i in indices}
        bp={i:tuple(arrays[f'{kind}_boundary_{i}']) for i in indices} if 'boundary' in recipe else None
        return Probabilities(fp,vp,bp)
    return read('inner',row['train']),read('outer',row['validation']),row


def grouped_bootstrap(records,indices,seed,replicates=32):
    groups={}
    for j,i in enumerate(indices):groups.setdefault(records[i]['sha256'],[]).append(j)
    strata=defaultdict(list)
    for ids in groups.values():strata[min(2,int(records[indices[ids[0]]]['event_count']))].append(ids)
    rng=np.random.default_rng(seed);counts=np.zeros((replicates,len(indices)),np.int64)
    for values in strata.values():
        for r in range(replicates):
            for j in rng.integers(0,len(values),len(values)):
                for position in values[j]:counts[r,position]+=1
    return counts


def calibrate(p,state):
    validate_calibration_state(state)
    if state is None:return p
    if set(p.frame)!=set(p.video):raise ValueError('unaligned calibration predictions')
    frame={};video={}
    for i,values in p.frame.items():
        frame[i],video[i]=apply_calibration_state(values,p.video[i],state)
    return Probabilities(frame,video,p.boundary)


def fit_calibration(kind,records,indices,p):
    if kind=='identity':return None
    if kind=='separate':return fit_oof_calibration(records,indices,p.frame,p.video)
    from optimized_joint_calibration import fit_joint
    return fit_joint([p.frame[i] for i in indices],[p.video[i] for i in indices],[records[i]['labels'] for i in indices])


def calibration_modes(records,train,p,partitions):
    yield 'identity',p,None
    for kind in ['separate','joint']:
        cross_frame={};cross_video={}
        # Leave each inner validation partition out when fitting the low-param
        # meta layer. Base OOF predictors retain cross-training dependencies,
        # explicitly not an additional independent calibration evaluation.
        for partition in partitions:
            val=partition['validation'];fit=[i for i in train if i not in set(val)]
            calibration=fit_calibration(kind,records,fit,Probabilities({i:p.frame[i] for i in fit},{i:p.video[i] for i in fit},None))
            subset=calibrate(Probabilities({i:p.frame[i] for i in val},{i:p.video[i] for i in val},None),calibration)
            cross_frame.update(subset.frame);cross_video.update(subset.video)
        assert set(cross_frame)==set(train)
        full=fit_calibration(kind,records,train,p)
        yield kind,Probabilities(cross_frame,cross_video,p.boundary),full


def config_grid(v2,boundary,joint):
    if not v2:
        for c in decoder_grid():
            for strength in [0.,.5]:
                for seconds in ([0,.2,.4] if boundary else [0]):
                    for vt in [0,.35,.5,.65,.75]:yield {**c,'video_strength':strength,'video_threshold':vt,'boundary_seconds':seconds}
        return
    for threshold,ratio,smooth,gap,minimum,seed in itertools.product([.25,.35,.45,.55,.65,.75],[1.,.7],[0,.15],[0,.12],[.12,.25],[0,.1]):
        for strength in ([0] if joint else [0,.25,.5]):
            for floor in ([0] if strength==0 else [0,.5]):
                for seconds in ([0,.2,.4] if boundary else [0]):
                    for vt in ([0] if joint else [0,.5]):
                        yield {'kind':'recall-stable-v2','threshold':threshold,'low_ratio':ratio,'smooth_seconds':smooth,'gap_seconds':gap,'min_seconds':minimum,'seed_seconds':seed,'video_strength':strength,'video_floor':floor,'video_threshold':vt,'boundary_seconds':seconds}


def candidates_for_score(score,fps,high,ratio,seed):
    runs=spans(score>=high*ratio)
    strong=np.zeros(len(score),bool);minimum=max(1,int(math.ceil(seed*fps-1e-9)))
    for s,e in spans(score>=high):
        if e-s+1>=minimum:strong[s:e+1]=True
    return [(s,e) for s,e in runs if strong[s:e+1].any()]


def modify_candidates(candidates,fps,gap,minimum,v2):
    maximum=max(0,int(round(gap*fps)));merged=[]
    for s,e in candidates:
        if merged and s-merged[-1][1]-1<=maximum:merged[-1]=(merged[-1][0],e)
        else:merged.append((s,e))
    length=max(1,int(math.ceil(minimum*fps-1e-9)) if v2 else int(round(minimum*fps)))
    return [(s,e) for s,e in merged if e-s+1>=length]


def choose(records,indices,p,v2,seed,joint=False):
    truths=[VideoTruth(records[i]['labels']) for i in indices];normal=sum(t.normal for t in truths);positive=len(indices)-normal
    score_cache={};seed_cache={};prediction_cache={};stat_cache={};seen=set();short=[];best=None
    boot=grouped_bootstrap(records,indices,seed)
    bn=boot@np.asarray([t.normal for t in truths]);bp=boot@np.asarray([not t.normal for t in truths])
    empty=np.stack([video_statistics([],t) for t in truths])
    for ordinal,cfg in enumerate(config_grid(v2,p.boundary is not None,joint)):
        adjust=(cfg['video_strength'],cfg.get('video_floor',0),cfg['smooth_seconds'])
        if adjust not in score_cache:
            score_cache[adjust]=[effective_score(p.frame[i],records[i]['fps'],p.video[i],cfg) if v2 else rolling_mean(p.frame[i]*p.video[i]**cfg['video_strength'],max(1,int(round(cfg['smooth_seconds']*records[i]['fps'])))) for i in indices]
        kseed=(adjust,cfg['threshold'],cfg['low_ratio'],cfg.get('seed_seconds',0))
        if kseed not in seed_cache:
            seed_cache[kseed]=[candidates_for_score(s,records[i]['fps'],cfg['threshold'],cfg['low_ratio'],cfg.get('seed_seconds',0)) for s,i in zip(score_cache[adjust],indices)]
        kpred=(kseed,cfg['gap_seconds'],cfg['min_seconds'],cfg['boundary_seconds'])
        if kpred not in prediction_cache:
            preds=[modify_candidates(c,records[i]['fps'],cfg['gap_seconds'],cfg['min_seconds'],v2) for c,i in zip(seed_cache[kseed],indices)]
            if cfg['boundary_seconds']:
                preds=[refine_intervals(q,*p.boundary[i],records[i]['fps'],cfg) for q,i in zip(preds,indices)]
            prediction_cache[kpred]=preds
            stat_cache[kpred]=np.stack([video_statistics(q,t) for q,t in zip(preds,truths)])
        stats=stat_cache[kpred].copy();vt=cfg['video_threshold']
        if vt:
            drop=np.asarray([p.video[i]<vt for i in indices]);stats[drop]=empty[drop]
        fingerprint=stats.tobytes()
        if fingerprint in seen:continue
        seen.add(fingerprint);total=stats.sum(0)
        f03=2*total[0]/max(1,2*total[0]+total[1]+total[2]);f05=2*total[3]/max(1,2*total[3]+total[4]+total[5]);precision=total[0]/max(1,total[0]+total[1])
        if not v2:
            key=(.5*(f03+f05)-.2*total[9]/max(1,normal),f05,precision,-total[11],-ordinal)
            if best is None or key>best['key']:best={'config':cfg,'key':key,'stats':total.tolist()}
        else:
            pooled=utility(total,normal,positive);key=(pooled,f05,-total[10],-total[9],-total[11],-ordinal)
            row={'config':cfg,'key':key,'stats':total.tolist(),'pooled_utility':float(pooled),'matrix':stats}
            short.append(row);short.sort(key=lambda z:z['key'],reverse=True);del short[8:]
    if not v2:return best
    for row in short:
        totals=boot@row['matrix'];u=np.asarray([utility(s,n,pn) for s,n,pn in zip(totals,bn,bp)])
        q=float(np.percentile(u,20));row.update(bootstrap_q20=q,stable_utility=.75*row['pooled_utility']+.25*q)
    short.sort(key=lambda r:(r['stable_utility'],*r['key']),reverse=True)
    chosen=short[0];chosen.pop('matrix');chosen['shortlist']=[{k:v for k,v in r.items() if k not in ['matrix','shortlist']} for r in short]
    return chosen


def evaluate(records,indices,p,config):
    decoder=decode_v2 if config.get('kind')=='recall-stable-v2' else decode
    result={i:decoder(p.frame[i],records[i]['fps'],p.video[i],config) for i in indices}
    if config.get('boundary_seconds',0):result={i:refine_intervals(result[i],*p.boundary[i],records[i]['fps'],config) for i in indices}
    return result


def select_legacy_family(rows):
    if not rows:raise ValueError('empty legacy family selection')
    return max(rows,key=lambda r:r['key'][0])


def fold_task(fold):
    manifest,records,_=_STATE;val=manifest['outer_folds'][fold];train=[i for i in range(len(records)) if i not in set(val)]
    caches={};outer={};partitions=None
    for recipe in RECIPES:
        inner,out,row=load_fold(fold,recipe,records);caches[recipe]=inner;outer[recipe]=out
        assert row['train']==train and row['validation']==val
        if partitions is None:partitions=row['inner_partitions']
        else:assert partitions==row['inner_partitions']
    output=_SELECTION_OUTPUT/'folds'/f'fold{fold}.json';output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists(),'refusing old selector output; fresh run directory required'
    started=time.perf_counter();baseline=[];new=[];fixed={}
    # Old decisions are fully made before using the outer labels for reporting.
    for candidate in CANDIDATES:
        inner=blend_probabilities(candidate,caches,train);decision=choose(records,train,inner,False,20261002+fold)
        baseline.append({'candidate':candidate.name,**decision})
    old_selected=select_legacy_family(baseline)
    # Legacy decoder ties use secondary metrics, but family ties compare ONLY
    # objective and preserve declared candidate order (old nested_selection).
    for candidate in ALL_CANDIDATES:
        inner=blend_probabilities(candidate,caches,train)
        choices=[]
        for mode,calibrated,state in calibration_modes(records,train,inner,partitions):
            decision=choose(records,train,calibrated,True,20261002+fold,mode=='joint')
            choices.append({'candidate':candidate.name,'calibration_mode':mode,'calibration':state,**decision})
        selected=max(choices,key=lambda r:(r['stable_utility'],*r['key']))
        new.append(selected)
        print(json.dumps({'phase':'selected_inner','fold':fold,'candidate':candidate.name,'calibration':selected['calibration_mode'],'utility':selected['stable_utility']}),flush=True)
    chosen=max(new,key=lambda r:(r['stable_utility'],*r['key']))
    # No decoder or family changes after this point. Outer labels only metrics.
    candidate_map={c.name:c for c in ALL_CANDIDATES}
    oldp=evaluate(records,val,blend_probabilities(candidate_map[old_selected['candidate']],outer,val),old_selected['config'])
    newp=evaluate(records,val,calibrate(blend_probabilities(candidate_map[chosen['candidate']],outer,val),chosen['calibration']),chosen['config'])
    for row in new:
        fp=evaluate(records,val,calibrate(blend_probabilities(candidate_map[row['candidate']],outer,val),row['calibration']),row['config'])
        fixed[row['candidate']]={str(i):fp[i] for i in val}
    result={'fold':fold,'train_indices':train,'validation_indices':val,'inner_partitions':partitions,'baseline_selections':baseline,'selected_baseline':old_selected,'new_selections':new,'selected_new':chosen,
            'baseline_predictions':{str(i):oldp[i] for i in val},'new_predictions':{str(i):newp[i] for i in val},'fixed_new_predictions':fixed,'seconds':time.perf_counter()-started}
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else None)
    return {'fold':fold,'old':old_selected['candidate'],'new':chosen['candidate'],'seconds':result['seconds']}


def paired_uncertainty(records,old,new):
    ids=list(range(len(records)));counts=grouped_bootstrap(records,ids,20261002+991,2000)
    truth=[VideoTruth(r['labels']) for r in records]
    a=np.stack([video_statistics(old[i],truth[i]) for i in ids]);b=np.stack([video_statistics(new[i],truth[i]) for i in ids]);da=counts@a;db=counts@b
    result={}
    for key,offset in [('event_f1_03',0),('event_f1_05',3),('frame_f1',6)]:
        def vals(s):return 2*s[:,offset]/np.maximum(1,2*s[:,offset]+s[:,offset+1]+s[:,offset+2])
        diff=vals(db)-vals(da);result[key]={'mean_difference':float(diff.mean()),'percentile_95_interval':np.percentile(diff,[2.5,97.5]).tolist(),'bootstrap_probability_positive':float((diff>0).mean()),'not_confirmatory_p_value':True}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--wait-training',action='store_true');parser.add_argument('--output',type=Path,default=NEW/'selection');args=parser.parse_args()
    global _SELECTION_OUTPUT
    _SELECTION_OUTPUT=args.output.resolve()
    started=time.perf_counter();pending=list(range(5));futures={};results=[]
    (_SELECTION_OUTPUT/'sources').mkdir(parents=True,exist_ok=True)
    sources={}
    for name in ['run_recall_selection.py','optimized_recall_decoder.py','optimized_probability_calibration.py','optimized_joint_calibration.py','optimized_feature_view_v2.py','optimized_video_statistics.py','optimized_grouped_training.py','optimized_detector.py','optimized_calibration_state.py','optimized_training_provenance.py']:
        p=ROOT/'code'/name;q=_SELECTION_OUTPUT/'sources'/name
        if q.exists():assert digest(q)==digest(p)
        else:q.write_bytes(p.read_bytes())
        sources[str(p)]=digest(p)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(str(_SELECTION_OUTPUT),)) as executor:
        while pending or futures:
            for fold in list(pending):
                ready=all((NEW/'grouped_training/folds'/f'{fold}_{r}.json').exists() for r in RECIPES)
                if ready:futures[executor.submit(fold_task,fold)]=fold;pending.remove(fold)
                elif not args.wait_training:raise FileNotFoundError('grouped training incomplete')
            done=[future for future in futures if future.done()]
            for future in done:
                result=future.result();results.append(result);del futures[future];print(json.dumps({'phase':'fold_complete',**result}),flush=True)
            if pending or futures:time.sleep(2)
    manifest,records,profiles=load_grouped();folds=[json.loads((_SELECTION_OUTPUT/'folds'/f'fold{f}.json').read_text('utf-8')) for f in range(5)]
    old={};new={};fixed={c.name:{} for c in ALL_CANDIDATES}
    for f in folds:
        old.update({int(i):[tuple(x) for x in p] for i,p in f['baseline_predictions'].items()});new.update({int(i):[tuple(x) for x in p] for i,p in f['new_predictions'].items()})
        for name,p in f['fixed_new_predictions'].items():fixed[name].update({int(i):[tuple(x) for x in q] for i,q in p.items()})
    report={'schema_version':'algorithm-opt-grouped-recall-v2','status':'complete','role':'development_content_grouped_CV_not_blind','protocol_sha256':digest(NEW/'protocol_amended.json'),'source_sha256':sources,
      'dataset_manifest_sha256':digest(NEW/'dataset_manifest.json'),'base_dataset_npz_sha256':digest(OLD/'dataset.npz'),
      'candidates':[c.as_dict() for c in ALL_CANDIDATES],'folds':folds,
      'grouped_v1_baseline':{'metrics':_group_metrics(records,old),'predictions':predictions_json(records,old)},
      'primary_v2_nested':{'metrics':_group_metrics(records,new),'predictions':predictions_json(records,new),**prediction_summary(records,new)},
      'fixed_candidate_diagnostics':{name:_group_metrics(records,p) for name,p in fixed.items()},
      'paired_content_bootstrap':paired_uncertainty(records,old,new),'elapsed_seconds':time.perf_counter()-started}
    a=report['grouped_v1_baseline']['metrics']['all'];b=report['primary_v2_nested']['metrics']['all']
    checks={'event_f1_03_no_worse':b['iou_0.3']['f1']>=a['iou_0.3']['f1'],
      'event_f1_05_improved_02':b['iou_0.5']['f1']>=a['iou_0.5']['f1']+.02,
      'frame_f1_drop_at_most_005':b['frame']['f1']>=a['frame']['f1']-.005,
      'normal_fp_no_worse':b['normal']['false_positive_videos']<=a['normal']['false_positive_videos'],
      'positive_empty_improved_3':b['positive_videos_without_candidate']<=a['positive_videos_without_candidate']-3}
    report['statistical_promotion_checks']=checks;report['statistical_promotion_passed']=all(checks.values())
    path=_SELECTION_OUTPUT/'report.json';assert not path.exists()
    with path.open('x') as f:json.dump(report,f,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else None)
    print(json.dumps({'phase':'complete','baseline':a,'v2':b,'checks':checks,'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
