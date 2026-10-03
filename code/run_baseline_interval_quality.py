"""One baseline-targeted hypothesis: train inclusive-interval IoU completeness.

Keep the original RF/ET/TCN frame members fixed. Fit interval quality using strict
deep OOF for inner training; choose only four thresholds and disabled control.
No outer-best deployment selection, no altered annotations, no new weights.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
import optimized_baseline_crossfit as B
from optimized_grouped_training import ROOT, load_grouped
from optimized_compact_features_v4 import _raw, content_weights
from run_recall_selection import load_fold, grouped_bootstrap
from run_optimization_selection import Probabilities, blend_probabilities, CANDIDATES, _group_metrics, predictions_json
from optimized_locator import decode, metrics, spans, interval_iou
from optimized_video_statistics import VideoTruth, video_statistics, utility

OUT=B.OUT
HEAD=OUT/'interval_heads'
FIXED=next(c for c in CANDIDATES if c.name=='corrected_rf_et_rgb_tcn_40_40_20')
STATE=None
THRESHOLDS=(.25,.35,.45,.55)


def quality_module():
    import optimized_interval_quality as Q
    return Q


def evidence_view(record):
    # Raw compact view has no label/identity access or fitted RGB transform.
    x, flags, support, names, fnames = _raw('compact_corrected_motion_et',record,None)
    values=np.concatenate([np.where(support,x,0),flags],axis=1).astype(np.float32)
    return values,['raw/'+n for n in names]+['flag/'+n for n in fnames]


def default_bundle():
    d=json.loads((ROOT/'models/optimized_locator_v1.json').read_text('utf-8'))
    actual=[(m['recipe'],m['weight']) for m in d['members']]
    if actual!=list(B.MEMBERS):raise ValueError('default member composition differs')
    return d


def prepare():
    manifest,records,profiles=load_grouped();raw={};names=None
    for i,r in enumerate(records):
        x,n=evidence_view(r)
        if names is None:names=n
        if n!=names:raise ValueError('regional evidence names differ')
        raw[i]=x
    return manifest,records,profiles,raw,names


def freeze():
    if (OUT/'interval_protocol.json').exists() or HEAD.exists():raise FileExistsError('refuse head freeze overwrite')
    p=B.check_protocol(recheck_inputs=True);_,records,_,raw,names=prepare();Q=quality_module()
    sources={str(ROOT/'code'/n):B.digest(ROOT/'code'/n) for n in (
        'run_baseline_interval_quality.py','optimized_interval_quality.py','optimized_compact_features_v4.py',
        'optimized_baseline_crossfit.py','optimized_video_statistics.py','optimized_locator.py',
        'run_recall_selection.py','optimized_detector.py')}
    d=default_bundle()
    row={'schema_version':'baseline-interval-completeness-v1','role':'repeated_development_not_blind',
        'mechanism':'Replace threshold-only interval construction with quality regressed to max inclusive IoU; unchanged default frame members.',
        'hypothesis_evidence':{'events':85,'overlapping_but_IoU_below_0.5':40,'those_event_peak_at_least_0.5':38},
        'sources_sha256':sources,'baseline_protocol_sha256':p['receipt_sha256'],
        'raw_frame_evidence_names':names,'raw_transform':'compact._raw corrected-motion masked continuous plus flags; no RGB/PCA/labels/identity',
        'frame_evidence_width':len(names),'interval_feature_names':Q.interval_features(
            np.full(3,.5,np.float32),.5,24.,[(0,2)],np.zeros((3,len(names)),np.float32))[1],
        'proposal_rule':{'grid_seconds':.12,'max_grid_points':64,'mean_floor':.2,'peak_floor':.35,
            'short_seconds':[.04,.12,.24],'run_thresholds':[.25,.4,.55,.7],'endpoint_offsets_seconds':[-.12,0,.12],'maximum_candidates':4096},
        'quality_target':'max inclusive IoU with any positive GT span, normal target zero',
        'sampler':{'normal_limit':96,'strata_limits':[32,32,32],'strata_cutoffs':[.1,.5],
            'seed':'head_seed + 10007*row_index; used for sampling only, never a feature'},
        'head':'ET regressor192/depth8/leaf12/max_features.5; only one learner configuration',
        'selector':'score descending, lexicographic tie; reject any inclusive overlap; no video-wide veto; start-sorted output',
        'configurations':[{'enabled':False}]+[{'enabled':True,'threshold':t} for t in THRESHOLDS],
        'choice':'inner OOF pooled top3, then .75 pooled+.25 grouped-bootstrap utility q20; utility matches frozen project normal/empty penalties',
        'default_decoder':d['decoder'],
        'legacy_baseline':'grouped v1 historical model-selection pipeline; scores not relabeled as current model validation',
        'control':'same fixed default recipe and default decoder from held-out grouped base scores',
        'guards':{'event_f1_03_min':.5034013605442177,'event_f1_05_min':.34653061224489793,
            'frame_f1_min':.5586221701795472,'normal_fp_max':5,'positive_empty_max':12,
            'portable_error_max':2e-6,'independent_audit_required':True,'fresh_product_required':True},
        'new_quality_fits':24,'statistical_upgrade_only_is_not_deployment':True,
        'limitations':['Data already inspected; no independent holdout.',
            'Legacy baseline states were not saved; historical baseline origin has legacy audit limits.',
            'New base and interval model states retained, but local unkeyed hashes are not authenticated external logs.',
            'Offline future context is not causal streaming evidence.']}
    row['receipt_sha256']=B.object_hash(row)
    (HEAD/'sources').mkdir(parents=True);(HEAD/'fits').mkdir()
    for path,sha in sources.items():
        with (HEAD/'sources'/Path(path).name).open('xb') as f:f.write(Path(path).read_bytes())
    B.write_new(OUT/'interval_protocol.json',row)
    print(json.dumps({'status':'interval_protocol_frozen','receipt_sha256':row['receipt_sha256'],
        'feature_width':len(row['interval_feature_names']),'configurations':len(row['configurations'])}))


def check_protocol():
    B.check_protocol()
    row=json.loads((OUT/'interval_protocol.json').read_text('utf-8'));sha=row.pop('receipt_sha256')
    if B.object_hash(row)!=sha:raise ValueError('interval protocol signature differs')
    row['receipt_sha256']=sha
    for path,h in row['sources_sha256'].items():
        if B.digest(path)!=h:raise ValueError('interval source drift: '+path)
    return row


def init_worker():
    global STATE
    STATE=prepare()


def legacy(scope,kind,records):
    caches={r:load_fold(scope,r,records) for r,w in B.MEMBERS}
    meta=caches[B.MEMBERS[0][0]][2]
    ids=meta['train'] if kind=='inner' else meta['validation']
    probs=blend_probabilities(FIXED,{r:z[0 if kind=='inner' else 1] for r,z in caches.items()},ids)
    return probs,meta,[str((ROOT/'output/algorithm-opt-2026-10-02-v2/grouped_training/folds'/f'{scope}_{r}.json').relative_to(ROOT)) for r,w in B.MEMBERS]


def combine_jobs(job_ids,records):
    fp={};vp={};sources=[]
    for name in job_ids:
        f,v,m=B.load_job(name,records)
        if set(fp)&set(f):raise ValueError('deeper OOF duplicate coverage')
        fp.update(f);vp.update(v)
        sources.append(str((B.BASE/'predictions'/(name+'.json')).relative_to(ROOT)))
    return Probabilities(fp,vp,None),sources


def parts(scope,records):
    if scope<5:return legacy(scope,'inner',records)[1]['inner_partitions']
    jobs=B.check_protocol()['jobs']
    return [{'fit':j['fit'],'validation':j['predict']} for j in jobs if j['role']=='final_inner_base_validation']


def head_inputs(scope,inner,records):
    if inner is not None:
        part=parts(scope,records)[inner]
        train,fit_sources=combine_jobs([f's{scope}_i{inner}_d{d}' for d in range(3)],records)
        train_ids=part['fit'];pred_ids=part['validation']
        if scope<5:
            p,_,predict_sources=legacy(scope,'inner',records)
            pred=Probabilities({i:p.frame[i] for i in pred_ids},{i:p.video[i] for i in pred_ids},None)
        else:pred,predict_sources=combine_jobs([f'final_inner_{inner}'],records)
    else:
        if scope==5:
            train,fit_sources=combine_jobs([f'final_inner_{j}' for j in range(3)],records)
            train_ids=list(range(len(records)));pred=train;pred_ids=train_ids;predict_sources=fit_sources
        else:
            train,meta,fit_sources=legacy(scope,'inner',records);train_ids=meta['train']
            pred,_,predict_sources=legacy(scope,'outer',records);pred_ids=meta['validation']
    if set(train.frame)!=set(train_ids) or set(pred.frame)!=set(pred_ids):raise ValueError('head input coverage differs')
    if scope!=5 or inner is not None:
        if {records[i]['sha256'] for i in train_ids}&{records[i]['sha256'] for i in pred_ids}:
            raise ValueError('head-fit content leaks prediction')
    return train,pred,train_ids,pred_ids,fit_sources,predict_sources


def sample_indices(target,normal,seed):
    target=np.asarray(target);rng=np.random.default_rng(seed)
    if normal:
        pool=np.arange(len(target));return np.sort(rng.choice(pool,min(96,len(pool)),replace=False)).astype(np.int32)
    groups=(np.flatnonzero(target<.1),np.flatnonzero((target>=.1)&(target<.5)),np.flatnonzero(target>=.5))
    return np.sort(np.concatenate([rng.choice(g,min(32,len(g)),replace=False) for g in groups])).astype(np.int32)


def fit_head(scope,inner):
    manifest,records,profiles,raw,raw_names=STATE or prepare();protocol=check_protocol();Q=quality_module()
    name=f's{scope}_'+('outer' if inner is None else f'inner{inner}')
    if scope==5 and inner is None:name='final_fullfit'
    base=HEAD/'fits'/name
    if any(base.with_suffix(s).exists() for s in ('.json','.npz','.model.json')):raise FileExistsError('refuse quality fit overwrite')
    start=time.perf_counter();train,pred,train_ids,pred_ids,fit_sources,predict_sources=head_inputs(scope,inner,records)
    seed=20278000+scope*100+(3 if inner is None else inner)
    content=content_weights(records,train_ids);feature_rows=[];target_rows=[];weight_rows=[];samples=[]
    for i in train_ids:
        bank=Q.proposal_bank(train.frame[i],records[i]['fps'])
        x,names=Q.interval_features(train.frame[i],train.video[i],records[i]['fps'],bank,raw[i])
        if names!=protocol['interval_feature_names']:raise ValueError('interval feature names differ')
        y=Q.quality_targets(bank,records[i]['labels']);normal=not np.any(records[i]['labels'])
        keep=sample_indices(y,normal,seed+10007*i)
        if len(keep):
            w=Q.quality_weights(y[keep],normal,content[i]);feature_rows.append(x[keep]);target_rows.append(y[keep]);weight_rows.append(w)
        samples.append({'index':i,'candidate_count':len(bank),'selected_ids':keep.tolist(),
            'seed':seed+10007*i,'features_sha256':__import__('hashlib').sha256(x[keep].tobytes()).hexdigest(),
            'targets_sha256':__import__('hashlib').sha256(y[keep].tobytes()).hexdigest(),
            'weight_mass':float(w.sum()) if len(keep) else 0.})
    if not feature_rows:raise ValueError('no quality training proposals')
    model=Q.fit_quality_head(np.concatenate(feature_rows),np.concatenate(target_rows),np.concatenate(weight_rows),seed);portable=Q.export_quality_head(model)
    arrays={};error=0.;counts={}
    for i in pred_ids:
        bank=Q.proposal_bank(pred.frame[i],records[i]['fps']);x,n=Q.interval_features(pred.frame[i],pred.video[i],records[i]['fps'],bank,raw[i])
        if n!=protocol['interval_feature_names']:raise ValueError('predict interval schema differs')
        scores=np.asarray(model.predict(x),np.float32) if len(x) else np.zeros(0,np.float32)
        pure=Q.portable_predict_quality(portable,x)
        error=max(error,float(np.max(np.abs(scores-pure))) if len(scores) else 0.)
        if error>2e-6:raise ValueError('quality portable parity failed')
        arrays[f'intervals_{i}']=np.asarray(bank,np.int32).reshape(-1,2);arrays[f'scores_{i}']=scores
        arrays[f'frame_{i}']=pred.frame[i];arrays[f'video_{i}']=np.asarray(pred.video[i],np.float64)
        counts[i]=len(bank)
    with base.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    B.write_new(base.with_suffix('.model.json'),{'schema_version':'interval-quality-state-v1',
        'model':portable,'interval_feature_names':protocol['interval_feature_names'],
        'raw_evidence_names':raw_names,'fit_seed':seed,'fit_content_sha256':sorted({records[i]['sha256'] for i in train_ids})})
    row={'status':'complete','scope':scope,'inner':inner,'full_fit':scope==5 and inner is None,
        'seed':seed,'fit':train_ids,'predict':pred_ids,
        'fit_content_sha256':sorted({records[i]['sha256'] for i in train_ids}),
        'predict_content_sha256':sorted({records[i]['sha256'] for i in pred_ids}),
        'fit_base_metadata':fit_sources,'predict_base_metadata':predict_sources,
        'baseline_protocol_sha256':B.check_protocol()['receipt_sha256'],
        'interval_protocol_sha256':protocol['receipt_sha256'],'sampling':samples,
        'training_proposal_rows':sum(len(y) for y in target_rows),'prediction_candidate_counts':counts,
        'npz_sha256':B.digest(base.with_suffix('.npz')),'model_sha256':B.digest(base.with_suffix('.model.json')),
        'portable_max_error':error,'seconds':time.perf_counter()-start}
    row['metadata_sha256']=B.object_hash(row);B.write_new(base.with_suffix('.json'),row);check_protocol()
    return {'name':name,'seconds':row['seconds'],'rows':row['training_proposal_rows'],'portable_max_error':error}


def train_heads(workers):
    check_protocol();jobs=[(s,j) for s in range(6) for j in range(3)]+[(s,None) for s in range(5)]
    start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        tasks=[pool.submit(fit_head,s,j) for s,j in jobs]
        for task in as_completed(tasks):
            results.append(task.result());print(json.dumps({'quality_jobs_complete':len(results),'total':len(jobs)}),flush=True)
    B.write_new(OUT/'quality_training_report.json',{'status':'complete','new_quality_fits':len(results),
        'results':results,'elapsed_seconds':time.perf_counter()-start})


def load_head(name,records):
    base=HEAD/'fits'/name;row=json.loads(base.with_suffix('.json').read_text('utf-8'));h=row.pop('metadata_sha256')
    if B.object_hash(row)!=h:raise ValueError('quality metadata signature differs')
    row['metadata_sha256']=h
    if row['interval_protocol_sha256']!=check_protocol()['receipt_sha256'] or B.digest(base.with_suffix('.npz'))!=row['npz_sha256'] or B.digest(base.with_suffix('.model.json'))!=row['model_sha256']:
        raise ValueError('quality model/cache evidence changed')
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as z:
        expected={f'{k}_{i}' for k in ('intervals','scores','frame','video') for i in row['predict']}
        if set(z.files)!=expected:raise ValueError('quality OOF output keys differ')
        items={i:{'intervals':z[f'intervals_{i}'].copy(),'scores':z[f'scores_{i}'].copy(),
            'frame':z[f'frame_{i}'].copy(),'video':float(z[f'video_{i}'])} for i in row['predict']}
    for i,item in items.items():
        if item['frame'].shape!=(records[i]['frames'],) or item['scores'].shape!=(len(item['intervals']),):raise ValueError('quality cache shapes differ')
        if not np.isfinite(item['scores']).all() or np.any((item['scores']<0)|(item['scores']>1)):raise ValueError('quality probabilities invalid')
    return items,row


def decode_items(items,records,cfg):
    Q=quality_module();d=check_protocol()['default_decoder'];out={}
    for i,item in items.items():
        out[i]=Q.select_intervals(item['intervals'],item['scores'],cfg['threshold']) if cfg['enabled'] else decode(item['frame'],records[i]['fps'],item['video'],d)
    return out


def choose(records,ids,items,seed):
    configs=check_protocol()['configurations'];truths=[VideoTruth(records[i]['labels']) for i in ids]
    normal=sum(t.normal for t in truths);positive=len(ids)-normal;rows=[]
    boot=grouped_bootstrap(records,ids,seed);bn=boot@np.asarray([t.normal for t in truths]);bp=boot@np.asarray([not t.normal for t in truths])
    for ordinal,cfg in enumerate(configs):
        predictions=decode_items({i:items[i] for i in ids},records,cfg)
        matrix=np.stack([video_statistics(predictions[i],t) for i,t in zip(ids,truths)])
        total=matrix.sum(0);u=float(utility(total,normal,positive));m=metrics([predictions[i] for i in ids],[records[i]['labels'] for i in ids])
        rows.append({'ordinal':ordinal,'config':cfg,'pooled_utility':u,'metrics':m,'matrix':matrix,
                     'rank':(u,m['iou_0.5']['f1'],-m['normal']['false_positive_videos'],-m['positive_videos_without_candidate'],-ordinal)})
    top=sorted(rows,key=lambda z:z['rank'],reverse=True)[:3]
    for row in top:
        totals=boot@row['matrix'];u=np.asarray([utility(s,n,pn) for s,n,pn in zip(totals,bn,bp)])
        row['bootstrap_q20']=float(np.percentile(u,20));row['stable_utility']=.75*row['pooled_utility']+.25*row['bootstrap_q20']
    selected=max(top,key=lambda z:(z['stable_utility'],*z['rank']))
    return {'config':selected['config'],'ordinal':selected['ordinal'],
        'stable_utility':selected['stable_utility'],'pooled_utility':selected['pooled_utility'],
        'inner_diagnostic_metrics_not_validation':selected['metrics'],
        'all_configurations':[{k:v for k,v in r.items() if k not in ('matrix','rank')} for r in rows]}


def merge_inner(scope,records):
    items={};metadata=[]
    for j in range(3):
        sub,row=load_head(f's{scope}_inner{j}',records)
        if set(items)&set(sub):raise ValueError('quality inner duplicate coverage')
        items.update(sub);metadata.append(row)
    return items,metadata


def guards(m):
    g=check_protocol()['guards']
    return {'event_f1_03_no_worse':m['iou_0.3']['f1']>=g['event_f1_03_min'],
        'event_f1_05_improved_02':m['iou_0.5']['f1']>=g['event_f1_05_min'],
        'frame_f1_drop_at_most_005':m['frame']['f1']>=g['frame_f1_min'],
        'normal_fp_no_worse':m['normal']['false_positive_videos']<=g['normal_fp_max'],
        'positive_empty_improved_3':m['positive_videos_without_candidate']<=g['positive_empty_max']}


def select_all():
    manifest,records,_,_,_=prepare();protocol=check_protocol();folds=[];pred={};control={}
    for scope in range(5):
        inner,meta=merge_inner(scope,records);chosen=choose(records,sorted(inner),inner,20279000+scope)
        outer,row=load_head(f's{scope}_outer',records)
        new=decode_items(outer,records,chosen['config']);old=decode_items(outer,records,{'enabled':False})
        pred.update(new);control.update(old)
        folds.append({'fold':scope,'train':sorted(inner),'validation':sorted(outer),'chosen':chosen,
            'predictions':{str(i):p for i,p in new.items()},'control_predictions':{str(i):p for i,p in old.items()}})
    if set(pred)!=set(range(len(records))):raise ValueError('formal OOF coverage incomplete')
    m=metrics([pred[i] for i in range(len(records))],[r['labels'] for r in records]);oldm=metrics([control[i] for i in range(len(records))],[r['labels'] for r in records])
    g=guards(m)
    baseline=json.loads((ROOT/'output/algorithm-opt-2026-10-02-v2/selection_audited/report.json').read_text('utf-8'))['grouped_v1_baseline']
    B.write_new(OUT/'report.json',{'schema_version':'baseline-targeted-interval-quality-v1','status':'complete',
        'role':'repeated_development_strict_deeper_OOF_not_blind','interval_protocol_sha256':protocol['receipt_sha256'],
        'baseline_protocol_sha256':B.check_protocol()['receipt_sha256'],'folds':folds,
        'historical_grouped_v1_baseline':baseline,'fixed_default_control':{'metrics':_group_metrics(records,control),'predictions':predictions_json(records,control)},
        'primary_interval_quality_nested':{'metrics':_group_metrics(records,pred),'predictions':predictions_json(records,pred)},
        'statistical_promotion_checks':g,'statistical_promotion_passed':all(g.values()),
        'default_promoted':False,'single_fixed_default_model_is_not_OOF':True})
    inner,meta=merge_inner(5,records);final=choose(records,sorted(inner),inner,20279005)
    B.write_new(OUT/'deployment_selection.json',{'status':'complete','role':'direct_inner3_deployment_choice_not_validation',
        'chosen':final,'uses_outer_probabilities':False,'uses_outer_metrics_for_selection':False,
        'inner_partitions':[{'fit':r['fit'],'validation':r['predict']} for r in meta],
        'coverage_per_row':1,'interval_protocol_sha256':protocol['receipt_sha256']})
    print(json.dumps({'status':'complete','formal_metrics':m,'fixed_default_control_metrics':oldm,
        'guards':g,'statistical_promotion_passed':all(g.values()),'final_config':final['config']}))


def diagnose_bank():
    _,records,_,raw,_=prepare();Q=quality_module()
    with np.load(OUT/'diagnostic_baseline_probabilities.npz',allow_pickle=False) as z:
        events=[];counts=[];empty=[]
        for i,r in enumerate(records):
            p=z[f'fixed_frame_{i}'];bank=Q.proposal_bank(p,r['fps']);counts.append(len(bank))
            for a,b in spans(r['labels']>0):events.append(max((interval_iou((a,b),q) for q in bank),default=0.))
            if not bank and np.any(r['labels']):empty.append(i)
    result={'role':'proposal_ceiling_diagnostic_uses_GT_not_achievable_result','events':len(events),
        'events_with_candidate_IoU_at_least_0.5':sum(x>=.5 for x in events),
        'events_with_candidate_IoU_at_least_0.7':sum(x>=.7 for x in events),
        'candidate_counts':counts,'positive_videos_without_proposals':empty,
        'not_precision_or_final_recall':True}
    B.write_new(OUT/'proposal_bank_diagnostic.json',result);print(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','diagnose','train','select','full-fit'));p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='diagnose':diagnose_bank()
    elif a.action=='train':train_heads(a.workers)
    elif a.action=='select':select_all()
    else:fit_head(5,None)

if __name__=='__main__':main()