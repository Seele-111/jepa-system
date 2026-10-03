#!/usr/bin/env python3
"""Mask-only v13 video-head repair; cached frame probabilities bitwise preserved."""
from __future__ import annotations
import argparse,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import run_spatial_jepa_v13 as OLD
import run_feature_blocks_v10 as V10
import run_event_compact_experiment_v8 as V8
import optimized_spatial_jepa_v13 as SP
from optimized_spatial_jepa_v13r import fit_repaired_video
from optimized_grouped_training import ROOT,digest
from run_compact_experiment_v4 import write_new,object_hash,decode_predictions
from run_optimization_selection import Probabilities,predictions_json,prediction_summary
from run_algorithm_optimization import _group_metrics
from run_recall_selection import paired_uncertainty

OUT=ROOT/'output/algorithm-opt-v13r-spatial-mask-repair';_STATE=None
SOURCES=tuple(dict.fromkeys(('run_spatial_jepa_v13r.py','optimized_spatial_jepa_v13r.py')+OLD.SOURCES))


def protocol():
    return {'schema_version':'spatial-jepa-mask-repair-v13r','role':'repeated_development_not_blind_or_confirmatory',
        'reason':'Independent source review found v13 spatial video statistics include unsupported edge values while frame features are masked. Preserve frozen v13 and repair only video aggregation.',
        'intervention':'Only mean/std/p10/p90 of spatial video channels are computed over spatial_support==True; empty support returns zero; separate support statistics retained.',
        'frame_probabilities':'Exactly copy all frozen v13 frame tensors bitwise; no frame model refits or new frame features.',
        'video_model':'Unchanged ET128/depth4/leaf2/max_features.75/class-balanced/normal mass2 and identical seed+1100.',
        'selection':'Unchanged v13 2candidate selection and40configs with V8.choose; no tuning after repaired outer evaluation.',
        'partition':'Identical frozen v13 outer5/inner3 and final direct inner3.',
        'new_nested_video_fits':20,'new_final_video_fits':3,'new_frame_fits':0,
        'promotion_guards':V10.protocol()['promotion_guards'],
        'limitations':OLD.protocol()['limitations']+['This repairs observed support semantics; it does not guarantee metric improvement.']}


def current_evidence():
    parent=OLD.check_receipt();paths=[OLD.OUT/('final_selection' if f==5 else 'training')/f'{f}_{SP.RECIPE}{ext}' for f in range(6) for ext in ['.npz','.json']]
    return {'parent_receipt_sha256':parent['receipt_sha256'],'input_hashes':parent['input_hashes'],'inputs_sha256':parent['inputs_sha256'],
        'raw_spatial':parent['raw_spatial'],'control_artifacts':parent['control_artifacts'],
        'reused_parent_frame_artifacts':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'sources':{n:digest(ROOT/'code'/n) for n in SOURCES},'protocol_sha256':digest(OUT/'protocol.json'),
        'baseline_sha256':digest(V8.BASELINE),'default_models':parent['default_models']}


def freeze():
    write_new(OUT/'protocol.json',protocol());row=current_evidence();row['receipt_sha256']=object_hash(row);write_new(OUT/'training_receipt.json',row)
    for n in SOURCES:
        path=OUT/'sources'/n;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    print('FROZEN',row['receipt_sha256'],flush=True)


def check_receipt():
    row=json.loads((OUT/'training_receipt.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='receipt_sha256'}
    if row['receipt_sha256']!=object_hash(core) or core!=current_evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):raise ValueError('v13r frozen evidence differs')
    if any(digest(OUT/'sources'/n)!=h for n,h in row['sources'].items()):raise ValueError('v13r source snapshot differs')
    return row


def init_worker():
    global _STATE
    _STATE=(*SP.load_spatial_grouped(),check_receipt())


def fit_task(args):
    scope,final=args;manifest,records,_,receipt=_STATE;start=time.perf_counter();ids=list(range(len(records)));val=[] if final else manifest['outer_folds'][scope];train=[i for i in ids if i not in set(val)]
    parts=V10.partitions(records,train,scope);ip,op,parent=OLD.load_new(scope,records,OLD.check_receipt(),final)
    if parts!=parent['inner_partitions']:raise ValueError('parent repair partitions differ')
    arrays={};fits=[]
    path=OUT/('final_selection' if final else 'training')/f'{scope}_{SP.RECIPE}.npz'
    if path.exists() or path.with_suffix('.json').exists():raise ValueError('refuse existing repair artifact')
    for k,part in enumerate(parts+([] if final else [{'fit':train,'validation':val,'seed':20261002+scope*53+99}])):
        vp,state=fit_repaired_video(records,part['fit'],part['validation'],part['seed']);prefix='inner' if k<3 else 'outer';fp=ip.frame if k<3 else op.frame
        for i in part['validation']:
            arrays[f'{prefix}_frame_{i}']=fp[i].copy();arrays[f'{prefix}_video_{i}']=np.asarray(vp[i])
        fits.append({'partition':k if k<3 else 'outer','seed':part['seed'],'fit_content_sha256':state['fit_content_sha256'],
            'transform_sha256':object_hash(state['transform']),'video_model_sha256':state['model_sha256'],'max_portable_error':state['max_portable_error']})
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    row={'scope':scope,'final_selection':final,'recipe':SP.RECIPE,'train':train,'validation':val,'inner_partitions':parts,
        'fit_evidence':fits,'training_receipt_sha256':receipt['receipt_sha256'],'parent_frame_npz_sha256':parent['npz_sha256'],
        'npz_sha256':digest(path),'frame_probabilities_reused_unchanged':True,'seconds':time.perf_counter()-start}
    row['signature']=object_hash(row);write_new(path.with_suffix('.json'),row)
    return {'scope':scope,'final_selection':final,'seconds':row['seconds']}


def train_all(workers,final=False):
    start=time.perf_counter();check_receipt();results=[];jobs=[(5,True)] if final else [(f,False) for f in range(5)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        for future in as_completed([pool.submit(fit_task,j) for j in jobs]):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
    check_receipt();write_new(OUT/('final_training_report.json' if final else 'training_report.json'),
        {'status':'complete','new_video_fits':3 if final else 20,'new_frame_fits':0,'results':results,'elapsed_seconds':time.perf_counter()-start})


def load_repair(scope,records,receipt,final=False):
    path=OUT/('final_selection' if final else 'training')/f'{scope}_{SP.RECIPE}.npz';row=json.loads(path.with_suffix('.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='signature'}
    if row['signature']!=object_hash(core) or row['training_receipt_sha256']!=receipt['receipt_sha256'] or digest(path)!=row['npz_sha256']:raise ValueError('v13r cache signature differs')
    ip,op,parent=OLD.load_new(scope,records,OLD.check_receipt(),final)
    for key in ['train','validation','inner_partitions','scope','final_selection','recipe']:
        if row[key]!=parent[key]:raise ValueError('repair parent partitions differ')
    if row['parent_frame_npz_sha256']!=parent['npz_sha256'] or row['frame_probabilities_reused_unchanged'] is not True:raise ValueError('repair parent frame binding differs')
    parts=row['inner_partitions']+([] if final else [{'fit':row['train'],'validation':row['validation'],'seed':20261002+scope*53+99}])
    if len(parts)!=len(row['fit_evidence']):raise ValueError('repair fit coverage differs')
    for proof,part in zip(row['fit_evidence'],parts):
        a={records[i]['sha256'] for i in part['fit']};b={records[i]['sha256'] for i in part['validation']}
        if a&b or proof['fit_content_sha256']!=sorted(a) or proof['seed']!=part['seed'] or proof['max_portable_error']>2e-6:raise ValueError('repair fit content/seed/parity differs')
    with np.load(path,allow_pickle=False) as a:
        expected={f'{kind}_{head}_{i}' for kind,sub in [('inner',row['train']),('outer',row['validation'])] for i in sub for head in ['frame','video']}
        if set(a.files)!=expected:raise ValueError('repair probability coverage differs')
        data={k:a[k].copy() for k in a.files}
    for kind,sub,source in [('inner',row['train'],ip),('outer',row['validation'],op)]:
        for i in sub:
            if not np.array_equal(data[f'{kind}_frame_{i}'],source.frame[i]):raise ValueError('repair unexpectedly changed frame probabilities')
            for head,shape in [('frame',(records[i]['frames'],)),('video',())]:
                x=data[f'{kind}_{head}_{i}']
                if x.shape!=shape or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('repair probability tensor invalid')
    def read(kind,sub):return Probabilities({i:data[f'{kind}_frame_{i}'] for i in sub},{i:float(data[f'{kind}_video_{i}']) for i in sub},None)
    return read('inner',row['train']),read('outer',row['validation']),row


def select_rows(records,ids,ip,cp,scope):
    rows=[]
    for name,p in [(SP.RECIPE,ip),('control',cp)]:
        row=V8.choose(records,ids,p,20261002+scope);row['candidate']=name;rows.append(row)
    chosen=max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:],-[SP.RECIPE,'control'].index(z['candidate'])))
    return chosen,rows


def select_all():
    start=time.perf_counter();receipt=check_receipt();manifest,records,_=SP.load_spatial_grouped();selected={};fixed={n:{} for n in [SP.RECIPE,'control']};folds=[]
    for scope,val in enumerate(manifest['outer_folds']):
        train=[i for i in range(len(records)) if i not in set(val)];ip,op,meta=load_repair(scope,records,receipt);ci,co,cm=V10.load_fold(scope,OLD.CONTROL,records,V10.check_receipt())
        if meta['inner_partitions']!=cm['inner_partitions']:raise ValueError('repair/control partitions differ')
        chosen,rows=select_rows(records,train,ip,ci,scope);pred=decode_predictions(records,val,op if chosen['candidate']==SP.RECIPE else co,chosen['config']);selected.update(pred)
        for row in rows:fixed[row['candidate']].update(decode_predictions(records,val,op if row['candidate']==SP.RECIPE else co,row['config']))
        folds.append({'fold':scope,'train_indices':train,'validation_indices':val,'inner_partitions':meta['inner_partitions'],
            'primary':chosen,'all_inner_selections':rows,'primary_predictions':{str(i):pred[i] for i in val}})
        print(json.dumps({'fold':scope,'candidate':chosen['candidate'],'config':chosen['config']}),flush=True)
    baseline=json.loads(V8.BASELINE.read_text('utf-8'))['grouped_v1_baseline'];bp={x['index']:[tuple(s) for s in x['segments']] for x in baseline['predictions']};bm=_group_metrics(records,bp);m=_group_metrics(records,selected)
    if bm!=baseline['metrics']:raise ValueError('repair baseline reconstruction differs')
    guards=V10.promotion_guards(m,bm)
    row={'schema_version':'spatial-jepa-mask-repair-v13r','status':'complete','role':protocol()['role'],
        'protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],'folds':folds,
        'grouped_v1_baseline':baseline,'primary_v13r_nested':{'metrics':m,'predictions':predictions_json(records,selected),'summary':prediction_summary(records,selected)},
        'fixed_candidate_diagnostics_not_for_promotion':{n:{'role':'fixed_recipe_outer_diagnostic_not_deployment_selection','metrics':_group_metrics(records,pred),'predictions':predictions_json(records,pred)} for n,pred in fixed.items()},
        'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),
        'paired_content_bootstrap':paired_uncertainty(records,bp,selected),'elapsed_seconds':time.perf_counter()-start}
    check_receipt();write_new(OUT/'report.json',row);print(json.dumps({'primary':m['all'],'guards':guards}),flush=True)


def final_select():
    receipt=check_receipt();_,records,_=SP.load_spatial_grouped();ids=list(range(len(records)));ip,_,meta=load_repair(5,records,receipt,True);ci,_,cm=V10.load_fold(5,OLD.CONTROL,records,V10.check_receipt(),final=True)
    if meta['inner_partitions']!=cm['inner_partitions']:raise ValueError('repair final control partition differs')
    chosen,rows=select_rows(records,ids,ip,ci,5)
    row={'status':'complete','role':'inner3_only_deployment_choice_not_validation','chosen':chosen,'all_inner_selections':rows,
        'inner_partitions':meta['inner_partitions'],'uses_outer_probabilities':False,'uses_outer_metrics_for_selection':False,'coverage_per_row':1,
        'training_receipt_sha256':receipt['receipt_sha256']}
    check_receipt();write_new(OUT/'deployment_selection.json',row);print(json.dumps({'final_candidate':chosen['candidate'],'config':chosen['config']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','train','select','final-train','final-select'],required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('workers must be1..4')
    if a.phase=='freeze':freeze()
    elif a.phase=='train':train_all(a.workers)
    elif a.phase=='select':select_all()
    elif a.phase=='final-train':train_all(a.workers,True)
    else:final_select()
if __name__=='__main__':main()
