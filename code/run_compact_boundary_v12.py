#!/usr/bin/env python3
"""Bounded boundary refinement/acceptance on matched rich ET OOF proposals."""
from __future__ import annotations
import argparse, io, json, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import run_event_compact_experiment_v8 as V8
import run_feature_blocks_v10 as V10
from optimized_grouped_training import ROOT,load_grouped,digest
from optimized_compact_boundary_v12 import fit_content_boundary,choose_boundary,decode_boundary,configurations
from run_compact_experiment_v4 import write_new,object_hash,decode_predictions
from run_optimization_selection import predictions_json,prediction_summary
from run_algorithm_optimization import _group_metrics
from run_recall_selection import paired_uncertainty

OUT=ROOT/'output/algorithm-opt-v12-compact-boundary'
SOURCES=tuple(dict.fromkeys(('run_compact_boundary_v12.py','optimized_compact_boundary_v12.py','optimized_boundary_head.py')+V10.SOURCES))
_STATE=None


def protocol():
    return {'schema_version':'compact-boundary-v12','role':'repeated_development_not_blind_or_confirmatory',
        'hypothesis':'Matched events have early starts/late ends in the OOF error ledger; train existing boundary heads on compact rich label-free features and verify endpoint evidence locally.',
        'proposal_source':'Frozen v8 rich ET; StageA unchanged V8.choose and40configs',
        'boundary_model':'Existing ET128/depth7/leaf5/.5 start and end heads; targets +/-floor(.06*fps), base mass1 perSHA split aliases and frames, inverse weighted class balance; seed=base_seed+2200.',
        'feature_view':'v10 rich view bitwise identical to frozen v4; train-only RGB PCA16',
        'selection':'StageB evaluates exactly10 policies with content-weighted existing utility,32 stratified content bootstrap q20, .75pooled+.25q20; all10 eligible, fixed tie order. Same StageA/StageB in outer nested and final inner3.',
        'boundary_policies':list(configurations()),'new_nested_boundary_pair_fits':20,'new_final_boundary_pair_fits':3,'new_frame_model_fits':0,
        'partitions':'Frozen SHA grouped outer5/inner3; final scope5 direct inner3, no averaging other experiments OOF.',
        'promotion_guards':V10.protocol()['promotion_guards'],
        'limitations':V10.protocol()['limitations']+['Boundary refinement cannot recover absent ET proposals or explicitly split merged events.', 'Minimum endpoint evidence may reject true events and must be measured.']}


def current_evidence():
    a=V8.check_receipt();b=V10.check_receipt()
    paths=[V8.OUT/'training'/f'{f}_{V10.PARENT_RICH}{ext}' for f in range(5) for ext in ('.npz','.json')]
    paths += [V10.OUT/'final_selection'/f'5_{V10.RICH}{ext}' for ext in ('.npz','.json')]
    return {'input_hashes':b['input_hashes'],'inputs_sha256':b['inputs_sha256'],
        'parent_receipts':{'v8':a['receipt_sha256'],'v10':b['receipt_sha256']},
        'consumed_artifacts':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'sources':{n:digest(ROOT/'code'/n) for n in SOURCES},'protocol_sha256':digest(OUT/'protocol.json'),
        'baseline_sha256':digest(V8.BASELINE),'default_models':b['default_models']}


def freeze():
    write_new(OUT/'protocol.json',protocol());row=current_evidence();row['receipt_sha256']=object_hash(row)
    write_new(OUT/'training_receipt.json',row)
    for n in SOURCES:
        target=OUT/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    print('FROZEN',row['receipt_sha256'],flush=True)


def check_receipt():
    row=json.loads((OUT/'training_receipt.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='receipt_sha256'}
    if object_hash(core)!=row['receipt_sha256'] or core!=current_evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):raise ValueError('v12 frozen evidence differs')
    if any(digest(OUT/'sources'/n)!=h for n,h in row['sources'].items()):raise ValueError('v12 snapshot differs')
    return row


def init_worker():
    global _STATE
    _STATE=(*load_grouped(),check_receipt())


def fit_task(args):
    scope,final=args;manifest,records,_,receipt=_STATE;ids=list(range(len(records)))
    val=[] if final else manifest['outer_folds'][scope];train=[i for i in ids if i not in set(val)]
    parts=V10.partitions(records,train,scope);arrays={};evidence=[];start=time.perf_counter()
    path=OUT/('final_selection' if final else 'training')/f'{scope}_boundary.npz'
    if path.exists() or path.with_suffix('.json').exists():raise ValueError('refuse existing boundary fit')
    for k,part in enumerate(parts+([] if final else [{'fit':train,'validation':val,'seed':20261002+scope*53+99}])):
        probs,state=fit_content_boundary(records,part['fit'],part['validation'],part['seed']+2200)
        prefix='inner' if k<3 else 'outer'
        for i,(s,e) in probs.items():arrays[f'{prefix}_start_{i}']=s;arrays[f'{prefix}_end_{i}']=e
        evidence.append({'partition':k if k<3 else 'outer','seed':part['seed']+2200,
            'fit_content_sha256':state['fit_content_sha256'],'transform_sha256':object_hash(state['transform']),
            'models_sha256':object_hash(state['boundary_models']),'base_content_mass':state['base_content_mass']})
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    import sklearn
    row={'scope':scope,'final_selection':final,'train':train,'validation':val,'inner_partitions':parts,
        'fit_evidence':evidence,'training_receipt_sha256':receipt['receipt_sha256'],'npz_sha256':digest(path),
        'seconds':time.perf_counter()-start,'runtime':{'numpy':np.__version__,'sklearn':sklearn.__version__}}
    row['signature']=object_hash(row);write_new(path.with_suffix('.json'),row)
    return {'scope':scope,'final_selection':final,'seconds':row['seconds']}


def train_all(workers,*,final=False):
    start=time.perf_counter();check_receipt();jobs=[(5,True)] if final else [(f,False) for f in range(5)];results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        for future in as_completed([pool.submit(fit_task,j) for j in jobs]):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
    check_receipt();write_new(OUT/('final_training_report.json' if final else 'training_report.json'),
        {'status':'complete','boundary_pair_fits':3 if final else 20,'results':results,'elapsed_seconds':time.perf_counter()-start})


def load_heads(scope,records,receipt,*,final=False):
    path=OUT/('final_selection' if final else 'training')/f'{scope}_boundary.npz';row=json.loads(path.with_suffix('.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='signature'}
    if object_hash(core)!=row['signature'] or row['training_receipt_sha256']!=receipt['receipt_sha256'] or digest(path)!=row['npz_sha256']:raise ValueError('boundary cache signature differs')
    manifest=json.loads((V10.NEW/'dataset_manifest.json').read_text('utf-8'));val=[] if final else manifest['outer_folds'][scope]
    train=[i for i in range(len(records)) if i not in set(val)];parts=V10.partitions(records,train,scope)
    if row['scope']!=scope or row['final_selection']!=final or row['train']!=train or row['validation']!=val or row['inner_partitions']!=parts:raise ValueError('boundary partition differs')
    expected=parts+([] if final else [{'fit':train,'validation':val,'seed':20261002+scope*53+99}])
    if len(row['fit_evidence'])!=len(expected):raise ValueError('boundary fit coverage differs')
    for fit,part in zip(row['fit_evidence'],expected):
        a={records[i]['sha256'] for i in part['fit']};b={records[i]['sha256'] for i in part['validation']}
        if a&b or fit['fit_content_sha256']!=sorted(a) or fit['seed']!=part['seed']+2200 or fit['base_content_mass']!=len(a):raise ValueError('boundary train proof differs')
    keys={f'{kind}_{head}_{i}' for kind,sub in [('inner',train),('outer',val)] for i in sub for head in ['start','end']}
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=keys:raise ValueError('boundary tensor coverage differs')
        arrays={k:archive[k].copy() for k in archive.files}
    for kind,sub in [('inner',train),('outer',val)]:
        for i in sub:
            for head in ['start','end']:
                x=arrays[f'{kind}_{head}_{i}']
                if x.shape!=(records[i]['frames'],) or x.dtype!=np.float32 or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('boundary tensor invalid')
    def read(kind,sub):return {i:(arrays[f'{kind}_start_{i}'],arrays[f'{kind}_end_{i}']) for i in sub}
    return read('inner',train),read('outer',val),row


def select_all():
    start=time.perf_counter();receipt=check_receipt();manifest,records,_=load_grouped();primary={};control={};folds=[]
    for scope,val in enumerate(manifest['outer_folds']):
        train=[i for i in range(len(records)) if i not in set(val)]
        ip,op,meta=V8.load_new_fold(scope,V10.PARENT_RICH,records,V8.check_receipt());ih,oh,hmeta=load_heads(scope,records,receipt)
        if meta['inner_partitions']!=hmeta['inner_partitions']:raise ValueError('boundary/base partition differs')
        chosen=choose_boundary(records,train,ip,ih,scope);pred,trace=decode_boundary(records,val,op,oh,chosen['config'],trace=True)
        primary.update(pred);control.update(decode_predictions(records,val,op,chosen['config']['proposal']))
        folds.append({'fold':scope,'train_indices':train,'validation_indices':val,'inner_partitions':meta['inner_partitions'],
            'primary':chosen,'primary_predictions':{str(i):pred[i] for i in val},'outer_boundary_trace':{str(i):trace[i] for i in val}})
        print(json.dumps({'fold':scope,'policy':chosen['config']['boundary']}),flush=True)
    baseline=json.loads(V8.BASELINE.read_text('utf-8'))['grouped_v1_baseline'];bp={p['index']:[tuple(x) for x in p['segments']] for p in baseline['predictions']}
    bm=_group_metrics(records,bp);m=_group_metrics(records,primary);guards=V10.promotion_guards(m,bm)
    if bm!=baseline['metrics']:raise ValueError('baseline differs')
    row={'schema_version':'compact-boundary-v12','status':'complete','role':protocol()['role'],'folds':folds,
        'protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],
        'grouped_v1_baseline':baseline,'primary_v12_nested':{'metrics':m,'predictions':predictions_json(records,primary),'summary':prediction_summary(records,primary)},
        'boundary_disabled_control':{'metrics':_group_metrics(records,control),'predictions':predictions_json(records,control)},
        'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),
        'paired_content_bootstrap':paired_uncertainty(records,bp,primary),'elapsed_seconds':time.perf_counter()-start}
    check_receipt();write_new(OUT/'report.json',row);print(json.dumps({'primary':m['all'],'guards':guards}),flush=True)


def final_select():
    receipt=check_receipt();_,records,_=load_grouped();ids=list(range(len(records)))
    p,_,meta=V10.load_fold(5,V10.RICH,records,V10.check_receipt(),final=True);heads,_,hmeta=load_heads(5,records,receipt,final=True)
    if meta['inner_partitions']!=hmeta['inner_partitions']:raise ValueError('final boundary/base partitions differ')
    chosen=choose_boundary(records,ids,p,heads,5)
    row={'status':'complete','role':'inner3_only_deployment_choice_not_validation','chosen':chosen,
        'inner_partitions':hmeta['inner_partitions'],'uses_outer_probabilities':False,'uses_outer_metrics_for_selection':False,
        'coverage_per_row':1,'selection_function':'optimized_compact_boundary_v12.choose_boundary','training_receipt_sha256':receipt['receipt_sha256']}
    check_receipt();write_new(OUT/'deployment_selection.json',row);print(json.dumps({'final_candidate':chosen['candidate'],'config':chosen['config']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','train','select','final-train','final-select'],required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('workers must be1..4')
    if a.phase=='freeze':freeze()
    elif a.phase=='train':train_all(a.workers)
    elif a.phase=='select':select_all()
    elif a.phase=='final-train':train_all(a.workers,final=True)
    else:final_select()

if __name__=='__main__':main()
