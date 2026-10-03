#!/usr/bin/env python3
"""Retrain all old/new singles on content-held-out folds; CPU only, fresh namespace."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from optimized_grouped_training import NEW,ROOT,RECIPES,load_grouped,grouped_folds,fit_member,digest

_STATE=None


def init_worker():
    global _STATE
    os.environ['CUDA_VISIBLE_DEVICES']=''
    _STATE=load_grouped()


def task(args):
    fold,recipe=args;manifest,records,profiles=_STATE
    out=NEW/'grouped_training';file=out/'folds'/f'{fold}_{recipe}.npz';meta=file.with_suffix('.json')
    sources={str(ROOT/'code'/n):digest(ROOT/'code'/n) for n in ['optimized_grouped_training.py','run_recall_training.py','run_algorithm_optimization.py','optimized_feature_view.py','optimized_feature_view_v2.py','optimized_locator.py','optimized_temporal_head.py','optimized_boundary_head.py']}
    signature=digest(NEW/'dataset_manifest.json')+json.dumps(sources,sort_keys=True)
    if file.exists() and meta.exists():
        row=json.loads(meta.read_text('utf-8'));assert row['signature']==signature and row['npz_sha256']==digest(file)
        return {'fold':fold,'recipe':recipe,'seconds':row['seconds'],'status':'verified_reuse'}
    assert not file.exists() and not meta.exists(),'partial existing fit; use a new namespace'
    started=time.perf_counter();val=manifest['outer_folds'][fold];train=[i for i in range(len(records)) if i not in set(val)]
    splits=grouped_folds(records,train,3,20261002+fold*31);arrays={};inner_partitions=[]
    for k,inner_val in enumerate(splits):
        inner_train=[i for i in train if i not in set(inner_val)]
        fp,vp,state=fit_member(recipe,records,inner_train,inner_val,20261002+fold*53+k)
        for i in inner_val:
            arrays[f'inner_frame_{i}']=fp[i];arrays[f'inner_video_{i}']=np.asarray(vp[i])
            if state[-1]['boundary_probabilities'] is not None:
                arrays[f'inner_boundary_{i}']=np.stack(state[-1]['boundary_probabilities'][i])
        inner_partitions.append({'fit':inner_train,'validation':inner_val,'seed':20261002+fold*53+k})
    fp,vp,state=fit_member(recipe,records,train,val,20261002+fold*53+99)
    for i in val:
        arrays[f'outer_frame_{i}']=fp[i];arrays[f'outer_video_{i}']=np.asarray(vp[i])
        if state[-1]['boundary_probabilities'] is not None:arrays[f'outer_boundary_{i}']=np.stack(state[-1]['boundary_probabilities'][i])
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('xb') as stream:np.savez_compressed(stream,**arrays)
    row={'fold':fold,'recipe':recipe,'train':train,'validation':val,'inner_partitions':inner_partitions,
         'content_holdout':True,'seconds':time.perf_counter()-started,'signature':signature,'sources':sources,'npz_sha256':digest(file)}
    with meta.open('x',encoding='utf-8') as stream:json.dump(row,stream,indent=2)
    return {'fold':fold,'recipe':recipe,'seconds':row['seconds'],'status':'fresh'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    started=time.perf_counter();out=NEW/'grouped_training';out.mkdir(exist_ok=True)
    source_dir=out/'sources';source_dir.mkdir(exist_ok=True)
    for name in ['optimized_grouped_training.py','run_recall_training.py','run_algorithm_optimization.py','optimized_feature_view.py','optimized_feature_view_v2.py','optimized_locator.py','optimized_temporal_head.py','optimized_boundary_head.py']:
        p=ROOT/'code'/name;target=source_dir/name
        if target.exists():assert digest(target)==digest(p)
        else:target.write_bytes(p.read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as executor:
        jobs={executor.submit(task,(f,r)):(f,r) for f in range(5) for r in RECIPES}
        for future in as_completed(jobs):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    report={'status':'complete','recipes':RECIPES,'folds':5,'inner_folds':3,'content_grouped':True,'results':results,'elapsed_seconds':time.perf_counter()-started}
    p=out/'report.json'
    if p.exists():p=out/('report_replay_'+str(int(time.time()))+'.json')
    with p.open('x') as stream:json.dump(report,stream,indent=2)
    print('COMPLETE',report['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
