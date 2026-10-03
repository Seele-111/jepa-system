#!/usr/bin/env python3
"""Nested spatial-JEPA ablation against the selected v10 compact control."""
from __future__ import annotations
import argparse, io, json, multiprocessing as mp, time, hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import run_event_compact_experiment_v8 as V8
import run_feature_blocks_v10 as V10
import optimized_spatial_jepa_v13 as SP
from optimized_grouped_training import ROOT,NEW,digest,grouped_folds
from optimized_training_provenance import collect_new_fit_inputs
from run_compact_experiment_v4 import write_new,object_hash,decode_predictions
from run_optimization_selection import Probabilities,predictions_json,prediction_summary
from run_algorithm_optimization import _group_metrics
from run_recall_selection import paired_uncertainty

OUT=ROOT/'output/algorithm-opt-v13-spatial-jepa';RECIPES=(SP.RECIPE,'blocks_corrected_motion_et');CONTROL='blocks_corrected_motion_et';_STATE=None
SOURCES=tuple(dict.fromkeys(('run_spatial_jepa_v13.py','optimized_spatial_jepa_v13.py')+V10.SOURCES))


def protocol():
 return {'schema_version':'spatial-jepa-v13','role':'repeated_development_not_blind_or_confirmatory','hypothesis':'Spatial concentration and location of raw JEPA patch surprise may distinguish local physical errors from normal motion; compare with identical compact event-balanced ET control.','new_recipe':SP.RECIPE,'control_recipe':CONTROL,'features':'verified raw vjepa/ijepa heatmaps, 17 spatial descriptors per model, anchor interpolation and support flags, appended to frozen v10 corrected+motion view without RGB; no labels/identity/GT position.','objective':'Exactly v8 event_weights; ET192 depth8 leaf16 max_features .5; video ET128 depth4 leaf2 max_features .75 normal mass2.','partition':'Frozen SHA-grouped outer5/inner3 and direct final inner3; control caches reuse v10 partitions; aliases/conflicting labels retained.','selection':'Exactly V8.choose per candidate: top8 pooled content utility and 32 bootstrap q20; same procedure for outer inner3 and final inner3, no outer-best.','new_nested_member_fits':20,'new_final_member_fits':3,'promotion_guards':V10.protocol()['promotion_guards'],'limitations':V10.protocol()['limitations']+['Raw heatmaps are sparse anchors interpolated over frames and remain offline/non-causal.','Spatial descriptors cannot recover events absent from the frame detector.']}

def _raw_evidence():
 rm=SP.RAW_ROOT/'manifest.json';j=json.loads(rm.read_text('utf-8'));files={p.relative_to(ROOT).as_posix():digest(p) for p in [rm]}
 for e in j['videos']:
  p=SP.RAW_ROOT/e['raw_directory']/'signals.npz';files[p.relative_to(ROOT).as_posix()]=digest(p)
 return {'manifest_sha256':digest(rm),'files':files}

def current_evidence():
 controls=[V10.OUT/('final_selection' if scope==5 else 'training')/f'{scope}_{CONTROL}{ext}' for scope in range(6) for ext in ['.npz','.json']]
 src=collect_new_fit_inputs();return {'input_hashes':dict(src.input_hashes),'inputs_sha256':src.inputs_sha256,'input_sources':dict(src.sources),'parents':{'v8':V8.check_receipt()['receipt_sha256'],'v10':V10.check_receipt()['receipt_sha256']},'raw_spatial':_raw_evidence(),'control_artifacts':{p.relative_to(ROOT).as_posix():digest(p) for p in controls},'sources':{n:digest(ROOT/'code'/n) for n in SOURCES},'protocol_sha256':digest(OUT/'protocol.json'),'baseline_sha256':digest(V8.BASELINE),'default_models':{n:digest(ROOT/'models'/n) for n in ('optimized_locator_v1.json','optimized_motion_locator_v1.json')}}

def freeze():
 write_new(OUT/'protocol.json',protocol());row=current_evidence();row['receipt_sha256']=object_hash(row);write_new(OUT/'training_receipt.json',row)
 for n in SOURCES:
  t=OUT/'sources'/n;t.parent.mkdir(parents=True,exist_ok=True)
  with t.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
 print('FROZEN',row['receipt_sha256'],flush=True)

def check_receipt():
 row=json.loads((OUT/'training_receipt.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='receipt_sha256'}
 if object_hash(core)!=row['receipt_sha256'] or core!=current_evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):raise ValueError('v13 frozen evidence differs')
 if any(digest(OUT/'sources'/n)!=h for n,h in row['sources'].items()):raise ValueError('v13 source snapshot differs')
 return row

def parts(records,ids,scope):
 vals=grouped_folds(records,ids,3,20261002+scope*31);return [{'fit':[i for i in ids if i not in set(v)],'validation':v,'seed':20261002+scope*53+k} for k,v in enumerate(vals)]

def init_worker():
 global _STATE;_STATE=(*SP.load_spatial_grouped(),check_receipt())

def fit_task(args):
 scope,final=args;manifest,records,_,receipt=_STATE;ids=list(range(len(records)));val=[] if final else manifest['outer_folds'][scope];train=[i for i in ids if i not in set(val)];ps=parts(records,train,scope);arrays={};evidence=[];start=time.perf_counter();path=OUT/('final_selection' if final else 'training')/f'{scope}_{SP.RECIPE}.npz'
 if path.exists() or path.with_suffix('.json').exists():raise ValueError('refuse existing v13 fit')
 from optimized_spatial_jepa_v13 import fit_spatial_member
 for k,part in enumerate(ps):
  fp,vp,state=fit_spatial_member(SP.RECIPE,records,part['fit'],part['validation'],part['seed'])
  for i in part['validation']:arrays[f'inner_frame_{i}']=fp[i];arrays[f'inner_video_{i}']=np.asarray(vp[i])
  evidence.append({'partition':k,'fit_content_sha256':state['fit_content_sha256'],'transform_sha256':object_hash(state['transform'])})
 if not final:
  fp,vp,state=fit_spatial_member(SP.RECIPE,records,train,val,20261002+scope*53+99)
  for i in val:arrays[f'outer_frame_{i}']=fp[i];arrays[f'outer_video_{i}']=np.asarray(vp[i])
  evidence.append({'partition':'outer','fit_content_sha256':state['fit_content_sha256'],'transform_sha256':object_hash(state['transform'])})
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:np.savez_compressed(f,**arrays)
 import sklearn
 row={'scope':scope,'final_selection':final,'recipe':SP.RECIPE,'train':train,'validation':val,'inner_partitions':ps,'fit_evidence':evidence,'outer_seed':None if final else 20261002+scope*53+99,'training_receipt_sha256':receipt['receipt_sha256'],'npz_sha256':digest(path),'seconds':time.perf_counter()-start,'runtime':{'numpy':np.__version__,'sklearn':sklearn.__version__}};row['signature']=object_hash(row);write_new(path.with_suffix('.json'),row);return {'scope':scope,'final_selection':final,'seconds':row['seconds']}

def load_new(scope,records,receipt,final=False):
 path=OUT/('final_selection' if final else 'training')/f'{scope}_{SP.RECIPE}.npz';row=json.loads(path.with_suffix('.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='signature'}
 if object_hash(core)!=row['signature'] or row['training_receipt_sha256']!=receipt['receipt_sha256'] or digest(path)!=row['npz_sha256']:raise ValueError('v13 cache signature differs')
 manifest=json.loads((NEW/'dataset_manifest.json').read_text());val=[] if final else manifest['outer_folds'][scope];ids=list(range(len(records)));train=[i for i in ids if i not in set(val)];expected=parts(records,train,scope)
 if row['scope']!=scope or row['final_selection']!=final or row['recipe']!=SP.RECIPE or row['train']!=train or row['validation']!=val or row['inner_partitions']!=expected:raise ValueError('v13 partition differs')
 if len(row['fit_evidence'])!=3+(0 if final else 1):raise ValueError('v13 evidence coverage differs')
 for f,p in zip(row['fit_evidence'],expected+([] if final else [{'fit':train,'validation':val,'seed':20261002+scope*53+99}])):
  a={records[i]['sha256'] for i in p['fit']};b={records[i]['sha256'] for i in p['validation']}
  if a&b or f['fit_content_sha256']!=sorted(a):raise ValueError('v13 fit proof differs')
 keys={f'{kind}_{head}_{i}' for kind,sub in [('inner',train),('outer',val)] for i in sub for head in ['frame','video']}
 with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as a:
  if set(a.files)!=keys:raise ValueError('v13 cache coverage differs')
  data={k:a[k].copy() for k in a.files}
 def read(kind,sub):
  return Probabilities({i:data[f'{kind}_frame_{i}'] for i in sub},{i:float(data[f'{kind}_video_{i}']) for i in sub},None)
 for kind,sub in [('inner',train),('outer',val)]:
  for i in sub:
   for head,shape in [('frame',(records[i]['frames'],)),('video',())]:
    value=data[f'{kind}_{head}_{i}']
    if value.shape!=shape or not np.isfinite(value).all() or np.any((value<0)|(value>1)):raise ValueError('v13 invalid probability tensor')
 return read('inner',train),read('outer',val),row

def train_all(workers,final=False):
 start=time.perf_counter();check_receipt();jobs=[(5,True)] if final else [(f,False) for f in range(5)];out=[]
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
  for fu in as_completed([pool.submit(fit_task,j) for j in jobs]):r=fu.result();out.append(r);print(json.dumps(r),flush=True)
 check_receipt();write_new(OUT/('final_training_report.json' if final else 'training_report.json'),{'status':'complete','fits':3 if final else 20,'results':out,'elapsed_seconds':time.perf_counter()-start})

def guards(m,b):return V10.promotion_guards(m,b)

def select_all():
 start=time.perf_counter();receipt=check_receipt();manifest,records,_=SP.load_spatial_grouped();primary={};fixed={name:{} for name in [SP.RECIPE,'control']};folds=[]
 for scope,val in enumerate(manifest['outer_folds']):
  train=[i for i in range(len(records)) if i not in set(val)];ip,op,meta=load_new(scope,records,receipt);ci,co,cmeta=V10.load_fold(scope,CONTROL,records,V10.check_receipt())
  if meta['inner_partitions']!=cmeta['inner_partitions']:raise ValueError('v13/control partitions differ')
  rows=[]
  for name,p in [(SP.RECIPE,ip),('control',ci)]:
   row=V8.choose(records,train,p,20261002+scope);row['candidate']=name;rows.append(row)
  chosen=max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:],-['spatial_jepa_corrected_motion_et','control'].index(z['candidate'])));chosen_p=op if chosen['candidate']==SP.RECIPE else co;pred=decode_predictions(records,val,chosen_p,chosen['config']);primary.update(pred)
  for row in rows:
   fixed[row['candidate']].update(decode_predictions(records,val,op if row['candidate']==SP.RECIPE else co,row['config']))
  folds.append({'fold':scope,'train_indices':train,'validation_indices':val,'inner_partitions':meta['inner_partitions'],'primary':chosen,'all_inner_selections':rows,'primary_predictions':{str(i):pred[i] for i in val}});print(json.dumps({'fold':scope,'candidate':chosen['candidate'],'config':chosen['config']}),flush=True)
 baseline=json.loads(V8.BASELINE.read_text())['grouped_v1_baseline'];bp={x['index']:[tuple(s) for s in x['segments']] for x in baseline['predictions']};bm=_group_metrics(records,bp);m=_group_metrics(records,primary)
 if bm!=baseline['metrics']:raise ValueError('baseline differs')
 g=guards(m,bm);row={'schema_version':'spatial-jepa-v13','status':'complete','role':protocol()['role'],'protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],'folds':folds,'grouped_v1_baseline':baseline,'primary_v13_nested':{'metrics':m,'predictions':predictions_json(records,primary),'summary':prediction_summary(records,primary)},'fixed_candidate_diagnostics_not_for_promotion':{name:{'role':'fixed_recipe_outer_diagnostic_not_deployment_selection','metrics':_group_metrics(records,pred),'predictions':predictions_json(records,pred)} for name,pred in fixed.items()},'statistical_promotion_checks':g,'statistical_promotion_passed':all(g.values()),'paired_content_bootstrap':paired_uncertainty(records,bp,primary),'elapsed_seconds':time.perf_counter()-start};check_receipt();write_new(OUT/'report.json',row);print(json.dumps({'primary':m['all'],'guards':g}),flush=True)

def final_select():
 receipt=check_receipt();_,records,_=SP.load_spatial_grouped();ids=list(range(len(records)));ip,_,meta=load_new(5,records,receipt,True);ci,_,cmeta=V10.load_fold(5,CONTROL,records,V10.check_receipt(),final=True)
 if meta['inner_partitions']!=cmeta['inner_partitions']:raise ValueError('final partitions differ')
 rows=[]
 for name,p in [(SP.RECIPE,ip),('control',ci)]:row=V8.choose(records,ids,p,20261007);row['candidate']=name;rows.append(row)
 chosen=max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:],-['spatial_jepa_corrected_motion_et','control'].index(z['candidate'])));out={'status':'complete','role':'inner3_only_deployment_choice_not_validation','chosen':chosen,'all_inner_selections':rows,'inner_partitions':meta['inner_partitions'],'uses_outer_probabilities':False,'uses_outer_metrics_for_selection':False,'coverage_per_row':1,'training_receipt_sha256':receipt['receipt_sha256']};check_receipt();write_new(OUT/'deployment_selection.json',out);print(json.dumps({'final_candidate':chosen['candidate'],'config':chosen['config']}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','train','select','final-train','final-select'],required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
 if not 1<=a.workers<=4:raise ValueError('workers must be1..4')
 if a.phase=='freeze':freeze()
 elif a.phase=='train':train_all(a.workers)
 elif a.phase=='select':select_all()
 elif a.phase=='final-train':train_all(a.workers,True)
 else:final_select()
if __name__=='__main__':main()
