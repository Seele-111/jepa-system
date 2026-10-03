#!/usr/bin/env python3
"""Bounded independent audit of unchanged baseline + one interval-IoU head.
No fitting/search/production selectors. --snapshot never waits or claims a pass.
Final acceptance requires --replay-base-states. Only trusted local states allowed.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
from io import BytesIO
import json
from pathlib import Path
import pickle
import time
import sys
import numpy as np
import audit_feature_blocks_v10 as A
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('output/baseline-targeted-interval-quality')
REPAIR=OUT/'interval-schema-repair'
BASE=OUT/'base_crossfit'
HEAD=REPAIR/'interval_heads'
MEMBERS=(('corrected_motion_rf',.4),('corrected_motion_et',.4),('rgb_motion_tcn',.2))
BASE_PIN='c93e70a55ce57125f1384f6798fdaebd567f654289c9e9fd346f01778693aba1'
BASE_RECEIPT='6876cfb51fddfe2be357bfafa3e118e9da00e8a96923444edb600595384e18cf'
PARENT_RECEIPT='7489f640a8627a7354f3c342647bc2f7c7f6384177bc950cbcbaebec3320c7d4'
HEAD_RECEIPT='594f9ee635d2f43f510254bde73f178371c0a9920668b99e20b8d80b4eb390c2'
HEAD_PIN='b48fd6b9126cb3835719434e989f99139b2d4ec77a373f1506bfd84400d94e1b'
CORE_PIN='82871634aa795b389c49bfb996f148e79074c1a3ae4101d9a57f94fcc4dca4ac'
HELPER_PIN='f1ad533abe2bc46e454f73bfdd0ab007afdd8d83924b3ce5696ae54fddc97700'
CONFIGS=({'enabled':False},)+tuple({'enabled':True,'threshold':t} for t in (.25,.35,.45,.55))
GUARDS={'event_f1_03_min':.5034013605442177,'event_f1_05_min':.34653061224489793,
 'frame_f1_min':.5586221701795472,'normal_fp_max':5,'positive_empty_max':12,
 'portable_error_max':2e-6,'independent_audit_required':True,'fresh_product_required':True}
TOL=2e-6
MAX_STATE_BYTES=128*1024*1024
require,compare=A.require,A.compare
LIMITS=[
 '77 already-inspected rows/76 contents: not a new blind or confirmatory test.',
 'SHA grouping proves exact-content isolation, not near-duplicate/source/scene isolation.',
 'Legacy base states were not saved: native metadata/cache hashes do not enable historical model replay.',
 'New model-state replay proves state-to-score consistency, not authenticated execution of fit/partition/seed. Unkeyed hashes are not authentication.',
 'Only locally self-generated trusted pickle states may be replayed, after hash validation. Allowlisted unpickling does not make third-party pickle safe.',
 'Original alias label conflicts retained; metrics still count 77 rows and only 14 normal rows.',
 'Pinned feature transforms/proposal generator/default decoder are audited dependencies. Targets, sampling, weights, portable head traversal, selection/statistics/utility are independent.',
 'Portable head has no sklearn fit/RNG state or full training trace. Static fitter/seed wrapper/tree count/depth are checked, not proof of historical seed/weight use. Weight-mass evidence does not authenticate the entire historical weight vector.',
 'Saved PCA mean/components are checked and replayed without refit; randomized PCA seed execution is not recoverable from components alone.',
 'Offline centered/global context is not causal streaming. No fresh-pixel/product/latency checks. Oracle proposal coverage never affects selection or performance claims.',
]

def relative(value):
    value=str(value).replace('\\','/')
    for prefix in ('/mnt/e/jepa-system/','E:/jepa-system/','e:/jepa-system/'):
        if value.startswith(prefix):value=value[len(prefix):];break
    A.safe_path(ROOT,value)
    return Path(value)

def signed(row,key):
    require(A.is_hash(row.get(key)) and row[key]==A.value_hash({k:v for k,v in row.items() if k!=key}),'signature mismatch: '+key)

def partition(records,fit,predict,excluded=(),full=False):
    for ids in (fit,predict,excluded):
        require(isinstance(ids,(list,tuple)) and len(set(ids))==len(ids) and all(type(i) is int and 0<=i<len(records) for i in ids),'invalid partition indices')
    hs=[{records[i]['sha256'] for i in ids} for ids in (fit,predict,excluded)]
    require(not (set(fit)|set(predict))&set(excluded) and not (hs[0]|hs[1])&hs[2],'excluded row/SHA leaked')
    require((full and fit==predict) or (not full and not set(fit)&set(predict) and not hs[0]&hs[1]),'fit/predict row/SHA leaked')

def folds(records,ids,seed):return A.seeded_folds(records,A.content_groups(records,ids),3,seed)

def inner_parts(records,outer,scope):
    ids=[i for i in range(len(records)) if scope==5 or i not in set(outer[scope])]
    vals=folds(records,ids,20273000 if scope==5 else 20261002+scope*31)
    return [{'fit':[i for i in ids if i not in set(val)],'validation':val,'seed':20273100+j if scope==5 else 20261002+scope*53+j} for j,val in enumerate(vals)]

def expected_jobs(records,outer):
    jobs=[]
    for scope in range(6):
        for inner,part in enumerate(inner_parts(records,outer,scope)):
            fit,val=part['fit'],part['validation']
            if scope==5:jobs.append({'id':f'final_inner_{inner}','role':'final_inner_base_validation','scope':scope,'inner':inner,'deep':None,'fit':fit,'predict':val,'excluded':[],'seed':20273200+inner})
            for deep,dv in enumerate(folds(records,fit,20274000+scope*10+inner)):
                jobs.append({'id':f's{scope}_i{inner}_d{deep}','role':'deep_oof_base_for_quality_training','scope':scope,'inner':inner,'deep':deep,'fit':[i for i in fit if i not in set(dv)],'predict':dv,'excluded':sorted(set((outer[scope] if scope<5 else [])+val)),'seed':20275000+scope*100+inner*10+deep})
    for row in jobs:
        partition(records,row['fit'],row['predict'],row['excluded'])
        require(sorted(row['fit']+row['predict']+row['excluded'])==list(range(len(records))),'job partition not exhaustive')
    require(len(jobs)==57,'57 base jobs required')
    return jobs

def validate_scores(arrays,records,ids):
    require(set(arrays)=={f'{k}_{i}' for k in ('frame','video') for i in ids},'probability OOF keys mismatch')
    fp,vp={},{}
    for i in ids:
        f,v=arrays[f'frame_{i}'],arrays[f'video_{i}']
        require(f.dtype==np.float32 and f.shape==(records[i]['frames'],) and v.shape==() and v.dtype.kind=='f','probability shape/dtype mismatch')
        require(np.isfinite(f).all() and np.all((f>=0)&(f<=1)) and np.isfinite(v).all() and 0<=float(v)<=1,'invalid probability value')
        fp[i],vp[i]=f,float(v)
    return A.Probabilities(fp,vp)

def merge_scores(rows,ids):
    fp,vp={},{}
    for p in rows:
        require(not set(fp)&set(p.frame),'duplicate OOF row');fp.update(p.frame);vp.update(p.video)
    require(set(fp)==set(vp)==set(ids),'missing/extra OOF row')
    return A.Probabilities(fp,vp)

def load_legacy(e,records,outer):
    result={}
    for scope in range(5):
        val=outer[scope];train=[i for i in range(len(records)) if i not in set(val)];blended=[];sources=[]
        for recipe,weight in MEMBERS:
            path=A.GROUPED/'grouped_training/folds'/f'{scope}_{recipe}.json';row=e.json(path);sources.append(path.as_posix())
            for k,v in (('fold',scope),('recipe',recipe),('train',train),('validation',val),('inner_partitions',inner_parts(records,outer,scope)),('content_holdout',True)):compare(row[k],v,'legacy/'+k)
            for p,h in row['sources'].items():e.read(relative(p),h)
            require(row['signature']==e.hashes[(A.GROUPED/'dataset_manifest.json').as_posix()]+json.dumps(row['sources'],sort_keys=True),'legacy native signature mismatch')
            arrays=A.read_archive(e.read(path.with_suffix('.npz'),row['npz_sha256']),str(path))
            require(set(arrays)=={f'{k}_{kind}_{i}' for k,ids in (('inner',train),('outer',val)) for kind in ('frame','video') for i in ids},'legacy NPZ coverage mismatch')
            values=[validate_scores({key[len(kind)+1:]:v for key,v in arrays.items() if key.startswith(kind+'_')},records,ids) for kind,ids in (('inner',train),('outer',val))]
            blended.append((weight,values))
        pairs=[]
        for k,ids in enumerate((train,val)):
            # Native float32 frame arithmetic; no upcast before blending.
            fp={i:sum(w*p[k].frame[i] for w,p in blended).astype(np.float32) for i in ids}
            vp={i:float(sum(w*p[k].video[i] for w,p in blended)) for i in ids};pairs.append(A.Probabilities(fp,vp))
        result[scope]=(*pairs,sources)
    return result

class LocalStateUnpickler(pickle.Unpickler):
    ALLOWED={'sklearn.ensemble._forest':{'RandomForestClassifier','ExtraTreesClassifier'},'sklearn.tree._classes':{'DecisionTreeClassifier','ExtraTreeClassifier'},'sklearn.tree._tree':{'Tree'},'numpy':{'ndarray','dtype'},'numpy.core.multiarray':{'_reconstruct','scalar'},'numpy._core.multiarray':{'_reconstruct','scalar'},'numpy.core.numeric':{'_frombuffer'},'numpy._core.numeric':{'_frombuffer'},'builtins':{'set','frozenset'}}
    def find_class(self,module,name):
        require(name in self.ALLOWED.get(module,set()),'unapproved pickle global: '+module+'.'+name)
        return super().find_class(module,name)

def read_state(e,path,digest):
    data=e.read(path,digest) # MUST hash before any deserialization.
    with gzip.GzipFile(fileobj=BytesIO(data),mode='rb') as z:payload=z.read(MAX_STATE_BYTES+1)
    require(len(payload)<=MAX_STATE_BYTES,'decompressed state bound exceeded')
    stream=BytesIO(payload);state=LocalStateUnpickler(stream).load()
    require(stream.tell()==len(payload),'trailing pickle data')
    require(isinstance(state,dict) and set(state)=={r for r,w in MEMBERS},'new state member schema mismatch')
    return state

def tree_parameters(model,recipe,seed,video=False):
    rf=recipe.endswith('_rf') and not video
    require(type(model).__module__=='sklearn.ensemble._forest' and type(model).__name__==('RandomForestClassifier' if rf else 'ExtraTreesClassifier'),'unexpected estimator type')
    expected={'n_estimators':128 if video else 160 if rf else 192,'max_depth':4 if video else 8 if rf else 9,
     'min_samples_leaf':2 if video else 10 if rf else 8,'max_features':.75 if video else .5 if rf else .6,
     'n_jobs':4,'class_weight':'balanced' if video else None,'random_state':seed+1100 if video else seed,
     'criterion':'gini','bootstrap':rf,'min_samples_split':2,'min_weight_fraction_leaf':0.,
     'max_leaf_nodes':None,'min_impurity_decrease':0.,'ccp_alpha':0.,'warm_start':False,
     'oob_score':False,'max_samples':None,'monotonic_cst':None}
    params=model.get_params(deep=False)
    for k,v in expected.items():compare(params[k],v,'saved base parameter/'+k)
    require(np.array_equal(model.classes_,[0,1]) and len(model.estimators_)==expected['n_estimators'],'saved classes/tree count mismatch')
    require(all(t.tree_.max_depth<=expected['max_depth'] for t in model.estimators_),'saved base depth mismatch')

def replay_base(e,row,scores,records):
    from optimized_feature_view import feature_view
    from optimized_temporal_head import predict_temporal
    states=read_state(e,relative(row['model_state_path']),row['model_state_sha256'])
    fp={i:np.zeros(records[i]['frames'],np.float32) for i in row['predict']};vp={i:0. for i in row['predict']}
    for recipe,weight in MEMBERS:
        state=states[recipe]
        require(set(state)=={'frame_model','video_model','transform','boundary_models'} and state['boundary_models'] is None,'unverified base/boundary schema')
        tr=state['transform'];fm,vm=state['frame_model'],state['video_model']
        compare(tr['recipe'],recipe,'saved transform recipe');compare(tr['feature_view'],'shared-label-free-v1','saved feature view')
        tree_parameters(vm,recipe,row['seed'],True)
        if recipe.endswith('_tcn'):
            compare(fm['schema'],'optimized-temporal-head-v1','TCN schema')
            for k,v in (('seed',row['seed']),('steps',400),('selection','none_final_fixed_step'),('device','cpu'),
                ('video_count',len(row['fit'])),('optimizer','AdamW'),('learning_rate',.002),('weight_decay',.0001),
                ('gradient_clip_norm',5.),('cpu_threads',2),('batch_size',min(8,len(row['fit']))),
                ('sampling','uniform_videos_without_replacement_per_step'),('loss','equal_video_masked_bce_with_logits'),
                ('frame_positive_weight',1.25),('temporal_variation_weight',0.)):
                compare(fm['training'][k],v,'saved TCN/'+k)
            compare(fm['architecture'],{'channels':32,'kernel_size':3,'dilations':[1,2,4],'dropout':[.15,.2,.2],
                'output_kernel_size':1,'receptive_field_samples':15,'padding':'zero_same','mask_hidden_padding':True},'saved TCN architecture')
            compare(fm['normalization'],{'fit':'training_videos_only','population_std':True,
                'statistics_axis':'all_resampled_training_frames','std_floor':1e-6,'small_std_scale':1.},'TCN normalization contract')
            for k,v in (('target_hz',10.),('grid','regular_10hz_plus_exact_endpoint'),('features','linear'),
                ('labels','nearest_frame_half_up'),('output','linear_to_original_frame_axis')):
                compare(fm['fps_profile'][k],v,'TCN resampling/'+k)
            compare(fm['fps_profile']['source_fps'],[records[i]['fps'] for i in row['fit']],'TCN fit FPS')
            compare(fm['fps_profile']['source_frame_counts'],[records[i]['frames'] for i in row['fit']],'TCN fit frame counts')
            pool=np.concatenate([records[i]['rgb'] for i in row['fit']]);pca=tr['pca']
            mean,components=np.asarray(pca['mean']),np.asarray(pca['components'])
            require(mean.shape==(pool.shape[1],) and components.shape==(min(16,*pool.shape),pool.shape[1]) and np.isfinite(mean).all() and np.isfinite(components).all(),'PCA shape/value mismatch')
            require(np.allclose(mean,pool.mean(0),rtol=0,atol=2e-6) and np.allclose(components@components.T,np.eye(len(components)),rtol=0,atol=2e-5),'PCA fit mean/orthogonality mismatch')
            from optimized_temporal_head import _resample_features
            sampled=[_resample_features(feature_view(recipe,records[i],pca)[0],records[i]['fps'])[0] for i in row['fit']]
            compare(fm['fps_profile']['resampled_frame_counts'],[len(x) for x in sampled],'TCN resampled fit counts')
            count=sum(len(x) for x in sampled)
            total=sum((x.sum(0,dtype=np.float64) for x in sampled),np.zeros(sampled[0].shape[1],np.float64))
            mu=total/count
            var=sum((((x.astype(np.float64)-mu)**2).sum(0) for x in sampled),np.zeros_like(mu))/count
            std=np.sqrt(var);std=np.where(std<1e-6,1.,std)
            require(np.array_equal(np.asarray(fm['mean'],np.float32),mu.astype(np.float32)) and
                    np.array_equal(np.asarray(fm['std'],np.float32),std.astype(np.float32)),'TCN train-only normalization mismatch')
        else:
            tree_parameters(fm,recipe,row['seed']);require(tr['pca'] is None,'non-RGB member has PCA')
        for i in row['predict']:
            x,v,names=feature_view(recipe,records[i],tr['pca']);compare(names,tr['frame_feature_names'],'saved feature order')
            f=predict_temporal(fm,x,records[i]['fps']) if recipe.endswith('_tcn') else fm.predict_proba(x)[:,1].astype(np.float32)
            v=float(vm.predict_proba(v[None])[:,1][0])
            require(f.shape==fp[i].shape and np.isfinite(f).all() and np.all((f>=0)&(f<=1)) and np.isfinite(v) and 0<=v<=1,'invalid replay probability')
            fp[i]=fp[i]+weight*f;vp[i]+=weight*v
    error=max([float(np.max(np.abs(fp[i]-scores.frame[i]))) for i in row['predict']]+[abs(vp[i]-scores.video[i]) for i in row['predict']],default=0.)
    require(error<=TOL,'base state score replay mismatch: '+row['id'])
    return {'id':row['id'],'member_states':3,'heldout_rows':len(row['predict']),'PCA_checks':1,'max_abs_error':error,'model_state_sha256':row['model_state_sha256']}

def load_bases(e,protocol,records,outer,replay):
    compare(protocol['jobs'],expected_jobs(records,outer),'independently reconstructed 57 jobs')
    caches,proofs={},[]
    for job in protocol['jobs']:
        path=BASE/'predictions'/(job['id']+'.json');row=e.json(path);signed(row,'metadata_sha256')
        for k,v in job.items():compare(row[k],v,'new base/'+k)
        compare(row['baseline_protocol_sha256'],BASE_RECEIPT,'base protocol binding')
        for k in ('fit','predict','excluded'):compare(row[k+'_content_sha256'],sorted({records[i]['sha256'] for i in row[k]}),'exact SHA/'+k)
        mp=BASE/'models'/(job['id']+'.pkl.gz');compare(row['model_state_path'],mp.as_posix(),'model path binding')
        e.read(mp,row['model_state_sha256'])
        p=validate_scores(A.read_archive(e.read(path.with_suffix('.npz'),row['npz_sha256']),str(path)),records,job['predict']);caches[job['id']]=p
        if replay:proofs.append(replay_base(e,row,p,records))
    training=e.json(OUT/'base_training_report.json')
    require(training['status']=='complete' and training['new_member_fits']==171 and len(training['results'])==57 and {r['id'] for r in training['results']}==set(caches),'base training coverage mismatch')
    for row in training['results']:compare(row['model_state_sha256'],e.hashes[(BASE/'models'/(row['id']+'.pkl.gz')).as_posix()],'training state hash')
    return caches,{'status':'replayed' if replay else 'not_requested','jobs_replayed':len(proofs),'member_states_replayed':3*len(proofs),'heldout_score_rows_replayed':sum(p['heldout_rows'] for p in proofs),'max_abs_error':max((p['max_abs_error'] for p in proofs),default=None),'jobs':proofs}

def independent_targets(bank,labels):
    events=A.label_spans(labels);out=[]
    for s,e in A.segments(bank,len(labels),'target IoU'):
        best=0.
        for a,b in events:
            intersection=max(0,min(e,b)-max(s,a)+1);best=max(best,intersection/(e-s+b-a+2-intersection))
        out.append(best)
    return np.asarray(out,np.float32)

def sample_ids(y,normal,seed):
    require(y.ndim==1 and np.isfinite(y).all() and np.all((y>=0)&(y<=1)),'invalid target tensor')
    rng=np.random.default_rng(seed)
    groups=[np.arange(len(y))] if normal else [np.flatnonzero(y<.1),np.flatnonzero((y>=.1)&(y<.5)),np.flatnonzero(y>=.5)]
    return np.sort(np.concatenate([rng.choice(g,min(96 if normal else 32,len(g)),replace=False) for g in groups])).astype(np.int32)

def independent_weights(y,normal,content):
    require(np.isfinite(content) and content>0 and (not normal or not np.any(y)),'invalid content weight/normal target')
    w=np.zeros(len(y),np.float64)
    groups=[np.ones(len(y),bool)] if normal else [y<.1,(y>=.1)&(y<.5),y>=.5]
    for mask in groups:
        if mask.any():w[mask]=(2*content if normal else content)/np.count_nonzero(mask)
    return w.astype(np.float32)

def portable_head(model,x,expected_trees=192):
    require(set(model)=={'kind','n_features','trees'} and model['kind']=='interval-quality-et-v1' and type(model['n_features']) is int and 0<model['n_features']<=4096,'invalid quality model schema')
    require(x.ndim==2 and x.dtype==np.float32 and x.shape[1]==model['n_features'] and np.isfinite(x).all(),'invalid quality feature tensor')
    require(isinstance(model['trees'],list) and len(model['trees'])==expected_trees and 1<=expected_trees<=192,'wrong quality tree count')
    result=np.zeros(len(x),np.float64)
    for tree in model['trees']:
        require(set(tree)=={'left','right','feature','threshold','value'},'invalid tree schema')
        arrays={k:np.asarray(v) for k,v in tree.items()};n=len(arrays['left'])
        require(0<n<=511 and all(a.shape==(n,) and a.dtype.kind in 'iuf' and np.isfinite(a).all() for a in arrays.values()),'invalid tree arrays')
        for k in ('left','right','feature'):require(arrays[k].dtype.kind in 'iu','noninteger tree indices')
        l,r,f,t,v=(arrays[k] for k in ('left','right','feature','threshold','value'))
        require(np.all((v>=0)&(v<=1)),'invalid quality leaf values')
        seen=set();pending=[(0,0)]
        while pending:
            node,depth=pending.pop()
            require(0<=node<n and node not in seen and depth<=8,'cyclic/shared/outside/deep tree');seen.add(node)
            if l[node]==r[node]==-1:require(f[node]==t[node]==-2,'invalid leaf sentinel')
            else:
                require(l[node]!=r[node] and 0<=f[node]<x.shape[1],'invalid branch')
                pending.extend([(int(l[node]),depth+1),(int(r[node]),depth+1)])
        require(len(seen)==n,'unreachable tree nodes')
        nodes=np.zeros(len(x),np.int64)
        for _ in range(9):
            active=np.flatnonzero(l[nodes]!=-1)
            if not len(active):break
            cur=nodes[active];nodes[active]=np.where(x[active,f[cur]]<=t[cur],l[cur],r[cur])
        require(np.all(l[nodes]==-1),'nonterminating tree');result+=v[nodes]
    return (result/expected_trees).astype(np.float32)

def select_intervals(bank,scores,threshold,frames):
    spans=A.segments(bank.tolist() if isinstance(bank,np.ndarray) else bank,frames,'quality selection')
    require(scores.ndim==1 and len(scores)==len(spans) and np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1)),'invalid quality scores')
    kept=[]
    for j in sorted(range(len(spans)),key=lambda i:(-float(scores[i]),*spans[i])):
        if scores[j]>=threshold and not any(max(spans[j][0],a)<=min(spans[j][1],b) for a,b in kept):kept.append(spans[j])
    return sorted(kept)

def utility(s,normal,positive):
    def f1(a,b,c):return 2*a/max(1,2*a+b+c)
    return .35*f1(*s[:3])+.65*f1(*s[3:6])-.20*s[9]/max(1,normal)-.15*s[10]/max(1,positive)

def decode_items(items,records,cfg,decoder):
    from optimized_locator import decode
    require(cfg in CONFIGS,'non-preregistered configuration')
    return {i:select_intervals(item['intervals'],item['scores'],cfg['threshold'],records[i]['frames']) if cfg['enabled'] else decode(item['frame'],records[i]['fps'],item['video'],decoder) for i,item in items.items()}

def choose_independent(records,ids,items,seed,decoder):
    require(set(ids)==set(items) and len(set(ids))==len(ids),'selection OOF coverage mismatch')
    normal=np.asarray([not records[i]['labels'].any() for i in ids],np.int64);positive=1-normal
    boot=A.bootstrap_counts(records,ids,seed);bn,bp=boot@normal,boot@positive;rows=[]
    for ordinal,cfg in enumerate(CONFIGS):
        p=decode_items(items,records,cfg,decoder);mat=np.stack([A.row_statistics(p[i],records[i]['labels']) for i in ids]);m=A.metric_dict(mat.sum(0),int(normal.sum()));u=float(utility(mat.sum(0),int(normal.sum()),int(positive.sum())))
        rows.append({'ordinal':ordinal,'config':dict(cfg),'pooled_utility':u,'metrics':m,'matrix':mat,'rank':(u,m['iou_0.5']['f1'],-m['normal']['false_positive_videos'],-m['positive_videos_without_candidate'],-ordinal)})
    top=sorted(rows,key=lambda r:r['rank'],reverse=True)[:3]
    for row in top:
        values=[utility(s,int(n),int(p)) for s,n,p in zip(boot@row['matrix'],bn,bp)]
        row['bootstrap_q20']=float(np.percentile(values,20));row['stable_utility']=.75*row['pooled_utility']+.25*row['bootstrap_q20']
    best=max(top,key=lambda r:(r['stable_utility'],*r['rank']))
    return {'config':best['config'],'ordinal':best['ordinal'],'stable_utility':best['stable_utility'],'pooled_utility':best['pooled_utility'],'inner_diagnostic_metrics_not_validation':best['metrics'],'all_configurations':[{k:v for k,v in r.items() if k not in ('matrix','rank')} for r in rows]}

def repair_source_equal(parent,repaired):
    before,after=ast.parse(parent),ast.parse(repaired);counts=Counter()
    class Normalize(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='OUT':
                require(ast.unparse(node.value)=="B.OUT / 'interval-schema-repair'",'repair namespace differs')
                node.value=ast.parse('B.OUT',mode='eval').body;counts['namespace']+=1
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value=='run_baseline_interval_quality_repaired.py':node.value='run_baseline_interval_quality.py';counts['source']+=1
            return node
        def visit_Compare(self,node):
            if isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Name) and node.left.func.id=='list' and len(node.left.args)==1 and isinstance(node.left.args[0],ast.Name) and node.left.args[0].id in ('names','n'):
                require(len(node.comparators)==1 and ast.unparse(node.comparators[0])=="protocol['interval_feature_names']",'normalization outside schema guard')
                node.left=node.left.args[0];counts['schema']+=1
            return self.generic_visit(node)
        def visit_Dict(self,node):
            remove=[]
            expected={'repair_parent_protocol_sha256':PARENT_RECEIPT,'repair_scope':'sequence type normalization only; zero original quality fits; identical algorithm/configs'}
            for j,(k,v) in enumerate(zip(node.keys,node.values)):
                if isinstance(k,ast.Constant) and k.value in expected:
                    require(isinstance(v,ast.Constant) and v.value==expected[k.value],'repair provenance field differs')
                    remove.append(j);counts['provenance']+=1
            node.keys=[v for j,v in enumerate(node.keys) if j not in remove];node.values=[v for j,v in enumerate(node.values) if j not in remove]
            return self.generic_visit(node)
    after=Normalize().visit(after)
    require(counts==Counter(namespace=1,source=1,schema=2,provenance=2),'repair edit scope/count mismatch')
    require(ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False),'algorithm/partition/source changed beyond schema repair')
    return {'status':'passed','parent_repaired_AST_equal_after_exact_allowed_edits':True,'allowed_edit_counts':dict(counts)}

def verify_protocols(e):
    base=e.json(OUT/'baseline_protocol.json',BASE_PIN);signed(base,'receipt_sha256');compare(base['receipt_sha256'],BASE_RECEIPT,'base receipt')
    compare(base['members'],[{'recipe':r,'weight':w} for r,w in MEMBERS],'unchanged base composition')
    require(base['rows']==77 and base['content_groups']==76 and base['new_member_fits']==171,'base scope changed')
    require(A.value_hash(base['input_sha256'])==base['input_inventory_sha256']==A.INPUTS_PIN,'input inventory digest mismatch')
    for p,h in base['input_sha256'].items():e.read(relative(p),h)
    for p,h in base['source_sha256'].items():
        local=relative(p);e.read(local,h);e.read(BASE/'sources'/local.name,h)
    for p,h in base['default_sha256'].items():e.read(relative(p),h)
    parent=e.json(OUT/'interval_protocol.json');signed(parent,'receipt_sha256');compare(parent['receipt_sha256'],PARENT_RECEIPT,'failed parent receipt')
    head=e.json(REPAIR/'interval_protocol.json',HEAD_PIN);signed(head,'receipt_sha256');compare(head['receipt_sha256'],HEAD_RECEIPT,'repaired head receipt')
    for protocol,folder in ((parent,OUT/'interval_heads'),(head,HEAD)):
        for p,h in protocol['sources_sha256'].items():
            local=relative(p);e.read(local,h);e.read(folder/'sources'/local.name,h)
    old=deepcopy(parent);new=deepcopy(head)
    old.pop('receipt_sha256');new.pop('receipt_sha256')
    compare(new.pop('repair_parent_protocol_sha256'),PARENT_RECEIPT,'repair parent receipt')
    compare(new.pop('repair_scope'),'sequence type normalization only; zero original quality fits; identical algorithm/configs','repair scope')
    op='/mnt/e/jepa-system/code/run_baseline_interval_quality.py';npth='/mnt/e/jepa-system/code/run_baseline_interval_quality_repaired.py'
    old['sources_sha256'].pop(op);new['sources_sha256'].pop(npth);compare(new,old,'parent/repair protocol algorithm equivalence')
    review=repair_source_equal(e.read(relative(op)).decode('utf-8-sig'),e.read(relative(npth)).decode('utf-8-sig'))
    failure=e.json(OUT/'schema_failure_evidence.json')
    for k,v in (('status','failed_before_any_quality_fit'),('original_interval_protocol_sha256',PARENT_RECEIPT),('original_runner_sha256',parent['sources_sha256'][op]),('quality_fit_artifacts_observed',0),('base_fits_reused_without_change',171)):compare(failure[k],v,'preserved failure/'+k)
    require(not any(A.safe_path(e.root,OUT/'interval_heads/fits').iterdir()),'failed parent namespace unexpectedly contains quality fits')
    for k,v in (('schema_version','baseline-interval-completeness-v1'),('role','repeated_development_not_blind'),('baseline_protocol_sha256',BASE_RECEIPT),('configurations',list(CONFIGS)),('guards',GUARDS),('frame_evidence_width',166),('new_quality_fits',24),('sampler',{'normal_limit':96,'strata_limits':[32,32,32],'strata_cutoffs':[.1,.5],'seed':'head_seed + 10007*row_index; used for sampling only, never a feature'})):
        compare(head[k],v,'head protocol/'+k)
    require(len(head['interval_feature_names'])==1535,'head width changed')
    compare(head['proposal_rule'],{'grid_seconds':.12,'max_grid_points':64,'mean_floor':.2,'peak_floor':.35,'short_seconds':[.04,.12,.24],'run_thresholds':[.25,.4,.55,.7],'endpoint_offsets_seconds':[-.12,0,.12],'maximum_candidates':4096},'proposal rule')
    default=e.json(Path('models/optimized_locator_v1.json'));compare(head['default_decoder'],default['decoder'],'unchanged default decoder')
    return base,head,review

def core_source_review(e,Q):
    code=e.read(Path('code/optimized_interval_quality.py'),CORE_PIN).decode('utf-8-sig');tree=ast.parse(code)
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in ('proposal_bank','interval_features','interval_feature_names'):
            require(not any(isinstance(n,ast.Constant) and n.value in ('labels','sha256','name','generator','row_id','index') for n in ast.walk(fn)),'feature/proposal label identity access')
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_quality_head')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ExtraTreesRegressor']
    require(len(calls)==1,'single ET head required')
    compare({k.arg:ast.unparse(k.value) for k in calls[0].keywords},{'n_estimators':'192','max_depth':'8','min_samples_leaf':'12','max_features':'0.5','n_jobs':'2','random_state':'seed'},'fixed head parameters')
    return {'fixed_single_ET_head':True,'identity_label_absent_from_feature_source':True,'core_sha256':CORE_PIN}

def head_inputs(records,outer,bases,legacy,scope,inner):
    if inner is not None:
        part=inner_parts(records,outer,scope)[inner];train,pred=part['fit'],part['validation']
        fs=[(BASE/'predictions'/f's{scope}_i{inner}_d{d}.json').as_posix() for d in range(3)]
        fit=merge_scores([bases[f's{scope}_i{inner}_d{d}'] for d in range(3)],train)
        if scope<5:
            p=legacy[scope][0];predict=A.Probabilities({i:p.frame[i] for i in pred},{i:p.video[i] for i in pred});ps=legacy[scope][2]
        else:predict=bases[f'final_inner_{inner}'];ps=[(BASE/'predictions'/f'final_inner_{inner}.json').as_posix()]
    elif scope<5:
        fit,predict,fs=legacy[scope];ps=fs;train,pred=sorted(fit.frame),sorted(predict.frame)
    else:
        train=pred=list(range(len(records)));fit=predict=merge_scores([bases[f'final_inner_{j}'] for j in range(3)],train)
        fs=ps=[(BASE/'predictions'/f'final_inner_{j}.json').as_posix() for j in range(3)]
    partition(records,train,pred,full=scope==5 and inner is None)
    return fit,predict,train,pred,fs,ps

def raw_evidence(records,protocol,Q):
    from optimized_compact_features_v4 import _raw
    result={};names=None
    for i,r in enumerate(records):
        x,flags,support,n,fn=_raw('compact_corrected_motion_et',r,None)
        values=np.concatenate([np.where(support,x,0),flags],axis=1).astype(np.float32)
        current=['raw/'+v for v in n]+['flag/'+v for v in fn]
        if names is None:names=current
        compare(current,names,'raw feature schema')
        altered={**r,'labels':~r['labels'],'sha256':'changed','name':'changed','generator':'changed'}
        ax,af,am,an,afn=_raw('compact_corrected_motion_et',altered,None)
        require(np.array_equal(values,np.concatenate([np.where(am,ax,0),af],axis=1).astype(np.float32)),'raw label/identity leakage')
        result[i]=values
    compare(names,protocol['raw_frame_evidence_names'],'raw protocol schema')
    compare(list(Q.interval_features(np.full(3,.5,np.float32),.5,24.,[(0,2)],np.zeros((3,len(names)),np.float32))[1]),protocol['interval_feature_names'],'interval protocol schema')
    return result,names

def load_head(e,protocol,records,outer,bases,legacy,raw,raw_names,Q,scope,inner):
    name=f's{scope}_inner{inner}' if inner is not None else f's{scope}_outer' if scope<5 else 'final_fullfit'
    path=HEAD/'fits'/name;row=e.json(path.with_suffix('.json'));signed_head(row)
    fit,pred,train,ids,fs,ps=head_inputs(records,outer,bases,legacy,scope,inner);seed=20278000+scope*100+(3 if inner is None else inner)
    for k,v in (('status','complete'),('scope',scope),('inner',inner),('full_fit',scope==5 and inner is None),('seed',seed),('fit',train),('predict',ids),('fit_base_metadata',fs),('predict_base_metadata',ps),('fit_content_sha256',sorted({records[i]['sha256'] for i in train})),('predict_content_sha256',sorted({records[i]['sha256'] for i in ids})),('baseline_protocol_sha256',BASE_RECEIPT),('interval_protocol_sha256',HEAD_RECEIPT)):compare(row[k],v,'head/'+name+'/'+k)
    content=dict(zip(train,A.content_weights(records,train)));samples=[];total=0
    for i in train:
        bank=Q.proposal_bank(fit.frame[i],records[i]['fps']);x,n=Q.interval_features(fit.frame[i],fit.video[i],records[i]['fps'],bank,raw[i]);compare(list(n),protocol['interval_feature_names'],'fit feature names')
        y=independent_targets(bank,records[i]['labels']);normal=not records[i]['labels'].any();keep=sample_ids(y,normal,seed+10007*i);w=independent_weights(y[keep],normal,content[i])
        samples.append({'index':i,'candidate_count':len(bank),'selected_ids':keep.tolist(),'seed':seed+10007*i,'features_sha256':A.sha(x[keep].tobytes()),'targets_sha256':A.sha(y[keep].tobytes()),'weight_mass':float(w.sum()) if len(keep) else 0.});total+=len(keep)
    compare(row['sampling'],samples,'head target/sampling/weight/feature digests');compare(row['training_proposal_rows'],total,'head fit rows')
    state=e.json(path.with_suffix('.model.json'),row['model_sha256'])
    for k,v in (('schema_version','interval-quality-state-v1'),('interval_feature_names',protocol['interval_feature_names']),('raw_evidence_names',raw_names),('fit_seed',seed),('fit_content_sha256',sorted({records[i]['sha256'] for i in train}))):compare(state[k],v,'portable head wrapper/'+k)
    arrays=A.read_archive(e.read(path.with_suffix('.npz'),row['npz_sha256']),str(path));require(set(arrays)=={f'{k}_{i}' for k in ('intervals','scores','frame','video') for i in ids},'head OOF keys mismatch')
    items={};counts={};error=0.
    for i in ids:
        bank=Q.proposal_bank(pred.frame[i],records[i]['fps']);x,n=Q.interval_features(pred.frame[i],pred.video[i],records[i]['fps'],bank,raw[i]);compare(list(n),protocol['interval_feature_names'],'prediction feature names')
        b,s,f,v=(arrays[f'{k}_{i}'] for k in ('intervals','scores','frame','video'))
        require(b.dtype==np.int32 and b.shape==(len(bank),2) and np.array_equal(b,np.asarray(bank).reshape(-1,2)),'head candidate mismatch')
        require(s.dtype==np.float32 and s.shape==(len(bank),) and np.isfinite(s).all() and np.all((s>=0)&(s<=1)),'head score shape/value mismatch')
        require(f.dtype==np.float32 and np.array_equal(f,pred.frame[i]) and v.shape==() and float(v)==pred.video[i],'wrong head base provenance')
        pure=portable_head(state['model'],x);delta=float(np.max(np.abs(pure-s))) if len(s) else 0.;require(delta<=TOL,'head portable score replay mismatch');error=max(error,delta)
        items[i]={'intervals':b,'scores':s,'frame':f,'video':float(v)};counts[str(i)]=len(bank)
    compare(row['prediction_candidate_counts'],counts,'head candidate counts');require(np.isfinite(row['portable_max_error']) and 0<=row['portable_max_error']<=TOL,'head attested parity invalid')
    return items,{'name':name,'fit':train,'predict':ids,'fit_base_metadata':fs,'predict_base_metadata':ps,'sample_rows':total,'portable_replay_max_abs_error':error,'seed':seed}

def producer_row(row):
    result=deepcopy(row);ids=result.get('predict');counts=result.get('prediction_candidate_counts')
    require(isinstance(ids,list) and len(set(ids))==len(ids) and all(type(i) is int and 0<=i<77 for i in ids),'noncanonical head prediction ids')
    require(isinstance(counts,dict),'candidate counts must be a dictionary')
    restored={}
    for key,value in counts.items():
        require(type(key) is str and key.isdecimal() and str(int(key))==key and type(value) is int and 0<=value<=4096,'noncanonical count key/value')
        restored[int(key)]=value
    require(set(restored)==set(ids),'candidate count coverage mismatch')
    result['prediction_candidate_counts']=restored
    return result

def signed_head(row):signed(producer_row(row),'metadata_sha256')

def verify_reader(e,names):
    receipt=e.json(REPAIR/'selection_reader_receipt.json')
    compare(receipt['interval_protocol_sha256'],HEAD_RECEIPT,'selection reader head receipt')
    compare(receipt['status'],'frozen_before_selection','selection reader status')
    compare(receipt['role'],'strict_JSON_integer_key_restore_not_algorithm_change','selection reader role')
    compare(receipt['models_or_probabilities_retrained'],False,'reader retrained flag')
    compare(receipt['stored_metadata_or_signatures_rewritten'],False,'reader rewritten flag')
    digest='328ce219cae94ca71891b62360f1fee0ee5fac1508ccf8fedecd49467e044e04'
    compare(receipt['adapter_sha256'],digest,'reader source pin');e.read(Path('code/select_baseline_interval_quality.py'),digest)
    rows=receipt['original_quality_metadata_files'];expected={(HEAD/'fits'/(n+'.json')).as_posix() for n in names}
    require(len(rows)==23 and {r['file'] for r in rows}==expected,'reader metadata coverage mismatch')
    plain=0
    for proof in rows:
        row=e.json(relative(proof['file']),proof['sha256']);signed_head(row)
        raw={k:v for k,v in row.items() if k!='metadata_sha256'};matched=A.value_hash(raw)==row['metadata_sha256'];plain+=int(matched)
        compare(proof['plain_string_key_hash_matches'],matched,'reader raw string hash claim')
        compare(proof['producer_integer_key_hash_matches'],True,'reader producer hash claim')
    return {'metadata_files':23,'producer_signatures_reproduced':23,'plain_string_signatures_matched':plain,'integer_key_semantics_only':True,'stored_metadata_unchanged':True,'reader_source_sha256':digest}

def paired_differences(records,control,primary):
    ids=list(range(len(records)));normal=np.asarray([not r['labels'].any() for r in records],np.int64);positive=1-normal
    old=np.stack([A.row_statistics(control[i],records[i]['labels']) for i in ids]);new=np.stack([A.row_statistics(primary[i],records[i]['labels']) for i in ids])
    boot=A.bootstrap_counts(records,ids,20279099,replicates=256)
    def vector(s,n,p):
        m=A.metric_dict(s,int(n))
        return np.asarray([m['iou_0.3']['f1'],m['iou_0.5']['f1'],m['frame']['f1'],m['normal']['false_positive_rate'],m['normal']['false_positive_videos'],m['positive_videos_without_candidate'],utility(s,int(n),int(p))],np.float64)
    delta=vector(new.sum(0),normal.sum(),positive.sum())-vector(old.sum(0),normal.sum(),positive.sum())
    draws=np.stack([vector(b@new,b@normal,b@positive)-vector(b@old,b@normal,b@positive) for b in boot])
    keys=('F1_IoU03','F1_IoU05','frame_F1','normal_FPR','normal_FP_rows','positive_empty_rows','utility')
    return {'role':'descriptive_paired_content_bootstrap_not_blind_significance','replicates':256,'seed':20279099,'deltas':{key:{'primary_minus_control':float(delta[j]),'q025':float(np.percentile(draws[:,j],2.5)),'q975':float(np.percentile(draws[:,j],97.5))} for j,key in enumerate(keys)},'does_not_refit_or_reselect':True}

def run_audit(root=ROOT,replay=False,snapshot=False):
    e=A.Evidence(root);e.read(Path('code/audit_feature_blocks_v10.py'),HELPER_PIN)
    prior_path=REPAIR/'independent_audit.json';prior_pin='b177c3a67bd159d83486613009bb915706ef1036951c770fbb5973bf99694241'
    prior=e.json(prior_path,prior_pin)
    compare(prior['status'],'failed','preserved auditor failure status')
    compare(prior['error'],'IndexError: index 41 is out of bounds for axis 0 with size 40','preserved auditor failure cause')
    compare(prior['auditor_source_sha256'],'7e6dda30b311c56b04ce70b3d93258f66eaf581df38ef8d0eebf69a30141a158','preserved failed auditor source')
    auditor_repair={'prior_failed_report':prior_path.as_posix(),'prior_failed_report_sha256':prior_pin,'prior_auditor_source_sha256':prior['auditor_source_sha256'],
      'cause':'Independent content_weights returns a subset-position array; load_head incorrectly indexed it by global row id.',
      'fix':'Map independently computed positional weights to exact train row ids before sampling/weight checks; no production source, metadata, model or NPZ changes.',
      'regression':'Sparse train ids [9,7] with identical SHA receive 0.5 content weights each; incorrect 1.0 mass is rejected.',
      'prior_failure_preserved':True,'retraining_performed':False}
    base,protocol,repair_review=verify_protocols(e)
    records,outer,aliases=A.load_records(e);compare(base['legacy_outer5_inner3'],outer,'outer content partitions')
    legacy=load_legacy(e,records,outer);bases,replay_result=load_bases(e,base,records,outer,replay)
    qpath=Path('code/optimized_interval_quality.py');Q=A.module_from_bytes(e.read(qpath,CORE_PIN),e.root/qpath,'_audited_quality_core')
    review=core_source_review(e,Q)
    missing=[(REPAIR/n).as_posix() for n in ('quality_training_report.json','report.json','deployment_selection.json','selection_reader_receipt.json') if not A.safe_path(e.root,REPAIR/n).is_file()]
    common={'rows':77,'content_groups':76,'aliases_retained':aliases,'base_jobs_verified':57,'base_state_replay':replay_result,'baseline_receipt_sha256':BASE_RECEIPT,'parent_head_receipt_sha256':PARENT_RECEIPT,'head_receipt_sha256':HEAD_RECEIPT,'repair_equivalence':repair_review,'auditor_repair':auditor_repair,'core_source_review':review,'deployment_approved':False,'limitations':LIMITS}
    if snapshot or missing:
        e.recheck()
        return {**common,'status':'pending','scope':'bounded_snapshot_not_formal_head_selection','missing_formal_evidence':missing,'formal_head_selection_audited':False,'consumed_artifact_sha256':dict(sorted(e.hashes.items()))}
    raw,raw_names=raw_evidence(records,protocol,Q);heads={};proofs=[]
    for scope in range(6):
        for inner in range(3):
            heads[(scope,inner)],proof=load_head(e,protocol,records,outer,bases,legacy,raw,raw_names,Q,scope,inner);proofs.append(proof)
        if scope<5:
            heads[(scope,None)],proof=load_head(e,protocol,records,outer,bases,legacy,raw,raw_names,Q,scope,None);proofs.append(proof)
    training=e.json(REPAIR/'quality_training_report.json')
    require(training['status']=='complete' and training['new_quality_fits']==23 and len(training['results'])==23 and {r['name'] for r in training['results']}=={r['name'] for r in proofs},'23 head training report coverage mismatch')
    for row in training['results']:
        proof=next(p for p in proofs if p['name']==row['name']);compare(row['rows'],proof['sample_rows'],'training sampled rows');require(0<=row['portable_max_error']<=TOL,'training parity summary invalid')
    reader=verify_reader(e,[p['name'] for p in proofs])
    report=e.json(REPAIR/'report.json');deployment=e.json(REPAIR/'deployment_selection.json')
    for k,v in (('schema_version','baseline-targeted-interval-quality-v1'),('status','complete'),('role','repeated_development_strict_deeper_OOF_not_blind'),('interval_protocol_sha256',HEAD_RECEIPT),('baseline_protocol_sha256',BASE_RECEIPT),('default_promoted',False),('single_fixed_default_model_is_not_OOF',True)):compare(report[k],v,'formal report/'+k)
    require(len(report['folds'])==5,'outer fold coverage mismatch');primary={};control={};choices=[]
    for scope in range(5):
        ids=sorted(legacy[scope][0].frame);items={i:item for j in range(3) for i,item in heads[(scope,j)].items()}
        require(len(items)==sum(len(heads[(scope,j)]) for j in range(3)),'duplicate head inner OOF')
        choice=choose_independent(records,ids,items,20279000+scope,protocol['default_decoder']);held=heads[(scope,None)]
        new=decode_items(held,records,choice['config'],protocol['default_decoder']);old=decode_items(held,records,CONFIGS[0],protocol['default_decoder'])
        compare(report['folds'][scope],{'fold':scope,'train':ids,'validation':sorted(held),'chosen':choice,'predictions':{str(i):[list(v) for v in rows] for i,rows in new.items()},'control_predictions':{str(i):[list(v) for v in rows] for i,rows in old.items()}},'outer choice/'+str(scope))
        require(not set(primary)&set(new),'duplicate outer prediction');primary.update(new);control.update(old);choices.append(choice)
    require(set(primary)==set(control)==set(range(77)),'formal OOF incomplete')
    compare(report['primary_interval_quality_nested'],{'metrics':A.grouped_metrics(records,primary),'predictions':A.prediction_rows(records,primary)},'primary independent metrics')
    compare(report['fixed_default_control'],{'metrics':A.grouped_metrics(records,control),'predictions':A.prediction_rows(records,control)},'fixed control independent metrics')
    m=A.grouped_metrics(records,primary)['all']
    guards={'event_f1_03_no_worse':m['iou_0.3']['f1']>=GUARDS['event_f1_03_min'],'event_f1_05_improved_02':m['iou_0.5']['f1']>=GUARDS['event_f1_05_min'],'frame_f1_drop_at_most_005':m['frame']['f1']>=GUARDS['frame_f1_min'],'normal_fp_no_worse':m['normal']['false_positive_videos']<=GUARDS['normal_fp_max'],'positive_empty_improved_3':m['positive_videos_without_candidate']<=GUARDS['positive_empty_max']}
    compare(report['statistical_promotion_checks'],guards,'unchanged guards');compare(report['statistical_promotion_passed'],all(guards.values()),'guard result')
    baseline=e.json(A.BASELINE,A.BASELINE_PIN)['grouped_v1_baseline'];compare(report['historical_grouped_v1_baseline'],baseline,'historical baseline binding')
    items={i:item for j in range(3) for i,item in heads[(5,j)].items()};require(len(items)==sum(len(heads[(5,j)]) for j in range(3))==77,'final inner3 coverage')
    final=choose_independent(records,list(range(77)),items,20279005,protocol['default_decoder'])
    compare(deployment,{'status':'complete','role':'direct_inner3_deployment_choice_not_validation','chosen':final,'uses_outer_probabilities':False,'uses_outer_metrics_for_selection':False,'inner_partitions':[{'fit':r['fit'],'validation':r['validation']} for r in inner_parts(records,outer,5)],'coverage_per_row':1,'interval_protocol_sha256':HEAD_RECEIPT},'final direct inner3 choice')
    full_path=HEAD/'fits/final_fullfit';full_files=[A.safe_path(e.root,full_path.with_suffix(s)).exists() for s in ('.json','.npz','.model.json')]
    if all(full_files):
        _,proof=load_head(e,protocol,records,outer,bases,legacy,raw,raw_names,Q,5,None);proofs.append(proof);fullfit={'status':'diagnostic_replayed'}
    else:
        require(not any(full_files),'partial fullfit state')
        skip=e.json(REPAIR/'final_fit_skipped.json')
        require(not final['config']['enabled'] and not all(guards.values()),'fullfit skipped without rejection evidence')
        for k,v in (('status','intentionally_not_run'),('planned_fullfit_heads',1),('actual_fullfit_heads',0),('formal_inner_outer_heads_complete',23),('default_promoted',False)):compare(skip[k],v,'fullfit skip/'+k)
        fullfit={'status':'intentionally_skipped_verified','reason':skip['reason']}
    paired=paired_differences(records,control,primary);e.recheck()
    return {**common,'status':'passed' if replay else 'incomplete','formal_head_selection_audited':True,'scope':'independent_nested_interval_quality_repair_audit','base_state_replay_required_for_acceptance':True,'metadata_reader_semantics':reader,'heads_replayed':len(proofs),'head_proofs':proofs,'fullfit':fullfit,'outer_choices':choices,'final_direct_inner3_choice':final,'primary_metrics':A.grouped_metrics(records,primary),'fixed_control_metrics':A.grouped_metrics(records,control),'paired_differences':paired,'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),'findings':[{'priority':'P1','kind':'promotion_rejected_not_audit_mismatch','detail':'Formal guard failure; keep unchanged default classifier/model and reject quality head.'}] if not all(guards.values()) else [],'consumed_artifact_sha256':dict(sorted(e.hashes.items()))}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--snapshot',action='store_true');parser.add_argument('--replay-base-states',action='store_true');args=parser.parse_args(argv)
    if args.output is None and not args.snapshot:parser.error('--output required unless --snapshot')
    stream=None
    if args.output is not None:
        target=args.output.expanduser().resolve()
        if target.parent!=(ROOT/REPAIR).resolve() or not target.name.startswith('independent_audit') or target.suffix!='.json':print('FAIL: use NEW repair/independent_audit*.json',file=sys.stderr);return 2
        try:stream=target.open('x',encoding='utf-8')
        except OSError as exc:print('FAIL: exclusive output: '+str(exc),file=sys.stderr);return 2
    start=time.perf_counter()
    try:
        result=run_audit(replay=args.replay_base_states,snapshot=args.snapshot);code=0 if result['status']=='passed' else 2
    except Exception as exc:result={'status':'failed','error':type(exc).__name__+': '+str(exc),'deployment_approved':False,'limitations':LIMITS};code=1
    result.update(schema_version='independent-baseline-interval-quality-audit-v1',finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,auditor_source_sha256=A.sha(Path(__file__).read_bytes()))
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n'
    if stream is not None:
        with stream:stream.write(text)
        print(json.dumps({'status':result['status'],'output':str(target),'exit_code':code}))
    else:print(text)
    return code

if __name__=='__main__':raise SystemExit(main())
