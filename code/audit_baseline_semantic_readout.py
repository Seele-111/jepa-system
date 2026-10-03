#!/usr/bin/env python3
"""Independent bounded semantic two-mode audit; no fitting or producer selectors.
Only NEW controlled-readout/independent_audit.json may be written, exclusively.
--snapshot performs source/partition/feature checks without formal pass or output.
"""
from __future__ import annotations
import argparse, ast, gzip, json, sys, time, traceback
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import audit_feature_blocks_v10 as A
import audit_baseline_interval_quality as I
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('output/baseline-targeted-semantic-readout'); RUN=OUT/'controlled-readout'; FEATURES=OUT/'load-repair'; PRECISION=RUN/'video-precision-repair'
MODES=('default_control','semantic_content'); ALGORITHMS=('rf','et')
KIND='semantic-content-readout-v1'; ROLE='training_content_only_equal_content_covariance'
PROTOCOL_PIN='a5d3bf8135ff4e2625297837633a4198f8f078a719eae104e8e879193af204d7'
RECEIPT='590cbab6a3e28b3ccc0d56bd06f45aecb1f41719e51da854e717b652c8ee9475'
CORE_PIN='719bc0ab61b0bf9a0cff0e1a4b1d491650f3bf181ec64e1a25e91a86143af5f2'
RUNNER_PIN='ed62e8f16a888f74eeda7bccc552822aebeb0c13a7ce1e256f31dd368b158fff'
MANIFEST_PIN='17abe917655b12050455e38335f7b68aa66fb7c67b27e1b4441a9908d2ed263c'
MANIFEST_RECEIPT='82eb8f9b6b650fff9118bdf434bdac272468acfa9241b520aebdf1e1ade51944'
EXTRACTION_PIN='3277e06659cd1b486b55f19031accd6dcb798901a12686e92a6dd32ecab70e59'
EXTRACTION_RECEIPT='521a6d118f2639841cd5c566397cf0e5071aea863d7974491e6ec4d183a72aef'
READER_PIN='172336704b4e92a28247de9174204b187536966ca2ad4dcd58475f729dd08a8a'
READER_RECEIPT='e53f17806dee53cf2126a61337b8546d6ad1d4940e96d0c9dfca79134dfa41cb'
READER_FILE_PIN='7ba367c5e6e00c2920101966ad32633604e56eac640ff2de75f36349231737bb'
REPORT_PIN='031a74c32b85ae671bff1d3b365875f9567fd1b74a46a757f54640029d3e1694'
DEPLOYMENT_PIN='5d2365d483a798ec38d6e21de2444930e1f4dc0fc4de02a385e595be3450373c'
HELPER_PIN='f1ad533abe2bc46e454f73bfdd0ab007afdd8d83924b3ce5696ae54fddc97700'
INTERVAL_HELPER_PIN='2ee268a1d77c1b3599cc6d4804c02d54297aeef9e7d4fb5dc75df4056d04955c'
GUARDS={k:I.GUARDS[k] for k in ('event_f1_03_min','event_f1_05_min','frame_f1_min','normal_fp_max','positive_empty_max')}
TOL=2e-6; MAX_BYTES=128*1024*1024
require,compare=A.require,A.compare
LIMITS=[
 '77 repeated-development rows/76 contents are not a new blind test; 14 normals and existing alias label conflicts remain.',
 'Exact SHA isolation is not near-duplicate, scene or generator isolation.',
 'Legacy outer/inner RF/ET/TCN states were not saved: metadata/cache hashes cannot replay or authenticate historical fits.',
 'New state-to-score replay and parameter/PCA consistency are not authenticated fit execution, partition/seed/weight-use proof. Unkeyed hashes are not authentication.',
 'PCA checks recompute supported train-only equal-SHA-content central covariance and top eigen projections, not historical execution; degenerate eigenspaces have no unique basis.',
 'Only trusted locally generated pickle may be replayed after hashing and bounded allowlisted loading. No third-party pickle safety claim.',
 'Encoder/source provenance is frozen receipt-bound; this bounded audit checks stored 1408 features/tubelet interpolation, not fresh pixels or a new checkpoint load/GPU execution. Encoder weights are not independently rehashed here.',
 'Base feature_view, augmentation, video summaries, TCN predictor and formal decoder are pinned audited dependencies. Semantic projection, tree traversal and selection/statistics are independent.',
 'Per-video positive-balanced classifier weighting is source-audited, not authenticated from fit traces. PCA has different equal-content weighting intentionally.',
 'Original final TCN video float32 quantization is faithfully reproduced; repaired float64 path is separately replayed. Original bitwise unchanged TCN claims are not valid.',
 'No fullfit, production export, GPU/product/latency check, new method or posthoc event-ledger-based selection.'
]

def expected_jobs(records,outer):
    jobs=[]
    for s in range(5):
        for j,p in enumerate(I.inner_parts(records,outer,s)):
            jobs.append(dict(id=f's{s}_inner{j}',scope=s,inner=j,role='inner_oof',fit=p['fit'],predict=p['validation'],excluded=outer[s],seed=20261002+s*53+j))
        jobs.append(dict(id=f's{s}_outer',scope=s,inner=None,role='outer_validation',fit=[i for i in range(len(records)) if i not in set(outer[s])],predict=outer[s],excluded=[],seed=20261002+s*53+99))
    for j,p in enumerate(I.inner_parts(records,outer,5)):
        jobs.append(dict(id=f'final_inner_{j}',scope=5,inner=j,role='final_deployment_inner_oof',fit=p['fit'],predict=p['validation'],excluded=[],seed=20273200+j))
    for p in jobs:
        I.partition(records,p['fit'],p['predict'],p['excluded'])
        require(p['fit'] and p['predict'] and sorted(p['fit']+p['predict']+p['excluded'])==list(range(len(records))),'job not exhaustive/nonempty')
    require(len(jobs)==23 and len({p['id'] for p in jobs})==23,'23 unique jobs required')
    return jobs

def source_check(e):
    core=ast.parse(e.read(Path('code/optimized_semantic_readout.py'),CORE_PIN).decode('utf-8'))
    functions={n.name:n for n in core.body if isinstance(n,ast.FunctionDef)}
    for name in ('_view','_semantic','feature_view_semantic','_pca'):
        require(not any(isinstance(n,ast.Constant) and n.value in ('labels','sha256','name','generator','row_id','index') for n in ast.walk(functions[name])),'inference label/identity access')
    actual=[]
    for n in ast.walk(functions['fit_semantic_member']):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('RandomForestClassifier','ExtraTreesClassifier'):
            actual.append((n.func.id,{k.arg:ast.unparse(k.value) for k in n.keywords}))
    expected=[('RandomForestClassifier',dict(n_estimators='160',max_depth='8',min_samples_leaf='10',max_features='0.5',n_jobs='4',random_state='seed')),
     ('ExtraTreesClassifier',dict(n_estimators='192',max_depth='9',min_samples_leaf='8',max_features='0.6',n_jobs='4',random_state='seed')),
     ('ExtraTreesClassifier',dict(n_estimators='128',max_depth='4',min_samples_leaf='2',max_features='0.75',class_weight="'balanced'",n_jobs='4',random_state='seed + 1100'))]
    key=lambda r:(r[0],int(r[1]['n_estimators']))
    compare(sorted(actual,key=key),sorted(expected,key=key),'fixed original learner source')
    e.read(Path('code/run_baseline_semantic_readout.py'),RUNNER_PIN)
    return {'core_sha256':CORE_PIN,'runner_sha256':RUNNER_PIN,'inference_label_identity_absent':True,
     'fixed_original_RF_ET_video_parameters':True,'frame_loss':'original per-video positive-balanced, source-audited; not authenticated execution',
     'PCA':'supported train-only equal SHA content, independent central covariance check'}

def protocol_check(e):
    e.read(Path('code/audit_feature_blocks_v10.py'),HELPER_PIN);e.read(Path('code/audit_baseline_interval_quality.py'),INTERVAL_HELPER_PIN)
    p=e.json(RUN/'protocol.json',PROTOCOL_PIN);I.signed(p,'receipt_sha256');compare(p['receipt_sha256'],RECEIPT,'protocol receipt')
    for k,v in (('schema_version','baseline-targeted-semantic-readout-v1'),('role','repeated_development_not_blind'),('rows',77),('content_groups',76),('modes',list(MODES)),('guards',GUARDS),('base_protocol_receipt',I.BASE_RECEIPT),('runtime_defaults_modified',False),
       ('members',[{'algorithm':'rf','weight':.4},{'algorithm':'et','weight':.4},{'cached_recipe':'rgb_motion_tcn','weight':.2}]),
       ('choice',{'order':list(MODES),'bootstrap_replicates':32,'q20':.2,'stable_utility':'.75 pooled + .25 grouped bootstrap q20','utility':'.35 F1@.3 + .65 F1@.5 - .20 normalFPR - .15 positiveEmptyRate','tie':'earlier declared mode; no decoder grid, calibration, candidate filter or label-dependent feature'})):
        compare(p[k],v,'protocol/'+k)
    for path,h in p['source_sha256'].items():
        local=I.relative(path);e.read(local,h);e.read(RUN/'sources'/local.name,h)
    base=e.json(I.OUT/'baseline_protocol.json',I.BASE_PIN);I.signed(base,'receipt_sha256');compare(base['receipt_sha256'],I.BASE_RECEIPT,'parent base receipt')
    compare(p['default_model_sha256'],base['default_sha256'],'default hash binding')
    for path,h in p['default_model_sha256'].items():e.read(I.relative(path),h)
    e.read(Path('code/optimized_detector.py'),p['default_detector_sha256'])
    default=e.json(Path('models/optimized_locator_v1.json'))
    compare([(m['recipe'],m['weight']) for m in default['members']],list(I.MEMBERS),'fixed blend');compare(p['default_decoder'],default['decoder'],'fixed decoder')
    prior=e.json(I.REPAIR/'report.json',p['baseline_diagnostic_report_sha256'])
    m=e.json(FEATURES/'feature_manifest.json',MANIFEST_PIN);I.signed(m,'receipt_sha256');compare(m['receipt_sha256'],MANIFEST_RECEIPT,'manifest receipt')
    compare(p['feature_manifest_sha256'],MANIFEST_PIN,'manifest pin');compare(p['feature_manifest_receipt'],MANIFEST_RECEIPT,'manifest binding')
    extraction=e.json(FEATURES/'extraction_protocol.json',EXTRACTION_PIN);I.signed(extraction,'receipt_sha256');compare(extraction['receipt_sha256'],EXTRACTION_RECEIPT,'extraction receipt')
    parent=e.json(OUT/'extraction_protocol.json',extraction['parent_protocol_sha256']);compare(m['profile'],parent,'feature profile');compare(extraction['profile'],parent,'repair extraction profile')
    compare(m['repair_protocol_receipt'],EXTRACTION_RECEIPT,'manifest extraction binding')
    require(m['status']=='complete' and parent['labels_used'] is False and parent['new_weights_downloaded'] is False and extraction['parent_features_completed']==0,'extraction boundary')
    e.read(A.GROUPED/'dataset_manifest.json',m['dataset_manifest_sha256'])
    for folder,sources in ((OUT,parent['source_sha256']),(FEATURES,extraction['source_sha256'])):
        for path,h in sources.items():local=I.relative(path);e.read(local,h);e.read(folder/'sources'/local.name,h)
    return p,base,m,prior

def feature_archive(a,entry,r,width=1408):
    require(set(a)=={'signals','fps','semantic_support','semantic_direct','sampled_frame_ids','tubelet_frame_ids','tubelet_vectors','source_sha256'},'semantic archive keys')
    x,s,d,ids,pairs,v=(a[k] for k in ('signals','semantic_support','semantic_direct','sampled_frame_ids','tubelet_frame_ids','tubelet_vectors'))
    t=r['frames'];require(x.dtype==v.dtype==np.float32 and x.shape==(t,width) and np.isfinite(x).all() and np.isfinite(v).all(),'semantic feature shape/value')
    require(s.dtype==d.dtype==np.bool_ and s.shape==d.shape==(t,) and np.all(~d|s),'semantic masks')
    require(ids.dtype==pairs.dtype==np.int64 and ids.ndim==1 and 2<=len(ids)<=32 and len(ids)%2==0 and pairs.shape==(len(ids)//2,2) and v.shape==(len(pairs),width),'tubelet shape/type')
    require(np.all(np.diff(ids)>0) and 0<=ids[0]<=ids[-1]<t and np.array_equal(ids.reshape(-1,2),pairs),'tubelet/sample identities')
    require(a['fps'].shape==() and float(a['fps'])==r['fps'] and a['source_sha256'].shape==() and str(a['source_sha256'])==r['sha256'],'feature FPS/SHA')
    compare(entry['sampled_frame_ids'],ids.tolist(),'sample ids');compare(entry['tubelet_frame_ids'],pairs.tolist(),'tubelet ids')
    axis=np.arange(t);require(np.array_equal(d,np.isin(axis,ids)) and np.array_equal(s,(axis>=ids[0])&(axis<=ids[-1])),'mask support/direct evidence')
    expected=np.stack([np.interp(axis,pairs.mean(1),v[:,j]) for j in range(width)],axis=1).astype(np.float32)
    error=float(np.max(np.abs(expected-x)));require(error<=TOL,'semantic interpolation')
    return dict(semantic=x,semantic_support=s,semantic_direct=d),error

def attach_features(e,records,m):
    require(len(m['videos'])==len(records),'feature rows');seen={};maximum=0.
    for i,(r,row) in enumerate(zip(records,m['videos'])):
        I.signed(row,'receipt_sha256')
        for k,v in (('index',i),('source_sha256',r['sha256']),('frames',r['frames']),('fps',r['fps']),('feature_width',1408),('file',f'features/v{i:03d}.npz'),('protocol_receipt',EXTRACTION_RECEIPT)):
            compare(row[k],v,'feature row/'+k)
        compare(row['alias_reused'],r['sha256'] in seen,'alias flag')
        compare(e.json(FEATURES/'features'/f'v{i:03d}.json'),row,'feature metadata/manifest')
        a=A.read_archive(e.read(FEATURES/row['file'],row['feature_sha256']),row['file']);fields,error=feature_archive(a,row,r);r.update(fields);maximum=max(maximum,error)
        if r['sha256'] in seen:compare(row['feature_sha256'],seen[r['sha256']],'alias feature identity')
        seen[r['sha256']]=row['feature_sha256']
    return dict(rows=len(records),content_groups=len(seen),width=1408,interpolation_max_abs_error=maximum)

def moments(records,train):
    require(train and len(set(train))==len(train),'PCA training coverage');groups=A.content_groups(records,train);width=records[train[0]]['semantic'].shape[1];samples={}
    for i in train:
        x,s=records[i]['semantic'],records[i]['semantic_support']
        require(x.dtype==np.float32 and x.ndim==2 and x.shape[1]==width and np.isfinite(x).all() and s.dtype==np.bool_ and s.shape==(len(x),) and s.any(),'PCA supported input')
        samples[i]=x[s].astype(np.float64)
    # Two-pass central covariance and SHA-grouped means: not producer raw second moment.
    mean=np.mean([np.mean([samples[i].mean(0) for i in group],axis=0) for group in groups],axis=0);cov=np.zeros((width,width),np.float64)
    for group in groups:
        for i in group:
            z=samples[i]-mean;cov+=(z.T@z)/(len(groups)*len(group)*len(z))
    cov=(cov+cov.T)*.5
    return dict(mean=mean,covariance=cov,eigenvalues=np.linalg.eigvalsh(cov)[::-1],fit=list(train),fit_content_sha256=sorted({records[i]['sha256'] for i in train}),content_groups=len(groups),supported_rows=sum(len(x) for x in samples.values()))

def check_pca(pca,proof,k=16):
    require(set(pca)=={'kind','mean','components','semantic_names','n_components','fit_role'},'PCA schema')
    compare(pca['kind'],KIND,'PCA kind');compare(pca['fit_role'],ROLE,'PCA role');require(type(pca['n_components']) is int and pca['n_components']==k,'PCA count')
    mean=np.asarray(pca['mean'],np.float32);c=np.asarray(pca['components'],np.float32);width=len(proof['mean'])
    require(mean.shape==(width,) and c.shape==(k,width) and np.isfinite(mean).all() and np.isfinite(c).all(),'PCA dimensions/value')
    compare(pca['semantic_names'],[f'sem/latent_{j:04d}' for j in range(width)],'PCA names')
    require(np.allclose(mean,proof['mean'],rtol=2e-6,atol=2e-6),'PCA supported train equal-content mean')
    c=c.astype(np.float64);cov=proof['covariance'];top=proof['eigenvalues'][:k];tol=2e-5*max(1.,float(np.linalg.norm(cov)))
    require(np.allclose(c@c.T,np.eye(k),rtol=0,atol=2e-5),'PCA orthogonality')
    projected=c@cov@c.T;residual=c@cov-top[:,None]*c
    require(np.allclose(projected,np.diag(top),rtol=2e-5,atol=tol) and np.linalg.norm(residual)<=tol,'PCA top covariance projection/residual')
    return {k:proof[k] for k in ('fit','fit_content_sha256','content_groups','supported_rows')}|dict(mean_max_error=float(np.max(np.abs(mean-proof['mean']))),projection_max_error=float(np.max(np.abs(projected-np.diag(top)))),eigen_residual=float(np.linalg.norm(residual)))

def view(r,pca):
    from optimized_feature_view import feature_view
    from optimized_locator import augment_signals,video_features
    base,bv,bnames=feature_view('corrected_motion_rf',r,None);x=r['semantic'];s,d=r['semantic_support'],r['semantic_direct']
    mean=np.asarray(pca['mean'],np.float32);c=np.asarray(pca['components'],np.float32)
    require(x.dtype==np.float32 and x.ndim==2 and len(x)==len(base) and mean.shape==(x.shape[1],) and c.ndim==2 and c.shape[1]==x.shape[1] and np.isfinite(x).all(),'view dimensions')
    require(s.dtype==d.dtype==np.bool_ and s.shape==d.shape==(len(x),) and np.all(~d|s),'view masks')
    projected=np.zeros((len(x),len(c)),np.float32);projected[s]=(x[s]-mean)@c.T
    aug,names=augment_signals(projected,[f'sem/pca_{j:04d}' for j in range(len(c))],r['fps']);flags=np.stack([d,s],axis=1).astype(np.float32)
    f=np.concatenate([base,aug,flags],axis=1).astype(np.float32);v=np.concatenate([bv,video_features(aug),video_features(flags)]).astype(np.float32)
    fn=list(bnames)+names+['sem/direct','sem/support'];vn=[]
    for key in ('motion','corrected'):vn.extend(f'base/{key}/{name}/{stat}' for stat in ('mean','std','p10','p90') for name in r[key+'_names'])
    for group in (names,['sem/direct','sem/support']):vn.extend(f'{name}/video_{stat}' for stat in ('mean','std','p10','p90') for name in group)
    require(len(fn)==f.shape[1] and len(vn)==len(v) and len(set(fn))==len(fn) and len(set(vn))==len(vn) and np.isfinite(f).all() and np.isfinite(v).all(),'view schema/value')
    return f,v,fn,vn

def prepare_forest(model,width,trees,depth):
    require(isinstance(model,dict) and set(model)=={'kind','n_features','trees'} and model['kind']=='forest' and type(model['n_features']) is int and model['n_features']==width,'forest schema/width')
    require(isinstance(model['trees'],list) and len(model['trees'])==trees and 1<=trees<=192,'forest tree count');prepared=[]
    for row in model['trees']:
        require(set(row)=={'left','right','feature','threshold','value'},'tree schema')
        a={k:np.asarray(v) for k,v in row.items()};n=len(a['left'])
        require(0<n<=2**(depth+1)-1 and all(v.shape==(n,) and v.dtype.kind in 'iuf' and np.isfinite(v).all() for v in a.values()),'tree bound/shape/finite')
        require(all(a[k].dtype.kind in 'iu' for k in ('left','right','feature')),'tree index dtype')
        l,r,f,t,v=(a[k] for k in ('left','right','feature','threshold','value'))
        require(np.all((v>=0)&(v<=1)),'tree probability range');seen=set();pending=[(0,0)]
        while pending:
            node,level=pending.pop();require(0<=node<n and node not in seen and level<=depth,'tree cycle/shared/outside/depth');seen.add(node)
            if l[node]==r[node]==-1:require(f[node]==t[node]==-2,'leaf sentinel')
            else:
                require(0<=l[node]<n and 0<=r[node]<n and l[node]!=r[node] and 0<=f[node]<width,'tree branch')
                pending.extend([(int(l[node]),level+1),(int(r[node]),level+1)])
        require(len(seen)==n,'unreachable tree node');prepared.append((l,r,f,t,v))
    return prepared,width,depth

def forest_scores(prepared,x,cast32=True):
    trees,width,depth=prepared
    require(x.dtype==np.float32 and x.ndim==2 and x.shape[1]==width and np.isfinite(x).all(),'forest input shape/value')
    result=np.zeros(len(x),np.float64)
    for l,r,f,t,v in trees:
        nodes=np.zeros(len(x),np.int64)
        for _ in range(depth+1):
            active=np.flatnonzero(l[nodes]!=-1)
            if not len(active):break
            current=nodes[active];nodes[active]=np.where(x[active,f[current]]<=t[current],l[current],r[current])
        require(np.all(l[nodes]==-1),'nonterminal tree replay');result+=v[nodes]
    result/=len(trees)
    return result.astype(np.float32) if cast32 else result

def native_bundle(model,algorithm,seed,width,video=False):
    I.tree_parameters(model,'corrected_motion_'+algorithm,seed,video)
    require(int(model.n_features_in_)==width and model.n_outputs_==1,'native dimensions/outputs')
    expected=np.random.RandomState(seed+1100 if video else seed).randint(np.iinfo(np.int32).max,size=len(model.estimators_))
    trees=[]
    for est,rng_seed in zip(model.estimators_,expected):
        require(est.random_state==int(rng_seed),'native per-tree seed')
        tree=est.tree_;a=np.asarray(tree.value)
        require(a.shape==(tree.node_count,1,2) and np.isfinite(a).all() and np.all(a>=0) and np.all(a[:,0,:].sum(1)>0),'native binary class masses')
        trees.append(dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),feature=tree.feature.tolist(),threshold=tree.threshold.tolist(),value=(a[:,0,1]/a[:,0,:].sum(1)).tolist()))
    return dict(kind='forest',n_features=width,trees=trees)

def read_states(e,path,digest):
    data=e.read(path,digest)  # Hash FIRST. Only the locally generated trusted envelope.
    with gzip.GzipFile(fileobj=BytesIO(data),mode='rb') as f:payload=f.read(MAX_BYTES+1)
    require(len(payload)<=MAX_BYTES,'native decompression bound');stream=BytesIO(payload);states=I.LocalStateUnpickler(stream).load()
    require(stream.tell()==len(payload) and isinstance(states,dict) and set(states)==set(ALGORITHMS),'native envelope/trailing bytes')
    return states

def read_portable(e,path,digest):
    data=e.read(path,digest)
    with gzip.GzipFile(fileobj=BytesIO(data),mode='rb') as f:payload=f.read(MAX_BYTES+1)
    require(len(payload)<=MAX_BYTES,'portable decompression bound');result=A.strict_json(payload,str(path))
    require(isinstance(result,dict) and set(result)==set(ALGORITHMS),'portable envelope');return result

def member_check(native,portable,algorithm,records,job):
    fields={'kind','algorithm','recipe','transform','frame_model','video_model','fit_content_sha256','full_fit'}
    require(set(native)==fields and set(portable)==fields|{'weight','boundary_models'} and portable['weight']==1. and portable['boundary_models'] is None,'member flat compatibility schema')
    for state in (native,portable):
        for k,v in (('kind',KIND),('algorithm',algorithm),('recipe','semantic_corrected_motion_'+algorithm),('fit_content_sha256',sorted({records[i]['sha256'] for i in job['fit']}))):compare(state[k],v,'member/'+k)
        require(state['full_fit'] is False,'unregistered full-fit member')
    tr=native['transform'];compare(tr,portable['transform'],'native/portable transform binding')
    require(set(tr)=={'feature_view','base_recipe','pca','frame_feature_names','video_feature_names'} and tr['feature_view']==KIND and tr['base_recipe']=='corrected_motion_rf','transform schema')
    x,v,fn,vn=view(records[job['predict'][0]],tr['pca']);compare(fn,tr['frame_feature_names'],'feature order');compare(vn,tr['video_feature_names'],'video order')
    require(np.array_equal(native['frame_model'].classes_,np.unique(np.concatenate([records[i]['labels'] for i in job['fit']])).astype(int)) and np.array_equal(native['video_model'].classes_,np.unique([int(records[i]['labels'].any()) for i in job['fit']])),'native fit class evidence')
    models={}
    for key,width,n,depth in (('frame_model',len(fn),160 if algorithm=='rf' else 192,8 if algorithm=='rf' else 9),('video_model',len(vn),128,4)):
        bundle=native_bundle(native[key],algorithm,job['seed'],width,key=='video_model');compare(bundle,portable[key],'native/portable exact trees')
        models[key]=(prepare_forest(bundle,width,n,depth),prepare_forest(portable[key],width,n,depth))
    return tr,models

def verify_job(e,records,job,proof=None):
    path=RUN/'fits'/job['id'];m=e.json(path.with_suffix('.json'));I.signed(m,'receipt_sha256')
    for k,v in job.items():compare(m[k],v,'new fit/'+k)
    I.partition(records,m['fit'],m['predict'],m['excluded']);compare(m['protocol_receipt'],RECEIPT,'fit protocol');compare(m['new_member_fits'],2,'member fits')
    for k in ('fit','predict','excluded'):compare(m[k+'_content_sha256'],sorted({records[i]['sha256'] for i in job[k]}),'fit exact SHA/'+k)
    require(set(m['artifacts'])=={'.npz','.pkl.gz','.portable.json.gz'},'fit artifact inventory')
    arrays=A.read_archive(e.read(path.with_suffix('.npz'),m['artifacts']['.npz']),str(path))
    require(set(arrays)=={f'{alg}_{k}_{i}' for alg in ALGORITHMS for k in ('frame','video') for i in job['predict']},'heldout NPZ coverage')
    native=read_states(e,path.with_suffix('.pkl.gz'),m['artifacts']['.pkl.gz']);portable=read_portable(e,path.with_suffix('.portable.json.gz'),m['artifacts']['.portable.json.gz'])
    if proof is None:proof=moments(records,job['fit'])
    outputs={};pca=[];errors={'native':0.,'portable':0.}
    for alg in ALGORITHMS:
        tr,models=member_check(native[alg],portable[alg],alg,records,job);pca.append(check_pca(tr['pca'],proof));fp={};vp={}
        for i in job['predict']:
            f,v=arrays[f'{alg}_frame_{i}'],arrays[f'{alg}_video_{i}']
            require(f.dtype==np.float32 and f.shape==(records[i]['frames'],) and v.dtype==np.float64 and v.shape==() and np.isfinite(f).all() and np.all((f>=0)&(f<=1)) and np.isfinite(v) and 0<=float(v)<=1,'heldout probability shape/dtype/value')
            x,y,fn,vn=view(records[i],tr['pca']);compare(fn,tr['frame_feature_names'],'heldout names');compare(vn,tr['video_feature_names'],'heldout video names')
            for slot,name in ((0,'native'),(1,'portable')):
                pf=forest_scores(models['frame_model'][slot],x);pv=float(forest_scores(models['video_model'][slot],y[None],slot==1)[0])
                error=max(float(np.max(np.abs(pf-f))),abs(pv-float(v)));require(error<=TOL,name+' heldout replay mismatch');errors[name]=max(errors[name],error)
            fp[i]=f;vp[i]=float(v)
        outputs[alg]=A.Probabilities(fp,vp)
    compare(native['rf']['transform']['pca'],native['et']['transform']['pca'],'paired train PCA')
    require(type(m['portable_max_error']) in (int,float) and np.isfinite(m['portable_max_error']) and 0<=m['portable_max_error']<=TOL,'attested portable error')
    return outputs,dict(id=job['id'],fit=job['fit'],predict=job['predict'],excluded=job['excluded'],seed=job['seed'],native_members_replayed=2,portable_members_replayed=2,heldout_member_video_rows=2*len(job['predict']),native_max_abs_error=errors['native'],portable_max_abs_error=errors['portable'],metadata_portable_max_error=m['portable_max_error'],fit_receipt=m['receipt_sha256'],PCA=pca,artifact_sha256=m['artifacts'])

def legacy_scores(e,records,outer):
    result={}
    for scope,(inside,outside,sources) in I.load_legacy(e,records,outer).items():
        path=A.GROUPED/'grouped_training/folds'/f'{scope}_rgb_motion_tcn.json';m=e.json(path);a=A.read_archive(e.read(path.with_suffix('.npz'),m['npz_sha256']),str(path));tcn=[]
        for prefix,ids in (('inner',sorted(inside.frame)),('outer',sorted(outside.frame))):tcn.append(I.validate_scores({k[len(prefix)+1:]:v for k,v in a.items() if k.startswith(prefix+'_')},records,ids))
        result[scope]=(inside,outside,*tcn,sources)
    return result

def final_base(e,records,job,base):
    old=next(j for j in I.expected_jobs(records,base['legacy_outer5_inner3']) if j['id']==job['id']);path=I.BASE/'predictions'/job['id'];m=e.json(path.with_suffix('.json'));I.signed(m,'metadata_sha256')
    for k,v in old.items():compare(m[k],v,'saved final base/'+k)
    for k in ('fit','predict','seed'):compare(m[k],job[k],'paired final base/'+k)
    for k in ('fit','predict','excluded'):compare(m[k+'_content_sha256'],sorted({records[i]['sha256'] for i in old[k]}),'final base SHA')
    compare(m['baseline_protocol_sha256'],I.BASE_RECEIPT,'final base receipt');compare(m['model_state_path'],(I.BASE/'models'/(job['id']+'.pkl.gz')).as_posix(),'final base path')
    control=I.validate_scores(A.read_archive(e.read(path.with_suffix('.npz'),m['npz_sha256']),str(path)),records,job['predict'])
    replay=I.replay_base(e,m,control,records)  # THREE final states only; never all 57 old jobs.
    states=I.read_state(e,I.relative(m['model_state_path']),m['model_state_sha256'])
    from optimized_feature_view import feature_view
    from optimized_temporal_head import predict_temporal
    frame={};raw={};rounded={};max_control=0.
    for i in job['predict']:
        video={}
        for recipe,_ in I.MEMBERS:
            state=states[recipe];x,v,n=feature_view(recipe,records[i],state['transform']['pca']);compare(n,state['transform']['frame_feature_names'],'saved final feature order')
            video[recipe]=float(state['video_model'].predict_proba(v[None])[:,1][0])
            if recipe=='rgb_motion_tcn':frame[i]=predict_temporal(state['frame_model'],x,records[i]['fps'])
        raw[i]=video['rgb_motion_tcn'];rounded[i]=float(np.float32(raw[i]));max_control=max(max_control,abs(sum(w*video[r] for r,w in I.MEMBERS)-control.video[i]))
    require(max_control<=2e-12,'native final video blend/control mismatch')
    return (control,A.Probabilities(frame,rounded),A.Probabilities(frame,raw)),replay|dict(native_control_video_replay_max_error=max_control,TCN_video_changed_rows=sum(raw[i]!=rounded[i] for i in raw),TCN_video_max_difference=max(abs(raw[i]-rounded[i]) for i in raw))

def fuse(semantic,tcn,ids):
    require(set(semantic)==set(ALGORITHMS) and all(set(p.frame)==set(p.video)==set(ids) for p in semantic.values()) and set(ids)<=set(tcn.frame) and set(ids)<=set(tcn.video),'fusion coverage')
    return A.Probabilities({i:(.4*semantic['rf'].frame[i]+.4*semantic['et'].frame[i]+.2*tcn.frame[i]).astype(np.float32) for i in ids},{i:.4*semantic['rf'].video[i]+.4*semantic['et'].video[i]+.2*tcn.video[i] for i in ids})

def decode_scores(records,ids,scores,decoder):
    from optimized_locator import decode
    require(set(scores.frame)==set(scores.video)==set(ids),'decode coverage')
    return {i:decode(scores.frame[i],records[i]['fps'],scores.video[i],decoder) for i in ids}

def choose_independent(records,ids,predictions,seed):
    require(ids and len(set(ids))==len(ids) and set(predictions)==set(MODES) and all(set(p)==set(ids) for p in predictions.values()),'two-mode selection coverage')
    scoped=A.RestrictedRecords(records,ids);normal=np.asarray([not scoped[i]['labels'].any() for i in ids],np.int64);positive=1-normal
    boot=A.bootstrap_counts(scoped,ids,seed);bn,bp=boot@normal,boot@positive;rows=[]
    for ordinal,mode in enumerate(MODES):
        mat=np.stack([A.row_statistics(predictions[mode][i],scoped[i]['labels']) for i in ids]);total=mat.sum(0);pooled=float(I.utility(total,int(normal.sum()),int(positive.sum())))
        q20=float(np.percentile([I.utility(s,int(n),int(p)) for s,n,p in zip(boot@mat,bn,bp)],20))
        rows.append(dict(mode=mode,ordinal=ordinal,pooled_utility=pooled,bootstrap_q20=q20,stable_utility=.75*pooled+.25*q20,inner_diagnostic_metrics_not_validation=A.metric_dict(total,int(normal.sum()))))
    return max(rows,key=lambda row:(row['stable_utility'],-row['ordinal']))|{'configurations':rows}

def guards(m):
    return dict(event_f1_03_no_worse=m['iou_0.3']['f1']>=GUARDS['event_f1_03_min'],event_f1_05_improved_02=m['iou_0.5']['f1']>=GUARDS['event_f1_05_min'],frame_f1_drop_at_most_005=m['frame']['f1']>=GUARDS['frame_f1_min'],normal_fp_no_worse=m['normal']['false_positive_videos']<=GUARDS['normal_fp_max'],positive_empty_improved_3=m['positive_videos_without_candidate']<=GUARDS['positive_empty_max'])

def result(records,pred):return dict(metrics=A.grouped_metrics(records,pred),predictions=A.prediction_rows(records,pred))
def serialized(pred):return {str(i):[list(pair) for pair in spans] for i,spans in pred.items()}

def verify_results(e,p,records,jobs,semantic,legacy,final,prior):
    report=e.json(RUN/'report.json',REPORT_PIN);deploy=e.json(RUN/'deployment_selection.json',DEPLOYMENT_PIN)
    for k,v in (('schema_version','baseline-targeted-semantic-readout-results-v1'),('status','complete'),('role','repeated_development_not_blind'),('protocol_receipt',RECEIPT),('default_promoted',False),('single_full_fit_model_is_not_OOF',True)):compare(report[k],v,'report/'+k)
    require(report['default_promoted'] is False and len(report['folds'])==5,'report fold/default contract')
    fixed={mode:{} for mode in MODES};primary={};choices=[]
    for s in range(5):
        train=sorted(legacy[s][0].frame);val=sorted(legacy[s][1].frame)
        members={alg:I.merge_scores([semantic[f's{s}_inner{j}'][alg] for j in range(3)],train) for alg in ALGORITHMS}
        scores=fuse(members,legacy[s][2],train)
        preds={MODES[0]:decode_scores(records,train,legacy[s][0],p['default_decoder']),MODES[1]:decode_scores(records,train,scores,p['default_decoder'])}
        chosen=choose_independent(records,train,preds,20273000+s)
        enabled=fuse(semantic[f's{s}_outer'],legacy[s][3],val)
        held={MODES[0]:decode_scores(records,val,legacy[s][1],p['default_decoder']),MODES[1]:decode_scores(records,val,enabled,p['default_decoder'])}
        compare(report['folds'][s],dict(fold=s,train=train,validation=val,chosen=chosen,predictions={mode:serialized(held[mode]) for mode in MODES}),'independent outer/'+str(s))
        require(not set(primary)&set(val),'outer duplicate OOF');primary.update(held[chosen['mode']]);choices.append(chosen)
        for mode in MODES:fixed[mode].update(held[mode])
    require(set(primary)==set(range(len(records))),'outer missing OOF')
    compare(report['primary_nested_representation_selection'],result(records,primary),'independent primary metrics/predictions')
    compare(report['fixed_default_control'],result(records,fixed[MODES[0]]),'independent control')
    compare(report['fixed_semantic_content_diagnostic_not_selected_deployment'],result(records,fixed[MODES[1]]),'independent fixed semantic')
    compare(report['fixed_default_control'],prior['fixed_default_control'],'unchanged default control from interval branch')
    compare(report['historical_grouped_v1_baseline'],prior['historical_grouped_v1_baseline'],'historical baseline binding')
    checks=guards(result(records,primary)['metrics']['all']);compare(report['statistical_promotion_checks'],checks,'unchanged guards');compare(report['statistical_promotion_passed'],all(checks.values()),'guard result')
    ids=list(range(len(records)));fj=[j for j in jobs if j['scope']==5]
    fc=I.merge_scores([final[j['id']][0] for j in fj],ids)
    old_parts=[fuse(semantic[j['id']],final[j['id']][1],j['predict']) for j in fj]
    new_parts=[fuse(semantic[j['id']],final[j['id']][2],j['predict']) for j in fj]
    old=I.merge_scores(old_parts,ids);new=I.merge_scores(new_parts,ids)
    before={MODES[0]:decode_scores(records,ids,fc,p['default_decoder']),MODES[1]:decode_scores(records,ids,old,p['default_decoder'])}
    after={MODES[0]:before[MODES[0]],MODES[1]:decode_scores(records,ids,new,p['default_decoder'])}
    chosen=choose_independent(records,ids,before,20273111);corrected=choose_independent(records,ids,after,20273111)
    partitions=[dict(fit=j['fit'],validation=j['predict'],seed=j['seed']) for j in fj]
    compare(deploy,dict(status='complete',role='final_inner_OOF_deployment_selection_optimistic_not_validation',protocol_receipt=RECEIPT,inner_partitions=partitions,chosen=chosen,predictions={mode:serialized(before[mode]) for mode in MODES},statistical_promotion_passed=all(checks.values()),default_promoted=False,outer_best_not_used=True),'independent original final')
    skip=e.json(RUN/'final_fit_skipped.json')
    require(not all(checks.values()) or chosen['mode']==MODES[0],'fullfit skip without rejection/default evidence')
    compare(skip,dict(status='skipped',reason='statistical guards failed' if not all(checks.values()) else 'inner OOF selected unchanged baseline',planned_unused_full_member_fits=2,default_promoted=False,export_created=False),'original skipfullfit')
    return dict(outer_choices=choices,original_final_choice=chosen,precision_repaired_final_choice=corrected,
     primary_metrics=result(records,primary)['metrics'],fixed_default_control_metrics=result(records,fixed[MODES[0]])['metrics'],fixed_semantic_metrics=result(records,fixed[MODES[1]])['metrics'],
     statistical_promotion_checks=checks,statistical_promotion_passed=all(checks.values()),fullfit='intentionally_skipped_verified',
     primary_minus_control_paired=I.paired_differences(records,fixed[MODES[0]],primary),fixed_semantic_minus_control_paired=I.paired_differences(records,fixed[MODES[0]],fixed[MODES[1]])),(before,after,old,new,partitions)

def close_probability(actual,expected,label):
    require(type(actual) in (int,float) and np.isfinite(actual) and abs(float(actual)-float(expected))<=2e-12,'precision '+label+' mismatch')

def check_precision(e,records,jobs,semantic,final,proofs,baseproofs,res,context):
    reader=e.json(PRECISION/'selection_reader_receipt.json',READER_FILE_PIN);I.signed(reader,'receipt_sha256')
    for k,v in (('schema_version','semantic-final-tcn-video-precision-reader-v1'),('receipt_sha256',READER_RECEIPT),('parent_protocol_receipt',RECEIPT),('original_runner_sha256',RUNNER_PIN),('adapter_source_sha256',READER_PIN),('parent_report_sha256',REPORT_PIN),('original_deployment_selection_sha256',DEPLOYMENT_PIN),('scope','final inner TCN video only: native predict_proba float64, instead of float32 cast'),('additional_member_fits',0),('additional_feature_extractions',0),('old_outputs_preserved',True)):
        compare(reader[k],v,'precision receipt/'+k)
    compare(reader['unchanged'],['46 fitted semantic members','23 fit partitions and seeds','PCA states','all frame probabilities','5 outer choices/predictions/metrics/guards','original default decoder','two-mode selection rule','default runtime'],'precision unchanged scope')
    source=e.read(Path('code/select_baseline_semantic_precision.py'),READER_PIN);e.read(PRECISION/'sources/select_baseline_semantic_precision.py',READER_PIN);e.read(PRECISION/'sources/run_baseline_semantic_readout.py',RUNNER_PIN)
    tree=ast.parse(source.decode('utf-8'));positive=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='original_positive')
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='astype' for n in ast.walk(positive)),'precision adapter still casts video')
    r=e.json(PRECISION/'deployment_selection.json');I.signed(r,'receipt_sha256')
    a=A.read_archive(e.read(PRECISION/'final_video_probabilities.npz',r['corrected_score_sha256']),'precision scores')
    require(set(a)=={f'{kind}_{i}' for i in range(len(records)) for kind in ('tcn_video_original_precision','enabled_video_original_precision')},'precision NPZ coverage')
    before,after,old,new,parts=context;maximum=0.;changed=0;fused=0.;binding=[]
    for job in jobs:
        if job['scope']!=5:continue
        name=job['id'];control,rounded,raw=final[name]
        proof=next(p for p in proofs if p['id']==name);bp=next(p for p in baseproofs if p['id']==name)
        binding.append(dict(job=name,base_model_state_sha256=bp['model_state_sha256'],semantic_fit_receipt=proof['fit_receipt']))
        for i in job['predict']:
            for key,value in ((f'tcn_video_original_precision_{i}',raw.video[i]),(f'enabled_video_original_precision_{i}',new.video[i])):
                v=a[key];require(v.dtype==np.float64 and v.shape==() and np.isfinite(v) and 0<=float(v)<=1,'precision scalar dtype/shape/value');close_probability(float(v),value,key)
            require(np.array_equal(old.frame[i],new.frame[i]) and np.array_equal(rounded.frame[i],raw.frame[i]),'precision repair changed frames')
            difference=abs(raw.video[i]-rounded.video[i]);maximum=max(maximum,difference);changed+=int(difference!=0);fused=max(fused,abs(old.video[i]-new.video[i]))
    changes={mode:[i for i in range(len(records)) if before[mode][i]!=after[mode][i]] for mode in MODES}
    evidence=dict(TCN_video_changed_rows=changed,TCN_video_max_difference=maximum,fused_video_max_difference=fused,
       native_control_video_replay_max_error=max(p['native_control_video_replay_max_error'] for p in baseproofs),changed_intervals=changes,
       choice_changed=res['original_final_choice']['mode']!=res['precision_repaired_final_choice']['mode'],frames_identical=True,all_outer_results_preserved=True,additional_member_fits=0)
    for key in ('TCN_video_max_difference','fused_video_max_difference','native_control_video_replay_max_error'):close_probability(r['evidence'][key],evidence[key],key)
    expected=dict(status='complete',role='final_inner_OOF_deployment_selection_optimistic_not_validation',parent_protocol_receipt=RECEIPT,precision_reader_receipt=READER_RECEIPT,inner_partitions=parts,chosen=res['precision_repaired_final_choice'],predictions={mode:serialized(after[mode]) for mode in MODES},model_bindings=binding,evidence=evidence,statistical_promotion_passed=res['statistical_promotion_passed'],default_promoted=False,outer_best_not_used=True,corrected_score_sha256=r['corrected_score_sha256'],receipt_sha256=r['receipt_sha256'])
    compare(r,expected,'independent precision repair report')
    skip=e.json(PRECISION/'final_fit_skipped.json')
    compare(skip,dict(status='skipped',reason='statistical guards failed' if not res['statistical_promotion_passed'] else 'inner OOF selected baseline',planned_unused_full_member_fits=2,default_promoted=False,export_created=False,precision_reader_receipt=READER_RECEIPT),'precision skipfullfit')
    return dict(status='verified',reader_receipt_sha256=READER_RECEIPT,adapter_source_sha256=READER_PIN,model_bindings=binding,evidence=evidence,skipfullfit='verified',original_P2='confirmed_then_repaired_without_fit',probability_atol=2e-12)

def run_audit(root=ROOT,snapshot=False):
    e=A.Evidence(root);p,base,m,prior=protocol_check(e);records,outer,aliases=A.load_records(e)
    compare(base['legacy_outer5_inner3'],outer,'legacy SHA outer partitions');jobs=expected_jobs(records,outer);compare(p['jobs'],jobs,'23 independent jobs')
    features=attach_features(e,records,m);source=source_check(e)
    missing=[(RUN/name).as_posix() for name in ('training_report.json','report.json','deployment_selection.json','final_fit_skipped.json') if not A.safe_path(e.root,RUN/name).is_file()]
    missing += [(PRECISION/name).as_posix() for name in ('selection_reader_receipt.json','deployment_selection.json','final_video_probabilities.npz','final_fit_skipped.json') if not A.safe_path(e.root,PRECISION/name).is_file()]
    common=dict(rows=len(records),content_groups=76,protocol_receipt_sha256=RECEIPT,aliases_retained=aliases,feature_checks=features,source_review=source,deployment_approved=False,limitations=LIMITS)
    if snapshot:
        e.recheck();return common|dict(status='pending',scope='snapshot_not_formal_replay_selection',missing_formal_evidence=missing,formal_selection_audited=False,native_semantic_members_replayed=0,consumed_artifact_sha256=dict(sorted(e.hashes.items())))
    require(not missing,'missing formal evidence: '+', '.join(missing))
    expected={j['id']+s for j in jobs for s in ('.json','.npz','.pkl.gz','.portable.json.gz')}
    require({p.name for p in A.safe_path(e.root,RUN/'fits').iterdir()}==expected,'missing/unregistered/partial/fullfit artifacts')
    legacy=legacy_scores(e,records,outer);semantic={};proofs=[]
    for job in jobs:semantic[job['id']],proof=verify_job(e,records,job);proofs.append(proof)
    training=e.json(RUN/'training_report.json')
    require(training['status']=='complete' and training['new_member_fits']==46 and len(training['results'])==23 and {p['id'] for p in training['results']}=={j['id'] for j in jobs},'23 jobs/46 members training summary')
    for row in training['results']:compare(row['portable_max_error'],next(p['metadata_portable_max_error'] for p in proofs if p['id']==row['id']),'training parity binding')
    compare(training['maximum_portable_error'],max(p['metadata_portable_max_error'] for p in proofs),'training max parity')
    final={};baseproofs=[]
    for job in jobs:
        if job['scope']==5:final[job['id']],proof=final_base(e,records,job,base);baseproofs.append(proof)
    results,context=verify_results(e,p,records,jobs,semantic,legacy,final,prior)
    precision=check_precision(e,records,jobs,semantic,final,proofs,baseproofs,results,context)
    e.recheck()
    return common|results|dict(status='passed',scope='independent_frozen_semantic_two_mode_and_precision_audit',formal_selection_audited=True,
       native_semantic_members_replayed=46,portable_semantic_members_replayed=46,semantic_jobs_replayed=23,
       heldout_semantic_member_video_rows=sum(p['heldout_member_video_rows'] for p in proofs),native_semantic_max_abs_error=max(p['native_max_abs_error'] for p in proofs),portable_semantic_max_abs_error=max(p['portable_max_abs_error'] for p in proofs),
       job_proofs=proofs,final_TCN_states_replayed=3,final_saved_base_replay=baseproofs,precision_repair=precision,
       findings=[dict(priority='P2',status='repaired_verified',kind='original_final_TCN_video_extra_float32_cast',detail='Original 77 scalar values quantized; source/receipt-bound selection reader restores native float64 without new fits. Original report preserved; not a bitwise unchanged original-TCN claim.')],
       full_legacy_57_base_reaudit_performed=False,consumed_artifact_sha256=dict(sorted(e.hashes.items())))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args(argv)
    if args.output is None and not args.snapshot:parser.error('--output required unless --snapshot')
    stream=None
    if args.output is not None:
        target=args.output.expanduser().resolve()
        if target!=(ROOT/RUN/'independent_audit.json').resolve():print('FAIL: only NEW controlled-readout/independent_audit.json authorized',file=sys.stderr);return 2
        try:stream=target.open('x',encoding='utf-8')  # No mkdir; never reserve the producer freeze namespace.
        except OSError as exc:print('FAIL: exclusive output: '+str(exc),file=sys.stderr);return 2
    start=time.perf_counter()
    try:report=run_audit(snapshot=args.snapshot);code=0 if report['status']=='passed' else 2
    except Exception as exc:
        report=dict(status='failed',error=type(exc).__name__+': '+str(exc),deployment_approved=False,limitations=LIMITS,diagnostic_locations=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)[-6:]]);code=1
    report.update(schema_version='independent-baseline-semantic-readout-audit-v1',finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,auditor_source_sha256=A.sha(Path(__file__).read_bytes()))
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n'
    if stream is not None:
        with stream:stream.write(text)
        print(json.dumps(dict(status=report['status'],output=str(target),exit_code=code)))
    else:print(text)
    return code

if __name__=='__main__':raise SystemExit(main())
