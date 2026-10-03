"""Frozen SHA-grouped data access and four new CPU fitting recipes."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from build_optimization_dataset import make_folds
from optimized_feature_view_v2 import feature_view_v2
from run_algorithm_optimization import fit_predict as fit_v1
from run_optimization_selection import load_dataset, SINGLE_RECIPES

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output'/'algorithm-opt-2026-10-02'
NEW=ROOT/'output'/'algorithm-opt-2026-10-02-v2'
NEW_RECIPES=('noglobal_corrected_motion_et','noglobal_corrected_motion_rf','local_corrected_motion_et','local_motion_et')
RECIPES=tuple(SINGLE_RECIPES)+NEW_RECIPES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_folds(records, indices, n_folds, seed):
    groups={}
    for i in indices:groups.setdefault(records[i]['sha256'],[]).append(i)
    ordered=list(groups.values());representatives=[records[g[0]] for g in ordered]
    folds=make_folds(representatives,n_folds,seed)
    result=[sorted(i for j in f for i in ordered[j]) for f in folds]
    assert sorted(sum(result,[]))==sorted(indices)
    for a in range(n_folds):
        for b in range(a):assert not {records[i]['sha256'] for i in result[a]} & {records[i]['sha256'] for i in result[b]}
    return result


def load_grouped():
    old,records,profiles,hashes=load_dataset(OLD)
    manifest=json.loads((NEW/'dataset_manifest.json').read_text('utf-8'))
    assert manifest['rows']==old['rows']
    local_manifest=json.loads((NEW/'local_motion/manifest.json').read_text('utf-8'))
    profile=local_manifest['profile'];names=json.loads((NEW/'local_motion/feature_names.json').read_text('utf-8'))
    assert profile['extractor_sha256']==digest(ROOT/'code/optimized_local_motion.py')
    assert local_manifest['status']=='complete'
    rows=local_manifest['videos']
    assert len(rows)==len(records)
    for i,(r,entry) in enumerate(zip(records,rows)):
        assert entry['row_id']==i and entry['source_sha256']==r['sha256']
        path=NEW/'local_motion'/entry['file'];assert digest(path)==entry['feature_sha256']
        with np.load(path,allow_pickle=False) as a:
            values=np.asarray(a['signals'],np.float32);mask=np.asarray(a['feature_valid'],bool)
            assert values.shape==mask.shape==(r['frames'],len(names))
            assert float(a['fps'])==r['fps'] and np.isfinite(values).all()
            r['local']=np.concatenate([values,mask.astype(np.float32)],axis=1)
            r['local_names']=['local/'+n for n in names]+['local/'+n+'_support_flag' for n in names]
    profiles['local']=profile
    for fold,val in enumerate(manifest['outer_folds']):
        train=[i for i in range(len(records)) if i not in set(val)]
        assert not {records[i]['sha256'] for i in train}&{records[i]['sha256'] for i in val}
    return manifest,records,profiles


def fit_member(recipe,records,train,predict,seed):
    assert not {records[i]['sha256'] for i in train}&{records[i]['sha256'] for i in predict} or set(train)==set(predict), 'content leakage'
    if recipe not in NEW_RECIPES:return fit_v1(recipe,records,train,predict,seed)
    from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
    views=[feature_view_v2(recipe,r) for r in records]
    frames=[v[0] for v in views];video=np.stack([v[1] for v in views]);names=views[0][2]
    assert all(v[2]==names for v in views)
    x=np.concatenate([frames[i] for i in train]);y=np.concatenate([records[i]['labels'] for i in train])
    weights=np.concatenate([np.full(len(frames[i]),1/len(frames[i])) for i in train])
    positive=weights[y>0].sum();negative=weights[y==0].sum()
    weights=np.where(y>0,weights*negative/max(1e-8,positive),weights);weights*=len(weights)/weights.sum()
    if recipe.endswith('_rf'):
        model=RandomForestClassifier(n_estimators=160,max_depth=8,min_samples_leaf=10,max_features=.5,n_jobs=2,random_state=seed)
    else:model=ExtraTreesClassifier(n_estimators=192,max_depth=9,min_samples_leaf=8,max_features=.6,n_jobs=2,random_state=seed)
    model.fit(x,y,sample_weight=weights)
    video_model=ExtraTreesClassifier(n_estimators=128,max_depth=4,min_samples_leaf=2,max_features=.75,class_weight='balanced',n_jobs=2,random_state=seed+1100)
    vy=np.array([int(np.any(r['labels'])) for r in records]);video_model.fit(video[train],vy[train])
    def positive(m,x):
        probabilities=m.predict_proba(x)
        matches=np.flatnonzero(m.classes_==1)
        return probabilities[:,int(matches[0])].astype(np.float32) if len(matches) else np.zeros(len(x),np.float32)
    fp={i:positive(model,frames[i]) for i in predict};vp={i:float(p) for i,p in zip(predict,positive(video_model,video[predict]))}
    transform={'recipe':recipe,'frame_feature_names':names,'pca':None,'feature_view':'shared-label-free-v2'}
    return fp,vp,(model,video_model,transform,frames,video,{'boundary_models':None,'boundary_probabilities':None})
