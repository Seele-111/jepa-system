"""Video existence gate from STRICT out-of-content temporal evidence, experiment v6."""
from __future__ import annotations
import numpy as np
from optimized_locator import rolling_mean, spans, video_features, export_model, portable_predict

GATE_KINDS = ('ridge_logistic_v6', 'shallow_forest_v6')


def gate_features(frame_probability, video_probability, record):
    p = np.asarray(frame_probability, np.float32)
    fps = float(record['fps']); vp = float(video_probability)
    if p.ndim != 1 or len(p) != record['frames'] or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isfinite(fps) or fps<=0 or not np.isfinite(vp) or not 0<=vp<=1:
        raise ValueError('invalid video evidence features')
    q = np.percentile(p, [10, 25, 50, 75, 90, 99])
    values = [vp, float(p.mean()), float(p.std()), *q.tolist(), float(p.max()),
              float(np.mean(np.sort(p)[-max(1,int(np.ceil(.1*len(p)))):])), float(q[-1]-q[0])]
    names = ['raw_video_probability','frame_mean','frame_std'] + [f'frame_p{k}' for k in [10,25,50,75,90,99]] + ['frame_max','frame_top10_mean','frame_contrast_p99_p10']
    for seconds in [.2, .6]:
        z = rolling_mean(p,max(1,int(round(seconds*fps))))
        values += [float(z.max()),float(np.percentile(z,90))]
        names += [f'frame_mean_{seconds}s_max',f'frame_mean_{seconds}s_p90']
    for threshold in [.3,.5,.7]:
        runs=spans(p>=threshold)
        values += [float(np.mean(p>=threshold)),float(max([e-s+1 for s,e in runs],default=0)/fps),float(len(runs))]
        names += [f'frame_mass_above_{threshold}',f'frame_longest_seconds_above_{threshold}',f'frame_components_above_{threshold}']
    # Semantic error magnitudes and physical units, not filename/generator hashes.
    corrected=np.asarray(record['corrected'],np.float32)
    if corrected.shape!=(len(p),14) or not np.isfinite(corrected).all():raise ValueError('invalid gate JEPA features')
    for j in [0,2,3,4,7,9,10,11]:
        x=corrected[:,j];values += [float(x.mean()),float(np.percentile(x,90))]
        names += [f'corrected_{j}_mean',f'corrected_{j}_p90']
    result=np.asarray(values,np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite gate features')
    return result,names


def fit_gate(kind, records, train, fp, vp, seed):
    if kind not in GATE_KINDS:raise ValueError('unknown gate kind')
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from optimized_compact_features_v4 import content_weights
    features=[gate_features(fp[i],vp[i],records[i]) for i in train]
    names=features[0][1]
    if any(v[1]!=names for v in features):raise ValueError('gate schema mismatch')
    x=np.stack([v[0] for v in features]);y=np.asarray([int(np.any(records[i]['labels'])) for i in train])
    cw=content_weights(records,train);weights=np.asarray([cw[i] for i in train])
    if len(np.unique(y))!=2:raise ValueError('gate requires both video classes')
    negative=weights[y==0].sum();positive=weights[y>0].sum()
    weights[y>0]*=negative/positive;weights*=len(weights)/weights.sum()
    if kind=='ridge_logistic_v6':
        mean=np.average(x,axis=0,weights=weights);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights));scale=np.maximum(scale,1e-6)
        z=np.clip((x-mean)/scale,-8,8).astype(np.float32)
        model=LogisticRegression(C=.3,solver='lbfgs',max_iter=1000,random_state=seed)
        model.fit(z,y,sample_weight=weights)
        state={'kind':kind,'feature_names':names,'mean':mean.tolist(),'scale':scale.tolist(),
               'coefficients':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])}
    else:
        model=ExtraTreesClassifier(n_estimators=160,max_depth=3,min_samples_leaf=3,
                                   max_features=.75,n_jobs=2,random_state=seed)
        model.fit(x,y,sample_weight=weights)
        state={'kind':kind,'feature_names':names,'model':export_model(model)}
    state['fit_content_sha256']=sorted({records[i]['sha256'] for i in train})
    return state


def apply_gate(state, frame_probability, video_probability, record):
    x,names=gate_features(frame_probability,video_probability,record)
    if state.get('feature_names')!=names:raise ValueError('video gate feature order mismatch')
    if state.get('kind')=='ridge_logistic_v6':
        mean=np.asarray(state['mean'],np.float32);scale=np.asarray(state['scale'],np.float32);coef=np.asarray(state['coefficients'],np.float64)
        if mean.shape!=x.shape or scale.shape!=x.shape or coef.shape!=x.shape or not all(np.isfinite(v).all() for v in [mean,scale,coef]) or np.any(scale<=0) or not np.isfinite(state['intercept']):raise ValueError('invalid logistic gate state')
        z=np.clip((x-mean)/scale,-8,8);value=float(z@coef+state['intercept'])
        return float(1/(1+np.exp(-np.clip(value,-700,700))))
    if state.get('kind')=='shallow_forest_v6':
        return float(portable_predict(state['model'],x[None])[0])
    raise ValueError('unsupported video gate state')
