#!/usr/bin/env python3
"""Portable learned temporal locator. Inference needs only NumPy, not sklearn."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA = "optimized-video-locator-v1"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    window = max(1, min(int(window), max(1, len(values))))
    if window == 1 or not len(values): return values.copy()
    left = window // 2
    pad = [(left, window - 1 - left)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, pad, mode="edge")
    if values.ndim == 1: return np.convolve(padded, np.ones(window)/window, mode="valid").astype(np.float32)
    cs = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64)])
    return ((cs[window:] - cs[:-window])/window).astype(np.float32)


def augment_signals(values: np.ndarray, names: list[str], fps: float, *, context: bool = True) -> tuple[np.ndarray,list[str]]:
    """Label-free within-video context; absolute input channels are never discarded."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(names) != values.shape[1] or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid frame features")
    if not np.isfinite(fps) or fps<=0:raise ValueError("invalid FPS")
    features, feature_names = [values], list(names)
    if context:
        delta = np.vstack([np.zeros((1, values.shape[1]), np.float32), np.diff(values,axis=0)])
        features += [delta*fps, np.abs(delta)*fps]
        feature_names += [n+"/delta_per_second" for n in names]+[n+"/abs_delta_per_second" for n in names]
        for seconds in [.12, .35, .75]:
            window = max(1, int(round(seconds*fps)))
            mean = rolling_mean(values, window)
            variance = np.maximum(0, rolling_mean(values*values, window)-mean*mean)
            features += [mean, np.sqrt(variance), values-mean]
            feature_names += [n+f"/mean_{seconds}s" for n in names]+[n+f"/std_{seconds}s" for n in names]+[n+f"/contrast_{seconds}s" for n in names]
        median=np.median(values,axis=0)
        q=np.percentile(values,[25,75],axis=0)
        scale=np.maximum((q[1]-q[0])/1.349, np.std(values,axis=0)*.1+1e-5)
        features.append(np.clip((values-median)/scale,-8,8))
        feature_names += [n+"/relative_robust_z" for n in names]
    # Global shape/scale supports video-level normal rejection; no video id is used.
    stats=np.concatenate([np.mean(values,axis=0),np.std(values,axis=0),np.percentile(values,10,axis=0),np.percentile(values,90,axis=0)])
    features.append(np.broadcast_to(stats.astype(np.float32),(len(values),len(stats))))
    feature_names += [n+"/global_"+stat for stat in ['mean','std','p10','p90'] for n in names]
    return np.concatenate(features,axis=1).astype(np.float32),feature_names


def video_features(values: np.ndarray) -> np.ndarray:
    values=np.asarray(values,dtype=np.float32)
    return np.concatenate([values.mean(axis=0),values.std(axis=0),np.percentile(values,10,axis=0),
                           np.percentile(values,90,axis=0)]).astype(np.float32)


def spans(binary: np.ndarray) -> list[tuple[int,int]]:
    change=np.diff(np.r_[False,np.asarray(binary,dtype=bool),False].astype(np.int8))
    return [(int(s),int(e-1)) for s,e in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1))]


def decode(probabilities: np.ndarray, fps: float, video_probability: float, config: dict[str,Any]) -> list[tuple[int,int]]:
    probabilities=np.asarray(probabilities,dtype=np.float32).reshape(-1)
    if not len(probabilities): return []
    if not np.isfinite(probabilities).all() or not np.isfinite(fps) or fps<=0: raise ValueError("invalid probability/FPS")
    if not np.isfinite(video_probability) or not 0<=video_probability<=1:raise ValueError("invalid video probability")
    if np.any((probabilities<0)|(probabilities>1)):raise ValueError("probabilities outside [0,1]")
    if video_probability < float(config.get("video_threshold",0)): return []
    strength=float(config.get("video_strength",0))
    if not np.isfinite(strength) or strength<0:raise ValueError("invalid video strength")
    if strength:probabilities=probabilities*float(video_probability)**strength
    smooth=rolling_mean(probabilities,max(1,int(round(float(config.get("smooth_seconds",0))*fps))))
    high=float(config["threshold"]); low=high*float(config.get("low_ratio",1.0))
    # Hysteresis components remain empty if no strong seed occurs.
    candidates=[(s,e) for s,e in spans(smooth>=low) if smooth[s:e+1].max()>=high]
    max_gap=max(0,int(round(float(config.get("gap_seconds",0))*fps)))
    merged=[]
    for start,end in candidates:
        if merged and start-merged[-1][1]-1<=max_gap:
            merged[-1]=(merged[-1][0],end)
        else: merged.append((start,end))
    minimum=max(1,int(round(float(config.get("min_seconds",.08))*fps)))
    return [(s,e) for s,e in merged if e-s+1>=minimum]


def interval_iou(a: tuple[int,int],b: tuple[int,int]) -> float:
    intersection=max(0,min(a[1],b[1])-max(a[0],b[0])+1)
    return intersection/max(1,(a[1]-a[0]+1)+(b[1]-b[0]+1)-intersection)


def metrics(predictions: list[list[tuple[int,int]]], labels: list[np.ndarray]) -> dict[str,Any]:
    """Same inclusive endpoints and prediction-order greedy matching as legacy evaluation."""
    output={}
    for threshold in [.3,.5]:
        tp=fp=fn=0
        for predicted,y in zip(predictions,labels):
            truth=spans(np.asarray(y)>0); matched=set()
            for candidate in predicted:
                options=[(interval_iou(candidate,gt),i) for i,gt in enumerate(truth) if i not in matched]
                # In a tie, preserve the first GT, matching the project's strict > update.
                score,idx=max(options,key=lambda v:(v[0],-v[1])) if options else (0,-1)
                if idx>=0 and score>=threshold: tp+=1; matched.add(idx)
                else: fp+=1
            fn+=len(truth)-len(matched)
        precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
        output[f"iou_{threshold}"]={"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,
                                    "f1":2*tp/max(1,2*tp+fp+fn)}
    normal=[i for i,y in enumerate(labels) if not np.any(y)]
    output["normal"]={"videos":len(normal),"false_positive_videos":sum(bool(predictions[i]) for i in normal)}
    output["normal"]["false_positive_rate"]=output["normal"]["false_positive_videos"]/max(1,len(normal))
    frame_tp=frame_fp=frame_fn=0
    for predicted,y in zip(predictions,labels):
        y=np.asarray(y)>0; p=np.zeros(len(y),bool)
        for s,e in predicted:
            lo=max(0,s);hi=min(len(y),e+1)
            if hi>lo:p[lo:hi]=True
        frame_tp+=np.sum(y&p);frame_fp+=np.sum(~y&p);frame_fn+=np.sum(y&~p)
    output["frame"]={"f1":float(2*frame_tp/max(1,2*frame_tp+frame_fp+frame_fn)),
                     "tp":int(frame_tp),"fp":int(frame_fp),"fn":int(frame_fn)}
    output["positive_videos_without_candidate"]=sum(bool(np.any(y)) and not p for y,p in zip(labels,predictions))
    output["predicted_segments"]=sum(map(len,predictions))
    return output


def objective(result: dict[str,Any]) -> float:
    return .5*(result['iou_0.3']['f1']+result['iou_0.5']['f1'])-.20*result['normal']['false_positive_rate']


def _tree_values(tree: dict, x: np.ndarray) -> np.ndarray:
    node=np.zeros(len(x),dtype=np.int32)
    left=np.asarray(tree['left'],np.int32);right=np.asarray(tree['right'],np.int32)
    features=np.asarray(tree['feature'],np.int32);threshold=np.asarray(tree['threshold'],np.float64)
    active=left[node]>=0
    while np.any(active):
        indices=np.flatnonzero(active);nodes=node[indices]
        go_left=x[indices,features[nodes]]<=threshold[nodes]
        node[indices]=np.where(go_left,left[nodes],right[nodes]);active=left[node]>=0
    return np.asarray(tree['value'],dtype=np.float64)[node]


def portable_predict(model: dict, values: np.ndarray) -> np.ndarray:
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=model['n_features'] or not np.isfinite(x).all():raise ValueError("model feature schema mismatch")
    if model['kind']=='constant':return np.full(len(x),model['probability'],dtype=np.float32)
    if model['kind']=='forest':
        if not model.get('trees'):raise ValueError('empty forest')
        return np.mean([_tree_values(t,x) for t in model['trees']],axis=0).astype(np.float32)
    if model['kind']=='gradient_boosting':
        logits=np.full(len(x),model['initial_log_odds'],dtype=np.float64)
        for t in model['trees']:logits+=model['learning_rate']*_tree_values(t,x)
        return (1/(1+np.exp(-np.clip(logits,-40,40)))).astype(np.float32)
    raise ValueError('unsupported model kind')


def export_model(model) -> dict:
    """Training-time bridge; inference JSON uses a versioned explicit schema."""
    n_features=int(model.n_features_in_)
    classes=np.asarray(model.classes_)
    if len(classes)==1:
        if classes[0] not in [0,1]:raise ValueError('only binary 0/1 labels supported')
        return {'kind':'constant','n_features':n_features,'probability':float(classes[0])}
    if not np.array_equal(classes,[0,1]):raise ValueError('only binary 0/1 labels supported')
    if type(model).__name__=='GradientBoostingClassifier':
        if isinstance(model.init_,str) and model.init_=='zero':initial=0.0
        elif hasattr(model.init_,'class_prior_'):
            prior=float(model.init_.class_prior_[1]);initial=float(np.log(prior/(1-prior)))
        else:raise ValueError('unsupported gradient boosting initializer')
        out={'kind':'gradient_boosting','n_features':n_features,'initial_log_odds':initial,
             'learning_rate':float(model.learning_rate),'trees':[]}
        trees=[est.tree_ for est in model.estimators_[:,0]]
        classification=False
    else:
        out={'kind':'forest','n_features':n_features,'trees':[]}
        trees=[est.tree_ for est in model.estimators_];classification=True
    for tree in trees:
        if classification:
            probabilities=tree.value[:,0,:]
            values=probabilities[:,1]/np.maximum(1e-30,probabilities.sum(axis=1))
        else:values=tree.value[:,0,0]
        out['trees'].append({'left':tree.children_left.tolist(),'right':tree.children_right.tolist(),
                             'feature':tree.feature.tolist(),'threshold':tree.threshold.tolist(),'value':values.tolist()})
    return out


def load_bundle(path: str|Path) -> dict:
    bundle=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(bundle,dict) or bundle.get('schema_version')!=SCHEMA:raise ValueError('unsupported locator bundle')
    if "publication" in bundle:
        from published_models import validate_bundle
        validate_bundle(bundle,path)
    return bundle
