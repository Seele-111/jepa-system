"""Content-balanced existing boundary heads on the frozen rich compact view.

Heads see only raw label-free features, not fitted frame predictions. Thus ordinary
inner train/validation isolation suffices; this is not learned OOF stacking.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from optimized_compact_features_v4 import content_weights, fit_rgb_pca
from optimized_feature_blocks_v10 import RECIPES
from optimized_feature_blocks_v10 import feature_view_blocks
from optimized_boundary_head import _boundary_targets, _fit_head, predict_boundary_heads, refine_intervals
import run_event_compact_experiment_v8 as V8
from optimized_video_statistics import VideoTruth, video_statistics
from run_recall_selection import grouped_bootstrap
from run_video_gate_experiment_v6 import utility
from run_compact_experiment_v4 import decode_predictions
import json

RECIPE=RECIPES[-1]


def fit_content_boundary(records,train,predict,seed,*,full_fit=False):
    if not train or not predict or len(set(train))!=len(train) or len(set(predict))!=len(predict):raise ValueError('invalid boundary partitions')
    a={records[i]['sha256'] for i in train};b={records[i]['sha256'] for i in predict}
    if (not full_fit and a&b) or (full_fit and set(train)!=set(predict)):raise ValueError('boundary content leakage')
    pca=fit_rgb_pca(records,train);ids=sorted(set(train)|set(predict))
    views={i:feature_view_blocks(RECIPE,records[i],pca) for i in ids}
    names=views[ids[0]][2]
    if any(v[2]!=names for v in views.values()):raise ValueError('boundary feature names differ')
    weights=content_weights(records,train)
    x=np.concatenate([views[i][0] for i in train]);targets=[_boundary_targets(records[i]['labels'],records[i]['fps']) for i in train]
    starts=np.concatenate([v[0] for v in targets]);ends=np.concatenate([v[1] for v in targets])
    base=np.concatenate([np.full(records[i]['frames'],weights[i]/records[i]['frames']) for i in train])
    models={'start_model':_fit_head(x,starts,base,seed),'end_model':_fit_head(x,ends,base,seed+1)}
    probabilities={i:predict_boundary_heads(models,views[i][0]) for i in predict}
    state={'recipe':RECIPE,'transform':{'feature_view':'compact-feature-blocks-v10','pca':pca,'frame_feature_names':names},
        'boundary_models':models,'fit_content_sha256':sorted(a),'base_content_mass':sum(weights.values())}
    return probabilities,state


def configurations():
    yield {'boundary_seconds':0.,'minimum_boundary_evidence':0.}
    for seconds in [.2,.4,.6]:
        for minimum in [0.,.3,.5]:yield {'boundary_seconds':seconds,'minimum_boundary_evidence':minimum}


def apply_boundary(intervals,heads,fps,cfg):
    if set(cfg)!={'boundary_seconds','minimum_boundary_evidence'}:raise ValueError('unknown boundary policy schema')
    minimum=float(cfg['minimum_boundary_evidence'])
    if not np.isfinite(minimum) or not 0<=minimum<=1:raise ValueError('invalid boundary evidence threshold')
    refined=refine_intervals(intervals,*heads,fps,cfg)
    result=[];trace=[]
    for s,e in refined:
        start,end=float(heads[0][s]),float(heads[1][e]);keep=minimum==0 or min(start,end)>=minimum
        if keep:result.append((s,e))
        trace.append({'span':[s,e],'start_evidence':start,'end_evidence':end,'accepted':bool(keep)})
    return result,trace


def decode_boundary(records,ids,primary,heads,cfg,*,trace=False):
    if set(cfg)!={'kind','proposal','boundary'} or cfg['kind']!='compact-boundary-v12':raise ValueError('unknown boundary decoder schema')
    proposals=decode_predictions(records,ids,primary,cfg['proposal']);result={};traces={}
    for i in ids:result[i],traces[i]=apply_boundary(proposals[i],heads[i],records[i]['fps'],cfg['boundary'])
    return (result,traces) if trace else result


def choose_boundary(records,ids,primary,heads,scope):
    if set(primary.frame)!=set(ids) or set(primary.video)!=set(ids) or set(heads)!=set(ids):raise ValueError('boundary inner coverage differs')
    seed=20261002+scope;base=V8.choose(records,ids,primary,seed)
    cw=content_weights(records,ids);weights=np.array([cw[i] for i in ids]);truths=[VideoTruth(records[i]['labels']) for i in ids]
    normals=np.array([t.normal for t in truths],float);pos=1-normals;nm,pm=weights@normals,weights@pos
    boot=grouped_bootstrap(records,ids,seed,32)*weights[None,:];bn,bp=boot@normals,boot@pos
    rows=[];seen=set()
    for ordinal,policy in enumerate(configurations()):
        cfg={'kind':'compact-boundary-v12','proposal':deepcopy(base['config']),'boundary':policy}
        pred=decode_boundary(records,ids,primary,heads,cfg);fingerprint=json.dumps([pred[i] for i in ids])
        if fingerprint in seen:continue
        seen.add(fingerprint);mat=np.stack([video_statistics(pred[i],t) for i,t in zip(ids,truths)]);total=weights@mat
        u=float(utility(total,nm,pm));f05=2*total[3]/max(1e-12,2*total[3]+total[4]+total[5])
        row={'config':cfg,'stats':total.tolist(),'pooled_utility':u,'key':[u,float(f05),-total[10],-total[9],-total[11],-ordinal]}
        row['bootstrap_q20']=float(np.quantile(utility(boot@mat,bn,bp),.2));row['stable_utility']=.75*u+.25*row['bootstrap_q20'];rows.append(row)
    result=deepcopy(max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:])))
    result['shortlist']=rows;result['base_selection']=base;result['candidate']='boundary_control' if result['config']['boundary']['boundary_seconds']==0 else 'compact_boundary_refine_accept'
    return result
