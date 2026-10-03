"""Bounded proposal-level verification; no learned meta-model or global veto.

The ET proposal remains unchanged when accepted. HGB evidence is read only inside
that proposal. Cutoffs are selected using inner OOF only; scores are not calibrated
risk probabilities. Strong ET peaks may rescue a locally unconfirmed proposal.
"""
from __future__ import annotations
from copy import deepcopy
import json
import numpy as np
from optimized_compact_features_v4 import content_weights
from optimized_video_statistics import VideoTruth, video_statistics
from run_recall_selection import grouped_bootstrap
from run_video_gate_experiment_v6 import utility
from run_compact_experiment_v4 import decode_predictions
import run_event_compact_experiment_v8 as V8

KIND='proposal-local-q75-v11'


def review_configurations():
    yield {'kind':KIND,'minimum':0.,'rescue_peak':None}
    for minimum in [.35,.45,.55]:
        for rescue in [None,.8]:
            yield {'kind':KIND,'minimum':minimum,'rescue_peak':rescue}


def verify_proposals(proposals, primary, reviewer, cfg):
    p=np.asarray(primary,np.float64); q=np.asarray(reviewer,np.float64)
    if p.ndim!=1 or q.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(q).all() or np.any((p<0)|(p>1)) or np.any((q<0)|(q>1)):
        raise ValueError('invalid verifier probability tensors')
    if cfg.get('kind')!=KIND or set(cfg)!={'kind','minimum','rescue_peak'}:
        raise ValueError('unknown verifier schema')
    minimum=float(cfg['minimum']); rescue=cfg['rescue_peak']
    if not np.isfinite(minimum) or not 0<=minimum<=1 or (rescue is not None and (not np.isfinite(rescue) or not 0<=rescue<=1)):
        raise ValueError('invalid verifier cutoff')
    accepted=[];trace=[]
    for raw in proposals:
        if len(raw)!=2 or any(isinstance(v,(bool,np.bool_)) or int(v)!=v for v in raw):
            raise ValueError('invalid proposal span')
        start,end=map(int,raw)
        if not 0<=start<=end<len(p):raise ValueError('proposal out of bounds')
        evidence=float(np.quantile(q[start:end+1],.75)); peak=float(p[start:end+1].max())
        passed=minimum==0 or evidence>=minimum
        rescued=not passed and rescue is not None and peak>=rescue
        keep=passed or rescued
        if keep:accepted.append((start,end))
        trace.append({'span':[start,end],'review_q75':evidence,'primary_peak':peak,
            'accepted':bool(keep),'rescued':bool(rescued)})
    return accepted,trace


def decode_review(records,ids,primary,reviewer,cfg,*,trace=False):
    if cfg.get('kind')!='proposal-review-v11' or set(cfg)!={'kind','proposal','review'}:
        raise ValueError('unknown proposal decoder schema')
    proposals=decode_predictions(records,ids,primary,cfg['proposal'])
    result={}; evidence={}
    for i in ids:
        result[i],evidence[i]=verify_proposals(proposals[i],primary.frame[i],reviewer.frame[i],cfg['review'])
    return (result,evidence) if trace else result


def choose_review(records,ids,primary,reviewer,scope):
    if set(primary.frame)!=set(ids) or set(primary.video)!=set(ids) or set(reviewer.frame)!=set(ids):
        raise ValueError('inner coverage differs')
    seed=20261002+scope
    base=V8.choose(records,ids,primary,seed)
    cw=content_weights(records,ids);weights=np.array([cw[i] for i in ids])
    truths=[VideoTruth(records[i]['labels']) for i in ids]
    normals=np.array([t.normal for t in truths],float);positive=1-normals
    nm,pm=weights@normals,weights@positive
    boot=grouped_bootstrap(records,ids,seed,32)*weights[None,:];bn,bp=boot@normals,boot@positive
    rows=[];seen=set()
    for ordinal,review in enumerate(review_configurations()):
        cfg={'kind':'proposal-review-v11','proposal':deepcopy(base['config']),'review':review}
        pred=decode_review(records,ids,primary,reviewer,cfg)
        fingerprint=json.dumps([pred[i] for i in ids])
        if fingerprint in seen:continue
        seen.add(fingerprint)
        mat=np.stack([video_statistics(pred[i],t) for i,t in zip(ids,truths)])
        total=weights@mat;u=float(utility(total,nm,pm));f05=2*total[3]/max(1e-12,2*total[3]+total[4]+total[5])
        row={'config':cfg,'stats':total.tolist(),'pooled_utility':u,
            'key':[u,float(f05),-total[10],-total[9],-total[11],-ordinal]}
        row['bootstrap_q20']=float(np.quantile(utility(boot@mat,bn,bp),.2))
        row['stable_utility']=.75*u+.25*row['bootstrap_q20'];rows.append(row)
    chosen=deepcopy(max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:])))
    chosen['shortlist']=rows;chosen['base_selection']=base
    chosen['candidate']='et_control' if chosen['config']['review']['minimum']==0 else 'et_proposals_hgb_local_review'
    return chosen
