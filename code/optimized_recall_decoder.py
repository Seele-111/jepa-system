"""Fixed-profile recall/stability decoder; NumPy-only and backward opt-in."""
from __future__ import annotations
import math
import numpy as np
from optimized_locator import rolling_mean, spans

KIND = 'recall-stable-v2'


def effective_score(probabilities, fps, video_probability, config):
    p = np.asarray(probabilities, np.float32).reshape(-1)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isfinite(fps) or fps<=0:
        raise ValueError('invalid frame probabilities/FPS')
    v = float(video_probability)
    if not np.isfinite(v) or not 0<=v<=1:
        raise ValueError('invalid video probability')
    strength = float(config.get('video_strength',0))
    floor = float(config.get('video_floor',0))
    smoothing = float(config.get('smooth_seconds',0))
    if not np.isfinite([strength,floor,smoothing]).all() or strength<0 or not 0<=floor<=1 or smoothing<0:
        raise ValueError('invalid score modulation')
    p = p*(floor+(1-floor)*v)**strength if strength else p
    return rolling_mean(p,max(1,int(round(smoothing*fps))))


def decode_v2(probabilities, fps, video_probability, config):
    score = effective_score(probabilities,fps,video_probability,config)
    high = float(config['threshold']); ratio = float(config.get('low_ratio',1))
    minimum = float(config.get('min_seconds',0)); seed = float(config.get('seed_seconds',0))
    gap = float(config.get('gap_seconds',0)); vt = float(config.get('video_threshold',0))
    if not np.isfinite([high,ratio,minimum,seed,gap,vt]).all() or not 0<=high<=1 or not 0<ratio<=1 or min(minimum,seed,gap)<0 or not 0<=vt<=1:
        raise ValueError('invalid recall decoder config')
    if not len(score) or video_probability<vt: return []
    strong = np.zeros(len(score),bool)
    required = max(1,int(math.ceil(seed*fps-1e-9)))
    for s,e in spans(score>=high):
        if e-s+1>=required: strong[s:e+1] = True
    candidates = [(s,e) for s,e in spans(score>=high*ratio) if strong[s:e+1].any()]
    max_gap = max(0,int(round(gap*fps))); merged=[]
    for s,e in candidates:
        if merged and s-merged[-1][1]-1<=max_gap: merged[-1]=(merged[-1][0],e)
        else: merged.append((s,e))
    min_frames = max(1,int(math.ceil(minimum*fps-1e-9)))
    return [(s,e) for s,e in merged if e-s+1>=min_frames]
