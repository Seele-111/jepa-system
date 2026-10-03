"""Portable OOF-fitted monotone logit calibration; never fits during inference."""
from __future__ import annotations
import numpy as np


def checked_probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid calibration probabilities')
    return p


def apply_calibration(values, state=None):
    p = checked_probabilities(values)
    if not state or state.get('kind') == 'identity':
        return p.astype(np.float32)
    if state.get('kind') != 'logit_affine_v1':
        raise ValueError('unknown probability calibration')
    a, b = float(state['slope']), float(state['intercept'])
    if not np.isfinite([a, b]).all() or a <= 0:
        raise ValueError('invalid monotone calibration parameters')
    p = np.clip(p, 1e-5, 1-1e-5)
    z = np.clip(a*(np.log(p)-np.log1p(-p))+b, -30, 30)
    return (1/(1+np.exp(-z))).astype(np.float32)


def fit_logit_calibration(probabilities, labels, weights=None, ridge=1.):
    """Caller supplies held-out base predictions. This is inner tuning, not
    independent calibrated business risk. Ridge shrinks toward identity; the
    positive slope preserves ranks. Weights recover an equal-video prior.
    """
    p = checked_probabilities(probabilities).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if not len(p) or p.shape != y.shape or p.shape != w.shape:
        raise ValueError('calibration shape mismatch')
    if not np.isfinite(y).all() or not np.all((y == 0) | (y == 1)):
        raise ValueError('invalid binary calibration labels')
    if not np.isfinite(w).all() or np.any(w <= 0) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('invalid calibration weights/ridge')
    if len(np.unique(y)) < 2:
        return {'kind':'identity', 'reason':'single_class_inner_OOF'}
    p = np.clip(p, 1e-5, 1-1e-5)
    x = np.column_stack([np.log(p)-np.log1p(-p), np.ones(len(p))])
    anchor = np.array([1.,0.]); beta = anchor.copy()
    def loss(v):
        z = np.clip(x @ v, -40,40)
        return float(np.sum(w*(np.logaddexp(0,z)-y*z))+.5*ridge*np.sum((v-anchor)**2))
    for _ in range(40):
        z = np.clip(x @ beta, -40,40); pred = 1/(1+np.exp(-z))
        grad = x.T @ (w*(pred-y)) + ridge*(beta-anchor)
        hess = x.T @ (x*(w*pred*(1-pred))[:,None]) + ridge*np.eye(2)
        step = np.linalg.solve(hess,grad)
        if np.linalg.norm(step) < 1e-8: break
        before = loss(beta)
        for power in range(16):
            proposed = beta-step/(2**power)
            proposed[0] = np.clip(proposed[0],.25,3.)
            proposed[1] = np.clip(proposed[1],-3.,3.)
            if loss(proposed) <= before:
                beta = proposed; break
        else: break
    return {'kind':'logit_affine_v1','slope':float(beta[0]),'intercept':float(beta[1]),
            'ridge':float(ridge),'fit_role':'outer_training_inner_OOF_only',
            'independent_probability_calibration':False}


def fit_oof_calibration(records, indices, frame, video):
    if set(indices) != set(frame) or set(indices) != set(video):
        raise ValueError('calibration must use exactly the held-out inner OOF partition')
    fp = np.concatenate([frame[i] for i in indices])
    labels = np.concatenate([records[i]['labels'] for i in indices])
    weights = np.concatenate([np.full(len(frame[i]),1/len(frame[i])) for i in indices])
    vp = np.array([video[i] for i in indices])
    vy = np.array([int(np.any(records[i]['labels'])) for i in indices])
    return {'frame':fit_logit_calibration(fp,labels,weights),
            'video':fit_logit_calibration(vp,vy)}
