"""Opt-in repair for v6: preserve fitted float64 statistics until cast boundary."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from optimized_video_gate_v6 import gate_features, fit_gate as original_fit_gate, apply_gate as original_apply_gate, GATE_KINDS

NUMERIC_REVISION = 'float64-standardization-f32-v6r'


def repaired_state(state):
    result = deepcopy(state)
    result['numeric_revision'] = NUMERIC_REVISION
    return result


def fit_gate(kind, records, train, fp, vp, seed):
    return repaired_state(original_fit_gate(kind, records, train, fp, vp, seed))


def apply_gate(state, frame_probability, video_probability, record):
    if state.get('numeric_revision') != NUMERIC_REVISION:
        raise ValueError('unsupported repaired gate revision')
    if state.get('kind') != 'ridge_logistic_v6':
        return original_apply_gate(state, frame_probability, video_probability, record)
    x, names = gate_features(frame_probability, video_probability, record)
    if state.get('feature_names') != names:
        raise ValueError('video gate feature order mismatch')
    mean = np.asarray(state['mean'], np.float64)
    scale = np.asarray(state['scale'], np.float64)
    coef = np.asarray(state['coefficients'], np.float64)
    intercept = float(state['intercept'])
    if (mean.shape != x.shape or scale.shape != x.shape or coef.shape != x.shape
            or not all(np.isfinite(v).all() for v in [mean, scale, coef])
            or np.any(scale <= 0) or not np.isfinite(intercept)):
        raise ValueError('invalid logistic gate state')
    # This is exactly the standardization/cast order used by the frozen fitter.
    z = np.clip((x - mean) / scale, -8, 8).astype(np.float32)
    value = float(z @ coef + intercept)
    return float(1 / (1 + np.exp(-np.clip(value, -700, 700))))
