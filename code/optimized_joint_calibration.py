"""NumPy-only joint monotone calibration for held-out inner OOF predictions.

This is inner tuning, not independent calibration or a business-risk probability.
Inference never fits a model. Scores are clipped only when taking their logits.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product

import numpy as np


_KIND = 'monotone_joint_logit_v1'
_EPS = 1e-5
_ANCHOR = np.array([1., 0., 0.])  # frame slope, video slope, intercept
_LOWER = np.array([.25, 0., -3.])
_UPPER = np.array([3., 3., 3.])


def _float_array(values, name):
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError('complex values are not supported')
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'invalid calibration {name}') from exc
    if not np.isfinite(array).all():
        raise ValueError(f'nonfinite calibration {name}')
    return array


def _probabilities(values, name):
    array = _float_array(values, name)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f'calibration {name} must be in [0, 1]')
    return array


def _video_probability(value):
    probability = _probabilities(value, 'video probability')
    if probability.ndim != 0:
        raise ValueError('calibration video probability must be a scalar')
    return float(probability)


def _items(values, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f'calibration {name} must be a sequence per video')
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f'calibration {name} must be a sequence per video') from exc


def _logit(probabilities):
    clipped = np.clip(probabilities, _EPS, 1 - _EPS)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(logits):
    # All logits are finite and bounded by the probability/parameter guards.
    return 1 / (1 + np.exp(-logits))


def _newton_direction(beta, gradient, hessian):
    """Minimize the three-dimensional Newton quadratic inside the exact box.

    Enumerating its 27 possible active sets avoids projected Newton steps that
    stall at c=0 when the unconstrained direction points outside the box.
    """
    lower, upper = _LOWER - beta, _UPPER - beta
    best = np.zeros(3)
    best_value = 0.
    for status in product((-1, 0, 1), repeat=3):
        status = np.asarray(status)
        free = np.flatnonzero(status == 0)
        fixed = np.flatnonzero(status != 0)
        direction = np.zeros(3)
        direction[status == -1] = lower[status == -1]
        direction[status == 1] = upper[status == 1]
        if free.size:
            rhs = -gradient[free] - hessian[np.ix_(free, fixed)] @ direction[fixed]
            direction[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs)
        if np.any(direction < lower - 1e-12) or np.any(direction > upper + 1e-12):
            continue
        direction = np.clip(direction, lower, upper)
        value = float(gradient @ direction + .5 * direction @ hessian @ direction)
        if value < best_value:
            best, best_value = direction, value
    return best


def _state(beta, *, identity=False, weighting='inverse_video_length'):
    state = {
        'kind': 'identity' if identity else _KIND,
        'slope': float(beta[0]),
        'video_slope': float(beta[1]),
        'intercept': float(beta[2]),
        'ridge': 1.,
        'weighting': weighting,
        'fit_role': 'outer_training_inner_OOF_only',
        'independent_probability_calibration': False,
        'business_risk_probability': False,
    }
    if identity:
        state['reason'] = 'single_class_inner_OOF'
    return state


def fit_joint(frame_probs_list, video_probs_list, labels_list,
              weights_per_video=None, ridge=1):
    """Fit sigmoid(b + a*logit(frame_q) + c*logit(video_v)) on inner OOF.

    Each video must have a nonempty 1-D frame array, an exactly shape-matched
    binary label array, and one scalar video probability. By default the BCE
    is the SUM of per-video mean frame BCEs (frame weights 1/video_length),
    without class balancing. Ridge is fixed at 1 and shrinks (a,c,b) to (1,0,0).

    Optional weights_per_video contains one positive scalar video mass or one
    positive, shape-matched frame-weight array per video. Scalars are divided
    by video length; arrays are explicit final BCE weights, not renormalized.
    Supplying arrays filled with 1/video_length reproduces the default.
    All inputs are checked before the single-class identity fallback.
    """
    ridge_value = _float_array(ridge, 'ridge')
    if ridge_value.ndim != 0 or float(ridge_value) != 1.:
        raise ValueError('joint calibration ridge is fixed at 1')
    frames = _items(frame_probs_list, 'frame probabilities')
    videos = _items(video_probs_list, 'video probabilities')
    labels = _items(labels_list, 'labels')
    if not frames or len(frames) != len(videos) or len(frames) != len(labels):
        raise ValueError('calibration per-video shape mismatch or empty input')
    supplied_weights = None
    if weights_per_video is not None:
        supplied_weights = _items(weights_per_video, 'weights')
        if len(supplied_weights) != len(frames):
            raise ValueError('calibration per-video weight shape mismatch')

    designs, targets, weights = [], [], []
    for index, (frame, video, label) in enumerate(zip(frames, videos, labels)):
        frame = _probabilities(frame, 'frame probabilities')
        if frame.ndim != 1 or not frame.size:
            raise ValueError('calibration frames must be nonempty 1-D arrays')
        video = _video_probability(video)
        label = _float_array(label, 'labels')
        if label.shape != frame.shape:
            raise ValueError('calibration frame/label shape mismatch')
        if not np.all((label == 0) | (label == 1)):
            raise ValueError('calibration labels must be binary')
        if supplied_weights is None:
            weight = np.full(frame.shape, 1 / frame.size)
        else:
            weight = _float_array(supplied_weights[index], 'weights')
            if weight.ndim == 0:
                weight = np.full(frame.shape, float(weight) / frame.size)
            elif weight.shape != frame.shape:
                raise ValueError('calibration frame/weight shape mismatch')
        if np.any(weight <= 0):
            raise ValueError('calibration weights must be positive')
        designs.append(np.column_stack([
            _logit(frame), np.full(frame.shape, _logit(video)), np.ones(frame.shape),
        ]))
        targets.append(label)
        weights.append(weight)

    x, y, w = np.concatenate(designs), np.concatenate(targets), np.concatenate(weights)
    # Reserve headroom for the Hessian's squared logit features and reductions.
    with np.errstate(over='ignore'):
        total_weight = float(np.sum(w))
    if not np.isfinite(total_weight) or total_weight > np.finfo(np.float64).max / 1024:
        raise ValueError('calibration total weight is too large')
    weighting = 'inverse_video_length' if supplied_weights is None else 'provided_weights'
    if np.all(y == y[0]):
        return _state(_ANCHOR, identity=True, weighting=weighting)

    def loss(beta):
        logits = x @ beta
        # Binary softplus avoids cancellation for confidently correct examples.
        bce = np.logaddexp(0, np.where(y == 1, -logits, logits))
        return float(w @ bce + .5 * np.sum((beta - _ANCHOR) ** 2))

    beta = _ANCHOR.copy()
    for _ in range(80):
        predictions = _sigmoid(x @ beta)
        gradient = x.T @ (w * (predictions - y)) + beta - _ANCHOR
        projected = beta - np.clip(beta - gradient, _LOWER, _UPPER)
        if np.max(np.abs(projected)) < 1e-8:
            break
        curvature = w * predictions * (1 - predictions)
        hessian = x.T @ (x * curvature[:, None]) + np.eye(3)
        direction = _newton_direction(beta, gradient, hessian)
        if np.max(np.abs(direction)) < 1e-10:
            break
        before, derivative = loss(beta), float(gradient @ direction)
        for power in range(32):
            scale = 2. ** -power
            proposed = np.clip(beta + scale * direction, _LOWER, _UPPER)
            if loss(proposed) <= before + 1e-4 * scale * derivative:
                beta = proposed
                break
        else:
            break
    return _state(beta, weighting=weighting)


def apply_joint(frame_array, video_scalar, state):
    """Apply a saved state without fitting; return float32 with unchanged shape.

    Empty frame arrays and scalar frame inputs are supported. Both probabilities
    are checked even for identity/None states; the video input must be scalar.
    """
    frame = _probabilities(frame_array, 'frame probabilities')
    video = _video_probability(video_scalar)
    if state is None:
        return frame.astype(np.float32)
    if not isinstance(state, Mapping):
        raise ValueError('calibration state must be a mapping')
    if not state or state.get('kind') == 'identity':
        return frame.astype(np.float32)
    if state.get('kind') != _KIND:
        raise ValueError('unknown joint probability calibration')
    try:
        parameters = [state[name] for name in ('slope', 'video_slope', 'intercept')]
    except KeyError as exc:
        raise ValueError('missing joint calibration parameters') from exc
    beta = _float_array(parameters, 'parameters')
    if beta.shape != (3,) or np.any(beta < _LOWER) or np.any(beta > _UPPER):
        raise ValueError('joint calibration parameters outside their bounds')
    logits = beta[2] + beta[0] * _logit(frame) + beta[1] * _logit(video)
    return np.asarray(_sigmoid(logits), dtype=np.float32)
