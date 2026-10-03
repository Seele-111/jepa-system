"""Strict dispatch for explicit versioned portable calibration schemas.

None is the legacy no-calibration state. Separate calibration has exactly the
frame/video branches (each may be None, identity or logit_affine_v1). Joint
calibration and its single-class identity fallback use an explicit kind. Saved
fit metadata is optional, but unrecognized fields and mixed schemas are errors.
The versioned v3 stack explicitly requires original member predictions.
Inference never fits, mutates a state, or guesses a future schema.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

from optimized_joint_calibration import apply_joint
from optimized_probability_calibration import apply_calibration


_JOINT_KIND = 'monotone_joint_logit_v1'
_AFFINE_KIND = 'logit_affine_v1'
_AFFINE_PARAMETERS = frozenset(('slope', 'intercept'))
_JOINT_PARAMETERS = _AFFINE_PARAMETERS | {'video_slope'}
_COMMON_METADATA = frozenset(('reason', 'ridge', 'fit_role',
                              'independent_probability_calibration'))
_JOINT_METADATA = _COMMON_METADATA | {'weighting', 'business_risk_probability'}


def _fields(state, allowed, required, path):
    if any(key not in allowed for key in state):
        raise ValueError(f'{path}: unknown fields or mixed calibration structure')
    missing = required.difference(state)
    if missing:
        raise ValueError(f'{path}: missing calibration fields: {", ".join(sorted(missing))}')


def _number(state, name, path):
    value = state[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{path}.{name}: calibration parameter must be a real scalar')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{path}.{name}: invalid calibration parameter') from exc
    if not np.isfinite(value):
        raise ValueError(f'{path}.{name}: nonfinite calibration parameter')
    return value


def _metadata(state, path):
    if 'ridge' in state and _number(state, 'ridge', path) <= 0:
        raise ValueError(f'{path}.ridge: calibration ridge must be positive')
    for name in ('reason', 'fit_role', 'weighting'):
        if name in state and not isinstance(state[name], str):
            raise ValueError(f'{path}.{name}: calibration metadata must be a string')
    for name in ('independent_probability_calibration', 'business_risk_probability'):
        if name in state and not isinstance(state[name], (bool, np.bool_)):
            raise ValueError(f'{path}.{name}: calibration metadata must be boolean')


def _transform(state, path, *, joint=False):
    if state is None:
        return 'identity'
    if not isinstance(state, Mapping):
        raise ValueError(f'{path}: calibration state must be a mapping or None')
    kind = state.get('kind')
    active_kind = _JOINT_KIND if joint else _AFFINE_KIND
    if not isinstance(kind, str) or kind not in ('identity', active_kind):
        raise ValueError(f'{path}: unknown or missing calibration kind')
    parameters = _JOINT_PARAMETERS if joint else _AFFINE_PARAMETERS
    metadata = _JOINT_METADATA if joint else _COMMON_METADATA
    required = {'kind'}
    # Bare identity needs no coefficients. If coefficients are saved (as by
    # fit_joint's single-class fallback), validate the whole parameter group.
    if kind != 'identity' or any(name in state for name in parameters):
        required |= parameters
    _fields(state, {'kind'} | parameters | metadata, required, path)
    _metadata(state, path)
    if parameters.issubset(state):
        slope = _number(state, 'slope', path)
        intercept = _number(state, 'intercept', path)
        if slope <= 0:
            raise ValueError(f'{path}.slope: calibration slope must be positive')
        if joint:
            video_slope = _number(state, 'video_slope', path)
            if not (.25 <= slope <= 3 and 0 <= video_slope <= 3 and -3 <= intercept <= 3):
                raise ValueError(f'{path}: joint calibration parameters outside their bounds')
    return kind


def validate_calibration_state(state):
    """Validate without probabilities; return an explicit dispatch kind.

    Only None is an implicit identity: empty/falsy containers are not aliases.
    Separate states must contain both frame and video, without a top-level kind.
    Affine branches require a positive finite slope and finite intercept.
    Versioned v3 stacks require their explicit member recipe profile and five
    coefficients. Joint states require all three finite scalar coefficients within apply_joint's
    bounds. Known fit metadata is accepted and checked; unknown keys are rejected
    at every level. Raises ValueError for every invalid saved state.
    """
    if state is None:
        return 'identity'
    if not isinstance(state, Mapping):
        raise ValueError('calibration: state must be a mapping or None')
    if state.get('kind') == 'monotone_stack_logit_v3':
        from optimized_stack_v3 import validate_state
        validate_state(state)
        return 'monotone_stack_logit_v3'
    if 'kind' in state:
        return _transform(state, 'calibration', joint=True)
    _fields(state, {'frame', 'video'}, {'frame', 'video'}, 'calibration')
    _transform(state['frame'], 'calibration.frame')
    _transform(state['video'], 'calibration.video')
    return 'separate'


def apply_calibration_state(frame_probabilities, video_probability, state=None, *,
                            member_frame=None, member_video=None):
    """Return (frame probabilities, video probability) for one video's state.

    None returns the original inputs unchanged, preserving the v1 no-calibration
    path exactly. Separate dispatch retains apply_calibration's float32 results.
    Joint (including explicit identity) changes only frame probabilities; the
    video scalar remains unmodified. V3 combines the original member scores,
    never attempting to reconstruct them from their arithmetic average.
    Validation precedes every transformation.
    """
    kind = validate_calibration_state(state)
    if state is None:
        return frame_probabilities, video_probability
    if kind == 'monotone_stack_logit_v3':
        if member_frame is None or member_video is None:
            raise ValueError('stack calibration needs original member predictions')
        from optimized_stack_v3 import apply_stack
        return apply_stack(member_frame, member_video, state)
    if kind == 'separate':
        frame = apply_calibration(frame_probabilities, state['frame'])
        video = float(apply_calibration([video_probability], state['video'])[0])
        return frame, video
    return apply_joint(frame_probabilities, video_probability, state), video_probability
