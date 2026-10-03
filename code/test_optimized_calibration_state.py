"""CPU regressions for strict, portable calibration validation and dispatch."""
from copy import deepcopy
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from optimized_calibration_state import apply_calibration_state, validate_calibration_state
from optimized_joint_calibration import apply_joint, fit_joint
from optimized_probability_calibration import apply_calibration, fit_oof_calibration


AFFINE = {'kind': 'logit_affine_v1', 'slope': 1.2, 'intercept': -.3}
JOINT = {'kind': 'monotone_joint_logit_v1', 'slope': 1.3,
         'video_slope': .7, 'intercept': -.2}


class CalibrationStateTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.array([[0., .2], [.7, 1.]], dtype=np.float64)
        self.video = np.float64(.789123456789)
        self.separate = {'frame': deepcopy(AFFINE), 'video': deepcopy(AFFINE)}

    def assert_invalid(self, state):
        with self.assertRaisesRegex(ValueError, 'calibration'):
            validate_calibration_state(state)
        with self.assertRaisesRegex(ValueError, 'calibration'):
            apply_calibration_state(self.frame, self.video, state)

    def test_none_preserves_v1_inputs_without_casting_or_mutation(self):
        self.assertEqual(validate_calibration_state(None), 'identity')
        frame, video = apply_calibration_state(self.frame, self.video)
        self.assertIs(frame, self.frame)
        self.assertIs(video, self.video)
        self.assertEqual(frame.dtype, np.float64)

    def test_separate_dispatch_matches_existing_probability_transforms(self):
        before = deepcopy(self.separate)
        original = self.frame.copy()
        self.assertEqual(validate_calibration_state(self.separate), 'separate')
        frame, video = apply_calibration_state(self.frame, self.video, self.separate)
        np.testing.assert_array_equal(frame, apply_calibration(self.frame, AFFINE))
        self.assertEqual(video, float(apply_calibration([self.video], AFFINE)[0]))
        self.assertEqual(frame.shape, self.frame.shape)
        self.assertEqual(frame.dtype, np.float32)
        self.assertEqual(self.separate, before)
        np.testing.assert_array_equal(self.frame, original)

    def test_explicit_none_and_identity_branches_are_legal(self):
        for frame_state in (None, {'kind': 'identity'}, AFFINE):
            for video_state in (None, {'kind': 'identity'}, AFFINE):
                state = {'frame': frame_state, 'video': video_state}
                with self.subTest(state=state):
                    frame, video = apply_calibration_state(self.frame, self.video, state)
                    np.testing.assert_array_equal(frame, apply_calibration(self.frame, frame_state))
                    self.assertEqual(video, float(apply_calibration([self.video], video_state)[0]))

    def test_joint_dispatch_preserves_video_and_matches_existing_formula(self):
        before = deepcopy(JOINT)
        self.assertEqual(validate_calibration_state(JOINT), JOINT['kind'])
        frame, video = apply_calibration_state(self.frame, self.video, JOINT)
        np.testing.assert_array_equal(frame, apply_joint(self.frame, self.video, JOINT))
        self.assertIs(video, self.video)
        self.assertEqual(JOINT, before)

    def test_bare_top_identity_is_explicit_and_preserves_video_precision(self):
        state = {'kind': 'identity'}
        frame, video = apply_calibration_state(self.frame, self.video, state)
        self.assertEqual(validate_calibration_state(state), 'identity')
        np.testing.assert_array_equal(frame, self.frame.astype(np.float32))
        self.assertIs(video, self.video)

    def test_actual_separate_fit_metadata_is_accepted(self):
        records = [{'labels': np.array([0, 0, 0])}, {'labels': np.array([0, 1, 1])}]
        frames = {0: np.array([.02, .15, .35]), 1: np.array([.2, .7, .9])}
        videos = {0: .3, 1: .8}
        state = fit_oof_calibration(records, [0, 1], frames, videos)
        self.assertEqual(validate_calibration_state(state), 'separate')
        self.assertEqual(state['frame']['kind'], AFFINE['kind'])
        self.assertEqual(state['video']['kind'], AFFINE['kind'])
        frame, video = apply_calibration_state(self.frame, self.video, state)
        np.testing.assert_array_equal(frame, apply_calibration(self.frame, state['frame']))
        self.assertEqual(video, float(apply_calibration([self.video], state['video'])[0]))

    def test_actual_joint_fit_and_single_class_identity_metadata_are_accepted(self):
        frames = [np.array([.1, .3, .7]), np.array([.2, .8, .9])]
        videos = [.3, .8]
        for labels in ([np.array([0, 0, 1]), np.array([0, 1, 1])],
                       [np.zeros(3), np.zeros(3)]):
            state = fit_joint(frames, videos, labels)
            with self.subTest(kind=state['kind']):
                self.assertEqual(validate_calibration_state(state), state['kind'])
                frame, video = apply_calibration_state(self.frame, self.video, state)
                np.testing.assert_array_equal(frame, apply_joint(self.frame, self.video, state))
                self.assertIs(video, self.video)

    def test_single_class_separate_identities_are_accepted(self):
        state = fit_oof_calibration([{'labels': np.zeros(3)}], [0],
                                    {0: np.array([.1, .2, .3])}, {0: .3})
        self.assertEqual(state['frame']['kind'], 'identity')
        self.assertEqual(state['video']['kind'], 'identity')
        self.assertEqual(validate_calibration_state(state), 'separate')

    def test_mapping_and_numpy_real_scalars_are_supported(self):
        state = MappingProxyType({
            'frame': MappingProxyType(dict(AFFINE, slope=np.float32(1.2))),
            'video': MappingProxyType(dict(AFFINE, intercept=np.int64(0))),
        })
        self.assertEqual(validate_calibration_state(state), 'separate')
        frame, video = apply_calibration_state(self.frame, self.video, state)
        self.assertTrue(np.isfinite(frame).all())
        self.assertTrue(np.isfinite(video))

    def test_valid_joint_bounds_and_unbounded_positive_affine_slope(self):
        for slope in (.25, 3.):
            for video_slope in (0., 3.):
                for intercept in (-3., 3.):
                    state = dict(JOINT, slope=slope, video_slope=video_slope, intercept=intercept)
                    with self.subTest(state=state):
                        self.assertEqual(validate_calibration_state(state), JOINT['kind'])
        for slope in (.01, 4.):
            state = {'frame': dict(AFFINE, slope=slope), 'video': None}
            self.assertEqual(validate_calibration_state(state), 'separate')

    def test_empty_and_nonmapping_top_states_are_not_implicit_identity(self):
        for state in ({}, [], (), '', 0, False, 1, 'identity', np.nan, np.array([0, 1])):
            with self.subTest(state=state):
                self.assert_invalid(state)

    def test_unknown_top_kinds_never_fall_back_to_separate_or_identity(self):
        for kind in ('other', 'logit_affine_v1', 'separate', 'future_monotone_stack_v3',
                     None, False, 1, ['identity'], {'kind': 'identity'}, np.array('identity')):
            for branches in ({}, {'frame': None, 'video': None}, self.separate):
                state = dict(branches, kind=kind)
                with self.subTest(kind=kind, branches=branches):
                    self.assert_invalid(state)

    def test_missing_separate_branches_are_rejected(self):
        for state in ({'frame': None}, {'video': None}, {'frame': AFFINE}, {'video': AFFINE}):
            with self.subTest(state=state):
                self.assert_invalid(state)

    def test_unknown_top_keys_and_mixed_structures_are_rejected(self):
        cases = [dict(self.separate, frames=None), dict(self.separate, ridge=1.),
                 {'calibration': None}, {'slope': 1., 'intercept': 0.},
                 dict(JOINT, frame=None), dict(JOINT, video=None), dict(JOINT, unknown=0),
                 {'kind': 'identity', 'frame': None, 'video': None},
                 {'kind': 'identity', 'unknown': None}]
        for state in cases:
            with self.subTest(state=state):
                self.assert_invalid(state)

    def test_missing_affine_and_joint_fields_are_rejected(self):
        for name in ('kind', 'slope', 'intercept'):
            branch = dict(AFFINE)
            del branch[name]
            for channel in ('frame', 'video'):
                state = deepcopy(self.separate)
                state[channel] = branch
                with self.subTest(channel=channel, missing=name):
                    self.assert_invalid(state)
        for name in ('kind', 'slope', 'video_slope', 'intercept'):
            state = dict(JOINT)
            del state[name]
            with self.subTest(missing=name):
                self.assert_invalid(state)

    def test_empty_nonmapping_or_unknown_branches_are_rejected(self):
        for branch in ({}, [], (), '', 0, False, 'identity', {'kind': 'other'},
                       {'kind': JOINT['kind']}, {'slope': 1., 'intercept': 0.},
                       {'kind': None}, {'kind': ['identity']}):
            for channel in ('frame', 'video'):
                state = deepcopy(self.separate)
                state[channel] = branch
                with self.subTest(channel=channel, branch=branch):
                    self.assert_invalid(state)

    def test_unknown_branch_keys_are_rejected_even_for_identity(self):
        for branch in (dict(AFFINE, video_slope=0.), dict(AFFINE, unknown=0),
                       dict(AFFINE, frame=None), {'kind': 'identity', 'unknown': None},
                       {'kind': 'identity', 'video_slope': 0.}):
            for channel in ('frame', 'video'):
                state = deepcopy(self.separate)
                state[channel] = branch
                with self.subTest(channel=channel, branch=branch):
                    self.assert_invalid(state)

    def test_nonfinite_parameters_and_ridge_are_rejected_at_all_levels(self):
        for bad in (np.nan, np.inf, -np.inf):
            for name in ('slope', 'intercept', 'ridge'):
                for channel in ('frame', 'video'):
                    state = deepcopy(self.separate)
                    state[channel][name] = bad
                    with self.subTest(channel=channel, name=name, bad=bad):
                        self.assert_invalid(state)
            for kind in (JOINT['kind'], 'identity'):
                for name in ('slope', 'video_slope', 'intercept', 'ridge'):
                    state = dict(JOINT, kind=kind, **{name: bad})
                    with self.subTest(kind=kind, name=name, bad=bad):
                        self.assert_invalid(state)
            for channel in ('frame', 'video'):
                for name in ('slope', 'intercept', 'ridge'):
                    state = deepcopy(self.separate)
                    state[channel] = dict(AFFINE, kind='identity', **{name: bad})
                    with self.subTest(identity=channel, name=name, bad=bad):
                        self.assert_invalid(state)

    def test_nonscalar_or_nonreal_parameters_are_rejected(self):
        for bad in (None, True, '1.2', 'nan', [1.], {'value': 1.}, 1j, np.array(1.), 10**400):
            for name in ('slope', 'intercept', 'ridge'):
                state = {'frame': dict(AFFINE, **{name: bad}), 'video': None}
                with self.subTest(name=name, bad=bad):
                    self.assert_invalid(state)
            for name in ('slope', 'video_slope', 'intercept', 'ridge'):
                state = dict(JOINT, **{name: bad})
                with self.subTest(joint=name, bad=bad):
                    self.assert_invalid(state)

    def test_invalid_parameter_bounds_are_rejected(self):
        for name, bad in (('slope', 0.), ('slope', -.1), ('ridge', 0.), ('ridge', -1.)):
            self.assert_invalid({'frame': dict(AFFINE, **{name: bad}), 'video': None})
        for name, bad in (('slope', .249), ('slope', 3.001), ('video_slope', -.01),
                          ('video_slope', 3.001), ('intercept', -3.001),
                          ('intercept', 3.001), ('ridge', 0.)):
            with self.subTest(name=name, bad=bad):
                self.assert_invalid(dict(JOINT, **{name: bad}))

    def test_partial_identity_parameter_groups_are_rejected(self):
        for state in ({'kind': 'identity', 'slope': 1.},
                      {'kind': 'identity', 'slope': 1., 'intercept': 0.},
                      {'kind': 'identity', 'video_slope': 0.}):
            with self.subTest(state=state):
                self.assert_invalid(state)
        for branch in ({'kind': 'identity', 'slope': 1.}, {'kind': 'identity', 'intercept': 0.}):
            self.assert_invalid({'frame': branch, 'video': None})

    def test_invalid_metadata_types_and_nested_structures_are_rejected(self):
        for name, bad in (('reason', {'unknown': 1}), ('fit_role', []), ('weighting', 1),
                          ('independent_probability_calibration', 'false'),
                          ('business_risk_probability', 0)):
            with self.subTest(name=name):
                self.assert_invalid(dict(JOINT, **{name: bad}))
        for name, bad in (('reason', None), ('fit_role', {}),
                          ('independent_probability_calibration', 0)):
            self.assert_invalid({'frame': dict(AFFINE, **{name: bad}), 'video': None})

    def test_entire_state_is_validated_before_any_transform(self):
        state = {'frame': dict(AFFINE), 'video': {'kind': 'unknown'}}
        with patch('optimized_calibration_state.apply_calibration') as affine, \
                patch('optimized_calibration_state.apply_joint') as joint:
            with self.assertRaises(ValueError):
                apply_calibration_state(self.frame, self.video, state)
        affine.assert_not_called()
        joint.assert_not_called()

    def test_valid_dispatch_still_checks_nonfinite_probabilities(self):
        for state in (self.separate, JOINT, {'kind': 'identity'}):
            for frame, video in (([np.nan], .5), ([.5], np.inf), ([1.1], .5), ([.5], -.1)):
                with self.subTest(state=state, frame=frame, video=video), self.assertRaises(ValueError):
                    apply_calibration_state(frame, video, state)


if __name__ == '__main__':
    unittest.main()
