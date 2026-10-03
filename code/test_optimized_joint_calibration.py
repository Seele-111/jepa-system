import itertools
import unittest
from unittest.mock import patch

import numpy as np

from optimized_joint_calibration import apply_joint, fit_joint


ANCHOR = np.array([1., 0., 0.])
LOWER = np.array([.25, 0., -3.])
UPPER = np.array([3., 3., 3.])


def coefficients(state):
    return np.array([state['slope'], state['video_slope'], state['intercept']])


def logit(probabilities):
    p = np.clip(probabilities, 1e-5, 1 - 1e-5)
    return np.log(p) - np.log1p(-p)


def objective_and_gradient(frames, videos, labels, beta, weights=None):
    # Independent reference objective; no fitting helpers are reused.
    loss = .5 * np.sum((beta - ANCHOR) ** 2)
    gradient = beta - ANCHOR
    for index, (frame, video, label) in enumerate(zip(frames, videos, labels)):
        frame, label = np.asarray(frame), np.asarray(label)
        x = np.column_stack([logit(frame), np.full(frame.size, logit(video)),
                             np.ones(frame.size)])
        w = np.full(frame.size, 1 / frame.size) if weights is None else weights[index]
        z = x @ beta
        loss += np.sum(w * np.logaddexp(0, np.where(label == 1, -z, z)))
        gradient += x.T @ (w * (1 / (1 + np.exp(-z)) - label))
    return float(loss), gradient


class JointCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.frames = [np.array([.05, .2, .65, .85]), np.array([.1, .4, .7]),
                       np.array([.15, .6, .8, .95])]
        self.videos = [.15, .6, .9]
        self.labels = [np.array([0, 0, 1, 1]), np.array([0, 1, 1]),
                       np.array([0, 1, 1, 1])]
        self.state = {
            'kind': 'monotone_joint_logit_v1', 'slope': 1.3,
            'video_slope': .7, 'intercept': -.2,
        }

    def assert_optimal(self, frames, videos, labels, state, weights=None):
        beta = coefficients(state)
        self.assertTrue(np.isfinite(beta).all())
        self.assertTrue(np.all((beta >= LOWER) & (beta <= UPPER)))
        loss, gradient = objective_and_gradient(frames, videos, labels, beta, weights)
        initial, _ = objective_and_gradient(frames, videos, labels, ANCHOR, weights)
        self.assertLessEqual(loss, initial + 1e-10)
        # Box KKT: the projected gradient is zero, including active boundaries.
        projected = beta - np.clip(beta - gradient, LOWER, UPPER)
        np.testing.assert_allclose(projected, 0, atol=2e-6, rtol=0)

    def test_fit_metadata_bounds_and_optimum(self):
        state = fit_joint(self.frames, self.videos, self.labels)
        self.assertEqual(state['kind'], 'monotone_joint_logit_v1')
        self.assertEqual(state['ridge'], 1.)
        self.assertEqual(state['weighting'], 'inverse_video_length')
        self.assertEqual(state['fit_role'], 'outer_training_inner_OOF_only')
        self.assertIs(state['independent_probability_calibration'], False)
        self.assertIs(state['business_risk_probability'], False)
        self.assert_optimal(self.frames, self.videos, self.labels, state)

    def test_formula_float32_shape_and_no_mutation(self):
        frame = np.array([[0., .2], [.7, 1.]])
        original = frame.copy()
        original_state = self.state.copy()
        result = apply_joint(frame, .8, self.state)
        z = -.2 + 1.3 * logit(frame) + .7 * logit(.8)
        np.testing.assert_allclose(result, 1 / (1 + np.exp(-z)), atol=1e-7)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, frame.shape)
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(self.state, original_state)

    def test_frame_and_video_monotonicity(self):
        frame = np.linspace(.01, .99, 31)
        low = apply_joint(frame, .1, self.state)
        high = apply_joint(frame, .9, self.state)
        self.assertTrue(np.all(np.diff(low) > 0))
        self.assertTrue(np.all(high > low))
        zero_video = dict(self.state, video_slope=0.)
        np.testing.assert_array_equal(apply_joint(frame, 0., zero_video),
                                      apply_joint(frame, 1., zero_video))

    def test_apply_empty_scalar_and_noncontiguous_shapes(self):
        for frame in (np.empty((2, 0, 3)), np.array(.4),
                      np.arange(24).reshape(2, 3, 4)[:, :, ::2] / 24):
            with self.subTest(shape=frame.shape):
                result = apply_joint(frame, .5, self.state)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, frame.shape)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())

    def test_single_class_is_exact_identity_and_still_declared_nonindependent(self):
        for value in (0, 1):
            with self.subTest(label=value):
                labels = [np.full(frame.shape, value) for frame in self.frames]
                state = fit_joint(self.frames, self.videos, labels)
                self.assertEqual(state['kind'], 'identity')
                self.assertEqual(state['reason'], 'single_class_inner_OOF')
                np.testing.assert_array_equal(coefficients(state), ANCHOR)
                self.assertIs(state['independent_probability_calibration'], False)
                self.assertIs(state['business_risk_probability'], False)
                frame = np.array([0., .2, .8, 1.])
                np.testing.assert_array_equal(apply_joint(frame, .9, state),
                                              frame.astype(np.float32))

    def test_none_and_empty_states_are_identity(self):
        frame = np.array([0., .3, 1.])
        for state in (None, {}, {'kind': 'identity'}):
            np.testing.assert_array_equal(apply_joint(frame, .4, state),
                                          frame.astype(np.float32))

    def test_deterministic_and_does_not_mutate_training_inputs(self):
        frames = [frame.copy() for frame in self.frames]
        labels = [label.copy() for label in self.labels]
        videos = self.videos.copy()
        first = fit_joint(frames, videos, labels)
        self.assertEqual(first, fit_joint(frames, videos, labels))
        for before, after in zip(self.frames + self.labels, frames + labels):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(self.videos, videos)

    def test_default_weights_match_explicit_inverse_lengths(self):
        weights = [np.full(frame.shape, 1 / frame.size) for frame in self.frames]
        default = fit_joint(self.frames, self.videos, self.labels)
        explicit = fit_joint(self.frames, self.videos, self.labels, weights)
        scalar = fit_joint(self.frames, self.videos, self.labels, [1., 1., 1.])
        np.testing.assert_array_equal(coefficients(default), coefficients(explicit))
        np.testing.assert_array_equal(coefficients(default), coefficients(scalar))
        self.assert_optimal(self.frames, self.videos, self.labels, explicit, weights)

    def test_video_duplication_of_frames_preserves_inverse_length_objective(self):
        frames = [frame.copy() for frame in self.frames]
        labels = [label.copy() for label in self.labels]
        frames[1], labels[1] = np.tile(frames[1], 23), np.tile(labels[1], 23)
        original = fit_joint(self.frames, self.videos, self.labels)
        repeated = fit_joint(frames, self.videos, labels)
        np.testing.assert_allclose(coefficients(original), coefficients(repeated),
                                   atol=1e-8, rtol=0)

    def test_natural_imbalanced_prior_not_class_balanced(self):
        frames = [np.full(10, .5)]
        labels = [np.array([1] + [0] * 9)]
        state = fit_joint(frames, [.5], labels)
        self.assert_optimal(frames, [.5], labels, state)
        self.assertAlmostEqual(state['slope'], 1.)
        self.assertAlmostEqual(state['video_slope'], 0.)
        p = float(apply_joint(np.array(.5), .5, state))
        self.assertLess(p, .5)
        self.assertGreater(p, .1)  # Ridge keeps the natural prior from overfitting.
        self.assertAlmostEqual(p - .1 + state['intercept'], 0, places=7)

    def test_constant_frame_scores_can_use_video_signal(self):
        frames = [np.full(8, .5), np.full(8, .5)] * 12
        videos = [.05, .95] * 12
        labels = [np.zeros(8), np.ones(8)] * 12
        state = fit_joint(frames, videos, labels)
        self.assert_optimal(frames, videos, labels, state)
        self.assertGreater(state['video_slope'], 0)
        self.assertAlmostEqual(state['slope'], 1.)
        self.assertLess(float(apply_joint(.5, .05, state)),
                        float(apply_joint(.5, .95, state)))

    def test_equal_and_collinear_scores_are_stable(self):
        cases = [
            ([np.full(12, .3), np.full(4, .3)], [.3, .3],
             [np.array([0] * 9 + [1] * 3), np.array([0, 1, 1, 1])]),
            ([np.full(4, .1), np.full(4, .9)] * 6, [.1, .9] * 6,
             [np.array([0, 0, 0, 1]), np.array([0, 1, 1, 1])] * 6),
        ]
        for frames, videos, labels in cases:
            with self.subTest(videos=videos):
                state = fit_joint(frames, videos, labels)
                self.assert_optimal(frames, videos, labels, state)
                self.assertEqual(state, fit_joint(frames, videos, labels))
                if len(set(videos)) == 1:
                    outputs = [apply_joint(frame, video, state)
                               for frame, video in zip(frames, videos)]
                    self.assertEqual(float(outputs[0][0]), float(outputs[1][0]))
                    self.assertTrue(np.all(outputs[0] == outputs[0][0]))

    def test_negative_video_association_hits_zero_not_negative_slope(self):
        frames = [np.array([.2, .4, .7, .9])] * 20
        videos = [.05, .95] * 10
        labels = [np.array([0, 1, 1, 1]), np.array([0, 0, 0, 1])] * 10
        state = fit_joint(frames, videos, labels)
        self.assert_optimal(frames, videos, labels, state)
        self.assertEqual(state['video_slope'], 0.)
        self.assertGreater(abs(state['intercept']), .01)

    def test_active_parameter_bounds(self):
        cases = [
            ([np.array([.1, .9])] * 80, [.5] * 80,
             [np.array([1, 0])] * 80, 'slope', .25),
            ([np.array([.49, .51])] * 600, [.5] * 600,
             [np.array([0, 1])] * 600, 'slope', 3.),
            ([np.full(10, .5)] * 600, [.49, .51] * 300,
             [np.zeros(10), np.ones(10)] * 300, 'video_slope', 3.),
            ([np.full(100, .5)] * 200, [.5] * 200,
             [np.array([0] + [1] * 99)] * 200, 'intercept', 3.),
            ([np.full(100, .5)] * 200, [.5] * 200,
             [np.array([1] + [0] * 99)] * 200, 'intercept', -3.),
        ]
        for frames, videos, labels, name, bound in cases:
            with self.subTest(parameter=name, bound=bound):
                state = fit_joint(frames, videos, labels)
                self.assertEqual(state[name], bound)
                self.assert_optimal(frames, videos, labels, state)

    def test_probability_endpoints_and_all_parameter_corners_stay_finite(self):
        frames = [np.array([0., 1., 0., 1.])]
        labels = [np.array([0, 1, 1, 0])]
        for video in (0., 1.):
            state = fit_joint(frames, [video], labels)
            self.assert_optimal(frames, [video], labels, state)
        for a, c, b in itertools.product((.25, 3.), (0., 3.), (-3., 3.)):
            state = dict(self.state, slope=a, video_slope=c, intercept=b)
            for video in (0., 1.):
                result = apply_joint(frames[0], video, state)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_custom_weighted_objective_and_no_weight_mutation(self):
        weights = [np.array([.2, .1, .4, .3]), np.array([2., 1., .5]),
                   np.array([.1, .2, .3, .4])]
        before = [w.copy() for w in weights]
        state = fit_joint(self.frames, self.videos, self.labels, weights)
        self.assert_optimal(self.frames, self.videos, self.labels, state, weights)
        for original, current in zip(before, weights):
            np.testing.assert_array_equal(original, current)
        masses = [2., .3, 4.]
        explicit = [np.full(frame.shape, mass / frame.size)
                    for frame, mass in zip(self.frames, masses)]
        scalar = fit_joint(self.frames, self.videos, self.labels, masses)
        self.assert_optimal(self.frames, self.videos, self.labels, scalar, explicit)

    def test_rejects_empty_and_mismatched_training_shapes(self):
        cases = [([], [], []), ([[.5]], [], [[0]]), ([[.5]], [.5], []),
                 ([[]], [.5], [[]]), ([.5], [.5], [[0]]),
                 ([[[.2, .3]]], [.5], [[[0, 1]]]),
                 ([[.2, .3]], [.5], [[0]]),
                 ([[.2, .3]], [.5], [[[0], [1]]]),
                 ([[.2, .3]], [[.5]], [[0, 1]]),
                 (None, [.5], [[0]]), ('frames', [.5], [[0]])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                fit_joint(*args)

    def test_rejects_invalid_probabilities_fit_and_apply_including_identity(self):
        bad = [np.nan, np.inf, -np.inf, -.01, 1.01, 1j, 'invalid', None]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    fit_joint([[value, .5]], [.5], [[0, 1]])
                with self.assertRaises(ValueError):
                    fit_joint([[.2, .5]], [value], [[0, 1]])
                for state in (self.state, None, {'kind': 'identity'}):
                    with self.assertRaises(ValueError):
                        apply_joint([value], .5, state)
                    with self.assertRaises(ValueError):
                        apply_joint([.5], value, state)
        for video in ([.5], np.array([.5]), [[.5]]):
            with self.assertRaises(ValueError):
                apply_joint(np.empty(0), video, None)

    def test_rejects_nonbinary_or_nonfinite_labels(self):
        for label in ([-1, 1], [0, 2], [0, .5], [0, np.nan], [0, np.inf],
                      [0, 1j], ['invalid', 1], [None, 1]):
            with self.subTest(labels=label), self.assertRaises(ValueError):
                fit_joint([[.2, .8]], [.5], [label])

    def test_rejects_bad_weights_even_before_single_class_fallback(self):
        for weights in ([], [1., 1.], [0.], [-1.], [np.nan], [np.inf],
                        [[1.]], [[1., 0.]], [[1., -1.]], [[1., np.nan]],
                        [[[1., 1.]]], [1j], [np.finfo(np.float64).max]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fit_joint([[.2, .8]], [.5], [[0, 0]], weights)
        with self.assertRaises(ValueError):
            fit_joint([[.2, .8]], [.5], [[0, 0]], 1.)

    def test_ridge_is_fixed_one_and_checked_before_single_class_fallback(self):
        for ridge in (0, -1, .5, 2, np.nan, np.inf, [1.], 1j, None):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                fit_joint([[.2, .8]], [.5], [[1, 1]], ridge=ridge)
        self.assertEqual(fit_joint(self.frames, self.videos, self.labels, ridge=1.),
                         fit_joint(self.frames, self.videos, self.labels))

    def test_invalid_joint_states(self):
        cases = [1, [], 'state', {'kind': 'other'}, {'kind': 'monotone_joint_logit_v1'}]
        for name, bad in (('slope', .249), ('slope', 3.001), ('video_slope', -.01),
                          ('video_slope', 3.001), ('intercept', -3.001),
                          ('intercept', 3.001), ('intercept', np.nan),
                          ('slope', np.inf), ('video_slope', [1.]), ('slope', 1j)):
            cases.append(dict(self.state, **{name: bad}))
        for state in cases:
            with self.subTest(state=state), self.assertRaises(ValueError):
                apply_joint([.3, .7], .6, state)

    def test_backtracking_reduces_an_oversized_newton_step(self):
        # Force an overlong *descent* direction so full steps are rejected.
        from optimized_joint_calibration import _newton_direction

        calls = []

        def oversized(beta, gradient, hessian):
            calls.append(1)
            return 8 * _newton_direction(beta, gradient, hessian)

        with patch('optimized_joint_calibration._newton_direction', oversized):
            state = fit_joint(self.frames, self.videos, self.labels)
        self.assertTrue(calls)
        self.assert_optimal(self.frames, self.videos, self.labels, state)

    def test_deterministic_random_cases_satisfy_box_kkt(self):
        rng = np.random.default_rng(7719)
        for case in range(12):
            frames = [rng.uniform(.001, .999, size=int(rng.integers(4, 30)))
                      for _ in range(8)]
            videos = rng.uniform(.001, .999, size=len(frames))
            labels = [rng.integers(0, 2, size=frame.size) for frame in frames]
            with self.subTest(case=case):
                state = fit_joint(frames, videos, labels)
                self.assert_optimal(frames, videos, labels, state)


if __name__ == '__main__':
    unittest.main()
