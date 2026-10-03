"""Focused unittest coverage; run with Python -B to avoid other file writes."""

from itertools import product
import math
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from optimized_duration_decoder_v4 import decode_duration


def make_config(**overrides):
    config = {
        "kind": "duration-logit-v4",
        "threshold": 0.5,
        "transition_seconds": 0.0,
        "min_seconds": 0.0,
    }
    config.update(overrides)
    return config


def transition_count(bits):
    previous = 0
    count = 0
    for bit in bits:
        count += abs(bit - previous)
        previous = bit
    return count


def frame_rewards(probabilities, fps, threshold):
    # Independent scalar implementation of the specified, unscaled energy.
    threshold_logit = math.log(threshold) - math.log1p(-threshold)
    rewards = []
    for probability in probabilities:
        clipped = max(1e-6, min(1.0 - 1e-6, float(probability)))
        rewards.append(
            (math.log(clipped) - math.log1p(-clipped) - threshold_logit) / fps
        )
    return rewards


def energy(bits, rewards, transition_seconds):
    return sum(bit * reward for bit, reward in zip(bits, rewards)) - (
        transition_seconds * transition_count(bits)
    )


def bits_to_spans(bits):
    spans = []
    start = None
    for frame, bit in enumerate(tuple(bits) + (0,)):
        if bit and start is None:
            start = frame
        elif not bit and start is not None:
            spans.append((start, frame - 1))
            start = None
    return spans


class DurationDecoderV4Tests(unittest.TestCase):
    def assert_energy_optimal(self, probabilities, fps, config):
        spans = decode_duration(probabilities, fps, 1.0, config)
        bits = [0] * len(probabilities)
        previous_end = -2
        for start, end in spans:
            self.assertIs(type(start), int)
            self.assertIs(type(end), int)
            self.assertLessEqual(0, start)
            self.assertLessEqual(start, end)
            self.assertLess(end, len(bits))
            self.assertGreater(start, previous_end + 1)
            bits[start : end + 1] = [1] * (end - start + 1)
            previous_end = end
        rewards = frame_rewards(probabilities, fps, config["threshold"])
        best = max(
            energy(candidate, rewards, config["transition_seconds"])
            for candidate in product((0, 1), repeat=len(bits))
        )
        actual = energy(bits, rewards, config["transition_seconds"])
        self.assertAlmostEqual(actual, best, delta=1e-12 * max(1.0, abs(best)))

    def test_exhaustive_binary_sequence_energy_oracle(self):
        # Every binary sequence for every p in {0.15, 0.5, 0.85}^T, T <= 5.
        # No seed or random sampling: 1,456 inputs, 37,324 candidate sequences.
        settings = ((1.0, 0.0), (7.0, 0.03), (0.25, 0.8), (120.0, 0.002))
        case_count = 0
        candidate_count = 0
        for length in range(6):
            for probabilities in product((0.15, 0.5, 0.85), repeat=length):
                for fps, transition_seconds in settings:
                    with self.subTest(p=probabilities, fps=fps, cost=transition_seconds):
                        self.assert_energy_optimal(
                            probabilities,
                            fps,
                            make_config(transition_seconds=transition_seconds),
                        )
                    case_count += 1
                    candidate_count += 2**length
        self.assertEqual(case_count, 1456)
        self.assertEqual(candidate_count, 37324)

    def test_longer_small_t_energy_oracle_nondefault_threshold(self):
        for length in (6, 7, 8):
            cases = (
                np.linspace(0.0, 1.0, length),
                [0.8 if frame % 2 else 0.2 for frame in range(length)],
                [((frame * 7 + 3) % 11) / 10.0 for frame in range(length)],
            )
            for probabilities in cases:
                with self.subTest(length=length, p=list(probabilities)):
                    self.assert_energy_optimal(
                        probabilities,
                        7.0,
                        make_config(threshold=0.37, transition_seconds=0.06),
                    )

    def test_exhaustive_deterministic_tie_order(self):
        # For p in {0.5, 0.75}, all nonzero rewards have the same magnitude.
        # Integer energies give an exact independent oracle for score ties.
        for length in range(7):
            for positive in product((0, 1), repeat=length):
                best = max(
                    product((0, 1), repeat=length),
                    key=lambda bits: (
                        sum(bit * reward for bit, reward in zip(bits, positive)),
                        -transition_count(bits),
                        tuple(-bit for bit in reversed(bits)),
                    ),
                )
                probabilities = [0.75 if reward else 0.5 for reward in positive]
                with self.subTest(positive=positive):
                    self.assertEqual(
                        decode_duration(probabilities, 3.0, 1.0, make_config()),
                        bits_to_spans(best),
                    )

    def test_empty_sequences(self):
        for probabilities in ([], (), np.array([], dtype=float)):
            with self.subTest(p=probabilities):
                self.assertEqual(decode_duration(probabilities, 30, 0, make_config()), [])

    def test_constant_sequences_and_no_forced_event(self):
        for probability, expected in (
            (0.0, []),
            (0.2, []),
            (0.5, []),
            (0.8, [(0, 19)]),
            (1.0, [(0, 19)]),
        ):
            with self.subTest(p=probability):
                self.assertEqual(
                    decode_duration([probability] * 20, 10, 1, make_config()), expected
                )
        self.assertEqual(
            decode_duration(
                [0.6] * 20, 30, 1, make_config(transition_seconds=100.0)
            ),
            [],
        )

    def test_single_frame_and_no_terminal_charge(self):
        # logit(0.75) > 1, but < 2: one entry is affordable, entry+exit is not.
        self.assertEqual(
            decode_duration([0.75], 1, 1, make_config(transition_seconds=1.0)),
            [(0, 0)],
        )
        self.assertEqual(
            decode_duration([0.75, 0.01], 1, 1, make_config(transition_seconds=1.0)),
            [],
        )
        for probability in (0.0, 0.5, 0.6, 1.0):
            with self.subTest(p=probability):
                expected = [(0, 0)] if probability > 0.5 else []
                self.assertEqual(
                    decode_duration([probability], 30, 1, make_config()), expected
                )

    def test_same_physical_duration_at_low_and_high_fps(self):
        for fps in (0.5, 1, 10, 120, 1000):
            count = int(2 * fps)
            with self.subTest(fps=fps):
                self.assertEqual(
                    decode_duration(
                        [0.8] * count,
                        fps,
                        1,
                        make_config(transition_seconds=1.0, min_seconds=2.0),
                    ),
                    [(0, count - 1)],
                )

    def test_fps_changes_integrated_single_frame_evidence(self):
        for fps, expected in ((0.25, [(0, 0)]), (1, [(0, 0)]), (120, [])):
            with self.subTest(fps=fps):
                self.assertEqual(
                    decode_duration([0.8], fps, 1, make_config(transition_seconds=1.0)),
                    expected,
                )

    def test_extreme_finite_fps_and_duration_are_handled(self):
        self.assertEqual(
            decode_duration([0.8], 1e-308, 1, make_config(transition_seconds=1.0)),
            [(0, 0)],
        )
        self.assertEqual(
            decode_duration([0.8], 1e308, 1, make_config(transition_seconds=1e308)),
            [],
        )
        self.assertEqual(
            decode_duration([0.8], 1e308, 1, make_config(min_seconds=1e308)), []
        )
        self.assertEqual(
            decode_duration([0.8], 1e308, 1, make_config()), [(0, 0)]
        )

    def test_transition_penalty_bridges_weak_negative_gap(self):
        probabilities = [0.8] * 10 + [0.49] + [0.8] * 10
        self.assertEqual(
            decode_duration(probabilities, 10, 1, make_config()), [(0, 9), (11, 20)]
        )
        self.assertEqual(
            decode_duration(
                probabilities, 10, 1, make_config(transition_seconds=0.01)
            ),
            [(0, 20)],
        )

    def test_equal_energy_prefers_fewer_transitions_then_zero(self):
        self.assertEqual(decode_duration([0.5] * 10, 10, 1, make_config()), [])
        self.assertEqual(
            decode_duration([0.75, 0.5, 0.75], 10, 1, make_config()), [(0, 2)]
        )
        # Do not add a leading zero-evidence frame; keep trailing one to avoid
        # an unnecessary exit when energy is equal, even with a zero cost.
        self.assertEqual(
            decode_duration([0.5, 0.75, 0.5], 10, 1, make_config()), [(1, 2)]
        )
        tied_cost = math.log(0.75) - math.log1p(-0.75)
        self.assertEqual(
            decode_duration([0.75], 1, 1, make_config(transition_seconds=tied_cost)),
            [],
        )

    def test_no_seed_or_high_confidence_anchor_required(self):
        probabilities = [0.55] * 20
        config = make_config(transition_seconds=0.3)
        with patch.object(np.random, "seed", side_effect=AssertionError("no seed")):
            for _ in range(3):
                self.assertEqual(
                    decode_duration(probabilities, 10, 1, config), [(0, 19)]
                )

    def test_short_spike_is_removed_only_after_decoding(self):
        probabilities = [0.01, 0.99, 0.4, 0.4, 0.01]
        self.assertEqual(
            decode_duration(probabilities, 1, 1, make_config()), [(1, 1)]
        )
        # A longer positive-energy event exists, but it is NOT the unconstrained
        # optimum. The post-filter must not reoptimize or expand the spike.
        self.assertEqual(
            decode_duration(probabilities, 1, 1, make_config(min_seconds=2)), []
        )
        self.assertEqual(
            decode_duration(
                [0.9, 0.1, 0.9, 0.9], 1, 1, make_config(min_seconds=2)
            ),
            [(2, 3)],
        )

    def test_inclusive_spans_and_ceil_minimum_duration(self):
        probabilities = [0.8, 0.8, 0.1, 0.8]
        for min_seconds, expected in (
            (0.0, [(0, 1), (3, 3)]),
            (0.05, [(0, 1), (3, 3)]),
            (0.0500000001, [(0, 1)]),
            (0.1, [(0, 1)]),
            (0.1000000001, []),
        ):
            with self.subTest(min_seconds=min_seconds):
                self.assertEqual(
                    decode_duration(
                        probabilities, 20, 1, make_config(min_seconds=min_seconds)
                    ),
                    expected,
                )
        self.assertEqual(
            decode_duration([0.8, 0.8], 1, 1, make_config(min_seconds=2.0000000005)),
            [(0, 1)],
        )
        self.assertEqual(
            decode_duration([0.8, 0.8], 1, 1, make_config(min_seconds=2.000000002)),
            [],
        )

    def test_video_gate_is_strictly_less_and_never_multiplies(self):
        probabilities = [0.8] * 3
        for video_probability in (0.0, 0.1, 1.0):
            with self.subTest(video_probability=video_probability):
                self.assertEqual(
                    decode_duration(probabilities, 30, video_probability, make_config()),
                    [(0, 2)],
                )
        for video_probability, expected in ((0.49, []), (0.5, [(0, 2)]), (1, [(0, 2)])):
            with self.subTest(video_probability=video_probability):
                self.assertEqual(
                    decode_duration(
                        probabilities,
                        30,
                        video_probability,
                        make_config(video_threshold=0.5),
                    ),
                    expected,
                )
        self.assertEqual(
            decode_duration(probabilities, 30, 0.99, make_config(video_threshold=1)), []
        )
        self.assertEqual(
            decode_duration(probabilities, 30, 1, make_config(video_threshold=1)),
            [(0, 2)],
        )

    def test_clipping_applies_to_probabilities_not_threshold(self):
        for probabilities in ([0.0, 1.0], [1e-6, 1.0 - 1e-6]):
            with self.subTest(p=probabilities):
                self.assertEqual(
                    decode_duration(probabilities, 1, 1, make_config()), [(1, 1)]
                )
        self.assertEqual(
            decode_duration([0.0], 1, 1, make_config(threshold=1e-9)), [(0, 0)]
        )
        self.assertEqual(
            decode_duration([1.0], 1, 1, make_config(threshold=1 - 1e-9)), []
        )

    def test_inputs_are_not_mutated_and_numpy_scalars_are_valid(self):
        probabilities = np.array([0.0, 0.8, 1.0], dtype=np.float32)
        before = probabilities.copy()
        probabilities.flags.writeable = False
        config = make_config(
            threshold=np.float64(0.5),
            transition_seconds=np.float32(0),
            min_seconds=np.int64(0),
            video_threshold=np.float32(0),
        )
        before_config = config.copy()
        self.assertEqual(
            decode_duration(
                probabilities,
                np.int64(30),
                np.float32(0.2),
                MappingProxyType(config),
            ),
            [(1, 2)],
        )
        np.testing.assert_array_equal(probabilities, before)
        self.assertEqual(config, before_config)
        self.assertEqual(decode_duration([0, 1], 30, 1, make_config()), [(1, 1)])

    def test_invalid_probability_values_shapes_and_types(self):
        cases = (
            [math.nan], [math.inf], [-math.inf], [-1e-12], [1 + 1e-12],
            [[0.5]], np.empty((0, 1)), [[], []], [[0.2], [0.3, 0.4]],
            0.5, np.array(0.5), None, "0.5", ["0.5"], [0.5 + 0j], [True],
            np.array([0.5], dtype=object), np.array([], dtype=object),
        )
        for probabilities in cases:
            with self.subTest(p=repr(probabilities)):
                with self.assertRaises(ValueError):
                    decode_duration(probabilities, 30, 1, make_config())

    def test_invalid_fps(self):
        for fps in (
            0, -1, math.nan, math.inf, -math.inf, True, np.bool_(False),
            "30", None, 30 + 0j, np.array(30), [30], 10**1000,
        ):
            with self.subTest(fps=repr(fps)):
                with self.assertRaises(ValueError):
                    decode_duration([0.8], fps, 1, make_config())

    def test_invalid_config_scalar_values(self):
        nonreal_or_nonfinite = (
            math.nan, math.inf, -math.inf, True, np.bool_(False),
            "0.5", None, 0.5 + 0j, np.array(0.5), [0.5],
        )
        invalid_by_field = {
            "threshold": nonreal_or_nonfinite + (0, 1, -0.1, 1.1),
            "transition_seconds": nonreal_or_nonfinite + (-1e-12,),
            "min_seconds": nonreal_or_nonfinite + (-1e-12,),
            "video_threshold": nonreal_or_nonfinite + (-1e-12, 1 + 1e-12),
        }
        for field, values in invalid_by_field.items():
            for value in values:
                with self.subTest(field=field, value=repr(value)):
                    with self.assertRaises(ValueError):
                        decode_duration([0.8], 30, 1, make_config(**{field: value}))

    def test_invalid_video_probability_even_with_default_gate(self):
        for video_probability in (
            -1e-12, 1 + 1e-12, math.nan, math.inf, -math.inf,
            True, np.bool_(False), "0.5", None, 0.5 + 0j, np.array(0.5), [0.5],
        ):
            with self.subTest(video_probability=repr(video_probability)):
                with self.assertRaises(ValueError):
                    decode_duration([0.8], 30, video_probability, make_config())

    def test_config_mapping_kind_required_fields_and_unknown_fields(self):
        for config in (None, [], "duration-logit-v4", 0):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    decode_duration([0.8], 30, 1, config)
        for kind in ("", "duration-logit-v3", "unknown", 4, None, np.array([1, 2])):
            with self.subTest(kind=repr(kind)):
                with self.assertRaises(ValueError):
                    decode_duration([0.8], 30, 1, make_config(kind=kind))
        for field in ("kind", "threshold", "transition_seconds", "min_seconds"):
            config = make_config()
            del config[field]
            with self.subTest(missing=field):
                with self.assertRaises(ValueError):
                    decode_duration([0.8], 30, 1, config)
        for field in ("unknown", "seed", "min_frames", "duration_ratio", "fps", None, 1):
            config = make_config()
            config[field] = 0
            with self.subTest(unknown=field):
                with self.assertRaises(ValueError):
                    decode_duration([0.8], 30, 1, config)

    def test_empty_or_closed_gate_does_not_bypass_validation(self):
        config = make_config(video_threshold=0.9)
        cases = (
            ([math.nan], 30, 0.0, config),
            ([[0.8]], 30, 0.0, config),
            ([0.8], 0, 0.0, config),
            ([0.8], 30, 0.0, make_config(video_threshold=0.9, min_seconds=-1)),
            ([], 0, 0.0, config),
            ([], 30, math.nan, config),
            ([], 30, 0.0, make_config(kind="unknown")),
            ([], 30, 0.0, make_config(unrecognized=0)),
        )
        for probabilities, fps, video_probability, current_config in cases:
            with self.subTest(p=probabilities, fps=fps, config=current_config):
                with self.assertRaises(ValueError):
                    decode_duration(probabilities, fps, video_probability, current_config)


if __name__ == "__main__":
    unittest.main()
