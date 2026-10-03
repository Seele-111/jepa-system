"""Selection safeguards and fractional content risk tests for v4."""
import copy
import unittest
import numpy as np
from run_compact_experiment_v4 import configurations, utility_v4, choose_v4, decode_predictions, blended, CANDIDATES
from run_optimization_selection import Probabilities


class CompactSelection(unittest.TestCase):
    def test_grid_is_small_frozen_and_no_probability_multiplication(self):
        grid = list(configurations()); self.assertEqual(len(grid), 40)
        self.assertTrue(all(c.get('video_strength', 0) == 0 for c in grid))
        self.assertTrue(all(c['min_seconds'] == .12 for c in grid))
        self.assertEqual(len(CANDIDATES), 5)

    def test_fractional_normal_mass_not_underpenalized(self):
        s = np.zeros(12); s[9] = .5
        self.assertAlmostEqual(float(utility_v4(s, .5, 1)), -.2)
        self.assertEqual(float(utility_v4(np.zeros(12), 0, 0)), 0)
        b = np.stack([s, s]); np.testing.assert_allclose(utility_v4(b, np.array([.5, 1]), np.ones(2)), [-.2, -.1])

    def test_chooser_ignores_unlisted_outer_labels_and_scores(self):
        records = []
        for i in range(5):
            labels = np.zeros(20, np.uint8)
            if i != 0: labels[5:15] = 1
            records.append({'sha256': str(i), 'fps': 10., 'frames': 20, 'event_count': int(i != 0), 'labels': labels})
        frame = {i: np.where(records[i]['labels'], .8, .1).astype(np.float32) for i in range(5)}
        p = Probabilities(frame, {i: .9 for i in range(5)}, None)
        before = choose_v4(records, [0, 1, 2, 3], p, 1)
        modified = copy.deepcopy(records); modified[4]['labels'][:] = 0; modified[4]['event_count'] = 0
        p.frame[4][:] = .99; p.video[4] = 0
        after = choose_v4(modified, [0, 1, 2, 3], p, 1)
        self.assertEqual(before, after)

    def test_blend_does_not_add_vote_or_change_probability_range(self):
        a = Probabilities({0: np.array([.2, .8], np.float32)}, {0: .4}, None)
        b = Probabilities({0: np.array([.8, .2], np.float32)}, {0: .9}, None)
        p = blended((('a', .7), ('b', .3)), {'a': a, 'b': b}, [0])
        np.testing.assert_allclose(p.frame[0], [.38, .62], atol=1e-7); self.assertAlmostEqual(p.video[0], .55)
        with self.assertRaises(ValueError): blended((('a', .8),), {'a': a}, [0])


if __name__ == '__main__': unittest.main()
