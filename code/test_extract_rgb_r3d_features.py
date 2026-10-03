import unittest
import numpy as np

from extract_rgb_r3d_features import centered_indices


class RGBFeatureTests(unittest.TestCase):
    def test_centered_indices_replicate_edges_and_align(self):
        x = centered_indices(4, 4)
        self.assertEqual(x.shape, (4, 4))
        np.testing.assert_array_equal(x[0], [0, 0, 0, 1])
        np.testing.assert_array_equal(x[-1], [1, 2, 3, 3])
        self.assertTrue(np.all((x >= 0) & (x < 4)))

    def test_explicit_centers_preserve_consecutive_clips(self):
        x = centered_indices(10, 4, np.asarray([0, 4, 9]))
        self.assertEqual(x.shape, (3, 4))
        np.testing.assert_array_equal(x[1], [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
