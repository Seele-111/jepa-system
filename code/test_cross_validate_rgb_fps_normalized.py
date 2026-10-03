import unittest

import numpy as np

from cross_validate_rgb_fps_normalized import smooth_seconds


class FpsNormalizationTest(unittest.TestCase):
    def test_low_fps_is_unchanged(self):
        x = np.arange(6, dtype=np.float32)[:, None]
        np.testing.assert_array_equal(smooth_seconds(x, 8, 8), x)

    def test_high_fps_preserves_shape(self):
        x = np.arange(12, dtype=np.float32)[:, None]
        y = smooth_seconds(x, 24, 8)
        self.assertEqual(y.shape, x.shape)
        self.assertGreater(float(y[3, 0]), float(y[0, 0]))


if __name__ == "__main__":
    unittest.main()
