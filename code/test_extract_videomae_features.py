import unittest

import numpy as np

from extract_videomae_features import centered_indices, interpolate_features


class VideoMAEFeatureHelpersTest(unittest.TestCase):
    def test_centered_indices_replicate_edges(self):
        got = centered_indices(3, clip_length=4, centers=np.asarray([0, 2]))
        np.testing.assert_array_equal(got, np.asarray([[0, 0, 0, 1], [0, 1, 2, 2]]))

    def test_interpolation_returns_annotation_grid(self):
        sparse = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        got = interpolate_features(sparse, np.asarray([0, 2]), 3)
        np.testing.assert_allclose(got, np.asarray([[0, 2], [1, 3], [2, 4]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
