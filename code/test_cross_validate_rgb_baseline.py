import unittest
import numpy as np

from cross_validate_rgb_baseline import nested_indices
from train_segment_locator import VideoRecord


class RGBBaselineCVTests(unittest.TestCase):
    def test_nested_indices_are_disjoint(self):
        records = [VideoRecord(str(i), np.zeros((8, 3), np.float32), np.full(8, i % 2, np.int64)) for i in range(20)]
        outer = [1, 5, 9, 13]
        fit_idx, cal_idx = nested_indices(records, outer, 0.25, 7)
        self.assertFalse(set(fit_idx) & set(cal_idx))
        self.assertFalse(set(outer) & (set(fit_idx) | set(cal_idx)))
        self.assertEqual(set(fit_idx) | set(cal_idx) | set(outer), set(range(20)))


if __name__ == "__main__":
    unittest.main()
