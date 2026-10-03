import unittest
import numpy as np

from cross_validate_jepa_ablation import mode_channels, subset_records
from train_segment_locator import VideoRecord


class JEPAAblationTests(unittest.TestCase):
    def test_subset_records_changes_only_channels(self):
        records = [VideoRecord("a", np.arange(12, dtype=np.float32).reshape(3, 4), np.array([0, 1, 0]))]
        view = subset_records(records, [1, 3])
        self.assertEqual(view[0].signals.shape, (3, 2))
        np.testing.assert_array_equal(view[0].signals[:, 0], [1, 5, 9])
        np.testing.assert_array_equal(view[0].labels, records[0].labels)

    def test_rank_modes_resolve_named_channels(self):
        names = ["true_vjepa_raw", "true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"]
        self.assertEqual(mode_channels("vjepa_rank", names), [1])
        self.assertEqual(mode_channels("rank_triplet", names), [1, 2, 3])


if __name__ == "__main__": unittest.main()
