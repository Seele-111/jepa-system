import unittest

import numpy as np

from cross_validate_strict_graph_reranker import (
    apply_graph_config,
    apply_protected_config,
    apply_replacement_config,
    nested_split,
)
from train_segment_locator import VideoRecord


class StrictGraphRerankerTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            VideoRecord(str(idx), np.zeros((4, 3), dtype=np.float32), np.asarray([0, idx % 2, 1, 1]))
            for idx in range(12)
        ]

    def test_nested_split_is_disjoint_and_complete(self):
        outer = [0, 1, 2]
        fit, calibration = nested_split(self.records, outer, 0.25, 42)
        self.assertFalse(set(fit) & set(calibration))
        self.assertFalse(set(outer) & (set(fit) | set(calibration)))
        self.assertEqual(set(fit) | set(calibration) | set(outer), set(range(12)))

    def test_disabled_configs_leave_predictions_unchanged(self):
        predictions = [[(0, 3)]]
        self.assertIs(apply_replacement_config(predictions, [[]], [np.asarray([])], {"enabled": False}), predictions)
        self.assertIs(apply_graph_config(predictions, [[]], [np.asarray([])], predictions, {"enabled": False}), predictions)
        self.assertIs(apply_protected_config(predictions, [[]], {"enabled": False}), predictions)


if __name__ == "__main__":
    unittest.main()
