import unittest

import numpy as np

from run_strict_topology_set_split_oof import event_topology_evidence
from train_segment_locator import VideoRecord


class RunStrictTopologySetSplitOOFTests(unittest.TestCase):
    def test_event_topology_evidence_returns_requested_stack(self):
        record = VideoRecord(
            "v0",
            np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            np.zeros(2, dtype=np.int64),
        )

        evidence = event_topology_evidence(
            record,
            feature_names=["a", "b", "c"],
            evidence_names=["c", "a"],
            reducer="stack",
        )

        self.assertEqual(evidence.shape, (2, 2))
        self.assertTrue(np.allclose(evidence[:, 0], [0.3, 0.6]))
        self.assertTrue(np.allclose(evidence[:, 1], [0.1, 0.4]))


if __name__ == "__main__":
    unittest.main()
