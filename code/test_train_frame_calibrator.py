import unittest

import numpy as np

from train_frame_calibrator import FrameTable, make_frame_table, records_from_probabilities
from train_segment_locator import VideoRecord


class FrameCalibratorTests(unittest.TestCase):
    def test_make_frame_table_concatenates_frames_and_keeps_slices(self):
        records = [
            VideoRecord("a", np.zeros((2, 4), dtype=np.float32), np.array([0, 1])),
            VideoRecord("b", np.ones((3, 4), dtype=np.float32), np.array([1, 0, 0])),
        ]

        table = make_frame_table(records, [0, 1])

        self.assertEqual(table.x.shape, (5, 4))
        np.testing.assert_array_equal(table.y, np.array([0, 1, 1, 0, 0]))
        self.assertEqual(table.slices, [(0, 0, 2), (1, 2, 5)])

    def test_records_from_probabilities_restores_video_records(self):
        labels = [np.array([0, 1]), np.array([1, 0, 0])]
        table = FrameTable(
            x=np.zeros((5, 4), dtype=np.float32),
            y=np.array([0, 1, 1, 0, 0]),
            slices=[(0, 0, 2), (1, 2, 5)],
            names=["a", "b"],
            labels=labels,
        )
        probs = np.array([0.1, 0.9, 0.8, 0.2, 0.3], dtype=np.float32)

        records = records_from_probabilities(table, probs)

        self.assertEqual(records[0]["name"], "a")
        np.testing.assert_allclose(records[1]["probs"], np.array([0.8, 0.2, 0.3], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
