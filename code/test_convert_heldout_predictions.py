import unittest
from unittest.mock import patch
import numpy as np

from convert_heldout_predictions import convert
from train_segment_locator import VideoRecord


class HeldoutConversionTests(unittest.TestCase):
    @patch("convert_heldout_predictions.load_signal_dataset")
    def test_conversion_aligns_labels_by_name(self, loader):
        loader.return_value = [VideoRecord("a.mp4", np.zeros((3, 3), np.float32), np.array([0, 1, 0]))]
        bundle = convert({"summary": "s.json", "predictions": [{"video_name": "a.mp4", "segments": [[1, 1]]}]}, "data")
        self.assertEqual(bundle["folds"][0]["labels"], [[0, 1, 0]])
        self.assertEqual(bundle["folds"][0]["predictions"], [[[1, 1]]])


if __name__ == "__main__": unittest.main()
