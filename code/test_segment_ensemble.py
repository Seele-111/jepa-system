import unittest

import numpy as np

from segment_ensemble import average_prediction_records, records_to_segments


class SegmentEnsembleTests(unittest.TestCase):
    def test_average_prediction_records_averages_probabilities_by_position(self):
        outputs = [
            [
                {"name": "a", "labels": np.array([0, 1]), "probs": np.array([0.2, 0.8], dtype=np.float32)},
                {"name": "b", "labels": np.array([1, 0]), "probs": np.array([0.6, 0.4], dtype=np.float32)},
            ],
            [
                {"name": "a", "labels": np.array([0, 1]), "probs": np.array([0.4, 0.6], dtype=np.float32)},
                {"name": "b", "labels": np.array([1, 0]), "probs": np.array([0.8, 0.2], dtype=np.float32)},
            ],
        ]

        averaged = average_prediction_records(outputs)

        self.assertEqual([item["name"] for item in averaged], ["a", "b"])
        np.testing.assert_allclose(averaged[0]["probs"], np.array([0.3, 0.7], dtype=np.float32))
        np.testing.assert_array_equal(averaged[1]["labels"], np.array([1, 0]))

    def test_average_prediction_records_rejects_mismatched_order(self):
        outputs = [
            [{"name": "a", "labels": np.array([0]), "probs": np.array([0.1], dtype=np.float32)}],
            [{"name": "b", "labels": np.array([0]), "probs": np.array([0.1], dtype=np.float32)}],
        ]

        with self.assertRaises(ValueError):
            average_prediction_records(outputs)

    def test_records_to_segments_uses_existing_postprocess_params(self):
        records = [
            {
                "name": "clip",
                "labels": np.array([0, 1, 1, 0], dtype=np.int64),
                "probs": np.array([0.1, 0.8, 0.7, 0.2], dtype=np.float32),
            }
        ]

        segments = records_to_segments(
            records,
            {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 1},
        )

        self.assertEqual(segments, [[(1, 2)]])


if __name__ == "__main__":
    unittest.main()
