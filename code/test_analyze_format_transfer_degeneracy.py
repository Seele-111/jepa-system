import unittest

from analyze_format_transfer_degeneracy import audit_run


class FormatTransferDegeneracyTests(unittest.TestCase):
    def test_full_span_model_matches_trivial_baseline(self):
        run = {
            "seed": 42,
            "target_predictions": [[[0, 3]], [[0, 2]]],
            "target_labels": [[1, 1, 1, 1], [0, 1, 1]],
        }
        report = audit_run(run)
        self.assertEqual(report["exact_full_span_prediction_rate"], 1.0)
        self.assertEqual(report["comparisons"]["0.3"]["segment_f1_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
