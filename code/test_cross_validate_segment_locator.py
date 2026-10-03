import unittest

from cross_validate_segment_locator import aggregate_fold_metrics


class CrossValidateSegmentLocatorTests(unittest.TestCase):
    def test_aggregate_fold_metrics_reports_means_stds_and_counts(self):
        folds = [
            {
                "validation": {
                    "frame": {"precision": 0.5, "recall": 0.25, "f1": 1.0, "tp": 1, "fp": 2, "fn": 3},
                    "segment": {"precision": 0.25, "recall": 0.5, "f1": 0.4, "tp": 4, "fp": 5, "fn": 6},
                }
            },
            {
                "validation": {
                    "frame": {"precision": 1.0, "recall": 0.75, "f1": 0.0, "tp": 7, "fp": 8, "fn": 9},
                    "segment": {"precision": 0.75, "recall": 1.0, "f1": 0.8, "tp": 10, "fp": 11, "fn": 12},
                }
            },
        ]

        aggregate = aggregate_fold_metrics(folds)

        self.assertAlmostEqual(aggregate["frame"]["precision_mean"], 0.75)
        self.assertAlmostEqual(aggregate["frame"]["f1_std"], 0.5)
        self.assertEqual(aggregate["frame"]["tp"], 8)
        self.assertAlmostEqual(aggregate["segment"]["recall_mean"], 0.75)
        self.assertEqual(aggregate["segment"]["fn"], 18)


if __name__ == "__main__":
    unittest.main()
