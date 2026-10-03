import unittest

import numpy as np

from evaluate_trivial_bundle_baselines import baseline_predictions, paired_bootstrap


class TrivialBundleBaselineTests(unittest.TestCase):
    def test_baseline_predictions(self):
        labels = [np.asarray([1, 0, 0]), np.asarray([0, 0])]
        self.assertEqual(baseline_predictions(labels, "full_span"), [[(0, 2)], [(0, 1)]])
        self.assertEqual(baseline_predictions(labels, "empty"), [[], []])

    def test_identical_predictions_have_zero_bootstrap_difference(self):
        labels = [np.asarray([1, 1]), np.asarray([0, 1])]
        predictions = baseline_predictions(labels, "full_span")
        report = paired_bootstrap(predictions, predictions, labels, 0.3, 20, 1)
        self.assertEqual(report["difference_ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
