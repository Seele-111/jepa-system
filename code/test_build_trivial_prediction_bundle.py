import unittest

from build_trivial_prediction_bundle import transform_bundle


class BuildTrivialPredictionBundleTests(unittest.TestCase):
    def test_full_span_preserves_names_and_labels(self):
        source = {"folds": [{"fold": 0, "val_names": ["a"], "predictions": [[]], "labels": [[0, 1, 0]]}]}
        output = transform_bundle(source, "full_span")
        self.assertEqual(output["folds"][0]["predictions"], [[[0, 2]]])
        self.assertEqual(output["folds"][0]["labels"], [[0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
