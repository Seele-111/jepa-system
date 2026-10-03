import unittest

import numpy as np

from jepa_prototype_rescue import fit_prototype_rescuer


class JEPAPrototypeRescueTests(unittest.TestCase):
    def test_prototype_rescuer_scores_positive_like_candidate_above_hard_negative(self):
        x = np.array(
            [
                [1.0, 0.9, 0.8, 0.95, 0.1, 0.1, 0.0, 0.0, 0.0],
                [0.9, 1.0, 0.7, 0.90, 0.1, 0.1, 0.0, 0.0, 0.0],
                [-1.0, -0.8, -0.7, 0.92, 0.8, 0.8, 0.0, 0.0, 0.0],
                [-0.9, -1.0, -0.8, 0.88, 0.8, 0.8, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        y = np.array([1, 1, 0, 0], dtype=np.int64)

        rescuer = fit_prototype_rescuer(x, y, max_positive_prototypes=4, max_negative_prototypes=4)
        probs = rescuer.predict_proba(
            np.array(
                [
                    [0.95, 0.95, 0.75, 0.93, 0.1, 0.1, 0.0, 0.0, 0.0],
                    [-0.95, -0.9, -0.75, 0.93, 0.8, 0.8, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )[:, 1]

        self.assertGreater(float(probs[0]), 0.6)
        self.assertLess(float(probs[1]), 0.4)
        self.assertGreater(float(probs[0]), float(probs[1]) + 0.3)

    def test_prototype_rescuer_returns_low_scores_without_positive_bank(self):
        x = np.array(
            [
                [-1.0, -0.8, 0.9],
                [-0.9, -1.0, 0.8],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0], dtype=np.int64)

        rescuer = fit_prototype_rescuer(x, y)
        probs = rescuer.predict_proba(np.array([[1.0, 1.0, 0.9]], dtype=np.float32))[:, 1]

        self.assertLess(float(probs[0]), 0.05)


if __name__ == "__main__":
    unittest.main()
