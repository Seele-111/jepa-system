import unittest
import torch

from train_jepa_tridet_lite import JEPATriDetLite


class RGBTriDetTests(unittest.TestCase):
    def test_boundary_detector_accepts_rgb_feature_width(self):
        model = JEPATriDetLite(in_features=12, hidden=16, strides=(1, 2), max_regression_bin=5, dropout=0.0)
        out = model(torch.randn(2, 9, 12), torch.ones(2, 9, dtype=torch.bool))
        self.assertEqual(tuple(out[0]["center_logits"].shape), (2, 9))
        self.assertEqual(tuple(out[0]["left_logits"].shape), (2, 9, 6))


if __name__ == "__main__": unittest.main()
