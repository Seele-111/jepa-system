import unittest

import numpy as np

from build_jepa_event_dataset import event_feature_names, make_event_features


class BuildJEPAEventDatasetTests(unittest.TestCase):
    def test_make_event_features_preserves_length_and_base_channels(self):
        signals = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.3, 0.5],
                [0.8, 0.1, 0.7],
                [0.2, 0.9, 0.4],
            ],
            dtype=np.float32,
        )

        features = make_event_features(signals, windows=[3])
        names = event_feature_names([3])

        self.assertEqual(features.shape[0], signals.shape[0])
        self.assertEqual(features.shape[1], len(names))
        np.testing.assert_allclose(features[:, :3], signals)
        self.assertTrue(np.isfinite(features).all())

    def test_make_event_features_adds_cross_jepa_agreement(self):
        signals = np.ones((5, 3), dtype=np.float32)

        features = make_event_features(signals, windows=[3])
        agreement_idx = event_feature_names([3]).index("vi_agreement")

        np.testing.assert_allclose(features[:, agreement_idx], np.ones(5), atol=1e-5)

    def test_make_event_features_adds_video_internal_calibration_features(self):
        signals = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.8, 0.5],
                [2.0, 0.6, 1.0],
                [3.0, 0.4, 0.5],
                [4.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        )

        names = event_feature_names([3])
        features = make_event_features(signals, windows=[3])

        for name in [
            "true_vjepa_raw_robust_z",
            "true_ijepa_dense_raw_rank",
            "dual_jepa_composite_second_delta",
            "dual_jepa_composite_long_contrast_w31",
        ]:
            self.assertIn(name, names)
        self.assertEqual(features.shape[1], len(names))
        self.assertTrue(np.isfinite(features).all())

        rank_idx = names.index("true_vjepa_raw_rank")
        self.assertTrue(np.all(np.diff(features[:, rank_idx]) >= 0.0))

    def test_feature_names_match_generated_column_order(self):
        signals = np.array(
            [
                [0.0, 3.0, 1.0],
                [2.0, 1.0, 2.0],
                [3.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )

        names = event_feature_names([3])
        features = make_event_features(signals, windows=[3])

        np.testing.assert_allclose(features[:, names.index("true_vjepa_raw_delta")], [0.0, 2.0, 1.0])
        np.testing.assert_allclose(features[:, names.index("true_vjepa_raw_abs_delta")], [0.0, 2.0, 1.0])
        np.testing.assert_allclose(features[:, names.index("true_ijepa_dense_raw_delta")], [0.0, -2.0, 1.0])
        np.testing.assert_allclose(features[:, names.index("true_ijepa_dense_raw_abs_delta")], [0.0, 2.0, 1.0])
        np.testing.assert_allclose(features[:, names.index("dual_jepa_composite_second_delta")], [0.0, 1.0, -3.0])
        np.testing.assert_allclose(features[:, names.index("dual_jepa_composite_abs_second_delta")], [0.0, 1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
