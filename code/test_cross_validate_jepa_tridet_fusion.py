import unittest

import numpy as np

from cross_validate_jepa_tridet_fusion import apply_protected_config, probability_records_to_segments


class JEPATriDetFusionTests(unittest.TestCase):
    def test_probability_records_to_segments_uses_existing_postprocess_params(self):
        records = [
            {
                "labels": np.zeros(8, dtype=np.int64),
                "probs": np.array([0.0, 0.1, 0.8, 0.9, 0.1, 0.0, 0.7, 0.8], dtype=np.float32),
            }
        ]
        params = {"threshold": 0.5, "smooth_window": 1, "min_gap": 0, "min_length": 2}

        segments = probability_records_to_segments(records, params)

        self.assertEqual(segments, [[(2, 3), (6, 7)]])

    def test_apply_protected_config_adds_only_non_overlapping_tridet_segments(self):
        base = [[(10, 20)]]
        tridet = [[(12, 18), (30, 35)]]
        config = {"enabled": True, "max_mainline_iou": 0.1, "selector_nms_iou": 0.3}

        fused = apply_protected_config(base, tridet, config)

        self.assertEqual(fused, [[(10, 20), (30, 35)]])


if __name__ == "__main__":
    unittest.main()
