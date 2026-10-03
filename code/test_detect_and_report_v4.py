import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    HAS_CV2 = False
else:
    HAS_CV2 = True

if HAS_CV2:
    from detect_and_report_v4 import compute_vjepa_attention, expand_sparse_frame_scores


@unittest.skipUnless(HAS_CV2, "detect_and_report_v4 requires cv2")
class DetectAndReportV4Tests(unittest.TestCase):
    def test_expand_sparse_frame_scores_interpolates_between_keyframes(self):
        dense, mask = expand_sparse_frame_scores([0, 4], [0.0, 1.0], total_frames=5)

        np.testing.assert_allclose(dense, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        np.testing.assert_array_equal(mask, np.ones(5, dtype=bool))

    def test_expand_sparse_frame_scores_fills_single_keyframe_score(self):
        dense, mask = expand_sparse_frame_scores([2], [0.7], total_frames=5)

        np.testing.assert_allclose(dense, np.full(5, 0.7))
        np.testing.assert_array_equal(mask, np.ones(5, dtype=bool))

    def test_expand_sparse_frame_scores_handles_empty_scores(self):
        dense, mask = expand_sparse_frame_scores([], [], total_frames=3)

        np.testing.assert_allclose(dense, np.zeros(3))
        np.testing.assert_array_equal(mask, np.zeros(3, dtype=bool))

    def test_compute_vjepa_attention_handles_shorter_heatmap_sequence(self):
        heatmaps = np.zeros((2, 24, 24), dtype=np.float32)
        heatmaps[0, 4:8, 4:8] = 1.0
        tubelet_to_frames = [(0, 1), (2, 3), (4, 5), (6, 7)]

        attention = compute_vjepa_attention(
            heatmaps,
            tubelet_to_frames,
            ijepa_frame_indices=[0, 6],
        )

        self.assertEqual(attention.shape, (2, 196))
        np.testing.assert_allclose(attention[1], np.full(196, 0.7))


if __name__ == "__main__":
    unittest.main()
