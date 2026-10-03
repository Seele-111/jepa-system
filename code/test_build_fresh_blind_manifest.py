import unittest

from build_fresh_blind_manifest import bucket, select_names


class FreshBlindManifestTests(unittest.TestCase):
    def test_bucket_and_zero_overlap_selection(self):
        card = {"videos": [
            {"video_name": "a.mp4", "frame_count": 49, "positive_frame_ratio": 0.0},
            {"video_name": "b.mp4", "frame_count": 121, "positive_frame_ratio": 0.8},
            {"video_name": "c.mp4", "frame_count": 61, "positive_frame_ratio": 0.4},
        ]}
        self.assertEqual(bucket(card["videos"][0]), "zero_short")
        names, _ = select_names(card, {"b.mp4"}, 2, 3)
        self.assertNotIn("b.mp4", names)
        self.assertEqual(len(names), 2)


if __name__ == "__main__": unittest.main()
