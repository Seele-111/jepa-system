import unittest

from analyze_bundle_strata import strata


class StrataTest(unittest.TestCase):
    def test_predeclared_groups(self):
        groups = strata({"videos": [{"video_name": "a.mp4", "positive_frame_ratio": 1, "event_count": 2, "fps": "24", "resolution": "x"}]})
        self.assertIn("a.mp4", groups["all_positive"])
        self.assertIn("a.mp4", groups["multi_event"])


if __name__ == "__main__":
    unittest.main()
