import csv
import tempfile
import unittest
from pathlib import Path

from build_videophy2_metadata import recover_rows, source_filename


class VideoPhyMetadataTests(unittest.TestCase):
    def test_url_filename_is_decoded(self):
        self.assertEqual(source_filename("https://example.test/A%20ball.mp4?x=1"), "A ball.mp4")

    def test_recovery_requires_unique_exact_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video_url", "model_name", "caption"])
                writer.writeheader()
                writer.writerow({"video_url": "https://x/a.mp4", "model_name": "wan", "caption": "a"})
            rows, report = recover_rows(path, ["a.mp4", "b.mp4"])
            self.assertEqual(rows[0]["generator"], "wan")
            self.assertEqual(report["matched"], 1)
            self.assertEqual(report["missing"], 1)


if __name__ == "__main__":
    unittest.main()
