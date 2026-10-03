import json
import tempfile
import unittest
from pathlib import Path

from build_dataset_card import build_card


class DatasetCardTests(unittest.TestCase):
    def test_card_preserves_unknown_metadata_and_counts_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(
                json.dumps(
                    {
                        "video_name": "a.mp4",
                        "total_frames": 10,
                        "annotations": [
                            {"start_frame": 2, "end_frame": 4, "category": "physics"},
                            {"start_frame": 4, "end_frame": 5, "category": "visual"},
                            {"start_frame": 7, "end_frame": 7, "category": "visual"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            card = build_card(root, "full215")
            self.assertEqual(card["summary"]["videos"], 1)
            self.assertEqual(card["summary"]["frames"], 10)
            self.assertEqual(card["summary"]["positive_frames"], 5)
            self.assertEqual(card["summary"]["events"], 2)
            self.assertEqual(card["summary"]["raw_annotations"], 3)
            self.assertEqual(card["summary"]["categories"], {"physics": 1, "visual": 2})
            self.assertEqual(card["videos"][0]["generator"], "unknown")

    def test_names_file_filters_full_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"):
                (root / f"{name}.json").write_text(
                    json.dumps({"video_name": f"{name}.mp4", "total_frames": 2, "annotations": []}),
                    encoding="utf-8",
                )
            names = root / "names.json"
            names.write_text(json.dumps(["b.mp4"]), encoding="utf-8")
            card = build_card(root, "full215", names_file=names)
            self.assertEqual([row["video_name"] for row in card["videos"]], ["b.mp4"])


if __name__ == "__main__":
    unittest.main()
