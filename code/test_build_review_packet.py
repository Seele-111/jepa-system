import json
import tempfile
import unittest
from pathlib import Path

from build_review_packet import (
    build_review_packet,
    infer_review_action,
    parse_segment,
)


class BuildReviewPacketTests(unittest.TestCase):
    def test_parse_segment_handles_strings_and_lists(self):
        self.assertEqual(parse_segment("[3, 7]"), (3, 7))
        self.assertEqual(parse_segment([9, 4]), (4, 9))
        self.assertIsNone(parse_segment(None))

    def test_infer_review_action_prioritizes_wide_prediction_splits(self):
        row = {
            "kind": "FN",
            "reason": "wide_prediction_matching_conflict",
            "best_prediction": "[0, 120]",
            "best_candidate": "[36, 60]",
            "best_candidate_iou": 0.73,
        }

        action = infer_review_action(row)

        self.assertEqual(action, "review_split_or_relabel_parent")

    def test_build_review_packet_writes_json_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos"
            videos.mkdir()
            (videos / "a.mp4").write_bytes(b"not a real video")
            priority = {
                "rows": [
                    {
                        "kind": "FN",
                        "priority": 200,
                        "fold": 0,
                        "video_idx": 12,
                        "video": "a.mp4",
                        "segment": "[10, 20]",
                        "reason": "wide_prediction_matching_conflict",
                        "best_prediction": "[0, 40]",
                        "best_prediction_iou": 0.25,
                        "best_candidate": "[11, 19]",
                        "best_candidate_iou": 0.82,
                        "best_candidate_rank": 5,
                        "covering_candidates": 8,
                    },
                    {
                        "kind": "FP",
                        "priority": 90,
                        "fold": 1,
                        "video_idx": 13,
                        "video": "b.mp4",
                        "segment": "[5, 8]",
                        "reason": "current_false_positive",
                        "best_prediction": "[5, 8]",
                    },
                ]
            }
            priority_path = root / "priority.json"
            priority_path.write_text(json.dumps(priority), encoding="utf-8")

            result = build_review_packet(
                priority_path=priority_path,
                output_dir=root / "packet",
                video_roots=[videos],
                top_k=10,
            )

            self.assertEqual(result["n_items"], 2)
            self.assertEqual(result["counts"]["FN"], 1)
            self.assertEqual(result["counts"]["FP"], 1)
            self.assertEqual(result["items"][0]["action"], "review_split_or_relabel_parent")
            self.assertEqual(result["items"][0]["video_path"], str(videos / "a.mp4"))
            self.assertTrue((root / "packet" / "review_packet.json").exists())
            self.assertTrue((root / "packet" / "review_packet.csv").exists())
            markdown = (root / "packet" / "review_packet.md").read_text(encoding="utf-8")
            self.assertIn("review_split_or_relabel_parent", markdown)
            self.assertIn("a.mp4", markdown)


if __name__ == "__main__":
    unittest.main()
