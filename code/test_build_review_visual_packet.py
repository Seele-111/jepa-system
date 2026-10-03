import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_review_visual_packet import build_visual_packet, frame_to_time, sample_segment_frames


class BuildReviewVisualPacketTests(unittest.TestCase):
    def test_sample_segment_frames_returns_start_middle_end(self):
        self.assertEqual(sample_segment_frames([10, 20], label_length=100, count=3), [10, 15, 20])
        self.assertEqual(sample_segment_frames([3, 3], label_length=10, count=3), [3])
        self.assertEqual(sample_segment_frames([-5, 99], label_length=12, count=3), [0, 6, 11])

    def test_frame_to_time_keeps_tail_frame_inside_video_duration(self):
        self.assertLess(frame_to_time(120, label_length=121, duration=5.041667), 5.041667 - 0.02)

    def test_build_visual_packet_uses_runner_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "event"
            data_dir.mkdir()
            np.savez_compressed(data_dir / "labels.npz", np.zeros(12, dtype=np.int64))
            (data_dir / "video_names.json").write_text(json.dumps(["a.mp4"]), encoding="utf-8")
            video = root / "a.mp4"
            video.write_bytes(b"fake video")
            review = {
                "items": [
                    {
                        "kind": "FN",
                        "priority": 200,
                        "video_idx": 0,
                        "video": "a.mp4",
                        "video_path": str(video),
                        "target_segment": [0, 2],
                        "current_prediction": [0, 11],
                        "best_jepa_candidate": [1, 3],
                        "action": "review_split_or_relabel_parent",
                    }
                ]
            }
            review_json = root / "review.json"
            review_json.write_text(json.dumps(review), encoding="utf-8")
            calls = []

            def fake_probe(path):
                self.assertEqual(path, video)
                return 12.0

            def fake_runner(video_path, timestamp, output_path):
                calls.append((video_path, timestamp, output_path))
                output_path.write_bytes(b"jpg")
                return True

            result = build_visual_packet(
                review_json=review_json,
                output_dir=root / "visual",
                data_dir=data_dir,
                top_k=1,
                probe_duration=fake_probe,
                extract_frame=fake_runner,
            )

            self.assertEqual(result["n_items"], 1)
            self.assertGreaterEqual(len(calls), 3)
            self.assertTrue((root / "visual" / "visual_review_manifest.json").exists())
            html = (root / "visual" / "visual_review.html").read_text(encoding="utf-8")
            self.assertIn("review_split_or_relabel_parent", html)
            self.assertIn("a.mp4", html)


if __name__ == "__main__":
    unittest.main()
