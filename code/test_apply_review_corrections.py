import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apply_review_corrections import apply_review_corrections, parse_segments_field


class ApplyReviewCorrectionsTests(unittest.TestCase):
    def test_parse_segments_field_accepts_single_and_nested_segments(self):
        self.assertEqual(parse_segments_field("[[1, 3], [7, 8]]"), [(1, 3), (7, 8)])
        self.assertEqual(parse_segments_field("[4, 2]"), [(2, 4)])
        self.assertEqual(parse_segments_field(""), [])

    def test_apply_review_corrections_replaces_only_explicit_video_segment_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "event"
            data_dir.mkdir()
            np.savez_compressed(
                data_dir / "signals.npz",
                np.zeros((6, 3), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),
            )
            np.savez_compressed(
                data_dir / "labels.npz",
                np.array([0, 1, 1, 0, 0, 0], dtype=np.int64),
                np.array([1, 1, 0, 0, 0], dtype=np.int64),
            )
            (data_dir / "video_names.json").write_text(json.dumps(["a.mp4", "b.mp4"]), encoding="utf-8")
            (data_dir / "summary.json").write_text(json.dumps({"feature_dim": 3}), encoding="utf-8")
            review_csv = root / "review.csv"
            with review_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["video", "video_idx", "review_decision", "corrected_error_segments"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video": "a.mp4",
                        "video_idx": "0",
                        "review_decision": "replace_video_segments",
                        "corrected_error_segments": "[[0, 0], [4, 5]]",
                    }
                )
                writer.writerow(
                    {
                        "video": "b.mp4",
                        "video_idx": "1",
                        "review_decision": "",
                        "corrected_error_segments": "[[0, 4]]",
                    }
                )

            result = apply_review_corrections(data_dir=data_dir, review_csv=review_csv, output_dir=root / "corrected")

            self.assertEqual(result["n_review_rows"], 2)
            self.assertEqual(result["n_changed_videos"], 1)
            labels = np.load(root / "corrected" / "labels.npz", allow_pickle=True)
            np.testing.assert_array_equal(labels["arr_0"], np.array([1, 0, 0, 0, 1, 1], dtype=np.int64))
            np.testing.assert_array_equal(labels["arr_1"], np.array([1, 1, 0, 0, 0], dtype=np.int64))
            summary = json.loads((root / "corrected" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["review_corrections"]["n_changed_videos"], 1)

    def test_apply_review_corrections_rejects_conflicting_replacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "event"
            data_dir.mkdir()
            np.savez_compressed(data_dir / "signals.npz", np.zeros((4, 3), dtype=np.float32))
            np.savez_compressed(data_dir / "labels.npz", np.zeros(4, dtype=np.int64))
            (data_dir / "video_names.json").write_text(json.dumps(["a.mp4"]), encoding="utf-8")
            review_csv = root / "review.csv"
            with review_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["video", "review_decision", "corrected_error_segments"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video": "a.mp4",
                        "review_decision": "replace_video_segments",
                        "corrected_error_segments": "[[0, 1]]",
                    }
                )
                writer.writerow(
                    {
                        "video": "a.mp4",
                        "review_decision": "replace_video_segments",
                        "corrected_error_segments": "[[2, 3]]",
                    }
                )

            with self.assertRaises(ValueError):
                apply_review_corrections(data_dir=data_dir, review_csv=review_csv, output_dir=root / "corrected")


if __name__ == "__main__":
    unittest.main()
