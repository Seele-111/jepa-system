import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from validate_review_corrections import validate_review_csv


class ValidateReviewCorrectionsTests(unittest.TestCase):
    def test_validate_review_csv_accepts_empty_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "event"
            data_dir.mkdir()
            np.savez_compressed(data_dir / "labels.npz", np.zeros(5, dtype=np.int64))
            (data_dir / "video_names.json").write_text(json.dumps(["a.mp4"]), encoding="utf-8")
            csv_path = root / "review.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video", "review_decision", "corrected_error_segments"])
                writer.writeheader()
                writer.writerow({"video": "a.mp4", "review_decision": "", "corrected_error_segments": ""})

            result = validate_review_csv(csv_path, data_dir)

            self.assertEqual(result["errors"], [])
            self.assertEqual(result["decision_rows"], 0)

    def test_validate_review_csv_reports_bad_decision_and_missing_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "event"
            data_dir.mkdir()
            np.savez_compressed(data_dir / "labels.npz", np.zeros(5, dtype=np.int64))
            (data_dir / "video_names.json").write_text(json.dumps(["a.mp4"]), encoding="utf-8")
            csv_path = root / "review.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video", "review_decision", "corrected_error_segments"])
                writer.writeheader()
                writer.writerow({"video": "a.mp4", "review_decision": "bad_value", "corrected_error_segments": ""})
                writer.writerow(
                    {
                        "video": "a.mp4",
                        "review_decision": "replace_video_segments",
                        "corrected_error_segments": "",
                    }
                )

            result = validate_review_csv(csv_path, data_dir)

            self.assertEqual(result["decision_rows"], 2)
            self.assertEqual(len(result["errors"]), 2)

    def test_validate_review_csv_reports_out_of_range_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "event"
            data_dir.mkdir()
            np.savez_compressed(data_dir / "labels.npz", np.zeros(5, dtype=np.int64))
            (data_dir / "video_names.json").write_text(json.dumps(["a.mp4"]), encoding="utf-8")
            csv_path = root / "review.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video", "review_decision", "corrected_error_segments"])
                writer.writeheader()
                writer.writerow(
                    {
                        "video": "a.mp4",
                        "review_decision": "replace_video_segments",
                        "corrected_error_segments": "[[3, 9]]",
                    }
                )

            result = validate_review_csv(csv_path, data_dir)

            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("out of range", result["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
