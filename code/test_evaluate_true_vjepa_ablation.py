import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

from evaluate_true_vjepa_ablation import (
    evaluate_method,
    extract_vjepa_scores,
    labels_from_annotations,
    load_annotations,
    smooth_scores,
    segment_metrics,
    segments_from_binary,
    select_threshold,
)


class TrueVJEPAEvaluationTests(unittest.TestCase):
    def test_labels_from_annotations_marks_inclusive_frame_ranges(self):
        labels = labels_from_annotations(
            6,
            [
                {"start_frame": 1, "end_frame": 2},
                {"start_frame": 4, "end_frame": 10},
            ],
        )

        np.testing.assert_array_equal(labels, np.array([0, 1, 1, 0, 1, 1]))

    def test_segments_from_binary_uses_inclusive_endpoints(self):
        segments = segments_from_binary(np.array([0, 1, 1, 0, 1]))

        self.assertEqual(segments, [(1, 2), (4, 4)])

    def test_segment_metrics_matches_each_ground_truth_once(self):
        metrics = segment_metrics(
            predicted=[(0, 4), (1, 3), (10, 12)],
            target=[(0, 4), (20, 25)],
            iou_threshold=0.3,
        )

        self.assertEqual(metrics.tp, 1)
        self.assertEqual(metrics.fp, 2)
        self.assertEqual(metrics.fn, 1)

    def test_select_threshold_finds_best_frame_f1(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])

        threshold, f1 = select_threshold(labels, scores, np.array([0.2, 0.5, 0.8]))

        self.assertEqual(threshold, 0.5)
        self.assertGreater(f1, 0.99)

    def test_evaluate_method_reports_frame_and_segment_metrics(self):
        records = [
            {
                "labels": np.array([0, 1, 1, 0]),
                "scores": np.array([0.1, 0.8, 0.7, 0.2]),
            },
            {
                "labels": np.array([0, 0, 1]),
                "scores": np.array([0.2, 0.3, 0.9]),
            },
        ]

        result = evaluate_method(records, "scores", threshold=0.5, iou_threshold=0.3)

        self.assertGreater(result["frame"]["f1"], 0.99)
        self.assertGreater(result["segment"]["f1"], 0.99)
        self.assertEqual(result["segment"]["tp"], 2)

    def test_load_annotations_accepts_video_field_and_repairs_bad_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ann_dir = root / "annotations"
            ann_dir.mkdir()
            video = root / "clip.mp4"
            video.write_bytes(b"fake")
            (ann_dir / "clip_annotations.json").write_text(
                json.dumps(
                    {
                        "video": "/mnt/c/Users/admin/Desktop/bad-mojibake/clip.mp4",
                        "video_name": "clip.mp4",
                        "total_frames": 4,
                        "annotations": [],
                    }
                ),
                encoding="utf-8",
            )
            (ann_dir / ".clip_backup.json").write_text("{}", encoding="utf-8")

            records = load_annotations(ann_dir, limit=None)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["video_path"], str(video))

    def test_load_annotations_finds_jepa_data_video_roots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "JEPA-data"
            ann_dir = root / "trainingdata" / "annotations-train"
            video_dir = root / "filtered_videos_train"
            ann_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)
            video = video_dir / "0626.mp4"
            video.write_bytes(b"fake")
            (ann_dir / "0626_annotations.json").write_text(
                json.dumps(
                    {
                        "video": r"D:\old\wrong\0626.mp4",
                        "video_name": "0626.mp4",
                        "total_frames": 61,
                        "annotations": [{"start_frame": 0, "end_frame": 10, "category": "visual"}],
                    }
                ),
                encoding="utf-8",
            )

            records = load_annotations(ann_dir, limit=None)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["video_path"], str(video))

    def test_testdata_prefers_filtered_videos_test_when_names_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "JEPA-data"
            ann_dir = root / "testdata" / "annotations-train"
            train_dir = root / "filtered_videos_train"
            test_dir = root / "filtered_videos_test"
            ann_dir.mkdir(parents=True)
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            (train_dir / "0626.mp4").write_bytes(b"train")
            test_video = test_dir / "0626.mp4"
            test_video.write_bytes(b"test")
            (ann_dir / "0626_annotations.json").write_text(
                json.dumps({"video_name": "0626.mp4", "total_frames": 3, "annotations": []}),
                encoding="utf-8",
            )

            records = load_annotations(ann_dir, limit=None)

            self.assertEqual(records[0]["video_path"], str(test_video))

    def test_extract_vjepa_scores_prefers_raw_objective_scores(self):
        scores = extract_vjepa_scores(
            {
                "physics_sig": [0.5, 1.0, 1.0],
                "physics_raw": [0.0, 0.2, 0.7],
            }
        )

        np.testing.assert_allclose(scores, np.array([0.0, 0.2, 0.7]))

    def test_smooth_scores_uses_centered_moving_average(self):
        scores = smooth_scores(np.array([0.0, 0.0, 1.0, 0.0, 0.0]), window=3)

        np.testing.assert_allclose(scores, np.array([0.0, 1 / 3, 1 / 3, 1 / 3, 0.0]))


if __name__ == "__main__":
    unittest.main()
