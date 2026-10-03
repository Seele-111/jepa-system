import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_clean_anchor_manifest import (
    annotation_to_labels,
    build_clean_anchor_manifest,
    load_desktop_annotations,
)


class BuildCleanAnchorManifestTests(unittest.TestCase):
    def test_annotation_to_labels_clips_ranges_and_ignores_categories(self):
        annotation = {
            "total_frames": 6,
            "annotations": [
                {"start_frame": -2, "end_frame": 1, "category": "physics"},
                {"start_frame": 4, "end_frame": 10, "severity": "high"},
            ],
        }

        labels = annotation_to_labels(annotation)

        np.testing.assert_array_equal(labels, np.array([1, 1, 0, 0, 1, 1], dtype=np.int64))

    def test_load_desktop_annotations_indexes_by_video_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "toy_annotations.json").write_text(
                json.dumps(
                    {
                        "video_name": "toy.mp4",
                        "total_frames": 4,
                        "annotations": [{"start_frame": 1, "end_frame": 2}],
                    }
                ),
                encoding="utf-8",
            )

            annotations = load_desktop_annotations(root)

        self.assertIn("toy.mp4", annotations)
        np.testing.assert_array_equal(annotations["toy.mp4"].labels, np.array([0, 1, 1, 0], dtype=np.int64))

    def test_build_clean_anchor_manifest_reports_matches_and_label_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann_root = root / "annotations"
            data_root = root / "event"
            ann_root.mkdir()
            data_root.mkdir()
            (ann_root / "a_annotations.json").write_text(
                json.dumps(
                    {
                        "video_name": "a.mp4",
                        "total_frames": 5,
                        "annotations": [{"start_frame": 1, "end_frame": 3}],
                    }
                ),
                encoding="utf-8",
            )
            (ann_root / "b_annotations.json").write_text(
                json.dumps(
                    {
                        "video_name": "b.mp4",
                        "total_frames": 4,
                        "annotations": [],
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(data_root / "labels.npz", np.array([0, 1, 0, 0, 0]), np.array([0, 0, 1, 0]))
            np.savez_compressed(data_root / "signals.npz", np.zeros((5, 3), dtype=np.float32), np.zeros((4, 3), dtype=np.float32))
            (data_root / "video_names.json").write_text(json.dumps(["a.mp4", "c.mp4"]), encoding="utf-8")
            out_path = root / "manifest.json"

            manifest = build_clean_anchor_manifest(
                annotations_dir=ann_root,
                event_data_dir=data_root,
                output_path=out_path,
            )
            self.assertTrue(out_path.exists())

        self.assertEqual(manifest["matched_videos"], 1)
        self.assertEqual(manifest["annotation_only_videos"], 1)
        self.assertEqual(manifest["dataset_only_videos"], 1)
        self.assertEqual(manifest["matched_positive_frames_annotation"], 3)
        self.assertEqual(manifest["matched_positive_frames_dataset"], 1)
        self.assertEqual(manifest["matched_label_disagreement_frames"], 2)


if __name__ == "__main__":
    unittest.main()
