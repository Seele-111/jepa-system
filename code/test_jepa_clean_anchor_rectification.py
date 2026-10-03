import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jepa_clean_anchor_rectification import (
    CleanAnchorRectifyConfig,
    compute_clean_anchor_frame_weights_for_fold,
    rectify_records_for_fold,
)
from train_segment_locator import VideoRecord


def _write_clean_dataset(root: Path) -> None:
    feature_names = ["evidence", "filler_a", "filler_b"]
    np.savez_compressed(
        root / "signals.npz",
        np.array([[0.05, 0.0, 0.0], [0.95, 0.0, 0.0], [0.9, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.05, 0.0, 0.0], [0.9, 0.0, 0.0], [0.85, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
    )
    np.savez_compressed(
        root / "labels.npz",
        np.array([0, 1, 1, 0], dtype=np.int64),
        np.array([0, 1, 1, 0], dtype=np.int64),
    )
    (root / "video_names.json").write_text(json.dumps(["clean_a.mp4", "clean_b.mp4"]), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps({"feature_names": feature_names, "feature_dim": 1, "n_videos": 2}),
        encoding="utf-8",
    )


class JEPACleanAnchorRectificationTests(unittest.TestCase):
    def test_rectify_records_for_fold_modifies_only_training_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_root = Path(tmp)
            _write_clean_dataset(clean_root)
            records = [
                VideoRecord(
                    "train",
                    np.array([[0.05, 0.0, 0.0], [0.95, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
                    np.array([1, 0, 1, 0]),
                ),
                VideoRecord(
                    "val",
                    np.array([[0.95, 0.0, 0.0], [0.05, 0.0, 0.0], [0.95, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
                    np.array([0, 1, 0, 1]),
                ),
            ]
            config = CleanAnchorRectifyConfig(
                evidence_names=["evidence"],
                teacher_model="prototype",
                positive_threshold=0.7,
                negative_threshold=0.3,
                add_threshold=0.8,
                remove_threshold=0.2,
                min_add_length=1,
                min_keep_length=1,
            )

            rectified, summary = rectify_records_for_fold(
                records,
                train_idx=[0],
                feature_names=["evidence", "filler_a", "filler_b"],
                clean_data_dir=clean_root,
                config=config,
            )

        np.testing.assert_array_equal(rectified[0].labels, np.array([0, 1, 0, 0], dtype=np.int64))
        np.testing.assert_array_equal(rectified[1].labels, records[1].labels)
        self.assertEqual(summary["changed_videos"], 1)
        self.assertEqual(summary["added_frames"], 1)
        self.assertEqual(summary["removed_frames"], 2)
        self.assertEqual(summary["teacher_positive_frames"], 4)
        self.assertEqual(summary["teacher_negative_frames"], 4)

    def test_learned_teacher_uses_multichannel_evidence_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_root = Path(tmp)
            feature_names = ["e0", "e1", "e2"]
            clean_signals = [
                np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
                np.array([[0.82, 0.08, 0.1], [0.08, 0.82, 0.1], [0.8, 0.12, 0.1], [0.12, 0.8, 0.1]], dtype=np.float32),
            ]
            clean_labels = [
                np.array([1, 0, 1, 0], dtype=np.int64),
                np.array([1, 0, 1, 0], dtype=np.int64),
            ]
            np.savez_compressed(clean_root / "signals.npz", *clean_signals)
            np.savez_compressed(clean_root / "labels.npz", *clean_labels)
            (clean_root / "video_names.json").write_text(json.dumps(["clean_a.mp4", "clean_b.mp4"]), encoding="utf-8")
            (clean_root / "summary.json").write_text(
                json.dumps({"feature_names": feature_names, "feature_dim": 3, "n_videos": 2}),
                encoding="utf-8",
            )
            records = [
                VideoRecord(
                    "train",
                    np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
                    np.array([1, 0, 1], dtype=np.int64),
                )
            ]
            config = CleanAnchorRectifyConfig(
                evidence_names=feature_names,
                teacher_model="extratrees",
                add_threshold=0.7,
                remove_threshold=0.3,
                min_add_length=1,
                min_keep_length=1,
            )

            rectified, summary = rectify_records_for_fold(
                records,
                train_idx=[0],
                feature_names=feature_names,
                clean_data_dir=clean_root,
                config=config,
            )

        np.testing.assert_array_equal(rectified[0].labels, np.array([0, 1, 0], dtype=np.int64))
        self.assertEqual(summary["teacher_model"], "extratrees")

    def test_soft_weights_downweight_teacher_label_conflicts_on_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_root = Path(tmp)
            _write_clean_dataset(clean_root)
            records = [
                VideoRecord(
                    "train",
                    np.array([[0.05, 0.0, 0.0], [0.95, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
                    np.array([1, 1, 0], dtype=np.int64),
                ),
                VideoRecord(
                    "val",
                    np.array([[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
                    np.array([0, 1], dtype=np.int64),
                ),
            ]
            config = CleanAnchorRectifyConfig(
                evidence_names=["evidence"],
                teacher_model="prototype",
                positive_min_weight=0.2,
                negative_min_weight=0.2,
            )

            weights, summary = compute_clean_anchor_frame_weights_for_fold(
                records,
                train_idx=[0],
                feature_names=["evidence", "filler_a", "filler_b"],
                clean_data_dir=clean_root,
                config=config,
            )

        self.assertEqual(sorted(weights), [0])
        self.assertLess(float(weights[0][0]), float(weights[0][1]))
        self.assertGreaterEqual(float(weights[0][2]), 0.9)
        self.assertEqual(summary["weighted_train_videos"], 1)
        self.assertEqual(summary["weighted_frames"], 3)


if __name__ == "__main__":
    unittest.main()
