import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge_jepa_event_datasets import merge_event_datasets
from train_segment_locator import load_signal_dataset


def _write_dataset(root: Path, names: list[str], frame_counts: list[int], feature_names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    signals = []
    labels = []
    for idx, frames in enumerate(frame_counts):
        base = np.full((frames, len(feature_names)), float(idx + 1), dtype=np.float32)
        label = np.zeros(frames, dtype=np.int64)
        if frames > 1:
            label[1] = 1
        signals.append(base)
        labels.append(label)
    np.savez_compressed(root / "signals.npz", *signals)
    np.savez_compressed(root / "labels.npz", *labels)
    (root / "video_names.json").write_text(json.dumps(names), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "n_videos": len(names),
                "frames": int(sum(frame_counts)),
                "positive_frames": int(sum(label.sum() for label in labels)),
                "feature_dim": len(feature_names),
                "feature_names": feature_names,
                "task": "binary_error_segment_localization",
            }
        ),
        encoding="utf-8",
    )


class MergeJepaEventDatasetsTests(unittest.TestCase):
    def test_merge_event_datasets_preserves_order_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_names = ["vjepa", "ijepa", "dual", "rank"]
            first = root / "first"
            second = root / "second"
            out = root / "merged"
            _write_dataset(first, ["a.mp4", "b.mp4"], [3, 4], feature_names)
            _write_dataset(second, ["c.mp4"], [5], feature_names)

            summary = merge_event_datasets([first, second], out)
            records = load_signal_dataset(out)

        self.assertEqual([record.name for record in records], ["a.mp4", "b.mp4", "c.mp4"])
        self.assertEqual(summary["n_videos"], 3)
        self.assertEqual(summary["frames"], 12)
        self.assertEqual(summary["positive_frames"], 3)
        self.assertEqual(summary["feature_dim"], 4)
        self.assertEqual(summary["feature_names"], feature_names)
        self.assertEqual(summary["sources"], [str(first), str(second)])

    def test_merge_event_datasets_rejects_duplicate_video_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_names = ["vjepa", "ijepa", "dual"]
            first = root / "first"
            second = root / "second"
            _write_dataset(first, ["a.mp4"], [3], feature_names)
            _write_dataset(second, ["a.mp4"], [4], feature_names)

            with self.assertRaisesRegex(ValueError, "duplicate video name"):
                merge_event_datasets([first, second], root / "merged")

    def test_merge_event_datasets_rejects_feature_name_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            _write_dataset(first, ["a.mp4"], [3], ["vjepa", "ijepa", "dual"])
            _write_dataset(second, ["b.mp4"], [4], ["vjepa", "ijepa", "changed"])

            with self.assertRaisesRegex(ValueError, "feature_names mismatch"):
                merge_event_datasets([first, second], root / "merged")

    def test_merge_event_datasets_rejects_signal_label_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "bad"
            _write_dataset(ds, ["a.mp4"], [3], ["vjepa", "ijepa", "dual"])
            np.savez_compressed(ds / "labels.npz", np.zeros(2, dtype=np.int64))

            with self.assertRaisesRegex(ValueError, "length mismatch"):
                merge_event_datasets([ds], root / "merged")


if __name__ == "__main__":
    unittest.main()
