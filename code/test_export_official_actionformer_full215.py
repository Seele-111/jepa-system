import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_official_actionformer_full215 import export_actionformer_dataset


class OfficialActionFormerExportTests(unittest.TestCase):
    def test_export_preserves_events_and_strict_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            names = ["a.mp4", "b.mp4", "c.mp4"]
            signals = {
                "x0": np.ones((4, 3), np.float32),
                "x1": np.ones((5, 3), np.float32) * 2,
                "x2": np.ones((6, 3), np.float32) * 3,
            }
            labels = {
                "x0": np.asarray([0, 1, 1, 0]),
                "x1": np.asarray([1, 1, 1, 1, 1]),
                "x2": np.asarray([0, 1, 0, 1, 0, 0]),
            }
            np.savez(data / "signals.npz", **signals)
            np.savez(data / "labels.npz", **labels)
            (data / "video_names.json").write_text(json.dumps(names), encoding="utf-8")
            split = {
                "folds": [
                    {
                        "fit_names": ["a.mp4"],
                        "calibration_names": ["b.mp4"],
                        "val_names": ["c.mp4"],
                    }
                ]
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")

            manifest = export_actionformer_dataset(data, split_path, root / "out")
            annotation = json.loads((root / "out" / "annotations" / "fold_0.json").read_text(encoding="utf-8"))
            first = annotation["database"][manifest["name_to_id"]["a.mp4"]]
            third = annotation["database"][manifest["name_to_id"]["c.mp4"]]
            self.assertEqual(first["subset"], "training")
            self.assertEqual(first["annotations"][0]["segment"], [1.0, 3.0])
            self.assertEqual([row["segment"] for row in third["annotations"]], [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(manifest["folds"][0]["fit_videos"], 1)
            self.assertTrue((root / "out" / "features" / "full215_0002.npy").exists())
            config = json.loads((root / "out" / "configs" / "fold_0_calibration.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["val_split"], ["calibration"])
            self.assertEqual(config["dataset"]["feat_stride"], 1)
            self.assertGreater(config["loader"]["num_workers"], 0)

    def test_export_rejects_split_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            np.savez(data / "signals.npz", x=np.ones((4, 3), np.float32))
            np.savez(data / "labels.npz", x=np.ones(4, np.int64))
            (data / "video_names.json").write_text(json.dumps(["a.mp4"]), encoding="utf-8")
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps({"folds": [{"fit_names": ["a.mp4"], "calibration_names": ["a.mp4"], "val_names": []}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlapping splits"):
                export_actionformer_dataset(data, split_path, root / "out")

    def test_fit_only_normalization_uses_no_calibration_or_outer_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            np.savez(
                data / "signals.npz",
                fit=np.asarray([[1, 2, 3], [3, 4, 5]], np.float32),
                calibration=np.full((2, 3), 100, np.float32),
                outer=np.full((2, 3), 200, np.float32),
            )
            np.savez(
                data / "labels.npz",
                fit=np.ones(2, np.int64),
                calibration=np.ones(2, np.int64),
                outer=np.ones(2, np.int64),
            )
            names = ["fit.mp4", "calibration.mp4", "outer.mp4"]
            (data / "video_names.json").write_text(json.dumps(names), encoding="utf-8")
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps({"folds": [{"fit_names": [names[0]], "calibration_names": [names[1]], "val_names": [names[2]]}]}),
                encoding="utf-8",
            )
            manifest = export_actionformer_dataset(data, split_path, root / "out", normalize_fit_per_fold=True)
            fit_id = manifest["name_to_id"][names[0]]
            calibration_id = manifest["name_to_id"][names[1]]
            fit_features = np.load(root / "out" / "features_fold_0" / f"{fit_id}.npy")
            calibration_features = np.load(root / "out" / "features_fold_0" / f"{calibration_id}.npy")
            self.assertTrue(np.allclose(fit_features.mean(axis=0), 0.0))
            self.assertTrue(np.allclose(fit_features.std(axis=0), 1.0))
            self.assertGreater(float(calibration_features.mean()), 50.0)
            self.assertEqual(manifest["normalization"], "fit-only channel z-score per fold")


if __name__ == "__main__":
    unittest.main()
