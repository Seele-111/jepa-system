import json
import tempfile
import unittest
from pathlib import Path

from build_aligned_prediction_export import main


class BuildAlignedPredictionExportTests(unittest.TestCase):
    def test_attaches_fold_indices_to_prediction_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "pred.json"
            idx = root / "idx.json"
            out = root / "out.json"
            pred.write_text(
                json.dumps({"folds": [{"fold": 9, "predictions": [[(1, 2)], []]}]}),
                encoding="utf-8",
            )
            idx.write_text(
                json.dumps({"folds": [{"fold": 0, "train_idx": [2, 3], "val_idx": [0, 1], "val_names": ["a", "b"]}]}),
                encoding="utf-8",
            )

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_aligned_prediction_export.py",
                    "--prediction-export",
                    str(pred),
                    "--fold-index-export",
                    str(idx),
                    "--output",
                    str(out),
                ]
                self.assertEqual(main(), 0)
            finally:
                sys.argv = old_argv

            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["folds"][0]["fold"], 0)
            self.assertEqual(data["folds"][0]["train_idx"], [2, 3])
            self.assertEqual(data["folds"][0]["val_idx"], [0, 1])
            self.assertEqual(data["folds"][0]["predictions"], [[[1, 2]], []])


if __name__ == "__main__":
    unittest.main()

