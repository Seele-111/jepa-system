import unittest
import json
import tempfile
from pathlib import Path

from bootstrap_bundle_difference import rows


class BootstrapHelpersTest(unittest.TestCase):
    def test_rows_requires_bundle(self):
        self.assertTrue(callable(rows))

    def test_rows_reads_nested_bundle(self):
        payload = {
            "bundles": {
                "vjepa": {
                    "folds": [{"val_names": ["a"], "predictions": [[[0, 1]]], "labels": [[1, 1]]}]
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = rows(path, "vjepa")
        self.assertEqual(list(result), ["a"])


if __name__ == "__main__":
    unittest.main()
