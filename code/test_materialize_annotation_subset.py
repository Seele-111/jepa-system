import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from materialize_annotation_subset import main


class MaterializeSubsetTests(unittest.TestCase):
    def test_materializes_named_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; source.mkdir(); output = root / "output"
            (source / "a_annotations.json").write_text("{}")
            names = root / "names.json"; names.write_text(json.dumps(["a.mp4"]))
            with patch("sys.argv", ["x", "--source-dir", str(source), "--names", str(names), "--output-dir", str(output)]):
                self.assertEqual(main(), 0)
            self.assertTrue((output / "a_annotations.json").exists())
            self.assertEqual(json.loads((output / "subset_manifest.json").read_text())["count"], 1)


if __name__ == "__main__": unittest.main()
