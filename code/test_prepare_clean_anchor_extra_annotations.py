import json
import tempfile
import unittest
from pathlib import Path

from prepare_clean_anchor_extra_annotations import prepare_extra_annotations


class PrepareCleanAnchorExtraAnnotationsTests(unittest.TestCase):
    def test_prepare_extra_annotations_copies_only_annotation_only_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            out = root / "out"
            source.mkdir()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"annotation_only_sample": ["a.mp4", "b.mp4"]}),
                encoding="utf-8",
            )
            for stem in ["a", "b", "c"]:
                (source / f"{stem}_annotations.json").write_text(
                    json.dumps({"video_name": f"{stem}.mp4", "total_frames": 3, "annotations": []}),
                    encoding="utf-8",
                )

            summary = prepare_extra_annotations(
                manifest_path=manifest_path,
                source_annotations_dir=source,
                output_annotations_dir=out,
            )

            self.assertEqual(summary["copied"], 2)
            self.assertEqual(summary["missing"], [])
            self.assertTrue((out / "a_annotations.json").exists())
            self.assertTrue((out / "b_annotations.json").exists())
            self.assertFalse((out / "c_annotations.json").exists())
            self.assertTrue((out / "extra_annotations_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
