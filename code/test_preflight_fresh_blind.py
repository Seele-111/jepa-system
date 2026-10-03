import json
import tempfile
import unittest
from pathlib import Path

from preflight_fresh_blind import preflight


class FreshBlindPreflightTests(unittest.TestCase):
    def test_ready_candidate_reports_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidate = root / "candidate"; candidate.mkdir()
            (candidate / "video_names.json").write_text(json.dumps(["x.mp4"]))
            (candidate / "fresh_overlap_gate.json").write_text(json.dumps({"zero_overlap": True}))
            artifact = root / "model.pt"; artifact.write_bytes(b"model")
            report = preflight(candidate, {"model": artifact})
            self.assertEqual(report["status"], "ready_for_one_time_inference")
            self.assertIn("sha256", report["artifacts"]["model"])


if __name__ == "__main__": unittest.main()
