"""Focused checks for demo provenance, rejection, output time ranges and API safety."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_detector as detector
import demo_app as web


class DetectorChecks(unittest.TestCase):
    def test_true_output_must_identify_both_predictors_and_match_frames(self):
        ts = {"composite": [.2, .3, .4], "total_frames": 3, "fps": 10,
              "jepa_usage": {"vjepa": "true_vjepa_predictor", "ijepa": "true_ijepa_predictor"}}
        np.testing.assert_allclose(detector._validated_true_score(ts, 3, 10), [.2, .3, .4])
        for update in [{"jepa_usage": {"vjepa": "encoder_proxy", "ijepa": "true_ijepa_predictor"}},
                       {"composite": []}, {"composite": [.2, float("nan"), .4]},
                       {"composite": [.2, .3]}, {"fps": 16}, {"total_frames": 4},
                       {"composite": [.2, .3, 4.0]}]:
            with self.subTest(update=update), self.assertRaises(detector.DemoDetectionError):
                detector._validated_true_score({**ts, **update}, 3, 10)

    def test_plateau_can_produce_empty_set(self):
        self.assertEqual(detector._make_segments(np.ones(40, dtype=np.float32) * .8, 10,
                         profile=detector.TRUE_DEMO_PROFILE, use_demo_gate=True), [])
        self.assertEqual(detector._make_segments(np.array([], dtype=np.float32), 10,
                         profile=detector.TRUE_DEMO_PROFILE, use_demo_gate=True), [])

    def test_time_range_uses_real_fps_and_exclusive_end(self):
        profile = {"smooth_window": 1, "score_threshold": .5, "min_gap": 0, "min_length": 1}
        segment = detector._make_segments(np.array([0, 1, 1, 0], dtype=np.float32), .5,
                                         profile=profile, use_demo_gate=False)[0]
        self.assertEqual((segment["start_frame"], segment["end_frame"]), (1, 2))
        self.assertEqual((segment["start_seconds"], segment["end_seconds"]), (2.0, 6.0))
        self.assertEqual(segment["frame_count"], 2)

    def test_motion_ties_do_not_invent_time_order(self):
        values = detector._rank01(np.array([0, 2, 2, 2], dtype=np.float32))
        self.assertEqual(values[1], values[2])
        self.assertEqual(values[2], values[3])
        np.testing.assert_allclose(detector._rank01(np.ones(4)), .5)

    def test_model_failure_is_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as root, \
             patch.object(detector, "_video_info", return_value=(5, 10, 16, 16)), \
             patch.object(detector, "_run_wsl_true_jepa", side_effect=detector.DemoDetectionError("WSL unavailable")), \
             patch.object(detector, "_extract_motion_score", return_value=(np.zeros(5), {})), \
             patch.object(detector, "_render_annotated_video", return_value={"browser_playable": True}):
            report = detector.analyze_video("example.mp4", root)
            self.assertEqual(report["method"], "cpu_motion_fallback")
            self.assertIn("WSL unavailable", report["fallback_reason"])
            self.assertEqual(report["segment_count"], 0)
            self.assertTrue(Path(root, "demo_report.json").is_file())

    def test_zero_vjepa_coverage_cannot_be_labelled_true_dual_jepa(self):
        ts = {"composite": [.5] * 5, "total_frames": 5, "fps": 10, "physics_raw": [0] * 5,
              "jepa_usage": {"vjepa": "true_vjepa_predictor", "ijepa": "true_ijepa_predictor"}}
        with tempfile.TemporaryDirectory() as root, \
             patch.object(detector, "_video_info", return_value=(5, 10, 16, 16)), \
             patch.object(detector, "_run_wsl_true_jepa", return_value={"timeseries": ts}), \
             patch.object(detector, "_extract_motion_score", return_value=(np.zeros(5), {})), \
             patch.object(detector, "_render_annotated_video", return_value={"browser_playable": True}):
            report = detector.analyze_video("short.mp4", root)
            self.assertEqual(report["method"], "cpu_motion_fallback")
            self.assertIn("no usable scored coverage", report["fallback_reason"])

    def test_render_failure_is_not_disguised_as_model_fallback(self):
        ts = {"composite": [.2] * 5, "total_frames": 5, "fps": 10, "physics_raw": [0, 0, .1, .2, .3],
              "jepa_usage": {"vjepa": "true_vjepa_predictor", "ijepa": "true_ijepa_predictor"}}
        run = {"timeseries": ts, "source_report_path": "report.json", "timeseries_path": "timeseries.json",
               "stdout_tail": "", "stderr_tail": ""}
        with tempfile.TemporaryDirectory() as root, \
             patch.object(detector, "_video_info", return_value=(5, 10, 16, 16)), \
             patch.object(detector, "_run_wsl_true_jepa", return_value=run), \
             patch.object(detector, "_extract_motion_score") as motion, \
             patch.object(detector, "_render_annotated_video", side_effect=RuntimeError("codec failed")):
            with self.assertRaisesRegex(RuntimeError, "codec failed"):
                detector.analyze_video("example.mp4", root)
            motion.assert_not_called()

    def test_chinese_path_conversion_without_shell(self):
        self.assertEqual(detector.windows_to_wsl_path(r"C:\Users\admin\Desktop\测试\0207.mp4"),
                         "/mnt/c/Users/admin/Desktop/测试/0207.mp4")


class PortablePathChecks(unittest.TestCase):
    def test_explicit_windows_slashes_and_dot_segments_are_normalized_on_any_host(self):
        self.assertEqual(detector.windows_to_wsl_path('E:/test/../videos/片段.mp4'),'/mnt/e/videos/片段.mp4')
        self.assertEqual(detector.windows_to_wsl_path(r'c:\videos\片段.mp4'),'/mnt/c/videos/片段.mp4')


class WebChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(web, "RUN_ROOT", self.root)
        self.root_patch.start()
        self.jobs_patch = patch.object(web, "jobs", {})
        self.jobs_patch.start()
        self.client = web.app.test_client()

    def tearDown(self):
        self.root_patch.stop()
        self.jobs_patch.stop()
        self.temp.cleanup()

    def test_local_page_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertTrue(self.client.get("/health").json["ok"])
        self.assertEqual(self.client.get("/health", headers={"Host": "attacker.invalid"}).status_code, 403)
        self.assertEqual(self.client.post("/api/upload", headers={"Origin": "https://attacker.invalid"}).status_code, 403)

    def test_invalid_inputs_do_not_queue_jobs(self):
        self.assertEqual(self.client.post("/api/upload").status_code, 400)
        data = {"video": (io.BytesIO(b"x"), "bad.txt")}
        self.assertEqual(self.client.post("/api/upload", data=data).status_code, 400)
        data = {"max_frames": "bad", "video": (io.BytesIO(b"x"), "video.mp4")}
        self.assertEqual(self.client.post("/api/upload", data=data).status_code, 400)
        self.assertFalse(web.jobs)
        self.assertFalse(list(self.root.iterdir()))

    def test_upload_name_is_never_used_as_file_path(self):
        with patch.object(web.executor, "submit") as submit:
            response = self.client.post("/api/upload", data={"prefer_true_jepa": "0",
                                        "video": (io.BytesIO(b"fake video"), "../../中文.mp4")})
            self.assertEqual(response.status_code, 202)
            job = web.jobs[response.json["job_id"]]
            self.assertEqual(Path(job["input_path"]).name, "input.mp4")
            self.assertEqual(job["video_name"], "中文.mp4")
            self.assertTrue(Path(job["input_path"]).is_relative_to(self.root))
            submit.assert_called_once()
            self.assertEqual(self.client.get("/api/result/" + job["id"]).status_code, 409)
            public = self.client.get("/api/status/" + job["id"]).json
            self.assertNotIn("dir", public)
            self.assertNotIn("input_path", public)

    def test_queue_is_bounded_and_single_worker(self):
        self.assertEqual(web.executor._max_workers, 1)
        web.jobs.update({str(i): {"status": "queued"} for i in range(4)})
        response = self.client.post("/api/upload", data={"video": (io.BytesIO(b"x"), "v.mp4")})
        self.assertEqual(response.status_code, 429)

    def test_finished_job_survives_restart_and_private_files_are_blocked(self):
        job_id = "abcd123456"
        job_dir = self.root / job_id
        job_dir.mkdir()
        original = job_dir / "input.mp4"
        original.write_bytes(b"1234")
        report = {"method": "cpu_motion_fallback", "segment_count": 0}
        (job_dir / "demo_report.json").write_text(json.dumps(report))
        job = {"id": job_id, "dir": str(job_dir), "input_path": str(original),
               "video_name": "input.mp4", "status": "done", "progress": 100}
        (job_dir / "job.json").write_text(json.dumps(job))
        response = self.client.get("/api/result/" + job_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["playback_mode"], "saved_inference")
        self.assertEqual(self.client.get(f"/files/{job_id}/job.json").status_code, 404)
        with self.client.get(f"/files/{job_id}/original", headers={"Range": "bytes=0-1"}) as ranged:
            self.assertEqual(ranged.status_code, 206)

    def test_interrupted_job_is_not_claimed_as_running(self):
        job_id = "1111111111"
        job_dir = self.root / job_id
        job_dir.mkdir()
        (job_dir / "job.json").write_text(json.dumps({"id": job_id, "status": "running", "dir": str(job_dir)}))
        self.assertEqual(self.client.get("/api/status/" + job_id).json["status"], "error")


if __name__ == "__main__":
    unittest.main()