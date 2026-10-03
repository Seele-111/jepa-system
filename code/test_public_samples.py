"""Public synthetic media and sample API checks; no worker or live HTTP service."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_app as web
import demo_detector as detector
import generate_public_samples as generator


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            frames.append(frame)
        return frames, fps
    finally:
        cap.release()


def object_center(frame):
    mask = (frame[:, :, 0] < 140) & (frame[:, :, 1] > 150) & (frame[:, :, 2] > 190)
    _, xs = np.where(mask)
    return float(xs.mean()) if xs.size else None


class GeneratorChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "public-samples"
        self.addCleanup(patch.stopall)
        patch.object(generator.shutil, "which", return_value=None).start()

    def test_generated_videos_are_decodable_and_contain_no_reports_or_labels(self):
        paths = generator.generate_public_samples(self.root)
        self.assertEqual({path.name for path in paths}, {s["name"] for s in generator.PUBLIC_SAMPLES.values()})
        self.assertEqual({path.name for path in self.root.iterdir()}, {path.name for path in paths})
        for path in paths:
            with self.subTest(video=path.name):
                frames, fps = read_frames(path)
                self.assertEqual(len(frames), generator.FRAME_COUNT)
                self.assertAlmostEqual(fps, generator.FPS)
                self.assertTrue(all(frame.shape == (180, 320, 3) for frame in frames))
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_pixel_scenarios_show_smooth_motion_jump_and_disappearance_not_predictions(self):
        generator.generate_public_samples(self.root)
        videos = {key: read_frames(self.root / sample["name"])[0]
                  for key, sample in generator.PUBLIC_SAMPLES.items()}
        continuous = videos["synthetic-continuous"]
        centers = [object_center(frame) for frame in continuous]
        self.assertTrue(all(center is not None for center in centers))
        self.assertLess(max(abs(b - a) for a, b in zip(centers, centers[1:])), 3)
        jump = videos["synthetic-jump"]
        self.assertGreater(object_center(jump[48]) - object_center(jump[47]), 120)
        blink = videos["synthetic-blink"]
        self.assertIsNotNone(object_center(blink[39]))
        self.assertTrue(all(object_center(frame) is None for frame in blink[40:56]))
        self.assertIsNotNone(object_center(blink[56]))

    def test_incomplete_existing_video_is_regenerated_without_touching_complete_samples(self):
        generator.generate_public_samples(self.root)
        path = self.root / "synthetic_jump.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), generator.FPS, generator.FRAME_SIZE)
        try:
            self.assertTrue(writer.isOpened())
            writer.write(generator._frame("synthetic-jump", 0))
        finally:
            writer.release()
        self.assertFalse(generator._valid_video(path))
        with patch.object(generator, "_generate_video", wraps=generator._generate_video) as generate:
            generator.generate_public_samples(self.root)
        generate.assert_called_once_with("synthetic-jump", path)
        self.assertTrue(generator._valid_video(path))

    def test_generation_reuses_complete_videos_without_reencoding(self):
        paths = generator.generate_public_samples(self.root)
        before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths}
        with patch.object(generator.cv2, "VideoWriter") as writer:
            self.assertEqual(generator.generate_public_samples(self.root), paths)
            writer.assert_not_called()
        self.assertEqual(before, {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths})

    def test_encoder_failure_leaves_no_partial_video_or_report(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with patch.object(generator.cv2, "VideoWriter", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "encoder"):
                generator.generate_public_samples(self.root)
        writer.release.assert_called_once()
        self.assertFalse(list(self.root.iterdir()))

    def test_optional_ffmpeg_timeout_keeps_valid_downloadable_media(self):
        with patch.object(generator.shutil, "which", return_value="local-ffmpeg"), \
             patch.object(generator.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 30)) as run:
            paths = generator.generate_public_samples(self.root)
        self.assertEqual(run.call_count, len(generator.PUBLIC_SAMPLES))
        self.assertTrue(all(generator._valid_video(path) for path in paths))
        self.assertFalse(list(self.root.glob(".*.mp4")))


class SampleApiChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.media = self.root / "public-samples"
        self.private = self.root / "originals"
        self.private.mkdir()
        self.original_samples = {key: {**sample, "dir": self.root / ("recorded-" + key)}
                                 for key, sample in web.SAMPLES.items()}
        patch.object(web, "RUN_ROOT", self.runs).start()
        patch.object(web, "PUBLIC_SAMPLE_ROOT", self.media).start()
        patch.object(web, "DATA_ROOT", self.private).start()
        patch.object(web, "SAMPLES", self.original_samples).start()
        patch.object(web, "jobs", {}).start()
        patch.object(generator.shutil, "which", return_value=None).start()
        self.submit = patch.object(web.executor, "submit").start()
        self.client = web.app.test_client()

    def catalog(self):
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        return {sample["id"]: sample for sample in response.json["samples"]}

    def queue(self, sample_id="synthetic-jump", **data):
        response = self.client.post(f"/api/samples/{sample_id}/analyze", data=data)
        self.assertEqual(response.status_code, 202)
        return web.jobs[response.json["job_id"]]

    def test_fresh_catalog_generates_only_public_media_and_never_runs_inference(self):
        with patch.object(web, "analyze_video") as analyze:
            catalog = self.catalog()
        self.assertEqual(set(catalog), set(generator.PUBLIC_SAMPLES))
        self.assertTrue(all(s["source"] == "synthetic" and s["input_available"] for s in catalog.values()))
        self.assertTrue(all(not s["result_ready"] and s["result_mode"] is None for s in catalog.values()))
        for sample in catalog.values():
            self.assertTrue({"id", "video_name", "title", "result_ready", "input_available"} <= sample.keys())
        self.assertFalse(web.jobs)
        self.assertFalse(list(self.runs.iterdir()))
        self.assertFalse(list(self.media.rglob("*.json")))
        self.submit.assert_not_called()
        analyze.assert_not_called()

    def test_home_page_is_dynamic_and_has_no_private_buttons_or_evaluation_scores(self):
        text = self.client.get("/").get_data(as_text=True)
        self.assertIn("公开 Synthetic", text)
        self.assertIn("requestJSON('/api/samples')", text)
        self.assertIn("预览不运行模型", text)
        self.assertNotIn('data-sample="0207"', text)
        self.assertNotIn('data-sample="0317"', text)
        for score in ("55.71%", "34.29%", "50.89%", "24/85", "22/63"):
            self.assertNotIn(score, text)
        self.assertFalse(self.media.exists())
        self.submit.assert_not_called()

    def test_public_preview_serves_range_requests_but_not_fake_report_artifacts(self):
        catalog = self.catalog()
        sample = catalog["synthetic-jump"]
        with self.client.get(sample["original_url"], headers={"Range": "bytes=0-15"}) as response:
            self.assertEqual(response.status_code, 206)
            self.assertEqual(len(response.data), 16)
        (self.media / "demo_report.json").write_text('{"method": "fake_fixture"}', encoding="utf-8")
        self.assertFalse(self.catalog()["synthetic-jump"]["result_ready"])
        self.assertEqual(self.client.get("/api/samples/synthetic-jump/result").status_code, 409)
        for filename in ("report", "annotated", "source", "job.json"):
            self.assertEqual(self.client.get("/samples/synthetic-jump/" + filename).status_code, 404)
        self.submit.assert_not_called()

    def test_sample_analysis_honors_all_modes_sampling_and_explicit_cpu_preference(self):
        for algorithm in ("optimized", "optimized_fast", "legacy"):
            with self.subTest(algorithm=algorithm):
                job = self.queue(algorithm=algorithm, prefer_true_jepa="0", max_frames="48", max_keyframes="20")
                args = self.submit.call_args.args
                self.assertEqual(args, (web._run_job, job["id"], Path(job["input_path"]), False, 48, 20, algorithm))
                self.assertEqual(job["sample_source"], "synthetic")
                self.assertEqual(job["sample_id"], "synthetic-jump")
                input_path = Path(job["input_path"])
                self.assertTrue(input_path.is_relative_to(self.runs))
                self.assertEqual(input_path.read_bytes(), (self.media / "synthetic_jump.mp4").read_bytes())
                stored = json.loads((Path(job["dir"]) / "job.json").read_text(encoding="utf-8"))
                self.assertEqual(stored["sample_source"], "synthetic")
                public = self.client.get("/api/status/" + job["id"]).json
                self.assertNotIn("input_path", public)
                self.assertNotIn("dir", public)
                self.assertEqual(self.client.get("/api/result/" + job["id"]).status_code, 409)

    def test_invalid_sample_parameters_do_not_generate_or_queue(self):
        for data in ({"algorithm": "invented"}, {"max_frames": "bad"}, {"max_keyframes": "1"}):
            self.assertEqual(self.client.post("/api/samples/synthetic-jump/analyze", data=data).status_code, 400)
        self.assertEqual(self.client.post("/api/samples/missing/analyze").status_code, 404)
        self.assertEqual(self.client.get("/api/samples/missing/result").status_code, 404)
        self.assertEqual(self.client.get("/samples/missing/original").status_code, 404)
        self.assertFalse(self.media.exists())
        self.assertFalse(list(self.runs.iterdir()))
        self.submit.assert_not_called()

    def test_sample_queue_keeps_existing_bound(self):
        self.catalog()
        web.jobs.update({str(i): {"status": "queued"} for i in range(4)})
        self.assertEqual(self.client.post("/api/samples/synthetic-jump/analyze").status_code, 429)
        self.assertFalse(list(self.runs.iterdir()))
        self.submit.assert_not_called()

    def test_generation_errors_are_actionable_and_do_not_create_results(self):
        with patch.object(web, "generate_public_samples", side_effect=RuntimeError("encoder unavailable")):
            response = self.client.get("/api/samples")
            self.assertEqual(response.status_code, 200)
            self.assertIn("generate_public_samples.py", response.json["generation_error"])
            self.assertTrue(all(not s["input_available"] and not s["result_ready"] for s in response.json["samples"]))
            self.assertEqual(self.client.post("/api/samples/synthetic-jump/analyze").status_code, 503)
            self.assertEqual(self.client.get("/samples/synthetic-jump/original").status_code, 503)
        self.assertFalse(web.jobs)
        self.submit.assert_not_called()

    def test_configured_original_samples_preserve_legacy_endpoints_and_provenance(self):
        sample = self.original_samples["0207"]
        original = self.private / sample["name"]
        original.write_bytes(b"original fixture")
        sample["dir"].mkdir()
        report = {"method": "true_vjepa_ijepa", "segment_count": 0}
        (sample["dir"] / "demo_report.json").write_text(json.dumps(report), encoding="utf-8")
        catalog = self.catalog()
        self.assertIn("0207", catalog)
        self.assertNotIn("0317", catalog)
        self.assertEqual(catalog["0207"]["source"], "local_original")
        self.assertEqual(catalog["0207"]["result_mode"], "recorded_demo")
        result = self.client.get("/api/samples/0207/result")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["playback_mode"], "recorded_demo")
        self.assertEqual(result.json["sample_source"], "local_original")
        self.assertEqual(result.json["method"], report["method"])
        with self.client.get("/samples/0207/original") as response:
            self.assertEqual(response.data, b"original fixture")
        job = self.queue("0207", algorithm="optimized_fast")
        self.assertEqual(self.submit.call_args.args[-1], "optimized_fast")
        self.assertTrue(self.submit.call_args.args[3])
        self.assertEqual(Path(job["input_path"]), original)
        self.assertEqual(job["sample_source"], "local_original")

    def test_original_recording_remains_accessible_without_its_input_video(self):
        sample = self.original_samples["0317"]
        sample["dir"].mkdir()
        (sample["dir"] / "demo_report.json").write_text('{"method": "fixture"}', encoding="utf-8")
        item = self.catalog()["0317"]
        self.assertFalse(item["input_available"])
        self.assertTrue(item["result_ready"])
        self.assertEqual(self.client.get("/api/samples/0317/result").status_code, 200)
        self.assertEqual(self.client.post("/api/samples/0317/analyze").status_code, 404)
        self.submit.assert_not_called()

    def test_real_explicit_cpu_analysis_can_be_replayed_after_restart(self):
        job = self.queue(algorithm="legacy", prefer_true_jepa="0")
        input_bytes = Path(job["input_path"]).read_bytes()
        args = self.submit.call_args.args
        with patch.object(detector, "_run_wsl_true_jepa") as wsl:
            web._run_job(*args[1:])
        wsl.assert_not_called()
        response = self.client.get("/api/result/" + job["id"])
        self.assertEqual(response.status_code, 200)
        report = response.json
        self.assertEqual(report["method"], "cpu_motion_fallback")
        self.assertEqual(report["playback_mode"], "new_inference")
        self.assertEqual(report["sample_source"], "synthetic")
        self.assertEqual(report["total_frames"], generator.FRAME_COUNT)
        self.assertAlmostEqual(report["fps"], generator.FPS)
        self.assertEqual(len(report["score_series"]), generator.FRAME_COUNT)
        for segment in report["segments"]:
            self.assertAlmostEqual(segment["start_seconds"], segment["start_frame"] / generator.FPS)
            self.assertAlmostEqual(segment["end_seconds"], (segment["end_frame"] + 1) / generator.FPS)
        with self.client.get(report["report_url"]) as response:
            downloaded = json.loads(response.data)
            self.assertEqual(downloaded["method"], "cpu_motion_fallback")
            self.assertEqual(downloaded["sample_source"], "synthetic")
        (self.media / "synthetic_jump.mp4").unlink()
        with self.client.get(report["original_url"]) as response:
            self.assertEqual(response.data, input_bytes)
        web.jobs.clear()
        item = self.catalog()["synthetic-jump"]
        self.assertTrue(item["result_ready"])
        self.assertEqual(item["result_mode"], "saved_inference")
        saved = self.client.get("/api/samples/synthetic-jump/result").json
        self.assertEqual(saved["playback_mode"], "saved_inference")
        self.assertEqual(saved["sample_source"], "synthetic")
        self.assertEqual(saved["method"], "cpu_motion_fallback")
        self.assertEqual(saved["report_url"], report["report_url"])
        self.assertEqual(self.client.get("/api/result/" + job["id"]).json["playback_mode"], "saved_inference")
        self.assertEqual(self.submit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
