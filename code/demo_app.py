#!/usr/bin/env python3
"""Local product demo; serial inference, persistent reports, and honest provenance."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from demo_detector import PROJECT_ROOT, _video_info, analyze_video
from generate_public_samples import PUBLIC_SAMPLE_ROOT, PUBLIC_SAMPLES, generate_public_samples
from jepa_runtime import settings
from optimized_locator import load_bundle

DEMO_PORT = int(os.environ.get("JEPA_DEMO_PORT", "5002"))

RUN_ROOT = PROJECT_ROOT / "output" / "demo-runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = settings().data_root
SAMPLES = {
    "0207": {"name": "0207.mp4", "title": "双片段定位样例", "dir": PROJECT_ROOT / "output" / "algorithm-opt-2026-10-02" / "product_smoke" / "final_0207"},
    "0317": {"name": "0317.mp4", "title": "正常拒识样例", "dir": PROJECT_ROOT / "output" / "algorithm-opt-2026-10-02" / "product_smoke" / "final_0317"},
}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
jobs: dict[str, dict] = {}
lock = threading.RLock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jepa-demo")
JOB_PATTERN = re.compile(r"^[0-9a-f]{10}$")
SAMPLE_FIELDS = ("sample_id", "sample_source", "sample_title", "sample_description")
sample_lock = threading.Lock()
PUBLIC_SAMPLE_ERROR = "公开样例生成失败，请显式运行 python code/generate_public_samples.py 并检查目录权限和视频编码器。"


def _save_job(job: dict):
    path = Path(job["dir"]) / "job.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _get_job(job_id: str) -> dict | None:
    if not JOB_PATTERN.fullmatch(job_id): return None
    with lock:
        if job_id not in jobs:
            path = RUN_ROOT / job_id / "job.json"
            if not path.is_file(): return None
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                # Disk state is not a running worker after an app restart.
                if job["status"] in {"queued", "running"}:
                    job.update(status="error", error="服务已重启，请重新分析视频", progress=100)
                job["restored_from_disk"] = True
                job["dir"] = str(RUN_ROOT / job_id)
                jobs[job_id] = job
            except (OSError, ValueError, KeyError): return None
        return dict(jobs[job_id])


def _public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in {"dir", "input_path"}}


@app.before_request
def local_only():
    if request.host.split(":")[0] not in {"127.0.0.1", "localhost"}:
        return jsonify(error="此演示仅允许本机访问"), 403
    if request.method == "POST":
        origin = request.headers.get("Origin")
        if origin and origin not in {"http://127.0.0.1:" + str(DEMO_PORT), "http://localhost:" + str(DEMO_PORT)}:
            return jsonify(error="不允许跨站提交演示任务"), 403


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="视频超过 500 MB 限制"), 413


@app.get("/health")
def health():
    return jsonify(ok=True, service="jepa-demo", schema_version="jepa-demo-report-v1",
                   processing="serial", demo_only=True)



@app.get("/api/capabilities")
def capabilities():
    # Package verification is not a claim that CUDA/upstream resources are ready.
    cfg = settings()
    modes = {}
    for name, path in (("optimized_fast", cfg.motion_bundle), ("optimized", cfg.full_bundle)):
        item = {"bundle_present": path.is_file(), "bundle_readable": False, "bundle_verified": False,
                "resource_readiness": "not_required" if name == "optimized_fast" else "not_checked",
                "usable": None}
        try:
            bundle = load_bundle(path)
            item["bundle_readable"] = True
            item["bundle_verified"] = bool(bundle.get("publication"))
            item["validation_level"] = "reviewed_public_release" if item["bundle_verified"] else "legacy_schema_only"
            if name == "optimized_fast" and item["bundle_verified"]:
                item["usable"] = True
        except (OSError, ValueError, KeyError):
            item.update(usable=False, error_code="invalid_or_missing_bundle")
        modes[name] = item
    modes["legacy"] = {"cpu_rules_available": True, "true_jepa_resource_readiness": "not_checked"}
    return jsonify(default_algorithm=cfg.default_algorithm, modes=modes,
                   model_resource_check="python code/model_doctor.py --model-runtime",
                   cuda_or_resources_verified=False)


@app.get("/")
def index():
    return render_template("demo.html", default_algorithm=settings().default_algorithm)


def _prepare_public_samples() -> str | None:
    with sample_lock:
        try:
            generate_public_samples(PUBLIC_SAMPLE_ROOT)
        except (OSError, RuntimeError):
            return PUBLIC_SAMPLE_ERROR
    return None


def _sample(sample_id: str) -> dict | None:
    if sample_id in PUBLIC_SAMPLES:
        return {**PUBLIC_SAMPLES[sample_id], "source": "synthetic", "dir": PUBLIC_SAMPLE_ROOT,
                "input_path": PUBLIC_SAMPLE_ROOT / PUBLIC_SAMPLES[sample_id]["name"]}
    if sample_id in SAMPLES:
        return {**SAMPLES[sample_id], "source": "local_original",
                "description": "本地配置的原始开发样例；录制报告与本次新推理分开展示，不是盲测。",
                "input_path": DATA_ROOT / SAMPLES[sample_id]["name"]}
    return None


def _sample_metadata(sample_id: str, sample: dict) -> dict:
    return {"sample_id": sample_id, "sample_source": sample["source"],
            "sample_title": sample["title"], "sample_description": sample["description"]}


def _saved_sample_job(sample_id: str) -> dict | None:
    # Only completed, explicitly queued sample jobs count as replayable inference.
    # A report dropped beside generated media is not a recorded model result.
    candidates = []
    for path in RUN_ROOT.glob("*/job.json"):
        job_id = path.parent.name
        if not JOB_PATTERN.fullmatch(job_id): continue
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if (job.get("sample_id") != sample_id or job.get("sample_source") != "synthetic"
                    or job.get("status") != "done" or not (path.parent / "demo_report.json").is_file()):
                continue
            candidates.append((float(job.get("created_at", 0)), job_id))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    for _, job_id in sorted(candidates, reverse=True):
        job = _get_job(job_id)
        if job and job["status"] == "done": return job
    return None


@app.get("/api/samples")
def samples():
    generation_error = _prepare_public_samples()
    items = []
    for sample_id in [*PUBLIC_SAMPLES, *SAMPLES]:
        sample = _sample(sample_id)
        input_available = sample["input_path"].is_file()
        saved = _saved_sample_job(sample_id) if sample["source"] == "synthetic" else None
        result_ready = saved is not None if sample["source"] == "synthetic" else (sample["dir"] / "demo_report.json").is_file()
        if sample["source"] == "local_original" and not (input_available or result_ready): continue
        items.append({"id": sample_id, "video_name": sample["name"], "title": sample["title"],
                      "result_ready": result_ready, "input_available": input_available,
                      "source": sample["source"], "description": sample["description"],
                      "original_url": f"/samples/{sample_id}/original",
                      "result_mode": ("saved_inference" if saved else "recorded_demo") if result_ready else None})
    return jsonify(samples=items, generation_error=generation_error)


def _sampling():
    try:
        frames = int(request.form.get("max_frames", "32"))
        keyframes = int(request.form.get("max_keyframes", "16"))
    except (ValueError, TypeError): return None
    if not 8 <= frames <= 256 or not 4 <= keyframes <= 128: return None
    return frames, keyframes


def _enqueue(input_path: Path, video_name: str, *, job_id: str, job_dir: Path,
             prefer_true: bool, sampling: tuple[int, int], algorithm: str = 'legacy'):
    job = {"id": job_id, "status": "queued", "progress": 0, "status_msg": "已入队，等待推理资源",
           "video_name": video_name, "dir": str(job_dir), "input_path": str(input_path),
           "created_at": time.time()}
    jobs[job_id] = job
    _save_job(job)
    executor.submit(_run_job, job_id, input_path, prefer_true, *sampling, algorithm)
    return jsonify(job_id=job_id, status="queued"), 202


def _busy():
    return sum(j["status"] in {"queued", "running"} for j in jobs.values()) >= 4


@app.post("/api/upload")
def upload():
    sampling = _sampling()
    if sampling is None: return jsonify(error="采样参数范围无效"), 400
    algorithm=request.form.get('algorithm','legacy')
    if algorithm not in {'legacy','optimized','optimized_fast'}:return jsonify(error='算法模式无效'),400
    file = request.files.get("video")
    if file is None or not file.filename: return jsonify(error="请选择视频文件"), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
        return jsonify(error="支持 MP4 / WebM / AVI / MOV / MKV"), 400
    with lock:
        if _busy(): return jsonify(error="演示队列已满，请等待当前分析完成"), 429
        job_id = uuid.uuid4().hex[:10]
        job_dir = RUN_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"input{suffix}"
        file.save(input_path)
        return _enqueue(input_path, Path(file.filename.replace("\\", "/")).name[:160],
                        job_id=job_id, job_dir=job_dir,
                        prefer_true=request.form.get("prefer_true_jepa", "1") not in {"0", "false", "False"},
                        sampling=sampling,algorithm=algorithm)


@app.post("/api/samples/<sample_id>/analyze")
def analyze_sample(sample_id: str):
    sample = _sample(sample_id)
    if sample is None: return jsonify(error="样例不存在"), 404
    sampling = _sampling()
    if sampling is None: return jsonify(error="采样参数范围无效"), 400
    algorithm=request.form.get('algorithm','legacy')
    if algorithm not in {'legacy','optimized','optimized_fast'}:return jsonify(error='算法模式无效'),400
    if sample["source"] == "synthetic":
        generation_error = _prepare_public_samples()
        if generation_error: return jsonify(error=generation_error), 503
    video = sample["input_path"]
    if not video.is_file(): return jsonify(error="样例视频不可用"), 404
    with lock:
        if _busy(): return jsonify(error="演示队列已满，请稍后重试"), 429
        job_id = uuid.uuid4().hex[:10]
        job_dir = RUN_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        if sample["source"] == "synthetic":
            # Keep the exact input used by this task, even if samples are regenerated later.
            input_path = job_dir / "input.mp4"
            try:
                shutil.copyfile(video, input_path)
            except OSError:
                return jsonify(error="无法保存样例输入，请检查任务目录权限"), 503
        else:
            input_path = video
        response = _enqueue(input_path, sample["name"], job_id=job_id, job_dir=job_dir,
                            prefer_true=request.form.get("prefer_true_jepa", "1") not in {"0", "false", "False"},
                            sampling=sampling,algorithm=algorithm)
        jobs[job_id].update(_sample_metadata(sample_id, sample))
        _save_job(jobs[job_id])
        return response


@app.get("/api/status/<job_id>")
def status(job_id: str):
    job = _get_job(job_id)
    if job is None: return jsonify(error="任务不存在"), 404
    return jsonify(_public_job(job))


def _report(root: Path, video_name: str, base_url: str, mode: str, sample: dict | None = None):
    try:
        report = json.loads((root / "demo_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError): return jsonify(error="结果文件不可用，请重新分析"), 409
    report.update(video_name=video_name, annotated_url=base_url + "/annotated",
                  original_url=base_url + "/original", report_url=base_url + "/report",
                  source_report_url=base_url + "/source" if report.get("source_report") else None,
                  playback_mode=mode)
    if sample:
        report.update({key: sample[key] for key in SAMPLE_FIELDS if key in sample})
    return jsonify(report)


@app.get("/api/result/<job_id>")
def result(job_id: str):
    job = _get_job(job_id)
    if job is None: return jsonify(error="任务不存在"), 404
    if job["status"] != "done": return jsonify(error="结果尚未就绪", status=job["status"]), 409
    return _report(Path(job["dir"]), job["video_name"], f"/files/{job_id}",
                   "saved_inference" if job.get("restored_from_disk") else "new_inference",
                   sample=job if job.get("sample_id") else None)


@app.get("/api/samples/<sample_id>/result")
def sample_result(sample_id: str):
    sample = _sample(sample_id)
    if sample is None: return jsonify(error="样例不存在"), 404
    if sample["source"] == "synthetic":
        job = _saved_sample_job(sample_id)
        if job is None: return jsonify(error="公开 synthetic 样例没有已保存的推理结果，请用当前模式分析"), 409
        return _report(Path(job["dir"]), job["video_name"], f"/files/{job['id']}", "saved_inference", sample=job)
    return _report(sample["dir"], sample["name"], f"/samples/{sample_id}", "recorded_demo",
                   sample=_sample_metadata(sample_id, sample))


def _artifact(root: Path, original: Path, filename: str):
    allowed = {"annotated": root / "annotated.mp4", "original": original, "report": root / "demo_report.json"}
    if filename == "source":
        try:
            report = json.loads((root / "demo_report.json").read_text(encoding="utf-8"))
            source = Path(report["source_report"]).resolve()
            if not source.is_relative_to(root.resolve()): return jsonify(error="非法来源路径"), 400
            allowed["source"] = source
        except (KeyError, OSError, ValueError): return jsonify(error="原始报告不可用"), 404
    path = allowed.get(filename)
    if path is None or not path.is_file(): return jsonify(error="文件不存在"), 404
    return send_file(path, conditional=True, as_attachment=filename in {"report", "source"},
                     download_name=("demo_report.json" if filename == "report" else path.name))


@app.get("/files/<job_id>/<filename>")
def files(job_id: str, filename: str):
    job = _get_job(job_id)
    if job is None: return jsonify(error="任务不存在"), 404
    return _artifact(Path(job["dir"]), Path(job["input_path"]), filename)


@app.get("/samples/<sample_id>/<filename>")
def sample_files(sample_id: str, filename: str):
    sample = _sample(sample_id)
    if sample is None: return jsonify(error="样例不存在"), 404
    if sample["source"] == "synthetic":
        if filename != "original": return jsonify(error="请通过已完成任务回看或下载真实推理产物"), 404
        generation_error = _prepare_public_samples()
        if generation_error: return jsonify(error=generation_error), 503
    return _artifact(sample["dir"], sample["input_path"], filename)


def _run_job(job_id: str, input_path: Path, prefer_true: bool, max_frames: int, max_keyframes: int, algorithm: str = 'legacy'):
    def progress(message: str, percent: int):
        with lock:
            jobs[job_id].update(status="running", progress=percent, status_msg=message)
            _save_job(jobs[job_id])
    try:
        progress("读取视频", 10)
        frames, fps, _, _ = _video_info(input_path)
        if frames / fps > 120 or frames > 10000:
            raise ValueError("当前演示限制为 120 秒 / 10000 帧以内的视频")
        report = analyze_video(input_path, Path(jobs[job_id]["dir"]), prefer_true_jepa=prefer_true,
                               max_frames=max_frames, max_keyframes=max_keyframes, progress_callback=progress,algorithm=algorithm)
        if jobs[job_id].get("sample_id"):
            report.update({key: jobs[job_id][key] for key in SAMPLE_FIELDS if key in jobs[job_id]})
            (Path(jobs[job_id]["dir"]) / "demo_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with lock:
            jobs[job_id].update(status="done", progress=100, status_msg="分析完成",
                                method=report["method"], verdict=report["verdict"],
                                segment_count=report["segment_count"], elapsed_seconds=report["elapsed_seconds"])
            _save_job(jobs[job_id])
    except Exception as exc:
        with lock:
            jobs[job_id].update(status="error", progress=100, error=str(exc), status_msg="分析失败")
            _save_job(jobs[job_id])


if __name__ == "__main__":
    print(f"JEPA demo app: http://127.0.0.1:{DEMO_PORT}")
    app.run(host="127.0.0.1", port=DEMO_PORT, debug=False, threaded=True)
