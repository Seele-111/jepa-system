#!/usr/bin/env python3
"""Local product-demo detector for the JEPA video error-localization project.

This module is deliberately separate from the historical research entry points.
It prefers the real V-JEPA/I-JEPA pipeline available in the local WSL image and
falls back to a clearly-labelled CPU motion probe when that path is unavailable.
The demo post-processing is conservative and is not a replacement for the
strict OOF evaluation protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import ntpath
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WSL_DISTRO = os.environ.get("JEPA_WSL_DISTRO", "Ubuntu-24.04")
DEFAULT_WSL_PYTHON = os.environ.get("JEPA_WSL_PYTHON", "python3")
DEFAULT_WSL_PIPELINE = os.environ.get("JEPA_WSL_PIPELINE", str(PROJECT_ROOT / "code" / "detect_and_report_v4.py"))

TRUE_DEMO_PROFILE = {
    "name": "true-jepa-demo-v1",
    "score_threshold": 0.40,
    "smooth_window": 3,
    "min_gap": 3,
    "min_length": 3,
    "min_segment_max": 0.50,
    "min_peak_prominence": 0.120,
}
MOTION_DEMO_PROFILE = {
    "name": "cpu-motion-fallback-v1",
    "score_threshold": 0.70,
    "smooth_window": 5,
    "min_gap": 4,
    "min_length": 5,
}

class DemoDetectionError(RuntimeError):
    """Raised when a demo detection step cannot produce a usable report."""


def _decode_process_bytes(data: bytes) -> str:
    if not data:
        return ""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in data[:80]:
        try:
            return data.decode("utf-16", errors="replace")
        except UnicodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _tail(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text


def windows_to_wsl_path(path: str | Path) -> str:
    """Convert a local Windows drive path without relying on a shell."""
    raw = os.fspath(path)
    # Recognize an explicit Windows drive before host-native abspath. On Linux
    # abspath otherwise prefixes cwd and loses the drive during offline checks.
    raw = ntpath.normpath(raw) if re.match(r"^[A-Za-z]:[\\/]", raw) else os.path.abspath(raw)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if match:
        drive, tail = match.groups()
        return f"/mnt/{drive.lower()}/{tail.replace(chr(92), '/')}"
    return raw.replace(chr(92), "/")


def _run_wsl_true_jepa(video_path: Path, output_dir: Path, *, max_frames: int = 32,
                       max_keyframes: int = 16, timeout: int = 900,
                       distro: str = DEFAULT_WSL_DISTRO) -> dict[str, Any]:
    """Run the existing true-JEPA detector inside the local WSL environment."""
    # A fresh run directory prevents old output from impersonating a new inference.
    output_dir = output_dir / "true-jepa" / uuid.uuid4().hex[:10]
    output_dir.mkdir(parents=True, exist_ok=False)
    from jepa_runtime import model_command, runtime_path
    command = model_command(PROJECT_ROOT / "code" / "demo_wsl_runner.py", [
        "--configured-resources", runtime_path(DEFAULT_WSL_PIPELINE), "0",
        "--video", runtime_path(video_path), "--output", runtime_path(output_dir),
        "--no-cache", "--threshold", "0.99", "--min-gap", "1", "--min-length", "1",
        "--max-frames", str(int(max_frames)), "--max-keyframes", str(int(max_keyframes)),
        "--use-true-vjepa", "--use-true-ijepa",
    ])
    try:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=int(timeout), check=False)
    except FileNotFoundError as exc:
        raise DemoDetectionError("wsl.exe is not available") from exc
    except subprocess.TimeoutExpired as exc:
        raise DemoDetectionError(f"true JEPA timed out after {timeout}s") from exc
    stdout, stderr = _decode_process_bytes(proc.stdout), _decode_process_bytes(proc.stderr)
    timeseries_path = output_dir / "vjepa_timeseries.json"
    report_path = output_dir / "report.json"
    if proc.returncode != 0:
        raise DemoDetectionError(f"true JEPA process failed (rc={proc.returncode}): {_tail(stderr or stdout)}")
    if not timeseries_path.is_file():
        raise DemoDetectionError(f"true JEPA finished without vjepa_timeseries.json: {_tail(stderr or stdout)}")
    try:
        timeseries = json.loads(timeseries_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoDetectionError(f"cannot read true-JEPA timeseries: {exc}") from exc
    source_report = {}
    if report_path.is_file():
        try:
            source_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"timeseries": timeseries, "source_report": source_report,
            "stdout_tail": _tail(stdout), "stderr_tail": _tail(stderr), "command": command,
            "source_report_path": report_path, "timeseries_path": timeseries_path}


def _validated_true_score(timeseries: dict[str, Any], total_frames: int, fps: float) -> np.ndarray:
    usage = timeseries.get("jepa_usage", {})
    if usage.get("vjepa") != "true_vjepa_predictor" or usage.get("ijepa") != "true_ijepa_predictor":
        raise DemoDetectionError("pipeline did not confirm both true JEPA predictors")
    score = np.asarray(timeseries.get("composite", []), dtype=np.float32)
    if score.ndim != 1 or len(score) != total_frames or not np.isfinite(score).all():
        raise DemoDetectionError("true-JEPA score is empty, non-finite, or not frame-aligned")
    if int(timeseries.get("total_frames", -1)) != total_frames:
        raise DemoDetectionError("true-JEPA frame count does not match the input video")
    source_fps = float(timeseries.get("fps", 0))
    if not np.isfinite(source_fps) or not np.isclose(source_fps, fps, rtol=0.001, atol=0.001):
        raise DemoDetectionError("true-JEPA FPS does not match the input video")
    if score.min() < -0.001 or score.max() > 1.001:
        raise DemoDetectionError("true-JEPA composite is outside the expected normalized range")
    return score


def _video_info(video_path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DemoDetectionError(f"cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if not np.isfinite(fps) or fps <= 0: fps = 16.0
    if count <= 0 or width <= 0 or height <= 0:
        raise DemoDetectionError(f"video metadata is incomplete: {video_path}")
    return count, fps, width, height


def _interpolate(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if length <= 0: return np.zeros(0, dtype=np.float32)
    if len(values) == length: return values
    if len(values) == 0: return np.zeros(length, dtype=np.float32)
    if len(values) == 1: return np.full(length, float(values[0]), dtype=np.float32)
    return np.interp(np.arange(length, dtype=np.float32),
                     np.linspace(0, length - 1, len(values), dtype=np.float32), values).astype(np.float32)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values, window = np.asarray(values, dtype=np.float32), max(1, int(window))
    if window <= 1 or len(values) <= 1: return values.copy()
    left, right = window // 2, window - 1 - window // 2
    return np.convolve(np.pad(values, (left, right), mode="edge"),
                       np.ones(window, dtype=np.float32) / window, mode="valid")


def _segments_from_binary(binary: np.ndarray, *, min_gap: int, min_length: int) -> list[tuple[int, int]]:
    binary = np.asarray(binary, dtype=bool).reshape(-1)
    segments, start = [], None
    for index, value in enumerate(np.r_[binary, False]):
        if value and start is None: start = index
        elif not value and start is not None:
            segments.append((start, index - 1)); start = None
    if int(min_gap) > 0 and len(segments) > 1:
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end - 1 <= int(min_gap): merged[-1] = (prev_start, end)
            else: merged.append((start, end))
        segments = merged
    return [(start, end) for start, end in segments if end - start + 1 >= int(min_length)]


def _confidence(max_score: float, volatility: float, prominence: float) -> str:
    evidence = max(float(max_score) / 0.70, float(volatility) / 0.10, float(prominence) / 0.15)
    return "high" if evidence >= 1.25 else "medium" if evidence >= 0.90 else "low"


def _make_segments(score: np.ndarray, fps: float, *, profile: dict[str, Any], use_demo_gate: bool) -> list[dict[str, Any]]:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    if not len(score): return []
    if not np.isfinite(score).all() or not np.isfinite(fps) or fps <= 0:
        raise DemoDetectionError("invalid score or FPS for segment generation")
    smoothed = _rolling_mean(score, int(profile["smooth_window"]))
    spans = _segments_from_binary(smoothed >= float(profile["score_threshold"]),
                                 min_gap=int(profile["min_gap"]), min_length=int(profile["min_length"]))
    prominence = np.maximum(score - _rolling_mean(score, 9), 0.0)
    output = []
    for start, end in spans:
        values = smoothed[start:end + 1]
        max_score, mean_score = float(values.max()), float(values.mean())
        volatility = float(values.std())
        peak_prominence = float(prominence[start:end + 1].max())
        if use_demo_gate:
            # True JEPA scores are min-max scaled per video.  A normal video
            # can therefore have a high plateau; require a localized positive
            # peak rather than accepting volatility alone.
            keep = max_score >= float(profile["min_segment_max"]) and (
                peak_prominence >= float(profile["min_peak_prominence"]))
            if not keep: continue
        output.append({
            "start_frame": int(start), "end_frame": int(end), "frame_count": int(end - start + 1),
            "timestamp_start": f"{start / fps:.3f}s",
            "timestamp_end": f"{end / fps:.3f}s",
            "start_seconds": float(start / fps), "end_seconds": float((end + 1) / fps),
            "time_range_convention": "start-inclusive/end-exclusive; frames are inclusive",
            "timestamp_end_kind": "last_frame_pts_inclusive",
            "timestamp_end_exclusive": f"{(end + 1) / fps:.3f}s",
            "boundary_source": "relative_threshold_with_gap_merge_not_semantic_boundary",
            "max_score": max_score, "mean_score": mean_score, "volatility": volatility,
            "peak_prominence": peak_prominence,
            "confidence": _confidence(max_score, volatility, peak_prominence),
            "confidence_kind": "uncalibrated_evidence_level_not_probability",
            "source": "true_jepa_composite" if use_demo_gate else "cpu_motion",
        })
    return output


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) <= 1 or float(values.max() - values.min()) <= 1e-8:
        return np.full(len(values), 0.5, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    # Average ranks for ties: equal motion must not acquire fake temporal order.
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = ((start + end - 1) / 2.0) / float(len(values) - 1)
    return ranks


def _extract_motion_score(video_path: Path) -> tuple[np.ndarray, dict[str, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise DemoDetectionError(f"cannot open video for CPU fallback: {video_path}")
    previous, differences, flows = None, [], []
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
        if previous is None:
            differences.append(0.0); flows.append(0.0)
        else:
            differences.append(float(np.mean(np.abs(gray.astype(np.float32) - previous))))
            flow = cv2.calcOpticalFlowFarneback(previous.astype(np.uint8), gray, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            flows.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
        previous = gray
    cap.release()
    if not differences: raise DemoDetectionError("CPU fallback extracted no frames")
    diff, flow = _rank01(np.asarray(differences)), _rank01(np.asarray(flows))
    return ((diff + flow) / 2.0).astype(np.float32), {
        "frame_diff_std": float(np.std(differences)), "flow_std": float(np.std(flows)),
        "frame_diff_max": float(np.max(differences)), "flow_max": float(np.max(flows)),
    }


def _render_annotated_video(video_path: Path, output_path: Path, score: np.ndarray,
                            segments: list[dict[str, Any]], *, method_label: str,
                            verdict: str, fps: float, width: int, height: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise DemoDetectionError(f"cannot render video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output_path.with_name("annotated_mpeg4.mp4")
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened(): cap.release(); raise DemoDetectionError(f"cannot create annotated video: {output_path}")
    spans = [(int(x["start_frame"]), int(x["end_frame"])) for x in segments]
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        current = float(score[min(index, len(score) - 1)]) if len(score) else 0.0
        active = any(start <= index <= end for start, end in spans)
        if active:
            overlay = np.zeros_like(frame); overlay[:, :] = (35, 35, 210)
            frame = cv2.addWeighted(frame, 0.72, overlay, 0.28, 0.0)
            status, color = "ANOMALY CANDIDATE", (60, 80, 255)
        else: status, color = "NO LOCALIZED EVENT", (90, 210, 110)
        cv2.rectangle(frame, (0, 0), (width, 46), (20, 25, 35), -1)
        title = f"{method_label} | {status}"
        text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        scale = 0.46 * min(1.0, max(0.2, (width - 24) / max(1, text_width)))
        cv2.putText(frame, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {index} / {len(score) - 1} | relative evidence {current:.3f}",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, min(scale, 0.40), (205, 215, 225), 1, cv2.LINE_AA)
        bar_height = min(48, max(28, height // 8)); bar_y = height - bar_height
        cv2.rectangle(frame, (0, bar_y), (width, height), (18, 20, 26), -1)
        for start, end in spans:
            x1 = int(start / max(1, len(score) - 1) * max(1, width - 1))
            x2 = int(end / max(1, len(score) - 1) * max(1, width - 1))
            cv2.rectangle(frame, (x1, bar_y + 8), (max(x1 + 2, x2), height - 10), (50, 70, 220), -1)
        cursor_x = int(index / max(1, len(score) - 1) * max(1, width - 1))
        cv2.line(frame, (cursor_x, bar_y + 3), (cursor_x, height - 3), (0, 220, 255), 2)
        cv2.putText(frame, "Red = candidate region | cursor = current frame", (12, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        writer.write(frame); index += 1
    cap.release(); writer.release()
    if index != len(score):
        raise DemoDetectionError(f"rendered {index} frames but expected {len(score)}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run([
            ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(raw_output),
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "21",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output_path),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
        if proc.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
            raw_output.unlink()
            return {"codec": "h264", "browser_playable": True, "frames": index, "audio_preserved": False}
    # Preserve a downloadable artifact, but do not claim browser playback works.
    raw_output.replace(output_path)
    return {"codec": "mpeg4", "browser_playable": False, "frames": index, "audio_preserved": False}


def _base_report(video_path: Path, *, method: str, method_label: str, score: np.ndarray,
                 segments: list[dict[str, Any]], fps: float, total_frames: int,
                 profile: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "jepa-demo-report-v1", "video": str(video_path), "video_name": video_path.name,
        "total_frames": int(total_frames), "fps": float(fps), "duration_seconds": total_frames / fps,
        "method": method, "method_label": method_label,
        "verdict": "anomalous" if segments else "normal_or_no_localized_evidence",
        "segments": segments, "segment_count": len(segments),
        "score_min": float(score.min()) if len(score) else 0.0, "score_max": float(score.max()) if len(score) else 0.0,
        "score_mean": float(score.mean()) if len(score) else 0.0, "score_series": [float(x) for x in score.tolist()],
        "profile": profile, "warnings": warnings, "demo_only": True,
        "research_protocol_note": "This is a product-demo path, not a fresh blind score and not a replacement for strict OOF evaluation.",
    }


def analyze_video(video_path: str | Path, output_dir: str | Path, *, prefer_true_jepa: bool = True,
                  max_frames: int = 32, max_keyframes: int = 16, timeout: int = 900,
                  progress_callback: Callable[[str, int], None] | None = None, algorithm: str = 'legacy') -> dict[str, Any]:
    """Analyze a video; degradation is explicit and rendering failure is not a model fallback."""
    if algorithm in {'optimized','optimized_fast'}:
        from optimized_detector import analyze_optimized_video,DEFAULT_BUNDLE
        from jepa_runtime import settings
        bundle_path=settings().full_bundle if algorithm=='optimized' else settings().motion_bundle
        return analyze_optimized_video(video_path,output_dir,bundle_path=bundle_path,timeout=timeout,progress_callback=progress_callback,algorithm=algorithm)
    if algorithm!='legacy':raise DemoDetectionError('unsupported algorithm')
    started = time.perf_counter()
    def progress(message: str, percent: int):
        if progress_callback: progress_callback(message, percent)

    video_path, output_dir = Path(video_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress("读取视频元数据", 10)
    total_frames, fps, width, height = _video_info(video_path)
    warnings, true_error, run = [], "", None
    if prefer_true_jepa:
        try:
            progress("真实 V-JEPA / I-JEPA 推理中（首次加载较慢）", 25)
            run = _run_wsl_true_jepa(video_path, output_dir, max_frames=max_frames,
                                     max_keyframes=max_keyframes, timeout=timeout)
            raw_score = _validated_true_score(run["timeseries"], total_frames, fps)
            channel_raw = np.asarray(run["timeseries"].get("physics_raw", []), dtype=np.float32)
            if len(channel_raw) != total_frames or not np.isfinite(channel_raw).all() or not np.any(channel_raw > 0):
                raise DemoDetectionError("V-JEPA has no usable scored coverage; input may be too short")
        except Exception as exc:
            run, true_error = None, str(exc)
            warnings.append(f"真实 JEPA 路径不可用，已降级到 CPU fallback：{true_error}")
    if run is not None:
        progress("生成局部候选与空集合拒识", 72)
        segments = _make_segments(raw_score, fps, profile=TRUE_DEMO_PROFILE, use_demo_gate=True)
        report = _base_report(video_path, method="true_vjepa_ijepa", method_label="True V-JEPA + I-JEPA",
                              score=raw_score, segments=segments, fps=fps, total_frames=total_frames,
                              profile=TRUE_DEMO_PROFILE, warnings=[
                                  "已核验真实 V-JEPA/I-JEPA predictor 身份和逐帧输出。",
                                  "分数是视频内部的相对证据，不是异常概率；未保留候选不等于保证正常。",
                                  "局部峰值门控仅用于演示，尚未完成独立校准或盲测。",
                              ])
        report["channel_support"] = {
            "vjepa_positive_raw_error_frames": int(np.count_nonzero(channel_raw > 0)),
            "total_frames": total_frames,
            "note": "positive-error support inferred from historical pipeline output, not an exact valid-token mask",
        }
        report["warnings"].append("V-JEPA 上下文帧未必有预测误差；缺少证据的帧不能当作已确认正常。")
        if any(s["frame_count"] / total_frames >= 0.8 for s in segments):
            report["warnings"].append("存在覆盖较长的候选段；当前边界来自阈值与间隔合并，请重点人工复核。")
        report.update({"source_report": str(run["source_report_path"]),
                       "source_timeseries": str(run["timeseries_path"]),
                       "jepa_usage": run["timeseries"]["jepa_usage"],
                       "detector_stdout_tail": run["stdout_tail"], "detector_stderr_tail": run["stderr_tail"]})
        render_label = "True JEPA demo"
    else:
        progress("CPU 运动信号降级分析（非 JEPA）", 45)
        raw_score, motion_meta = _extract_motion_score(video_path)
        if len(raw_score) != total_frames:
            raise DemoDetectionError("decoded motion frames do not match video metadata")
        progress("生成运动候选片段（非 JEPA）", 72)
        segments = _make_segments(raw_score, fps, profile=MOTION_DEMO_PROFILE, use_demo_gate=False)
        report = _base_report(video_path, method="cpu_motion_fallback", method_label="CPU motion fallback (not JEPA)",
                              score=raw_score, segments=segments, fps=fps, total_frames=total_frames,
                              profile=MOTION_DEMO_PROFILE, warnings=warnings + [
                                  "只使用灰度帧差和 Farneback 光流，不代表 JEPA 推理。",
                                  "运动变化不等于生成错误；此降级模式不能作为准确率结论。",
                              ])
        report.update({"fallback_stats": motion_meta, "fallback_reason": true_error})
        render_label = "CPU fallback"
    progress("生成浏览器可播放的标注视频", 88)
    report["annotated_video"] = str(output_dir / "annotated.mp4")
    report["video_encoding"] = _render_annotated_video(
        video_path, output_dir / "annotated.mp4", raw_score, segments, method_label=render_label,
        verdict=report["verdict"], fps=fps, width=width, height=height)
    if not report["video_encoding"]["browser_playable"]:
        report["warnings"].append("H.264 编码不可用，标注视频仅保证可下载；可播放原视频对照。")
    report["sampling"] = {"max_frames": int(max_frames), "max_keyframes": int(max_keyframes),
                          "random_seed": 0 if run is not None else None,
                          "reproducibility_note": "fixed seed on this host; not a cross-device bitwise guarantee"}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (output_dir / "demo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("完成", 100)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Local JEPA product-demo detector")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output" / "demo-cli"))
    parser.add_argument("--no-true-jepa", action="store_true")
    parser.add_argument("--algorithm",choices=["legacy","optimized","optimized_fast"],default="legacy")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-keyframes", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    report = analyze_video(args.video, args.output, prefer_true_jepa=not args.no_true_jepa,
                           max_frames=args.max_frames, max_keyframes=args.max_keyframes, timeout=args.timeout,algorithm=args.algorithm)
    print(json.dumps({"method": report["method"], "verdict": report["verdict"],
                      "segment_count": report["segment_count"], "segments": report["segments"],
                      "annotated_video": report["annotated_video"]}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
