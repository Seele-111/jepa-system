#!/usr/bin/env python3
"""Generate synthetic demo media, never labels, predictions or recorded reports.

Scene pixels are deterministic; encoded bytes can vary with the local codec.
Only existing OpenCV/numpy dependencies and optional local FFmpeg are used.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np

PUBLIC_SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "output" / "public-samples"
FPS = 16
FRAME_COUNT = 96
FRAME_SIZE = (320, 180)
PUBLIC_SAMPLES = {
    "synthetic-continuous": {
        "name": "synthetic_continuous.mp4", "title": "连续平滑运动 · Synthetic",
        "description": "程序绘制的几何形状连续平滑移动；不是已验证正常的视频。",
    },
    "synthetic-jump": {
        "name": "synthetic_jump.mp4", "title": "明显位置跳变 · Synthetic",
        "description": "程序绘制的几何形状在移动途中突然跳到另一位置；不保证模型检出。",
    },
    "synthetic-blink": {
        "name": "synthetic_blink.mp4", "title": "短暂消失与重现 · Synthetic",
        "description": "程序绘制的几何形状短暂消失后继续移动；不是带真实标注的评测样本。",
    },
}


def _frame(sample_id: str, index: int) -> np.ndarray:
    width, height = FRAME_SIZE
    frame = np.full((height, width, 3), (24, 30, 40), dtype=np.uint8)
    for x in range(0, width, 32):
        cv2.line(frame, (x, 40), (x, height - 28), (38, 46, 58), 1)
    for y in range(44, height - 28, 24):
        cv2.line(frame, (0, y), (width - 1, y), (38, 46, 58), 1)
    cv2.putText(frame, "SYNTHETIC / PROCEDURAL", (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (190, 200, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, sample_id.removeprefix("synthetic-").upper(), (12, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 200, 215), 1, cv2.LINE_AA)
    x = 48 + round(1.4 * index)
    if sample_id == "synthetic-jump":
        x = 40 + round(0.65 * index) + (140 if index >= FRAME_COUNT // 2 else 0)
    y = 96 + round(15 * math.sin(2 * math.pi * index / FRAME_COUNT))
    if sample_id != "synthetic-blink" or not 40 <= index < 56:
        cv2.circle(frame, (x, y), 16, (80, 200, 245), -1, cv2.LINE_AA)
    return frame


def _valid_video(path: Path) -> bool:
    if not path.is_file(): return False
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): return False
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if size != FRAME_SIZE or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != FRAME_COUNT:
            return False
        if not math.isclose(cap.get(cv2.CAP_PROP_FPS), FPS, rel_tol=0.001): return False
        first_ok, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_COUNT - 1)
        last_ok, _ = cap.read()
        return first_ok and last_ok
    finally:
        cap.release()


def _generate_video(sample_id: str, output_path: Path):
    token = uuid.uuid4().hex
    raw = output_path.with_name(f".{output_path.stem}-{token}-mpeg4.mp4")
    h264 = output_path.with_name(f".{output_path.stem}-{token}-h264.mp4")
    try:
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
        try:
            if not writer.isOpened(): raise RuntimeError("synthetic video encoder is unavailable")
            for index in range(FRAME_COUNT):
                writer.write(_frame(sample_id, index))
        finally:
            writer.release()
        if not _valid_video(raw): raise RuntimeError("synthetic video encoding is incomplete")
        chosen = raw
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                proc = subprocess.run([
                    ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(raw),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-an", str(h264),
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                if proc.returncode == 0 and _valid_video(h264): chosen = h264
            except (OSError, subprocess.TimeoutExpired):
                pass  # MPEG-4 remains downloadable if local H.264 encoding is unavailable.
        chosen.replace(output_path)
    except cv2.error as exc:
        raise RuntimeError("synthetic video generation failed") from exc
    finally:
        raw.unlink(missing_ok=True)
        h264.unlink(missing_ok=True)


def generate_public_samples(output_dir: str | Path = PUBLIC_SAMPLE_ROOT) -> list[Path]:
    """Create missing/incomplete videos only; do not invoke any inference path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    for sample_id, sample in PUBLIC_SAMPLES.items():
        path = output_dir / sample["name"]
        if not _valid_video(path): _generate_video(sample_id, path)
        videos.append(path)
    return videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PUBLIC_SAMPLE_ROOT)
    args = parser.parse_args()
    for path in generate_public_samples(args.output):
        print(f"synthetic video: {path}")
    print("Generated media only: no ground truth, model predictions or recorded results.")
    print("H.264 is attempted with local FFmpeg; otherwise browser playback is not guaranteed.")


if __name__ == "__main__":
    main()
