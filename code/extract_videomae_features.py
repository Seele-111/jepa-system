#!/usr/bin/env python3
"""Extract frozen VideoMAE clip features on the annotation frame grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel


def centered_indices(frame_count: int, clip_length: int = 16, centers: np.ndarray | None = None) -> np.ndarray:
    if frame_count <= 0 or clip_length <= 0:
        raise ValueError("frame_count and clip_length must be positive")
    centers = np.arange(frame_count, dtype=np.int64) if centers is None else np.asarray(centers, dtype=np.int64)
    offsets = np.arange(clip_length, dtype=np.int64) - clip_length // 2
    return np.clip(centers[:, None] + offsets[None, :], 0, frame_count - 1)


def read_video(path: Path, expected_frames: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no decodable frames: {path}")
    video = np.stack(frames)
    if expected_frames is not None:
        video = video[: min(len(video), expected_frames)]
    return video


def interpolate_features(sparse: np.ndarray, centers: np.ndarray, frame_count: int) -> np.ndarray:
    if len(centers) == frame_count:
        return sparse.astype(np.float32)
    target = np.arange(frame_count, dtype=np.float32)
    return np.stack(
        [np.interp(target, centers, sparse[:, col]) for col in range(sparse.shape[1])], axis=1
    ).astype(np.float32)


@torch.inference_mode()
def extract_video_features(
    model: VideoMAEModel,
    processor: VideoMAEImageProcessor,
    frames: np.ndarray,
    device: str,
    batch_size: int,
    clip_length: int = 16,
    feature_stride: int = 4,
) -> np.ndarray:
    centers = np.arange(0, len(frames), max(1, feature_stride), dtype=np.int64)
    if centers[-1] != len(frames) - 1:
        centers = np.append(centers, len(frames) - 1)
    indices = centered_indices(len(frames), clip_length, centers)
    outputs = []
    for start in range(0, len(indices), batch_size):
        clips = [[frames[idx] for idx in row] for row in indices[start : start + batch_size]]
        pixel_values = processor(clips, return_tensors="pt")["pixel_values"].to(device)
        hidden = model(pixel_values=pixel_values).last_hidden_state
        outputs.append(hidden.mean(dim=1).float().cpu().numpy())
    sparse = np.concatenate(outputs, axis=0).astype(np.float32)
    return interpolate_features(sparse, centers, len(frames))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-data-dir", required=True)
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model-id", default="MCG-NJU/videomae-base-finetuned-kinetics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--clip-length", type=int, default=16)
    ap.add_argument("--feature-stride", type=int, default=4)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    if args.clip_length != 16:
        raise ValueError("the preregistered VideoMAE model requires 16-frame clips")

    src = Path(args.source_data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = json.loads((src / "video_names.json").read_text(encoding="utf-8"))
    label_archive = np.load(src / "labels.npz", allow_pickle=True)
    labels = [np.asarray(label_archive[key]).reshape(-1).astype(np.int64) for key in label_archive.files]
    processor = VideoMAEImageProcessor.from_pretrained(args.model_id)
    model = VideoMAEModel.from_pretrained(args.model_id).eval().to(args.device)

    signals, used_labels, used_names, failures = [], [], [], []
    selected_names = names[: args.max_videos or None]
    for idx, name in enumerate(selected_names):
        path = args.video_dir / name
        if not path.exists():
            failures.append({"video_name": name, "error": "missing video"})
            continue
        try:
            frames = read_video(path, expected_frames=len(labels[idx]))
            n = min(len(frames), len(labels[idx]))
            features = extract_video_features(
                model, processor, frames[:n], args.device, args.batch_size, args.clip_length, args.feature_stride
            )
            signals.append(features)
            used_labels.append(labels[idx][:n])
            used_names.append(name)
            print(f"{len(used_names)}/{len(selected_names)} {name}: {features.shape}", flush=True)
        except Exception as exc:
            failures.append({"video_name": name, "error": f"{type(exc).__name__}: {exc}"})

    np.savez_compressed(args.output_dir / "signals.npz", *signals)
    np.savez_compressed(args.output_dir / "labels.npz", *used_labels)
    (args.output_dir / "video_names.json").write_text(json.dumps(used_names, indent=2), encoding="utf-8")
    feature_dim = int(signals[0].shape[1]) if signals else 0
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "task": "independent_frozen_videomae_temporal_detector",
                "model_id": args.model_id,
                "n_videos": len(used_names),
                "frames": int(sum(map(len, signals))),
                "feature_dim": feature_dim,
                "feature_names": [f"videomae_{idx}" for idx in range(feature_dim)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "requested": len(selected_names),
        "used": len(used_names),
        "failures": failures,
        "model_id": args.model_id,
        "clip_length": args.clip_length,
        "feature_stride": args.feature_stride,
        "device": args.device,
        "torch_version": torch.__version__,
    }
    (args.output_dir / "extraction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if len(used_names) == len(selected_names) else 2


if __name__ == "__main__":
    raise SystemExit(main())
