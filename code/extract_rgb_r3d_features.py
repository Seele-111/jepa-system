#!/usr/bin/env python3
"""Extract frozen RGB temporal features with a Kinetics R3D-18 encoder.

The output follows the project's ``signals.npz``/``labels.npz`` dataset
contract, so the RGB baseline can be evaluated with exactly the same matcher.
Each frame receives the feature of a centered 16-frame clip; edge clips are
replicated. Missing videos are recorded in ``extraction_report.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


def centered_indices(frame_count: int, clip_length: int = 16, centers: np.ndarray | None = None) -> np.ndarray:
    if frame_count <= 0 or clip_length <= 0:
        raise ValueError("frame_count and clip_length must be positive")
    centers = np.arange(frame_count, dtype=np.int64) if centers is None else np.asarray(centers, dtype=np.int64)
    offsets = np.arange(clip_length, dtype=np.int64) - clip_length // 2
    return np.clip(centers[:, None] + offsets[None, :], 0, frame_count - 1)


def read_video(path: Path, expected_frames: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no decodable frames: {path}")
    video = np.stack(frames)
    if expected_frames is not None and len(video) != expected_frames:
        n = min(len(video), expected_frames)
        video = video[:n]
    return video


@torch.inference_mode()
def extract_video_features(model, transform, frames: np.ndarray, device: str, batch_size: int, clip_length: int, feature_stride: int = 1) -> np.ndarray:
    centers = np.arange(0, len(frames), max(1, feature_stride), dtype=np.int64)
    if centers[-1] != len(frames) - 1:
        centers = np.append(centers, len(frames) - 1)
    indices = centered_indices(len(frames), clip_length, centers)
    outputs: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        clips = []
        for row in indices[start : start + batch_size]:
            clip = torch.from_numpy(frames[row]).permute(0, 3, 1, 2)
            clips.append(transform(clip))
        batch = torch.stack(clips).to(device)
        outputs.append(model(batch).float().cpu().numpy())
    sparse = np.concatenate(outputs, axis=0).astype(np.float32)
    if len(centers) == len(frames):
        return sparse
    target = np.arange(len(frames), dtype=np.float32)
    return np.stack([np.interp(target, centers, sparse[:, col]) for col in range(sparse.shape[1])], axis=1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-data-dir", required=True, help="dataset with labels.npz and video_names.json")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--clip-length", type=int, default=16)
    ap.add_argument("--feature-stride", type=int, default=4)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.source_data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads((src / "video_names.json").read_text(encoding="utf-8"))
    labels_archive = np.load(src / "labels.npz", allow_pickle=True)
    labels = [np.asarray(labels_archive[k]).reshape(-1).astype(np.int64) for k in labels_archive.files]
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(args.device)
    signals: list[np.ndarray] = []
    used_labels: list[np.ndarray] = []
    used_names: list[str] = []
    missing: list[str] = []
    for idx, name in enumerate(names[: args.max_videos or None]):
        path = Path(args.video_dir) / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            frames = read_video(path, expected_frames=len(labels[idx]))
            n = min(len(frames), len(labels[idx]))
            feat = extract_video_features(model, weights.transforms(), frames[:n], args.device, args.batch_size, args.clip_length, args.feature_stride)
            signals.append(feat)
            used_labels.append(labels[idx][:n])
            used_names.append(name)
            print(f"{name}: {feat.shape}", flush=True)
        except Exception as exc:
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    np.savez_compressed(out / "signals.npz", *signals)
    np.savez_compressed(out / "labels.npz", *used_labels)
    (out / "video_names.json").write_text(json.dumps(used_names, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"source": str(src), "n_videos": len(used_names), "frames": int(sum(map(len, signals))), "feature_dim": int(signals[0].shape[1]) if signals else 0, "feature_names": [f"r3d_{i}" for i in range(signals[0].shape[1])] if signals else [], "task": "rgb_temporal_vad"}, indent=2), encoding="utf-8")
    (out / "extraction_report.json").write_text(json.dumps({"requested": len(names[: args.max_videos or None]), "used": len(used_names), "missing": missing, "model": "torchvision_r3d_18_kinetics400", "clip_length": args.clip_length, "feature_stride": args.feature_stride}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
