#!/usr/bin/env python3
"""Extract frozen features with the official VideoMAE ViT-B K400 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

from extract_videomae_features import centered_indices, interpolate_features, read_video


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(repo: Path, checkpoint: Path, device: str):
    sys.path.insert(0, str(repo))
    from modeling_finetune import vit_base_patch16_224

    model = vit_base_patch16_224(
        pretrained=False,
        num_classes=400,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        init_scale=0.001,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if isinstance(state, dict) and "module" in state and isinstance(state["module"], dict):
        state = state["module"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model.head = torch.nn.Identity()
    return model.eval().to(device)


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    output = []
    mean = torch.tensor(IMAGENET_MEAN)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD)[None, :, None, None]
    for start in range(0, len(frames), 32):
        images = torch.from_numpy(frames[start : start + 32].copy()).permute(0, 3, 1, 2).float() / 255.0
        images = TF.resize(images, 224, antialias=True)
        images = TF.center_crop(images, [224, 224])
        output.append((images - mean) / std)
    return torch.cat(output)


@torch.inference_mode()
def extract_video_features(model, frames: np.ndarray, device: str, batch_size: int, feature_stride: int) -> np.ndarray:
    centers = np.arange(0, len(frames), max(1, feature_stride), dtype=np.int64)
    if centers[-1] != len(frames) - 1:
        centers = np.append(centers, len(frames) - 1)
    indices = centered_indices(len(frames), 16, centers)
    processed = preprocess_frames(frames)
    outputs = []
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.from_numpy(indices[start : start + batch_size])
        batch = processed[batch_indices].permute(0, 2, 1, 3, 4).contiguous().to(device)
        outputs.append(model.forward_features(batch).float().cpu().numpy())
    sparse = np.concatenate(outputs).astype(np.float32)
    return interpolate_features(sparse, centers, len(frames))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-data-dir", required=True, type=Path)
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--official-repo", default=Path("/home/zzy/VideoMAE"), type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--feature-stride", type=int, default=4)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = json.loads((args.source_data_dir / "video_names.json").read_text(encoding="utf-8"))
    archive = np.load(args.source_data_dir / "labels.npz", allow_pickle=True)
    labels = [np.asarray(archive[key]).reshape(-1).astype(np.int64) for key in archive.files]
    model = load_model(args.official_repo, args.checkpoint, args.device)
    selected = names[: args.max_videos or None]
    signals, used_labels, used_names, failures = [], [], [], []
    for idx, name in enumerate(selected):
        try:
            frames = read_video(args.video_dir / name, expected_frames=len(labels[idx]))
            n = min(len(frames), len(labels[idx]))
            features = extract_video_features(model, frames[:n], args.device, args.batch_size, args.feature_stride)
            signals.append(features)
            used_labels.append(labels[idx][:n])
            used_names.append(name)
            print(f"{len(used_names)}/{len(selected)} {name}: {features.shape}", flush=True)
        except Exception as exc:
            failures.append({"video_name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {name}: {failures[-1]['error']}", flush=True)
    np.savez_compressed(args.output_dir / "signals.npz", *signals)
    np.savez_compressed(args.output_dir / "labels.npz", *used_labels)
    (args.output_dir / "video_names.json").write_text(json.dumps(used_names, indent=2), encoding="utf-8")
    feature_dim = int(signals[0].shape[1]) if signals else 0
    (args.output_dir / "summary.json").write_text(json.dumps({"task": "official_videomae_k400_frozen_features", "n_videos": len(used_names), "frames": int(sum(map(len, signals))), "feature_dim": feature_dim, "feature_names": [f"videomae_{i}" for i in range(feature_dim)]}, indent=2), encoding="utf-8")
    report = {"requested": len(selected), "used": len(used_names), "failures": failures, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "official_repo": str(args.official_repo), "clip_length": 16, "feature_stride": args.feature_stride, "preprocess": {"short_side": 224, "center_crop": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}}
    (args.output_dir / "extraction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if len(used_names) == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
