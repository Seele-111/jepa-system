#!/usr/bin/env python3
"""
True I-JEPA masked-prediction surprise scorer.

This is the image-level counterpart to true V-JEPA scoring:
context encoder sees visible patches, predictor predicts masked target patches,
and the anomaly signal is predictor-vs-target feature error.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IJEPA_ROOT = Path("/home/zzy/ijepa-main")
CKPT_FULL = IJEPA_ROOT / "checkpoints" / "IN22K-vit.g.16-600e.pth.tar"
CKPT_SLIM_BF16 = IJEPA_ROOT / "checkpoints" / "ijepa_true_slim_bf16.pt"
DEVICE = torch.device("cuda")
GRID = 14
NUM_PATCHES = GRID * GRID


class IJEPASurprise:
    """Compute true I-JEPA predictor error for a list of frames."""

    def __init__(self, mask_seed: int = 0):
        self.mask_seed = mask_seed
        self.context_encoder = None
        self.target_encoder = None
        self.predictor = None
        self.transform = None
        self.mask_collator = None
        sys.path.insert(0, str(IJEPA_ROOT))

    def _load_models(self):
        if self.context_encoder is not None:
            return

        from ijepa_predict import ContextEncoder, TargetEncoder, transform
        from src.masks.multiblock import MaskCollator
        from src.models.vision_transformer import VisionTransformerPredictor

        print("[TrueIJEPA] Loading context/target encoders + predictor...")
        context_encoder = ContextEncoder().to(dtype=torch.bfloat16)
        target_encoder = TargetEncoder().to(dtype=torch.bfloat16)
        predictor = VisionTransformerPredictor(
            num_patches=NUM_PATCHES,
            embed_dim=1408,
            predictor_embed_dim=384,
            depth=16,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
        ).to(dtype=torch.bfloat16)

        ckpt_path = CKPT_SLIM_BF16 if CKPT_SLIM_BF16.exists() else CKPT_FULL
        print(f"  checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=False)
        self._load_strict_report(context_encoder, ckpt["encoder"], "encoder")
        self._load_strict_report(target_encoder, ckpt["target_encoder"], "target_encoder")
        self._load_strict_report(predictor, ckpt["predictor"], "predictor")
        del ckpt
        gc.collect()

        self.context_encoder = context_encoder.to(DEVICE, dtype=torch.bfloat16).eval()
        self.target_encoder = target_encoder.to(DEVICE, dtype=torch.bfloat16).eval()
        self.predictor = predictor.to(DEVICE, dtype=torch.bfloat16).eval()
        self.transform = transform
        self.mask_collator = MaskCollator(
            input_size=(224, 224),
            patch_size=16,
            pred_mask_scale=(0.15, 0.2),
            enc_mask_scale=(0.85, 1.0),
            aspect_ratio=(0.75, 1.5),
            nenc=1,
            npred=4,
            min_keep=10,
            allow_overlap=False,
        )
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    @staticmethod
    def _load_strict_report(model, state_dict, name: str):
        clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(clean, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            raise RuntimeError(
                f"I-JEPA {name} weights did not load cleanly: "
                f"missing={msg.missing_keys[:8]} unexpected={msg.unexpected_keys[:8]}"
            )
        print(f"  {name}: loaded {sum(v.numel() for v in clean.values())/1e6:.1f}M params")

    def _free_models(self):
        for attr in ("context_encoder", "target_encoder", "predictor"):
            module = getattr(self, attr)
            if module is not None:
                del module
                setattr(self, attr, None)
        torch.cuda.empty_cache()
        gc.collect()

    def _preprocess(self, frames: list[np.ndarray]) -> list[torch.Tensor]:
        tensors = []
        for frame in frames:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(pil))
        return tensors

    @torch.no_grad()
    def compute(self, frames: list[np.ndarray], batch_size: int = 2):
        """
        Return frame-level true I-JEPA surprise.

        Returns:
            heatmaps: [K, 14, 14] normalized masked-patch errors
            scores: [K] normalized mean predictor error per frame
        """
        if not frames:
            return np.zeros((0, GRID, GRID), dtype=np.float32), np.zeros(0, dtype=np.float32)

        self._load_models()
        all_scores: list[float] = []
        all_heatmaps: list[np.ndarray] = []

        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            batch_tensors = self._preprocess(batch_frames)

            # The collator uses torch.randint for block locations; set the seed
            # before each batch so repeated evaluations are reproducible.
            torch.manual_seed(self.mask_seed + start)
            imgs, masks_enc, masks_pred = self.mask_collator(batch_tensors)
            imgs = imgs.to(DEVICE, dtype=torch.bfloat16)
            masks_enc = [m.to(DEVICE) for m in masks_enc]
            masks_pred = [m.to(DEVICE) for m in masks_pred]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                target = self.target_encoder(imgs)
                target = F.layer_norm(target, (target.size(-1),))
                from src.masks.utils import apply_masks

                target_masked = apply_masks(target, masks_pred)
                context = self.context_encoder(imgs, masks_x=masks_enc)
                predicted = self.predictor(context, masks_enc, masks_pred)

            if predicted.shape != target_masked.shape:
                raise RuntimeError(
                    f"I-JEPA predictor shape {tuple(predicted.shape)} != target {tuple(target_masked.shape)}"
                )

            errors = F.smooth_l1_loss(
                predicted.float(),
                target_masked.float(),
                reduction="none",
            ).mean(dim=-1)

            num_masks = len(masks_pred)
            bsz = len(batch_frames)
            errors = errors.view(num_masks, bsz, -1).cpu()
            frame_scores = errors.mean(dim=(0, 2)).numpy()
            all_scores.extend(float(x) for x in frame_scores)

            for b in range(bsz):
                heatmap = np.zeros((GRID, GRID), dtype=np.float32)
                counts = np.zeros((GRID, GRID), dtype=np.float32)
                for m_idx, mask in enumerate(masks_pred):
                    patch_ids = mask[b].detach().cpu().numpy()
                    patch_errors = errors[m_idx, b].numpy()
                    for patch_id, patch_error in zip(patch_ids, patch_errors):
                        r, c = int(patch_id) // GRID, int(patch_id) % GRID
                        heatmap[r, c] += float(patch_error)
                        counts[r, c] += 1.0
                valid = counts > 0
                heatmap[valid] /= counts[valid]
                all_heatmaps.append(heatmap)

            del imgs, target, target_masked, context, predicted, errors
            torch.cuda.empty_cache()

        scores = np.asarray(all_scores, dtype=np.float32)
        heatmaps = np.asarray(all_heatmaps, dtype=np.float32)
        if scores.size and scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        if heatmaps.size and heatmaps.max() > heatmaps.min():
            heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-6)

        self._free_models()
        print(f"[TrueIJEPA] Done: {len(scores)} frames, scores [{scores.min():.3f}, {scores.max():.3f}]")
        return heatmaps, scores


if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/zzy/jepa_data/test_physics_break.mp4")
    frames = []
    for _ in range(8):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    scorer = IJEPASurprise()
    hm, sc = scorer.compute(frames, batch_size=2)
    print("heatmaps", hm.shape, "scores", sc)
