#!/usr/bin/env python3
"""
vjepa_predictor.py — 真正的 V-JEPA Predictor 预测
===================================================
提取 encoder 中间层特征(4层) → Predictor → 预测被 mask token → surprise

关键: V-JEPA Predictor 需要 4 层分层蒸馏特征(5632=1408*4)
"""
import sys, os, gc
import numpy as np
import torch
import cv2

sys.path.insert(0, '/home/zzy/vjepa2-main')
from src.masks.utils import apply_masks

DEVICE = torch.device("cuda")
FULL_CKPT = "/home/zzy/vjepa2-main/checkpoints/vjepa2_1_vitg_384.pt"
ENCODER_CKPT = "/home/zzy/vjepa2-main/checkpoints/encoder_only.pt"
IMG_SIZE, PATCH_SIZE = 384, 16
GRID = IMG_SIZE // PATCH_SIZE
SPATIAL_TOKENS = 576

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class VJEPASurprise:
    """真 JEPA: encoder中间层 → Predictor → surprise"""

    def __init__(self):
        self.encoder = None
        self.predictor = None
    def _load_encoder(self):
        if self.encoder is not None:
            return
        print("[Predictor] Loading encoder...")
        from src.hub.backbones import vjepa2_1_vit_giant_384
        enc, _ = vjepa2_1_vit_giant_384(pretrained=False)
        enc = enc.to(DEVICE, dtype=torch.bfloat16).eval()
        sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
        enc.load_state_dict(sd["encoder"], strict=True)
        del sd
        torch.cuda.empty_cache()

        self.encoder = enc
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def _free_encoder(self):
        if self.encoder:
            del self.encoder
            self.encoder = None
            torch.cuda.empty_cache()

    def _load_predictor(self):
        if self.predictor is not None:
            return
        print("[Predictor] Loading predictor (mmap)...")
        from src.hub.backbones import vjepa2_1_vit_giant_384
        _, pred = vjepa2_1_vit_giant_384(pretrained=False)
        ckpt = torch.load(FULL_CKPT, map_location="cpu", weights_only=False, mmap=True)
        psd = ckpt["predictor"]
        psd = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in psd.items()}
        pred.load_state_dict(psd, strict=True)
        del ckpt, psd
        gc.collect()
        pred = pred.to(DEVICE, dtype=torch.bfloat16).eval()
        self.predictor = pred
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def _free_predictor(self):
        if self.predictor:
            del self.predictor
            self.predictor = None
            torch.cuda.empty_cache()

    def preprocess(self, frames):
        T = len(frames)
        ss = int(256 / 224 * IMG_SIZE)
        proc = np.zeros((T, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for t, f in enumerate(frames):
            h, w = f.shape[:2]
            sc = ss / min(h, w)
            nh, nw = int(h * sc), int(w * sc)
            r = cv2.resize(f, (nw, nh))
            sh = (nh - IMG_SIZE) // 2
            sw = (nw - IMG_SIZE) // 2
            proc[t] = r[sh:sh + IMG_SIZE, sw:sw + IMG_SIZE] / 255.0
        tensor = torch.from_numpy(proc).permute(3, 0, 1, 2).float()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(DEVICE).to(torch.bfloat16)

    @torch.no_grad()
    def compute(self, frames, context_t=6, target_t=2, mask_ratio=0.5):
        """
        滑动窗口真 JEPA 预测。

        流程:
          1. Encoder 编码全部帧 → 提取 4 层中间特征
          2. 拼接 4 层特征 → [T_eff, 576, 5632]
          3. 滑动窗口: context → Predictor 预测 target 中被 mask 的 token
          4. L2(预测, 真实) → surprise heatmaps + scores

        返回: heatmaps [T_eff, 24, 24], scores [T_eff]
        """
        T_raw = len(frames)
        if T_raw < context_t + target_t:
            context_t = max(1, T_raw // 3)
            target_t = 1

        # Phase 1: target encoder once, using official hierarchical output.
        # In V-JEPA2.1, encoder(..., training=True) returns the 4-layer
        # distillation target expected by the predictor: [B, N, 5632].
        self._load_encoder()
        self._free_predictor()
        video = self.preprocess(frames)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            target_tokens = self.encoder(video, training=True)

        if target_tokens.ndim != 3 or target_tokens.shape[-1] != 5632:
            raise RuntimeError(f"Expected hierarchical target tokens [B,N,5632], got {target_tokens.shape}")

        T_eff = target_tokens.shape[1] // SPATIAL_TOKENS

        # Phase 2: Predictor sliding windows. Context tokens must be produced
        # by the encoder with context masks, not sliced from full target tokens.
        self._load_predictor()
        heatmaps = np.zeros((T_eff, GRID, GRID))
        scores = np.zeros(T_eff)
        counts = np.zeros(T_eff)
        stride = max(1, target_t)

        for t_start in range(0, T_eff - context_t - target_t + 1, stride):
            ctx_n = context_t * SPATIAL_TOKENS
            tgt_n = target_t * SPATIAL_TOKENS
            base = t_start * SPATIAL_TOKENS
            ctx_mask_full = torch.arange(base, base + ctx_n, device=DEVICE).unsqueeze(0)

            n_pred = max(1, tgt_n // 2)
            tgt_indices = torch.randperm(tgt_n, device=DEVICE)[:n_pred]
            tgt_mask_full = (tgt_indices + base + ctx_n).unsqueeze(0)

            if t_start == 0:
                print(f"  [Debug] ctx_n={ctx_n} tgt_n={tgt_n} n_pred={n_pred} model_n={self.predictor.num_patches}")
                print(f"  [Debug] target_tokens shape={target_tokens.shape}")
                print(f"  [Debug] ctx_mask={ctx_mask_full.shape} tgt_mask={tgt_mask_full.shape}")

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                context_tokens = self.encoder(video, masks=[ctx_mask_full], training=True)
                pred_result = self.predictor(context_tokens, [ctx_mask_full], [tgt_mask_full])

            if isinstance(pred_result, tuple):
                x_pred, _ = pred_result
                pred_tokens = x_pred[0].float().cpu()
            elif isinstance(pred_result, list):
                pred_tokens = pred_result[0][0].float().cpu()
            else:
                pred_tokens = pred_result[0].float().cpu()

            if t_start == 0:
                print(f"  [Debug] context_tokens shape={context_tokens.shape}")
                print(f"  [Debug] pred_tokens shape={pred_tokens.shape}")

            gt_tokens = target_tokens[0, tgt_mask_full[0]].float().cpu()
            gt_indices_cpu = tgt_indices.cpu()

            n = min(pred_tokens.shape[0], gt_tokens.shape[0])
            if n == 0:
                continue

            err = (pred_tokens[:n] - gt_tokens[:n]).norm(dim=-1).numpy()

            for i, token_idx in enumerate(gt_indices_cpu.numpy()):
                if i >= len(err):
                    break
                tubelet_off = int(token_idx) // SPATIAL_TOKENS
                spat_off = int(token_idx) % SPATIAL_TOKENS
                ti = t_start + context_t + tubelet_off
                if 0 <= ti < T_eff:
                    r, c = spat_off // GRID, spat_off % GRID
                    heatmaps[ti, r, c] += err[i]
                    scores[ti] += err[i]
                    counts[ti] += 1

            del context_tokens, pred_result
            torch.cuda.empty_cache()

        self._free_predictor()
        del video, target_tokens
        self._free_encoder()
        for t in range(T_eff):
            if counts[t] > 0:
                heatmaps[t] /= counts[t]
                scores[t] /= counts[t]
        if scores.max() > 0:
            scores /= scores.max()

        print(f"[Predictor] Done: {T_eff} tubelets, scores [{scores.min():.3f}, {scores.max():.3f}]")
        return heatmaps, scores


if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/zzy/vjepa2-main/ComfyUI_00002__overlay.mp4")
    frames = []
    for _ in range(32):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    print(f"Test: {len(frames)} frames")
    pred = VJEPASurprise()
    hm, sc = pred.compute(frames, context_t=6, target_t=2)
    print(f"Heatmaps: {hm.shape}, Scores: [{sc.min():.4f}, {sc.max():.4f}]")
