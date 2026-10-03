#!/usr/bin/env python3
"""depth_consistency_scorer.py — Depth Anything V2 Small 深度一致性评分器

逐帧深度图 → 帧间深度结构相关性 → 异常信号。
使用本地 depth_anything_v2_vits.pth 权重，避免网络下载。
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import logging

logger = logging.getLogger("jepa.depth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "/home/zzy/jepa_data/external_models/depth-anything_Depth-Anything-V2-Small/depth_anything_v2_vits.pth"


class DepthConsistencyScorer:
    def __init__(self, ckpt_path=None, device=None):
        self.device = device or DEVICE
        self.ckpt_path = ckpt_path or CKPT_PATH
        self.model = None
        self._input_size = 518

    def _ensure_loaded(self):
        if self.model is not None:
            return
        logger.info("[DepthDA2] Loading Depth Anything V2 Small...")
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from depth_anything_v2.dpt import DepthAnythingV2
        model_cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        self.model = DepthAnythingV2(**model_cfg)
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
        state = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        logger.info("[DepthDA2] Loaded.")

    @torch.no_grad()
    def score_video(self, frames: np.ndarray, frame_indices: list) -> np.ndarray:
        self._ensure_loaded()
        T = len(frames)
        if T < 2:
            return np.zeros(len(frame_indices))

        depth_maps = []
        for i in range(T):
            img = frames[i]
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (self._input_size, self._input_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            depth = self.model(img_tensor)
            # depth is [1, H, W] — resize to original size
            depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear',
                                 align_corners=False).squeeze().cpu().numpy()
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min)
            depth_maps.append(depth)

        scores = np.zeros(len(frame_indices))
        for i in range(min(T - 1, len(frame_indices) - 1)):
            d1, d2 = depth_maps[i], depth_maps[i+1]
            small_h, small_w = 32, 32
            d1s = cv2.resize(d1, (small_w, small_h))
            d2s = cv2.resize(d2, (small_w, small_h))
            corr = np.corrcoef(d1s.flatten(), d2s.flatten())[0, 1]
            if np.isnan(corr):
                corr = 0
            scores[i] = 1.0 - max(0, float(corr))
        if T > 1 and T <= len(frame_indices):
            scores[T-1] = scores[T-2]
        return scores


if __name__ == "__main__":
    scorer = DepthConsistencyScorer()
    scorer._ensure_loaded()
    print("Depth Consistency Scorer ready.")
