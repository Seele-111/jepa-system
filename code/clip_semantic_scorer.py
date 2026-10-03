#!/usr/bin/env python3
"""clip_semantic_scorer.py — CLIP ViT-B/32 语义一致性评分器

CLIP逐帧编码 → 相邻帧语义距离 → 异常信号。
使用本地 transformers 格式权重 (external_models/openai_clip-vit-base-patch32)。
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os, logging

logger = logging.getLogger("jepa.clip")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_MODEL = "/home/zzy/jepa_data/external_models/openai_clip-vit-base-patch32"


class CLIPSemanticScorer:
    def __init__(self, model_dir=None, device=None):
        self.device = device or DEVICE
        self.model_dir = model_dir or LOCAL_MODEL
        self.model = None
        self.processor = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        logger.info("[CLIP] Loading ViT-B/32 from local cache...")
        from transformers import CLIPModel, CLIPImageProcessor
        self.model = CLIPModel.from_pretrained(self.model_dir).to(self.device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(self.model_dir)
        logger.info("[CLIP] Loaded.")

    @torch.no_grad()
    def score_video(self, frames: np.ndarray, frame_indices: list) -> np.ndarray:
        """
        frames: [T, H, W, 3] uint8 BGR numpy
        Returns: semantic_shift scores (higher = more anomalous)
        """
        self._ensure_loaded()
        T = len(frames)
        if T < 2:
            return np.zeros(len(frame_indices))

        embeddings = []
        for i in range(T):
            img_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            emb = F.normalize(self.model.get_image_features(**inputs).pooler_output, dim=-1).cpu().numpy()
            embeddings.append(emb)

        embeddings = np.array(embeddings).squeeze(1)  # [T, 512]
        scores = np.zeros(len(frame_indices))
        for i in range(min(T - 1, len(frame_indices) - 1)):
            sim = float(np.dot(embeddings[i], embeddings[i+1]))
            scores[i] = 1.0 - max(0, sim)
        if T > 1 and T <= len(frame_indices):
            scores[T-1] = scores[T-2]
        return scores


if __name__ == "__main__":
    scorer = CLIPSemanticScorer()
    scorer._ensure_loaded()
    print("CLIP Semantic Scorer ready.")
