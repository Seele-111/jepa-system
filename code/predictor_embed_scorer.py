#!/usr/bin/env python3
"""
predictor_embed_scorer.py — 利用 V-JEPA Predictor 的 predictor_embed 层做时序评分
================================================================================
从 16GB 完整 checkpoint 提取 predictor_embed 权重（~2MB），
用它把 5632 维中间层拼接特征投影到 384 维"时序预测空间"，
在这个空间里计算帧间 cosine 距离 = 真正的 JEPA "惊讶值"。

原理: predictor_embed 是 V-JEPA 训练出来专门用于时序预测的投影层。
      在这个空间里，正常帧间过渡的 embedding 相近，异常突变则很远。

用法:
  from predictor_embed_scorer import PredictorEmbedScorer
  scorer = PredictorEmbedScorer()
  scores = scorer.score_temporal(tokens_5632, scales=[1,3,5])
"""
import sys, os, gc
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/home/zzy/vjepa2-main')
from app.vjepa_2_1.models import predictor as pred_module

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL_CKPT = "/home/zzy/vjepa2-main/checkpoints/vjepa2_1_vitg_384.pt"


class PredictorEmbedScorer:
    """用 predictor_embed 权重做时序异常评分（仅加载 2MB 投影层）"""
    # NOTE: This is a proxy scorer. It projects encoder features through the
    # predictor_embed MLP and measures distances; it does not run the JEPA
    # predictor with context/prediction masks.

    def __init__(self):
        # Directly extract predictor_embed weights from checkpoint (skip 15GB blocks)
        ckpt = torch.load(FULL_CKPT, map_location="cpu", weights_only=False, mmap=True)
        psd = ckpt["predictor"]
        # Build just the embed layer (nn.Sequential: Linear(1408*4→1408) + GELU + Linear(1408→384))
        self.embed = nn.Sequential(
            nn.Linear(5632, 1408),
            nn.GELU(),
            nn.Linear(1408, 384),
        )
        # Extract only predictor_embed weights
        embed_keys = [k for k in psd if "predictor_embed" in k]
        state = {}
        for k in embed_keys:
            clean_k = k.replace("module.backbone.predictor_embed.", "").replace("module.", "").replace("backbone.", "")
            # Map: "0.weight", "1.weight", "2.weight", "0.bias", ...
            state[clean_k] = psd[k]
        self.embed.load_state_dict(state)
        del ckpt, psd
        gc.collect()
        self.embed = self.embed.to(DEVICE, dtype=torch.bfloat16).eval()
        print(
            f"[PredictorEmbed] Loaded embed ({sum(p.numel() for p in self.embed.parameters()):,} params). "
            "Proxy mode: not running JEPA predictor masks."
        )

    @torch.no_grad()
    def score_temporal(self, tokens_5632, scales=[1, 3, 5]):
        """
        tokens_5632: [T, 5632] 每个 tubelet 的 4层拼接特征
        在 predictor_embed 投影空间里计算多尺度帧间距离。

        返回:
          scores: [T] 每帧的异常分数（越高越异常）
        """
        T = tokens_5632.shape[0]
        if T < 2:
            return np.zeros(T)

        # Project all tokens to predictor embedding space
        tokens_gpu = tokens_5632.to(DEVICE, dtype=torch.bfloat16)
        emb = self.embed(tokens_gpu)  # [T, 384]
        emb_norm = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()

        # Multi-scale cosine distance
        all_scores = np.zeros((T, len(scales)))
        for si, scale in enumerate(scales):
            for t in range(T):
                if t + scale < T:
                    sim = np.dot(emb_norm[t], emb_norm[t + scale])
                    all_scores[t, si] = 1.0 - sim
                else:
                    all_scores[t, si] = all_scores[max(0, t - scale), si] if t >= scale else 0

        # Weighted combination
        weights = np.array([1.0, 0.7 / 3, 0.4 / 5])[:len(scales)]
        scores = (all_scores * weights).sum(axis=1) / weights.sum()

        # Normalize
        if scores.max() > 0:
            scores /= scores.max()

        return scores


if __name__ == "__main__":
    # Quick test
    scorer = PredictorEmbedScorer()
    dummy = torch.randn(32, 5632)
    s = scorer.score_temporal(dummy, scales=[1, 3, 5])
    print(f"Test: {s.shape}, range=[{s.min():.4f}, {s.max():.4f}]")
