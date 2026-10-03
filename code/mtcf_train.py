#!/usr/bin/env python3
"""
mtcf_train.py — Multi-scale Temporal Conv Fusion 训练脚本

架构选择理由（数据约束下）：
- 50个标注视频 × 100帧 ≈ 5000 样本 → 小样本
- Transformer/Mamba 需要 100K+ 样本来抑制过拟合 → 不适合
- 1D 膨胀卷积 (dilated conv) 天然适合时序信号，参数少，inductive bias 强
- 3层膨胀 [1,2,4] → 感受野 15帧 → 覆盖 ~1秒 → 足够捕获短异常和排斥噪声

输入: [T, 6] 每帧6路信号 (vjepa, ijepa, flow, freq, depth, clip)
输出: [T, 4] softmax → (normal, physics_anomaly, visual_anomaly, other_anomaly)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, glob, argparse


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class MultiScaleFusion(nn.Module):
    """
    多尺度时序卷积融合头
    - 3层膨胀1D卷积 (dilations 1,2,4) → 感受野 = 1+2*(1+2+4) = 15 帧
    - 残差连接防止梯度消失
    - 输出头: 4类 (normal/physics/visual/other)
    """
    def __init__(self, n_signals=6, hidden=64, n_classes=4, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Conv1d(n_signals, hidden, 1)

        # 3层膨胀卷积，每层感受野翻倍
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, padding='same', dilation=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding='same', dilation=2)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size, padding='same', dilation=4)

        self.norm1 = nn.BatchNorm1d(hidden)
        self.norm2 = nn.BatchNorm1d(hidden)
        self.norm3 = nn.BatchNorm1d(hidden)

        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden // 2, n_classes, 1),
        )
        self.n_classes = n_classes

    def forward(self, x):
        """
        x: [B, T, 6] or [T, 6]
        returns: [B, T, n_classes] or [T, n_classes] logits
        """
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)  # [1, T, 6]

        x = x.permute(0, 2, 1)  # [B, 6, T]
        h = self.input_proj(x)  # [B, hidden, T]

        # Block 1 — dilation 1 (1-frame context)
        r1 = F.relu(self.norm1(self.conv1(h)))
        h1 = h + r1  # residual

        # Block 2 — dilation 2 (3-frame context)
        r2 = F.relu(self.norm2(self.conv2(h1)))
        h2 = h1 + r2

        # Block 3 — dilation 4 (7-frame context)
        r3 = F.relu(self.norm3(self.conv3(h2)))
        h3 = h2 + r3

        out = self.head(h3)  # [B, n_classes, T]
        out = out.permute(0, 2, 1)  # [B, T, n_classes]

        if single:
            out = out.squeeze(0)
        return out

    def predict(self, signals, return_probs=True):
        """signals: [T, 6] numpy → [T, n_classes] numpy"""
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(signals.astype(np.float32))
            logits = self.forward(t)
            if return_probs:
                return F.softmax(logits, dim=-1).numpy()
            return logits.numpy()

    def get_anomaly_score(self, signals):
        """signals: [T, 6] → [T] anomaly score (1 - P_normal)"""
        probs = self.predict(signals)  # [T, 4]
        return 1.0 - probs[:, 0]  # P(anomaly) = 1 - P(normal)


# ═══════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════

def temporal_smoothness_loss(logits):
    """惩罚相邻帧预测剧烈变化 — 异常应持续多帧"""
    probs = F.softmax(logits, dim=-1)  # [B, T, C]
    diff = (probs[:, 1:, :] - probs[:, :-1, :]).abs().mean()
    return diff


def segment_aware_loss(logits, labels, categories=None, cat_weights=None,
                       lambda_smooth=0.15, lambda_cat=0.3, lambda_seg=1.0):
    """
    段感知损失: BCE + 分类CE + 时序平滑 + 段连续性奖励

    lambda_seg: 鼓励模型在标注段内输出连贯的高分（而非逐帧抖动）
    """
    B, T, C = logits.shape

    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.25)
    p_anomaly = probs[:, :, 1:].sum(dim=-1).clamp(1e-6, 1 - 1e-6)

    # BCE (per-frame baseline)
    pos_w = torch.where(labels == 1,
                        torch.tensor(0.5, device=labels.device),
                        torch.tensor(1.0, device=labels.device))
    bce = F.binary_cross_entropy(p_anomaly, labels.float(), weight=pos_w, reduction='mean')

    loss = bce

    # Segment continuity: penalize score drops INSIDE labeled segments
    if labels.sum() > 0:
        # Within each contiguous segment of labels==1, penalize score variance
        seg_penalty = 0
        n_seg = 0
        in_seg = False; seg_start = 0
        for b in range(B):
            for t in range(T):
                if labels[b, t] == 1 and not in_seg:
                    seg_start = t; in_seg = True
                elif (labels[b, t] == 0 or t == T-1) and in_seg:
                    seg_end = t if labels[b, t] == 0 else t+1
                    seg_scores = p_anomaly[b, seg_start:seg_end]
                    if len(seg_scores) > 1:
                        # Penalize: (1.0 - mean_score) + variance
                        seg_penalty += (1.0 - seg_scores.mean()) + seg_scores.std()
                        n_seg += 1
                    in_seg = False
        if n_seg > 0:
            loss = loss + lambda_seg * seg_penalty / n_seg

    # Category CE
    if categories is not None and (labels == 1).any():
        cat_mask = (labels == 1)
        cat_logits = logits[cat_mask]
        cat_targets = categories[cat_mask].long()
        if cat_weights is not None:
            cat_weights = cat_weights.to(cat_logits.device)
        ce = F.cross_entropy(cat_logits, cat_targets, weight=cat_weights, reduction='mean')
        loss = loss + lambda_cat * ce

    # Temporal smoothness (global)
    loss = loss + lambda_smooth * temporal_smoothness_loss(logits)

    return loss


# ═══════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════

def load_signals_for_video(video_path, annotations):
    """
    对已标注视频提取6路信号。
    如果已有缓存直接用，否则调用 detect_and_report_v4 提取。

    Returns: signals [T, 6], labels [T], categories [T]
    """
    # Import detection pipeline (only extract signals, not run full pipeline)
    import sys
    sys.path.insert(0, '/home/zzy/jepa_data')
    # We'll call the existing pipeline and capture intermediate signals

    # For now: placeholder that shows the interface
    raise NotImplementedError("Will implement signal extraction from detect_and_report_v4")


def train_model(train_signals, train_labels, train_categories,
                val_signals=None, val_labels=None, val_categories=None,
                epochs=200, lr=1e-3, device='cuda'):
    """
    train_signals: list of [T_i, 6] numpy arrays
    train_labels: list of [T_i] binary arrays (0=normal, 1=anomalous)
    train_categories: list of [T_i] int arrays (0=normal, 1=physics, 2=visual, 3=other)
    """
    model = MultiScaleFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle
        indices = np.random.permutation(len(train_signals))
        for idx in indices:
            s = torch.from_numpy(train_signals[idx].astype(np.float32)).unsqueeze(0).to(device)
            l = torch.from_numpy(train_labels[idx].astype(np.int64)).unsqueeze(0).to(device)
            c = torch.from_numpy(train_categories[idx].astype(np.int64)).unsqueeze(0).to(device)
            mask = torch.from_numpy((train_labels[idx] == 1)).unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(s)
            loss = combined_loss(logits, l, c, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        log.append({'epoch': epoch, 'train_loss': avg_loss})

        # Validation
        if val_signals is not None and len(val_signals) > 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(len(val_signals)):
                    s = torch.from_numpy(val_signals[i].astype(np.float32)).unsqueeze(0).to(device)
                    l = torch.from_numpy(val_labels[i].astype(np.int64)).unsqueeze(0).to(device)
                    c = torch.from_numpy(val_categories[i].astype(np.int64)).unsqueeze(0).to(device)
                    mask = torch.from_numpy((val_labels[i] == 1)).unsqueeze(0).to(device)
                    logits = model(s)
                    val_loss += combined_loss(logits, l, c, mask).item()
            val_loss /= len(val_signals)
            log[-1]['val_loss'] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), '/home/zzy/jepa_data/mtcf_best.pt')

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}  train_loss={avg_loss:.4f}" +
                  (f"  val_loss={val_loss:.4f}" if val_signals else ""))

    # Load best
    if os.path.exists('/home/zzy/jepa_data/mtcf_best.pt'):
        model.load_state_dict(torch.load('/home/zzy/jepa_data/mtcf_best.pt'))

    return model, log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="/mnt/e/jepa-label/videos/annotations")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="/home/zzy/jepa_data/mtcf_best.pt")
    args = parser.parse_args()

    print("MTCF training ready.")
    print(f"Model params: {sum(p.numel() for p in MultiScaleFusion().parameters()):,}")
