#!/usr/bin/env python3
"""
detect_and_report_v4.py — JEPA 双路并行侦察兵
===============================================
V-JEPA 和 I-JEPA 独立并行检测，各自输出自己的"意见"，最终融合。

新架构:
  Phase 1 (并行提取):
    A. V-JEPA: 提取原始空间 token [T_eff, 576, 1408]
    B. I-JEPA: 提取全帧 patch 特征 [K, 196, 1408]（不再等 V-JEPA 标记）

  Phase 2 (独立打分):
    V-JEPA 时序预测: 对每个空间位置做帧间插值预测 → 预测误差 = "惊讶程度"
    I-JEPA 空间异常: 每个 patch vs 正常分布 → z-score = "空间异常程度"

  Phase 3 (融合):
    两路各自的 top-K 帧取并集 → 标记来源 (vjepa/ipepa/both)

  Phase 4 (输出):
    生成 V-JEPA 预测热力图 + I-JEPA 空间标注图 + 双栏报告

用法:
  python detect_and_report_v4.py --video <path> --output <dir>
"""

import os, sys, json, glob, argparse, math, logging
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from datetime import timedelta

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('jepa')

DEVICE = torch.device("cuda")
DATA_ROOT   = "/home/zzy/jepa_data"
CACHE_DIR   = os.path.join(DATA_ROOT, "features_cache_v3")  # 新缓存目录（避免和旧格式冲突）
VJEPA_CKPT  = "/home/zzy/vjepa2-main/checkpoints/encoder_only.pt"
IJEPA_CKPT  = "/home/zzy/ijepa-main/checkpoints/ijepa_encoder_only.pt"

# V-JEPA: 384px input, patch_size=16 → 24×24 grid = 576 spatial tokens
VJEPA_GRID  = 24
VJEPA_TOKENS = VJEPA_GRID * VJEPA_GRID  # 576
# I-JEPA: 224px input, patch_size=16 → 14×14 grid = 196 patches
IJEPA_GRID  = 14
IJEPA_PATCHES = IJEPA_GRID * IJEPA_GRID  # 196

# Global normal distribution (for video-external z-score)
GLOBAL_STATS_PATH = os.path.join(DATA_ROOT, "global_normal_stats.pt")
GLOBAL_STATS_PATH_V2 = os.path.join(DATA_ROOT, "global_normal_stats_v2.pt")
_global_stats = None

def load_global_stats():
    global _global_stats
    if _global_stats is not None:
        return _global_stats
    # 优先用 v2（分层特征基线），回退 v1
    for path in [GLOBAL_STATS_PATH_V2, GLOBAL_STATS_PATH]:
        if os.path.exists(path):
            _global_stats = torch.load(path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded global stats from {os.path.basename(path)}")
            return _global_stats
    return None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ═══════════════════════════════════════════════════════════════════════════
#  Part 0: 正常分布（复用已有，只取 V-JEPA 时序 + I-JEPA 空间统计）
# ═══════════════════════════════════════════════════════════════════════════

_normal_dist = None

def load_normal_dist():
    global _normal_dist
    if _normal_dist is not None:
        return _normal_dist

    dist_file = os.path.join(DATA_ROOT, "normal_distribution.pt")
    if os.path.exists(dist_file):
        print("[Load] Cached normal distribution")
        _normal_dist = torch.load(dist_file, map_location="cpu", weights_only=False)
        return _normal_dist

    print("[Build] Building normal distribution from cache_v2 features...")
    normal_files = glob.glob(os.path.join(DATA_ROOT, "features_cache_v2", "*_label0.pt"))

    # temporal stats (V-JEPA per-frame distances)
    all_temporal = []
    for fpath in normal_files:
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            vfeat = data["vjepa_perframe"]
            if vfeat.shape[0] < 3: continue
            vn = torch.nn.functional.normalize(vfeat, dim=-1)
            cos = (vn[:-1] * vn[1:]).sum(dim=-1)
            all_temporal.extend((1 - cos).tolist())
        except Exception as e:
            logger.debug(f"Skip {os.path.basename(fpath)}: {e}")
            continue

    all_temporal = np.array(all_temporal)
    if len(all_temporal) == 0:
        print("[Error] No normal data!")
        return None

    temporal_mu = float(all_temporal.mean())
    temporal_sigma = float(all_temporal.std())

    # spatial stats (I-JEPA per-position patch features)
    pos_feats = {p: [] for p in range(IJEPA_PATCHES)}
    loaded = 0
    for fpath in normal_files:
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            pfeat = data.get("ijepa_patch")
            if pfeat is None: continue
            for f in range(pfeat.shape[0]):
                for p in range(IJEPA_PATCHES):
                    pos_feats[p].append(pfeat[f, p].float().numpy())
            loaded += 1
        except Exception as e:
            logger.debug(f"Skip {os.path.basename(fpath)}: {e}")
            continue

    pos_stats = {}
    for p in range(IJEPA_PATCHES):
        feats = np.array(pos_feats[p])
        if len(feats) < 10: continue
        pos_stats[p] = {
            "mean": torch.from_numpy(feats.mean(axis=0)),
            "std":  torch.from_numpy(feats.std(axis=0)) + 1e-6,
        }

    print(f"  Temporal: μ={temporal_mu:.6f} σ={temporal_sigma:.6f}")
    print(f"  Spatial:  {len(pos_stats)}/{IJEPA_PATCHES} positions")

    _normal_dist = {
        "temporal_mu": temporal_mu,
        "temporal_sigma": temporal_sigma,
        "position_stats": pos_stats,
    }
    torch.save(_normal_dist, dist_file)
    return _normal_dist


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1A: V-JEPA — 提取原始空间 token
#  输入: 视频帧 list
#  输出: [T_eff, 576, 1408]  +  tubelet→frame 映射
# ═══════════════════════════════════════════════════════════════════════════

def extract_vjepa_tokens(video_path, use_cache=True, max_frames=64):
    """
    提取 V-JEPA 原始空间 token（不池化）。
    由于 V-JEPA 用 tubelet_size=2 处理视频，我们按 tubelet 分组。

    返回: {vjepa_tokens: [T_eff, 576, 1408], tubelet_to_frames: list, fps, total_frames}
    """
    safe_name = os.path.splitext(os.path.basename(video_path))[0][:80]
    # 用路径 hash 区分同名不同目录的视频
    import hashlib
    path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"{safe_name}_{path_hash}_vjepa_tokens.pt")

    if use_cache and os.path.exists(cache_file):
        print(f"[V-JEPA Cache] Loaded tokens from {os.path.basename(cache_file)}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    print(f"[V-JEPA] Extracting raw tokens: {safe_name[:40]}...")
    sys.path.insert(0, "/home/zzy/vjepa2-main")
    from src.hub.backbones import vjepa2_1_vit_giant_384

    os.makedirs(CACHE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 16.0
    if total_frames <= 0: cap.release(); return None

    # 均匀采样 max_frames 帧（覆盖全视频）
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, f = cap.read()
        if ret: frames.append((idx, f))
    cap.release()

    if len(frames) < 2: return None

    frame_indices = [idx for idx, _ in frames]
    raw_frames = [f for _, f in frames]

    # 加载 V-JEPA encoder
    encoder, _ = vjepa2_1_vit_giant_384(pretrained=False)
    encoder = encoder.to(DEVICE, dtype=torch.bfloat16).eval()
    sd = torch.load(VJEPA_CKPT, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(sd["encoder"], strict=True); del sd
    torch.cuda.empty_cache()

    def prep_v(frames, size=384):
        tensors = []
        for f in frames:
            f = cv2.resize(f, (int(256/224*size), int(256/224*size)))
            h, w = f.shape[:2]; y, x = (h-size)//2, (w-size)//2; f = f[y:y+size, x:x+size]
            t = torch.from_numpy(f).permute(2,0,1).float()/255.0
            t = (t - IMAGENET_MEAN.view(3,1,1)) / IMAGENET_STD.view(3,1,1)
            tensors.append(t)
        v = torch.stack(tensors, dim=0).unsqueeze(0)  # [1, BS, 3, 384, 384]
        return v.permute(0, 2, 1, 3, 4)  # [1, 3, BS, 384, 384]

    all_tokens = []
    all_hier_tokens = []

    BS = 4
    with torch.no_grad():
        for start in range(0, len(raw_frames), BS):
            end = min(start + BS, len(raw_frames))
            batch = prep_v(raw_frames[start:end]).to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                tokens = encoder(batch)
                hier_tokens = encoder(batch, training=True)
            all_tokens.append(tokens.squeeze(0).cpu().float())
            all_hier_tokens.append(hier_tokens.squeeze(0).cpu().float())
            del batch; torch.cuda.empty_cache()


    all_tokens = torch.cat(all_tokens, dim=0)
    T_eff = all_tokens.shape[0] // VJEPA_TOKENS
    vjepa_tokens = all_tokens.reshape(T_eff, VJEPA_TOKENS, 1408)

    # 拼接蒸馏特征：4层×1408 → [T_eff, 5632]
    # Official V-JEPA2.1 hierarchical distillation features: [T_eff, 5632]
    hierarchical_features = None
    if all_hier_tokens:
        hier_all = torch.cat(all_hier_tokens, dim=0)
        hierarchical_features = hier_all.reshape(T_eff, VJEPA_TOKENS, -1).mean(dim=1)

    tubelet_to_frames = []
    for t in range(T_eff):
        fi = t * 2
        f_start = frame_indices[fi] if fi < len(frame_indices) else frame_indices[-1]
        f_end = frame_indices[min(fi+1, len(frame_indices)-1)] if fi+1 < len(frame_indices) else f_start
        tubelet_to_frames.append((f_start, f_end))

    del encoder; torch.cuda.empty_cache()

    result = {
        "vjepa_tokens": vjepa_tokens,
        "hierarchical_features": hierarchical_features,  # JEPA 分层特征
        "tubelet_to_frames": tubelet_to_frames,
        "fps": fps,
        "total_frames": total_frames,
    }
    torch.save(result, cache_file)
    print(f"[V-JEPA] Tokens: {vjepa_tokens.shape}, saved to cache")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1B: I-JEPA — 全帧独立扫描
#  输入: 视频
#  输出: [K, 196, 1408] + frame_indices
# ═══════════════════════════════════════════════════════════════════════════

def extract_ijepa_all(video_path, use_cache=True, max_keyframes=32):
    """
    提取 I-JEPA patch 特征，对所有关键帧独立扫描（不再等待 V-JEPA 标记）。
    返回: {ijepa_patch: [K, 196, 1408], ijepa_frame_indices: list, fps, total_frames}
    """
    safe_name = os.path.splitext(os.path.basename(video_path))[0][:80]
    import hashlib
    path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    cache_file = os.path.join(CACHE_DIR, f"{safe_name}_{path_hash}_ijepa_all.pt")

    if use_cache and os.path.exists(cache_file):
        print(f"[I-JEPA Cache] Loaded from {os.path.basename(cache_file)}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    print(f"[I-JEPA] Extracting patches (all keyframes): {safe_name[:40]}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 16.0
    if total_frames <= 0: cap.release(); return None

    # 均匀采样关键帧（覆盖全视频）
    indices = np.linspace(0, total_frames - 1, min(max_keyframes, total_frames), dtype=int).tolist()

    frames = []
    valid_indices = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, f = cap.read()
        if ret:
            frames.append(f)
            valid_indices.append(idx)
    cap.release()

    if not frames:
        return None

    # ---- I-JEPA model ----
    sys.path.insert(0, "/home/zzy/ijepa-main")
    from src.models.vision_transformer import Block, PatchEmbed
    import torchvision.transforms as T

    ijepa_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    i_model = nn.Module()
    i_model.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=1408)
    i_model.pos_embed = nn.Parameter(torch.zeros(1, IJEPA_PATCHES, 1408), requires_grad=False)
    dpr = [0.4 * i / 39 for i in range(40)]
    i_model.blocks = nn.ModuleList([
        Block(dim=1408, num_heads=16, mlp_ratio=6144/1408, qkv_bias=True, drop_path=dpr[i])
        for i in range(40)
    ])
    i_model.norm = nn.LayerNorm(1408)
    sd2 = torch.load(IJEPA_CKPT, map_location="cpu", weights_only=False)
    clean = {k.replace("module.", "").replace("encoder.", ""): v for k, v in sd2.items()}
    i_model.load_state_dict(clean, strict=False); del sd2
    i_model.to(DEVICE, dtype=torch.bfloat16).eval(); torch.cuda.empty_cache()

    ijepa_patch = []
    with torch.no_grad():
        for frame in frames:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inp = ijepa_transform(pil).unsqueeze(0).to(DEVICE).to(torch.bfloat16)
            x = i_model.patch_embed(inp) + i_model.pos_embed
            for blk in i_model.blocks: x = blk(x)
            x = i_model.norm(x)  # [1, 196, 1408]
            ijepa_patch.append(x.cpu().half())
            del inp, x; torch.cuda.empty_cache()

    ijepa_patch = torch.stack(ijepa_patch, dim=0)  # [K, 1, 196, 1408]
    # squeeze the batch dim if present
    if ijepa_patch.dim() == 4 and ijepa_patch.shape[1] == 1:
        ijepa_patch = ijepa_patch.squeeze(1)  # → [K, 196, 1408]
    del i_model; torch.cuda.empty_cache()

    result = {
        "ijepa_patch": ijepa_patch,
        "ijepa_frame_indices": valid_indices,
        "fps": fps,
        "total_frames": total_frames,
    }
    torch.save(result, cache_file)
    print(f"[I-JEPA] Patches: {ijepa_patch.shape}, saved to cache")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2A: V-JEPA 多信号打分 (physics detection)
#  基于 dual_detector.py: 时间一致性 + 空间方差 + 能量变化
# ═══════════════════════════════════════════════════════════════════════════

def score_vjepa_prediction(vjepa_result):
    """
    保留原有插值预测热力图生成（用于可视化）。
    返回: tubelet_scores, heatmaps
    """
    tokens = vjepa_result["vjepa_tokens"]  # [T_eff, 576, 1408]
    T_eff = tokens.shape[0]

    if T_eff < 3:
        return np.zeros(T_eff), np.zeros((T_eff, VJEPA_GRID, VJEPA_GRID))

    tokens_n = torch.nn.functional.normalize(tokens.float(), dim=-1)
    tubelet_scores = np.zeros(T_eff)
    heatmaps = np.full((T_eff, VJEPA_GRID, VJEPA_GRID), np.nan)

    for t in range(1, T_eff - 1):
        pred = (tokens_n[t - 1] + tokens_n[t + 1]) / 2.0
        actual = tokens_n[t]
        cos_sim = (pred * actual).sum(dim=-1)
        errors = (1 - cos_sim).numpy()
        tubelet_scores[t] = float(errors.mean())
        heatmaps[t] = errors.reshape(VJEPA_GRID, VJEPA_GRID)

    if T_eff >= 2:
        errors_0 = (1 - (tokens_n[0] * tokens_n[1]).sum(dim=-1)).numpy()
        tubelet_scores[0] = float(errors_0.mean())
        heatmaps[0] = errors_0.reshape(VJEPA_GRID, VJEPA_GRID)
        errors_last = (1 - (tokens_n[-1] * tokens_n[-2]).sum(dim=-1)).numpy()
        tubelet_scores[-1] = float(errors_last.mean())
        heatmaps[-1] = errors_last.reshape(VJEPA_GRID, VJEPA_GRID)

    return tubelet_scores, heatmaps


def score_vjepa_multi(vjepa_tokens, tubelet_to_frames, total_frames,
                       predictor_embed=None, hierarchical_features=None):
    """
    V-JEPA 多路信号打分.
    hierarchical_features: [T, 5632] 真正的 4 层分层特征（替代假池化）
    """
    tokens = vjepa_tokens.float()
    T, S, D = tokens.shape
    if T < 2:
        return np.zeros(T)

    # 1. Temporal scoring
    scales = [1, 3, 5]
    if predictor_embed is not None and hierarchical_features is not None:
        # 真正的 JEPA: 用编码器 4 层分层特征喂给 predictor_embed
        hf = hierarchical_features.float()
        temporal_err = predictor_embed.score_temporal(hf[:T], scales=scales)
    elif predictor_embed is not None:
        # 回退: 假池化（旧方式）
        p1 = tokens.mean(dim=1)
        p2 = tokens.max(dim=1).values
        p3 = tokens.std(dim=1)
        p4 = (tokens[:, :288, :].mean(dim=1) - tokens[:, 288:, :].mean(dim=1))
        pooled_4x = torch.cat([p1, p2, p3, p4], dim=-1)
        temporal_err = predictor_embed.score_temporal(pooled_4x, scales=scales)
    else:
        temporal_multi = np.zeros((T, len(scales)))
        for si, scale in enumerate(scales):
            for t in range(T):
                if t + scale < T:
                    sim = F.cosine_similarity(
                        tokens[t].mean(0, keepdim=True),
                        tokens[t + scale].mean(0, keepdim=True)).item()
                    temporal_multi[t, si] = 1.0 - sim
                else:
                    temporal_multi[t, si] = temporal_multi[max(0, t-scale), si] if t >= scale else temporal_multi[min(T-1, t+1), si]
        weights_arr = np.array([1.0, 0.7/3, 0.4/5])[:len(scales)]
        temporal_err = (temporal_multi * weights_arr).sum(axis=1) / weights_arr.sum()

    # 2. Spatial: 每帧 token 方差（内部一致性）
    spatial_var = tokens.var(dim=1).mean(dim=1).numpy()  # [T]
    spatial_var = spatial_var / (spatial_var.mean() + 1e-6)

    # 3. Energy: L2 norm 突变
    energy = tokens.norm(dim=2).mean(dim=1).numpy()  # [T]
    energy_diff = np.abs(np.diff(energy, prepend=energy[0]))
    energy_diff = energy_diff / (energy.mean() + 1e-6)

    # 自适应权重: 信号方差越高 → 权重越高（信息量大）
    t_var = np.var(temporal_err) if np.var(temporal_err) > 0 else 1e-6
    s_var = np.var(spatial_var) if np.var(spatial_var) > 0 else 1e-6
    e_var = np.var(energy_diff) if np.var(energy_diff) > 0 else 1e-6
    total_var = t_var + s_var + e_var
    w_t = 0.3 + 0.4 * (t_var / total_var)  # 30% base + variance share
    w_s = 0.3 + 0.4 * (s_var / total_var)
    w_e = 0.2 + 0.4 * (e_var / total_var)
    w_sum = w_t + w_s + w_e
    w_t, w_s, w_e = w_t / w_sum, w_s / w_sum, w_e / w_sum

    physics_scores = w_t * temporal_err + w_s * spatial_var + w_e * energy_diff

    print(f"  [V-JEPA Multi] temporal_range=[{temporal_err.min():.4f},{temporal_err.max():.4f}] "
          f"spatial_range=[{spatial_var.min():.2f},{spatial_var.max():.2f}] "
          f"energy_range=[{energy_diff.min():.2f},{energy_diff.max():.2f}] "
          f"weights: t={w_t:.2f} s={w_s:.2f} e={w_e:.2f}")

    return physics_scores


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2B: I-JEPA 多信号打分 (corruption detection)
#  基于 dual_detector.py: 特征方差 + 空间不一致 + 能量 z-score
# ═══════════════════════════════════════════════════════════════════════════

def compute_vjepa_attention(vjepa_heatmaps, tubelet_to_frames,
                             ijepa_frame_indices):
    """
    将 V-JEPA 热力图映射为 I-JEPA patch 的软注意力权重 (0.3~1.0)。
    不再做硬门控，改为软引导：V-JEPA 热点区域权重高，其余区域保留基础权重。

    返回: [K, 196] 注意力权重，每个 patch ∈ [0.3, 1.0]
    """
    T_eff = len(tubelet_to_frames)
    H_eff = len(vjepa_heatmaps) if vjepa_heatmaps is not None else 0
    K = len(ijepa_frame_indices)
    attention = np.ones((K, IJEPA_PATCHES))  # 默认全关注

    if vjepa_heatmaps is None or T_eff == 0 or H_eff == 0:
        return attention * 0.7  # 无热力图时给中等权重

    for k, frame_idx in enumerate(ijepa_frame_indices):
        tubelet_idx = -1
        for t, (s, e) in enumerate(tubelet_to_frames):
            if s <= frame_idx <= e:
                tubelet_idx = t
                break
        if tubelet_idx < 0 or tubelet_idx >= T_eff or tubelet_idx >= H_eff:
            attention[k] = attention[k] * 0.7
            continue

        hm = np.nan_to_num(vjepa_heatmaps[tubelet_idx], nan=0.0)
        hmin, hmax = hm.min(), hm.max()
        if hmax <= hmin:
            attention[k] = attention[k] * 0.7
            continue
        hm_norm = (hm - hmin) / (hmax - hmin)

        # 下采样 24×24 → 14×14: max+avg 混合保留峰值
        hm_14 = np.zeros((IJEPA_GRID, IJEPA_GRID))
        for ri in range(IJEPA_GRID):
            for ci in range(IJEPA_GRID):
                rs = int(ri * VJEPA_GRID / IJEPA_GRID)
                re = int((ri + 1) * VJEPA_GRID / IJEPA_GRID + 0.5)
                cs = int(ci * VJEPA_GRID / IJEPA_GRID)
                ce = int((ci + 1) * VJEPA_GRID / IJEPA_GRID + 0.5)
                re, ce = min(re, VJEPA_GRID), min(ce, VJEPA_GRID)
                if re > rs and ce > cs:
                    block = hm_norm[rs:re, cs:ce]
                    hm_14[ri, ci] = 0.5 * block.mean() + 0.5 * block.max()  # max+avg hybrid

        weights = hm_14.flatten()

        # 软权重: 基础 0.3 + 热力引导 0.7
        # 热力图均匀 (cv < 0.25) → 全部给中等权重
        hm_cv = np.nanstd(weights) / (np.nanmean(weights) + 1e-6)
        if hm_cv < 0.25:
            attention[k] = attention[k] * 0.5
        else:
            soft_weights = 0.3 + 0.7 * weights / (weights.max() + 1e-6)
            attention[k] = soft_weights

    return attention


def score_ijepa_multi(ijepa_patch, vjepa_attention=None):
    """
    I-JEPA 多路信号打分 (corruption detection).
    基于 dual_detector.py: 视频内部相对比较。

    三路信号:
      1. feature_variance:      每帧 196 patch 特征方差（归一化）
      2. spatial_inconsistency: 帧内所有 patch 间平均 cosine dissimilarity
      3. energy_zscore:         每帧特征 norm 偏离全视频均值的 z-score

    返回:
      corruption_scores: [K] 综合 corruption 分数
    """
    pfeat = ijepa_patch.float()  # [K, 196, 1408]
    K, P, D = pfeat.shape

    if K < 2:
        return np.zeros(K)

    # 1. Feature variance
    fvar = pfeat.var(dim=1).mean(dim=1).numpy()  # [K]
    fvar = fvar / (fvar.mean() + 1e-6)

    # 2. Spatial inconsistency: mean off-diagonal cosine dissimilarity
    spatial_inc = np.zeros(K)
    for k in range(K):
        f = F.normalize(pfeat[k], dim=-1)  # [P, D]
        sim_matrix = f @ f.T  # [P, P]
        if P > 1:
            mean_sim = (sim_matrix.sum().item() - P) / (P * (P - 1))
        else:
            mean_sim = 1.0
        spatial_inc[k] = 1.0 - mean_sim

    # 3. Energy z-score
    energy = pfeat.norm(dim=2).mean(dim=1).numpy()  # [K]
    zscore = np.abs(energy - energy.mean()) / (energy.std() + 1e-6)

    # 自适应权重: 信号方差越高 → 权重越高
    f_var_v = np.var(fvar) if np.var(fvar) > 0 else 1e-6
    s_var_v = np.var(spatial_inc) if np.var(spatial_inc) > 0 else 1e-6
    z_var_v = np.var(zscore) if np.var(zscore) > 0 else 1e-6
    i_total_var = f_var_v + s_var_v + z_var_v
    w_f = 0.3 + 0.3 * (f_var_v / i_total_var)
    w_si = 0.3 + 0.3 * (s_var_v / i_total_var)
    w_z = 0.2 + 0.3 * (z_var_v / i_total_var)
    i_w_sum = w_f + w_si + w_z
    w_f, w_si, w_z = w_f / i_w_sum, w_si / i_w_sum, w_z / i_w_sum

    corruption_scores = w_f * fvar + w_si * spatial_inc + w_z * zscore

    # 4. Top-K patch anomaly: 只看最异常的 20% patch (捕捉小区域伪影)
    f_norm = F.normalize(pfeat, dim=-1)  # [K, P, D]
    mean_feat = f_norm.mean(dim=1, keepdim=True)  # [K, 1, D]
    patch_anomaly = 1.0 - (f_norm * mean_feat).sum(dim=-1)  # [K, 196]
    k = max(1, P // 5)  # top 20%
    topk_vals, _ = torch.topk(patch_anomaly, k=k, dim=1)
    topk_score = topk_vals.mean(dim=1).numpy()  # [K]
    topk_score = topk_score / (topk_score.max() + 1e-6)
    # Blend: 70% original + 30% top-K
    corruption_scores = 0.7 * corruption_scores + 0.3 * topk_score

    # Apply V-JEPA attention as soft weight
    if vjepa_attention is not None:
        attn_mean = vjepa_attention.mean(axis=1)  # [K] avg attention per frame
        corruption_scores = corruption_scores * attn_mean

    print(f"  [I-JEPA Multi] fvar_range=[{fvar.min():.2f},{fvar.max():.2f}] "
          f"spinc_range=[{spatial_inc.min():.4f},{spatial_inc.max():.4f}] "
          f"zscore_range=[{zscore.min():.1f},{zscore.max():.1f}] "
          f"weights: f={w_f:.2f} si={w_si:.2f} z={w_z:.2f}")

    return corruption_scores


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3: 连续分数融合 + Peak Detection
#  不再二值判定，输出连续异常分数，用峰值检测找精确异常段
# ═══════════════════════════════════════════════════════════════════════════

def expand_sparse_frame_scores(frame_indices, scores, total_frames):
    """Expand sparse keyframe scores to a dense frame timeline by linear interpolation."""
    dense = np.zeros(int(total_frames), dtype=np.float64)
    valid_mask = np.zeros(int(total_frames), dtype=bool)
    if total_frames <= 0:
        return dense, valid_mask

    pairs = [
        (int(fid), float(score))
        for fid, score in zip(frame_indices, scores)
        if 0 <= int(fid) < total_frames and np.isfinite(float(score))
    ]
    if not pairs:
        return dense, valid_mask

    pairs = sorted(pairs, key=lambda item: item[0])
    xs = np.asarray([item[0] for item in pairs], dtype=np.float64)
    ys = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if len(xs) == 1:
        dense[:] = ys[0]
    else:
        dense[:] = np.interp(np.arange(total_frames, dtype=np.float64), xs, ys)
    valid_mask[:] = True
    return dense, valid_mask


def composite_scoring(vjepa_scores, ijepa_scores,
                       tubelet_to_frames, ijepa_frame_indices,
                       total_frames, flow_scores=None, freq_scores=None,
                       depth_scores=None, clip_scores=None,
                       vjepa_normalization="global",
                       ijepa_normalization="global"):
    # 升采样 V-JEPA 到帧级
    vjepa_frame = np.zeros(total_frames)
    for t, (s, e) in enumerate(tubelet_to_frames):
        if t < len(vjepa_scores):
            s = max(0, min(s, total_frames - 1))
            e = max(0, min(e, total_frames - 1))
            vjepa_frame[s:e+1] = vjepa_scores[t]
    ijepa_frame, c_mask = expand_sparse_frame_scores(ijepa_frame_indices, ijepa_scores, total_frames)
    p_mask = vjepa_frame > 0
    p_z = np.zeros(total_frames); c_z = np.zeros(total_frames)

    # 全局基线归一化
    global_stats = load_global_stats() if (
        vjepa_normalization == "global" or ijepa_normalization == "global"
    ) else None
    if global_stats is not None:
        if p_mask.any() and vjepa_normalization == "global":
            p_z[p_mask] = (vjepa_frame[p_mask] - global_stats["v_physics_mu"]) / (global_stats["v_physics_sigma"] + 1e-6)
        if c_mask.any() and ijepa_normalization == "global":
            c_z[c_mask] = (ijepa_frame[c_mask] - global_stats["i_composite_mu"]) / (global_stats["i_composite_sigma"] + 1e-6)
    if p_mask.any() and not (global_stats is not None and vjepa_normalization == "global"):
        vals = vjepa_frame[p_mask]
        if vjepa_normalization == "per_video":
            p_z[p_mask] = (vals - vals.mean()) / (vals.std() + 1e-6)
        else:
            p_z[p_mask] = vals
    if c_mask.any() and not (global_stats is not None and ijepa_normalization == "global"):
        vals = ijepa_frame[c_mask]
        if ijepa_normalization == "per_video":
            c_z[c_mask] = (vals - vals.mean()) / (vals.std() + 1e-6)
        else:
            c_z[c_mask] = vals
    p_sig = 1.0 / (1.0 + np.exp(-p_z))
    c_sig = 1.0 / (1.0 + np.exp(-c_z))

    # #1 共识度融合: 两路都高 → 信任; 只有一路高 → 降权
    consensus = np.minimum(p_sig, c_sig) / (np.maximum(p_sig, c_sig) + 1e-6)
    consensus_weight = 0.5 + 0.5 * consensus  # 提高到 0.5 底线
    composite = np.maximum(p_sig, c_sig) * consensus_weight

    def _signal_quality(signal):
        """信号质量评估: 高分集中度越高 → 信号越干净 → 权重越高"""
        top = signal[signal > np.percentile(signal, 75)]
        if len(top) < 2: return 0.1
        cv = top.std() / (top.mean() + 1e-6)
        return min(1.0, 1.0 / (cv + 0.5))

    # #8 运动补偿: 高光流区域 V-JEPA 置信度降低（摄像机平移 ≠ 异常）
    if flow_scores is not None and len(flow_scores) == total_frames:
        flow_norm = flow_scores / (flow_scores.max() + 1e-6)
        motion_mask = 1.0 - 0.5 * flow_norm  # 最高运动区 V-JEPA 权重降 50%
        p_sig_masked = p_sig * motion_mask
        consensus_m = np.minimum(p_sig_masked, c_sig) / (np.maximum(p_sig_masked, c_sig) + 1e-6)
        cw_m = 0.3 + 0.7 * consensus_m
        composite = np.maximum(p_sig_masked, c_sig) * cw_m
        print(f"  [Motion] mask applied (range [{motion_mask.min():.2f}, {motion_mask.max():.2f}])")

    if flow_scores is not None and len(flow_scores) == total_frames:
        fz = (flow_scores - flow_scores.mean()) / (flow_scores.std() + 1e-6)
        fs = 1.0 / (1.0 + np.exp(-fz))
        flow_weight = 0.3 * _signal_quality(fs)  # #2 自适应权重
        composite = (1 - flow_weight) * composite + flow_weight * fs
        print(f"  [Flow] integrated: w={flow_weight:.3f}")
    if freq_scores is not None and len(freq_scores) == total_frames:
        frz = (freq_scores - freq_scores.mean()) / (freq_scores.std() + 1e-6)
        frs = 1.0 / (1.0 + np.exp(-frz))
        freq_weight = 0.15 * _signal_quality(frs)  # #2 自适应权重
        composite = (1 - freq_weight) * composite + freq_weight * frs
        print(f"  [Freq] integrated: w={freq_weight:.3f}")
    if depth_scores is not None and len(depth_scores) == total_frames:
        dz = (depth_scores - depth_scores.mean()) / (depth_scores.std() + 1e-6)
        ds = 1.0 / (1.0 + np.exp(-dz))
        depth_weight = 0.1 * _signal_quality(ds)
        composite = (1 - depth_weight) * composite + depth_weight * ds
        print(f"  [Depth] integrated: w={depth_weight:.3f}")
    if clip_scores is not None and len(clip_scores) == total_frames:
        cz = (clip_scores - clip_scores.mean()) / (clip_scores.std() + 1e-6)
        cs = 1.0 / (1.0 + np.exp(-cz))
        clip_weight = 0.1 * _signal_quality(cs)
        composite = (1 - clip_weight) * composite + clip_weight * cs
        print(f"  [CLIP] integrated: w={clip_weight:.3f}")
    return composite, p_sig, c_sig


def find_anomaly_segments(composite, threshold=0.4, min_gap=3, min_length=2,
                           fps=16.0):
    """
    增强版: 梯度过滤 + 双阈值（长段宽松，短段严格）。
    """

    # #10 梯度增强: 过滤孤立噪声尖峰
    gradient = np.abs(np.gradient(composite))
    grad_threshold = np.percentile(gradient, 80)
    above_grad = gradient > grad_threshold
    # 持续高分段即使梯度小也保留
    sustained = np.convolve((composite > threshold*0.8).astype(float),
                            np.ones(5)/5, mode='same') > 0.5
    valid_mask = (composite > threshold) & (above_grad | sustained)

    if not valid_mask.any():
        return []

    segments = []
    start = -1
    for i in range(len(valid_mask)):
        if valid_mask[i] and start == -1:
            start = i
        elif (not valid_mask[i] or i == len(valid_mask)-1) and start != -1:
            end = i if valid_mask[i] else i-1
            seg_score = composite[start:end+1]
            segments.append({
                "start_frame": start, "end_frame": end,
                "max_score": float(seg_score.max()),
                "mean_score": float(seg_score.mean()),
            })
            start = -1

    merged = []
    for seg in segments:
        if not merged: merged.append(seg)
        elif seg["start_frame"] - merged[-1]["end_frame"] <= min_gap:
            merged[-1]["end_frame"] = seg["end_frame"]
            merged[-1]["max_score"] = max(merged[-1]["max_score"], seg["max_score"])
            merged[-1]["mean_score"] = float(composite[merged[-1]["start_frame"]:merged[-1]["end_frame"]+1].mean())
        else: merged.append(seg)

    # #12 双阈值: 长段(min_length+)用基础阈值, 短段需更高分数
    filtered = []
    for seg in merged:
        length = seg["end_frame"] - seg["start_frame"] + 1
        if length >= min_length:
            filtered.append(seg)
        elif seg["max_score"] > threshold * 1.15:
            # 短段需要峰值 > 115% 阈值才保留
            filtered.append(seg)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Part 4: 可视化 — V-JEPA 预测热力图 + I-JEPA 空间标注
# ═══════════════════════════════════════════════════════════════════════════

def generate_vjepa_heatmap_frame(frame, heatmap_24x24):
    """
    在帧上叠加 V-JEPA 预测误差热力图（24×24 grid）。
    heatmap_24x24: [24, 24] 预测误差值（cosine distance）
    返回: 标注后的帧
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    error_grid = heatmap_24x24.copy()

    # 归一化到 [0, 1]
    error_min = np.nanmin(error_grid)
    error_max = np.nanmax(error_grid)
    if error_max > error_min:
        error_norm = (error_grid - error_min) / (error_max - error_min)
    else:
        error_norm = np.zeros_like(error_grid)

    cell_h = h / VJEPA_GRID
    cell_w = w / VJEPA_GRID

    for r in range(VJEPA_GRID):
        for c in range(VJEPA_GRID):
            if np.isnan(error_norm[r, c]):
                continue
            val = error_norm[r, c]
            y1, x1 = int(r * cell_h), int(c * cell_w)
            y2, x2 = int((r+1) * cell_h), int((c+1) * cell_w)

            # 颜色: 蓝(低) → 绿 → 黄 → 红(高)
            if val < 0.33:
                color = np.array([int(255 * (1 - val/0.33)), 255, int(255 * val/0.33)], dtype=np.float32)
            elif val < 0.66:
                v = (val - 0.33) / 0.33
                color = np.array([0, int(255 * (1-v)), 255], dtype=np.float32)
            else:
                v = (val - 0.66) / 0.34
                color = np.array([0, 0, int(255 * (1-v))], dtype=np.float32)

            # alpha blend
            alpha = 0.5
            roi = overlay[y1:y2, x1:x2]
            blended = (alpha * color + (1-alpha) * roi.astype(np.float32)).astype(np.uint8)
            overlay[y1:y2, x1:x2] = blended

    return overlay


def generate_ijepa_annotation_frame(frame, patch_scores_k, threshold=3.0):
    """
    在帧上标注 I-JEPA 空间异常 patch（14×14 grid）。
    patch_scores_k: [196] 该帧每个 patch 的帧内相对 z-score
    threshold: 帧内相对 z 阈值 (默认 3.0，即偏离帧中位数 3 个 MAD)
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    ph, pw = h / IJEPA_GRID, w / IJEPA_GRID

    for p in range(IJEPA_PATCHES):
        s = patch_scores_k[p]
        if s < threshold:
            continue
        r, c = p // IJEPA_GRID, p % IJEPA_GRID
        y1, x1 = int(r * ph), int(c * pw)
        y2, x2 = int((r+1) * ph), int((c+1) * pw)

        if s >= 5.0:
            box_color = (0, 0, 255)       # red: extreme outlier
        elif s >= 4.0:
            box_color = (0, 165, 255)     # orange: clear anomaly
        else:
            box_color = (0, 220, 255)     # yellow: mild anomaly
        thickness = 2 if s >= 5.0 else 1
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, thickness)

    return overlay


def _describe_patch_regions(patch_ids):
    """将 patch ID 映射到区域描述（复用旧版逻辑）"""
    from collections import Counter
    PATCH_REGIONS = {
        "top-left":     lambda r,c: r < 5 and c < 5,
        "top-center":   lambda r,c: r < 5 and 4 <= c < 10,
        "top-right":    lambda r,c: r < 5 and c >= 10,
        "mid-left":     lambda r,c: 4 <= r < 9 and c < 5,
        "center":       lambda r,c: 4 <= r < 9 and 4 <= c < 10,
        "mid-right":    lambda r,c: 4 <= r < 9 and c >= 10,
        "bottom-left":  lambda r,c: r >= 9 and c < 5,
        "bottom-center":lambda r,c: r >= 9 and 4 <= c < 10,
        "bottom-right": lambda r,c: r >= 9 and c >= 10,
    }
    regions = []
    for pid in patch_ids:
        r, c = pid // IJEPA_GRID, pid % IJEPA_GRID
        for name, check in PATCH_REGIONS.items():
            if check(r, c):
                regions.append(name)
                break
    return [r for r, _ in Counter(regions).most_common()]


# ═══════════════════════════════════════════════════════════════════════════
#  Part 5: 标注帧生成（双模型）
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_annotations(video_path, vjepa_result, ijepa_result,
                              vjepa_heatmaps,
                              ijepa_patch_relative,  # unused (kept for compat)
                              ijepa_frame_indices, fused_results,
                              output_dir):
    """
    生成标注帧：
      vjepa_heatmaps/  — V-JEPA 预测热力图
      ijepa_boxes/     — (deprecated, 生成空列表)
      combined/        — 合并标注 (heatmap + 来源标签)
    """
    os.makedirs(output_dir, exist_ok=True)
    vjepa_out = os.path.join(output_dir, "vjepa_heatmaps")
    ijepa_out = os.path.join(output_dir, "ijepa_boxes")
    combined_out = os.path.join(output_dir, "combined")
    for d in [vjepa_out, ijepa_out, combined_out]:
        os.makedirs(d, exist_ok=True)

    fps = vjepa_result.get("fps", 16.0)
    total_frames = vjepa_result.get("total_frames", 1)
    tubelet_to_frames = vjepa_result["tubelet_to_frames"]
    T_eff = len(tubelet_to_frames)

    # 收集需要生成标注的帧：按段长度比例均匀分布
    rep_frames = set()
    if fused_results:
        # 计算每段长度，按比例分配 24 帧预算（每段最少 3 帧）
        segment_lengths = [fr["end_frame"] - fr["start_frame"] + 1 for fr in fused_results]
        total_len = sum(segment_lengths)
        budget = 24
        for i, fr in enumerate(fused_results):
            s, e = fr["start_frame"], fr["end_frame"]
            seg_len = e - s + 1
            # 按长度比例分配，最少 3 帧
            alloc = max(3, int(budget * seg_len / total_len)) if total_len > 0 else 3
            alloc = min(alloc, seg_len)  # 不能超过段长度
            if alloc <= 3:
                rep_frames.add(s)
                rep_frames.add(e)
                if seg_len >= 3:
                    rep_frames.add((s + e) // 2)
            else:
                # 均匀采样 alloc 帧
                indices = np.linspace(s, e, alloc, dtype=int)
                for idx in indices:
                    rep_frames.add(int(idx))
    rep_frames = sorted(rep_frames)[:24]  # 硬上限

    if not rep_frames:
        return [], [], []

    cap = cv2.VideoCapture(video_path)

    vjepa_paths, ijepa_paths, combined_paths = [], [], []

    for frame_idx in rep_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue

        ts = str(timedelta(seconds=frame_idx / fps))

        # --- V-JEPA 热力图 ---
        vjepa_frame = frame.copy()
        # 找到对应的 tubelet
        tubelet_idx = None
        for t, (s, e) in enumerate(tubelet_to_frames):
            if s <= frame_idx <= e:
                tubelet_idx = t
                break
        if tubelet_idx is not None and tubelet_idx < len(vjepa_heatmaps):
            vjepa_frame = generate_vjepa_heatmap_frame(frame, vjepa_heatmaps[tubelet_idx])
            cv2.putText(vjepa_frame, f"V-JEPA Prediction Error | Frame {frame_idx} @ {ts}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(vjepa_frame, f"V-JEPA (no heatmap) | Frame {frame_idx} @ {ts}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        vjepa_path = os.path.join(vjepa_out, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(vjepa_path, vjepa_frame)
        vjepa_paths.append(vjepa_path)

        # --- I-JEPA 空间异常框 ---
        ijepa_frame = frame.copy()
        ijepa_k = None
        if ijepa_frame_indices:
            for k, fid in enumerate(ijepa_frame_indices):
                if fid == frame_idx:
                    ijepa_k = k
                    break
        if ijepa_k is not None and ijepa_patch_relative is not None:
            ijepa_frame = generate_ijepa_annotation_frame(frame, ijepa_patch_relative[ijepa_k])
        cv2.putText(ijepa_frame, f"I-JEPA Spatial Anomaly | Frame {frame_idx} @ {ts}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ijepa_path = os.path.join(ijepa_out, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(ijepa_path, ijepa_frame)
        ijepa_paths.append(ijepa_path)

        # --- Combined: 并排显示 ---
        # 左右并排
        combo = np.hstack([vjepa_frame, ijepa_frame])
        # 顶部加来源标签
        h_c, w_c = combo.shape[:2]
        mid_x = w_c // 2

        # 确定来源
        source_label = "both"
        for fr in fused_results:
            if fr["start_frame"] <= frame_idx <= fr["end_frame"]:
                source_label = fr["source"]
                break

        colors = {"vjepa": (255, 165, 0), "ijepa": (0, 255, 200), "both": (0, 255, 0)}
        color = colors.get(source_label, (255, 255, 255))
        cv2.putText(combo, f"DETECTED BY: {source_label.upper()}",
                    (10, h_c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        combo_path = os.path.join(combined_out, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(combo_path, combo)
        combined_paths.append(combo_path)

    cap.release()
    return vjepa_paths, ijepa_paths, combined_paths


def generate_segment_all_frames(video_path, vjepa_heatmaps, tubelet_to_frames,
                                  fused_results, fps, output_dir):
    """
    为每个异常段生成 ALL 帧：
      segments/seg_N/original/  — 原始帧（用于重新生成）
      segments/seg_N/           — 热力图叠加帧（用于查看）
    返回: list of {segment_index, frames: [paths], original_frames: [paths]}
    """
    seg_dir = os.path.join(output_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    segments_info = []

    for i, fr in enumerate(fused_results):
        s, e = fr["start_frame"], fr["end_frame"]
        out_dir = os.path.join(seg_dir, f"seg_{i}")
        orig_dir = os.path.join(out_dir, "original")
        os.makedirs(orig_dir, exist_ok=True)
        frame_paths = []
        orig_paths = []

        for frame_idx in range(s, e + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            ts_str = str(timedelta(seconds=frame_idx / max(fps, 1)))
            fname = f"frame_{frame_idx:05d}.png"

            # 1. Save original frame (no overlay) — for ComfyUI re-generation
            orig_path = os.path.join(orig_dir, fname)
            cv2.imwrite(orig_path, frame)
            orig_paths.append(os.path.join("original", fname))

            # 2. V-JEPA heatmap overlay for inspection
            tubelet_idx = None
            for t, (ts_t, te) in enumerate(tubelet_to_frames):
                if ts_t <= frame_idx <= te:
                    tubelet_idx = t; break

            annotated = frame.copy()
            if tubelet_idx is not None and tubelet_idx < len(vjepa_heatmaps):
                annotated = generate_vjepa_heatmap_frame(frame, vjepa_heatmaps[tubelet_idx])

            cv2.putText(annotated, f"Frame {frame_idx} @ {ts_str} | composite",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, annotated)
            frame_paths.append(fname)

        segments_info.append({
            "segment_index": i,
            "start_frame": s,
            "end_frame": e,
            "frame_count": len(frame_paths),
            "frames": frame_paths,
            "original_frames": orig_paths,
            "dir": f"seg_{i}",
        })

    cap.release()
    return segments_info


# ═══════════════════════════════════════════════════════════════════════════
#  Part 6: 生成双栏 LLM Prompt
# ═══════════════════════════════════════════════════════════════════════════

def generate_llm_prompt(video_name, vjepa_anomalies, ijepa_anomalies, fused_results):
    """生成双栏格式的 LLM prompt"""

    vjepa_text = ""
    for a in vjepa_anomalies:
        vjepa_text += (
            f"  - Tubelet {a['tubelet_start']}-{a['tubelet_end']}: "
            f"帧 {a['start_frame']}-{a['end_frame']} "
            f"({a['timestamp_start']}~{a['timestamp_end']}) "
            f"[score={a['score']:.4f}, z={a['z_score']:.1f}, {a['confidence']}]\n"
        )

    ijepa_text = ""
    for a in ijepa_anomalies[:10]:
        ijepa_text += (
            f"  - 帧 {a['real_frame']} @ {a['timestamp']}: "
            f"区域={a['dominant_region']}, "
            f"top patch={a['top_patches'][0] if a['top_patches'] else '?'} "
            f"[z={a['score']}, {a['confidence']}]\n"
        )

    fused_text = ""
    for fr in fused_results:
        fused_text += (
            f"  - 帧 {fr['start_frame']}-{fr['end_frame']} "
            f"({fr['timestamp_start']}~{fr['timestamp_end']}) "
            f"[source: {fr['source']}]\n"
        )

    prompt = f"""## AI 生成视频异常检测报告（双路并行 JEPA）

**视频**: {video_name}

---
### V-JEPA 时序预测 (Temporal Prediction Surprise)
V-JEPA 通过相邻帧插值预测来检测"意外"——高预测误差 = V-JEPA 认为这里不该长这样。
{vjepa_text if vjepa_text else "  V-JEPA 未检测到异常。视频时序模式正常。"}

---
### I-JEPA 空间检测 (Spatial Patch Anomaly)
I-JEPA 检测每个空间位置的 patch 特征是否偏离正常分布。
{ijepa_text if ijepa_text else "  I-JEPA 未检测到空间异常。"}

---
### 融合结果（并集）
{fused_text}

---
### 任务
请同时参考附带的 V-JEPA 预测热力图（蓝色=低误差，红色=V-JEPA 被吓到了）和 I-JEPA 空间异常框（红/橙/黄=异常程度递减），分析每个异常片段：
1. **V-JEPA 发现的异常**：时序预测误差高的片段，具体发生了什么变化？
2. **I-JEPA 发现的异常**：空间上异常的 patch 区域，画面有什么问题？
3. **两者的共识和分歧**：哪些帧被两个模型同时标记 vs 只有一个模型标记？这说明了什么？
4. **修复建议**：给出 ComfyUI 提示词修改建议。

请以 JSON 格式输出。
"""
    return prompt.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Part 7: 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="JEPA 双路并行侦察兵 v4 — 多信号连续评分")
    parser.add_argument("--video",        required=True,  help="输入视频路径")
    parser.add_argument("--output",       default="detection_report_v4", help="输出目录")
    parser.add_argument("--threshold",    type=float, default=0.4,
                        help="composite 分数阈值 (0~1, 默认 0.4)")
    parser.add_argument("--min-gap",      type=int, default=3,
                        help="异常段最小合并间隔帧数 (默认 3)")
    parser.add_argument("--min-length",   type=int, default=2,
                        help="异常段最小帧数 (默认 2)")
    parser.add_argument("--no-cache",     action="store_true", help="强制重新提取特征")
    parser.add_argument("--max-frames",   type=int, default=128, help="V-JEPA 采样帧数")
    parser.add_argument("--max-keyframes", type=int, default=64, help="I-JEPA 关键帧数")
    parser.add_argument("--use-referee",  action="store_true",
                        help="使用训练好的 Referee 模型 (referee_v2_best.pt) 替代手写规则")
    parser.add_argument("--use-predictor-embed", action="store_true",
                        help="使用 V-JEPA Predictor 的 predictor_embed 层做时序评分（需 16GB checkpoint）")
    parser.add_argument("--use-true-vjepa", action="store_true",
                        help="Use true V-JEPA masked predictor error instead of encoder-distance proxy")
    parser.add_argument("--use-true-ijepa", action="store_true",
                        help="Use true I-JEPA masked predictor error instead of encoder patch-statistics proxy")
    parser.add_argument("--use-optical-flow", action="store_true",
                        help="加入光流分析作为第三路物理一致性检测信号")
    parser.add_argument("--use-frequency", action="store_true",
                        help="加入 DCT 频域分析检测 AI 伪影")
    parser.add_argument("--use-depth", action="store_true",
                        help="加入 Depth Anything V2 深度一致性检测")
    parser.add_argument("--use-clip", action="store_true",
                        help="加入 CLIP ViT-B/32 语义一致性检测")
    parser.add_argument("--multi-scale", action="store_true",
                        help="多尺度融合：短clip+长clip各跑一次，段取交集减少假阳性")
    args = parser.parse_args()
    vjepa_usage = "true_vjepa_predictor" if args.use_true_vjepa else "encoder_proxy"
    if args.use_referee:
        vjepa_usage = "referee_on_encoder_tokens"
    elif args.use_predictor_embed:
        vjepa_usage = "predictor_embed_projection_proxy"
    jepa_usage = {
        "vjepa": vjepa_usage,
        "ijepa": "true_ijepa_predictor" if args.use_true_ijepa else "encoder_proxy",
    }

    video_name = os.path.basename(args.video)
    os.makedirs(args.output, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    #  Phase 1: 并行特征提取
    # ════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("[Phase 1] 并行特征提取")
    print("=" * 60)

    use_cache = not args.no_cache

    # V-JEPA token 提取
    vjepa_result = extract_vjepa_tokens(args.video, use_cache=use_cache, max_frames=args.max_frames)
    if vjepa_result is None:
        print("[Error] V-JEPA 特征提取失败")
        return

    # I-JEPA full-frame scan. For true I-JEPA we only need the same keyframe
    # indices; loading the old encoder proxy first creates an unnecessary
    # memory spike before the predictor path.
    if args.use_true_ijepa:
        cap_i = cv2.VideoCapture(args.video)
        i_total = int(cap_i.get(cv2.CAP_PROP_FRAME_COUNT))
        i_fps = cap_i.get(cv2.CAP_PROP_FPS)
        cap_i.release()
        if i_fps <= 0:
            i_fps = 16.0
        i_indices = np.linspace(0, max(0, i_total - 1), min(args.max_keyframes, max(1, i_total)), dtype=int).tolist()
        ijepa_result = {
            "ijepa_patch": None,
            "ijepa_frame_indices": i_indices,
            "fps": i_fps,
            "total_frames": i_total,
        }
    else:
        ijepa_result = extract_ijepa_all(args.video, use_cache=use_cache, max_keyframes=args.max_keyframes)
        if ijepa_result is None:
            print("[Warning] I-JEPA 特征提取失败，仅使用 V-JEPA")
            ijepa_result = {"ijepa_patch": None, "ijepa_frame_indices": [], "fps": 16.0, "total_frames": 0}

    fps = vjepa_result.get("fps", 16.0)
    total_frames = vjepa_result.get("total_frames", 1)

    # ════════════════════════════════════════════════════════════════
    #  Phase 2: 多信号打分
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[Phase 2] 多信号打分")
    print("=" * 60)

    # V-JEPA 热力图（可视化用）+ 打分
    print("[V-JEPA] Computing prediction heatmaps...")
    _, vjepa_heatmaps = score_vjepa_prediction(vjepa_result)

    if args.use_referee:
        # === Referee 模型打分 (训练好的双流分类器) ===
        print("[Referee] Using trained DualStreamReferee model...")
        sys.path.insert(0, DATA_ROOT)
        from referee_inference import RefereeScorer
        scorer = RefereeScorer()

        # Referee expects raw tokens in full shape
        referee_scores = scorer.score_video(
            vjepa_result["vjepa_tokens"],       # [T, 576, 1408]
            ijepa_result["ijepa_patch"]          # [K, 196, 1408]
        )
        print(f"  Referee scores: range=[{referee_scores.min():.3f}, {referee_scores.max():.3f}]")

        # Map to frame-level
        tubelet_to_frames = vjepa_result["tubelet_to_frames"]
        referee_frame = np.zeros(total_frames)
        for t, (s, e) in enumerate(tubelet_to_frames):
            if t < len(referee_scores):
                s = max(0, min(s, total_frames - 1))
                e = max(0, min(e, total_frames - 1))
                referee_frame[s:e+1] = referee_scores[t]

        # Use referee scores directly as composite
        composite = referee_frame
        p_sig = referee_frame  # placeholder
        c_sig = np.zeros_like(referee_frame)

        print(f"  Referee composite: range=[{composite.min():.3f}, {composite.max():.3f}]")
    else:
        # === 手写规则打分 ===
        print("[V-JEPA] Multi-signal physics scoring...")

        if args.use_true_vjepa:
            print("[TrueVJEPA] Computing masked predictor error...")
            from vjepa_predictor import VJEPASurprise
            frames = []
            cap_true = cv2.VideoCapture(args.video)
            total_for_sample = int(cap_true.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_n = min(args.max_frames, max(1, total_for_sample))
            frame_indices_true = np.linspace(0, max(0, total_for_sample - 1), sample_n, dtype=int).tolist()
            for idx in frame_indices_true:
                cap_true.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap_true.read()
                if ret:
                    frames.append(frame)
            cap_true.release()
            true_scorer = VJEPASurprise()
            true_heatmaps, physics_scores = true_scorer.compute(frames, context_t=6, target_t=2)
            vjepa_heatmaps = true_heatmaps
            print(f"  True V-JEPA predictor: {len(physics_scores)} tubelets, "
                  f"range=[{physics_scores.min():.4f},{physics_scores.max():.4f}]")
        else:
            # Optional: use predictor_embed for temporal scoring
            predictor_embed = None
            if args.use_predictor_embed:
                print("[PredictorEmbed] Loading predictor_embed weights...")
                sys.path.insert(0, DATA_ROOT)
                from predictor_embed_scorer import PredictorEmbedScorer
                predictor_embed = PredictorEmbedScorer()

            physics_scores = score_vjepa_multi(
                vjepa_result["vjepa_tokens"],
                vjepa_result["tubelet_to_frames"],
                total_frames,
                predictor_embed=predictor_embed,
                hierarchical_features=vjepa_result.get("hierarchical_features")
            )
            print(f"  V-JEPA physics: {len(physics_scores)} tubelets, "
                  f"range=[{physics_scores.min():.4f},{physics_scores.max():.4f}]")

        # I-JEPA 多信号 corruption 打分
        ijepa_scores = np.array([])
        if args.use_true_ijepa or ijepa_result["ijepa_patch"] is not None:
            print("[I-JEPA] Computing V-JEPA attention (soft guidance)...")
            vjepa_attn = compute_vjepa_attention(
                vjepa_heatmaps,
                vjepa_result["tubelet_to_frames"],
                ijepa_result["ijepa_frame_indices"]
            )
            print(f"  Attention: mean={vjepa_attn.mean():.3f} max={vjepa_attn.max():.3f}")

            if args.use_true_ijepa:
                print("[TrueIJEPA] Computing masked predictor error...")
                from ijepa_predictor import IJEPASurprise
                true_ijepa_frames = []
                cap_ijepa = cv2.VideoCapture(args.video)
                for idx in ijepa_result.get("ijepa_frame_indices", []):
                    cap_ijepa.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap_ijepa.read()
                    if ret:
                        true_ijepa_frames.append(frame)
                cap_ijepa.release()
                true_ijepa = IJEPASurprise()
                _, ijepa_scores = true_ijepa.compute(true_ijepa_frames, batch_size=2)
            else:
                print("[I-JEPA] Multi-signal corruption scoring...")
                ijepa_scores = score_ijepa_multi(
                    ijepa_result["ijepa_patch"],
                    vjepa_attention=vjepa_attn
                )
            print(f"  I-JEPA corruption: {len(ijepa_scores)} frames, "
                  f"range=[{ijepa_scores.min():.4f},{ijepa_scores.max():.4f}]")
        else:
            print("[I-JEPA] Skipped (no data)")

    # ════════════════════════════════════════════════════════════════
    #  Phase 3: 连续分数融合 + Peak Detection
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[Phase 3] 分数融合 + Peak Detection")
    print("=" * 60)

    # Optional optical flow scoring
    flow_scores = None
    if args.use_optical_flow:
        print("[OpticalFlow] Computing dense optical flow...")
        from optical_flow_scorer import OpticalFlowScorer
        of_scorer = OpticalFlowScorer(scale=0.5)
        raw_flow_scores, flow_details = of_scorer.score_video(
            args.video, start_frame=0, end_frame=total_frames - 1, step=1
        )
        # Interpolate to match total_frames
        flow_scores = np.interp(
            np.arange(total_frames),
            np.linspace(0, total_frames - 1, len(raw_flow_scores)),
            raw_flow_scores
        )
        print(f"  Flow scores: range=[{flow_scores.min():.3f}, {flow_scores.max():.3f}]")

    # Optional frequency domain scoring
    freq_scores = None
    if args.use_frequency:
        print("[Frequency] Computing DCT spectral features...")
        from frequency_scorer import FrequencyScorer
        fq_scorer = FrequencyScorer(scale=0.5, block_size=8)
        raw_freq_scores, freq_details = fq_scorer.score_video(
            args.video, start_frame=0, end_frame=total_frames - 1, step=2
        )
        freq_scores = np.interp(
            np.arange(total_frames),
            np.linspace(0, total_frames - 1, len(raw_freq_scores)),
            raw_freq_scores
        )
        print(f"  Freq scores: range=[{freq_scores.min():.3f}, {freq_scores.max():.3f}]")

    # Depth consistency
    depth_scores = None
    keyframes = None  # pre-declare for CLIP reuse
    if args.use_depth:
        print("[DepthDA2] Computing depth consistency...")
        from depth_consistency_scorer import DepthConsistencyScorer
        dp_scorer = DepthConsistencyScorer()
        # Use same keyframes as I-JEPA (every step=2 frames)
        keyframes = []
        cap = cv2.VideoCapture(args.video)
        frame_idx = 0; kf = 0
        while cap.isOpened() and kf < args.max_keyframes:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % 2 == 0 and kf < args.max_keyframes:
                keyframes.append(frame)
                kf += 1
            frame_idx += 1
        cap.release()
        keyframes = np.array(keyframes)
        raw_depth = dp_scorer.score_video(keyframes, list(range(len(keyframes))))
        depth_scores = np.interp(
            np.arange(total_frames),
            np.linspace(0, total_frames - 1, len(raw_depth)),
            raw_depth
        )
        print(f"  Depth scores: range=[{depth_scores.min():.3f}, {depth_scores.max():.3f}]")

    # CLIP semantic consistency
    clip_scores = None
    if args.use_clip:
        print("[CLIP] Computing semantic consistency...")
        from clip_semantic_scorer import CLIPSemanticScorer
        cp_scorer = CLIPSemanticScorer()
        if keyframes is None:
            cap = cv2.VideoCapture(args.video)
            keyframes = []
            frame_idx = 0; kf = 0
            while cap.isOpened() and kf < args.max_keyframes:
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % 2 == 0 and kf < args.max_keyframes:
                    keyframes.append(frame)
                    kf += 1
                frame_idx += 1
            cap.release()
            keyframes = np.array(keyframes)
        raw_clip = cp_scorer.score_video(keyframes, list(range(len(keyframes))))
        clip_scores = np.interp(
            np.arange(total_frames),
            np.linspace(0, total_frames - 1, len(raw_clip)),
            raw_clip
        )
        print(f"  CLIP scores: range=[{clip_scores.min():.3f}, {clip_scores.max():.3f}]")

    if not args.use_referee:
        vjepa_normalization = "per_video" if args.use_true_vjepa else "global"
        ijepa_normalization = "per_video" if args.use_true_ijepa else "global"
        composite, p_sig, c_sig = composite_scoring(
            physics_scores, ijepa_scores,
            vjepa_result["tubelet_to_frames"],
            ijepa_result.get("ijepa_frame_indices", []),
            total_frames,
            flow_scores=flow_scores,
            freq_scores=freq_scores,
            depth_scores=depth_scores,
            clip_scores=clip_scores,
            vjepa_normalization=vjepa_normalization,
            ijepa_normalization=ijepa_normalization
        )

    fused = find_anomaly_segments(
        composite,
        threshold=args.threshold,
        min_gap=args.min_gap,
        min_length=args.min_length,
        fps=fps
    )

    # Add timestamp and confidence labels
    from datetime import timedelta
    for seg in fused:
        seg["timestamp_start"] = str(timedelta(seconds=seg["start_frame"] / max(fps, 1)))
        seg["timestamp_end"] = str(timedelta(seconds=seg["end_frame"] / max(fps, 1)))
        seg["source"] = "composite"
        seg["confidence"] = ("high" if seg["max_score"] > 0.7 else
                             "medium" if seg["max_score"] > 0.5 else "low")

    print(f"  Threshold={args.threshold}, min_gap={args.min_gap}, min_length={args.min_length}")
    print(f"  Composite range=[{composite.min():.3f},{composite.max():.3f}]")
    print(f"  Anomaly segments: {len(fused)}")
    for seg in fused:
        print(f"    [{seg['confidence']}] 帧 {seg['start_frame']}-{seg['end_frame']} "
              f"({seg['timestamp_start']}~{seg['timestamp_end']}) "
              f"max={seg['max_score']:.3f} mean={seg['mean_score']:.3f}")

    # 改进3: 多尺度融合 (短clip + 长clip 取交集)
    if args.multi_scale:
        print(f"\n  [Multi-Scale] Running long-clip pass (max_frames=128)...")
        # Re-extract with more frames for finer temporal resolution
        vjepa_long = extract_vjep_tokens(args.video, use_cache=True, max_frames=128)
        physics_long = score_vjepa_multi(vjepa_long["vjepa_tokens"],
                                         vjepa_long["tubelet_to_frames"], total_frames,
                                         predictor_embed=predictor_embed)
        composite_long, _, _ = composite_scoring(
            physics_long, ijepa_scores,
            vjepa_long["tubelet_to_frames"],
            ijepa_result.get("ijepa_frame_indices", []),
            total_frames,
            flow_scores=flow_scores, freq_scores=freq_scores,
            depth_scores=depth_scores, clip_scores=clip_scores,
            vjepa_normalization=vjepa_normalization,
            ijepa_normalization=ijepa_normalization
        )
        fused_long = find_anomaly_segments(
            composite_long,
            threshold=args.threshold, min_gap=args.min_gap,
            min_length=args.min_length, fps=fps
        )
        # Add metadata
        for seg in fused_long:
            seg["timestamp_start"] = str(timedelta(seconds=seg["start_frame"] / max(fps, 1)))
            seg["timestamp_end"] = str(timedelta(seconds=seg["end_frame"] / max(fps, 1)))
            seg["source"] = "composite_long"
            seg["confidence"] = ("high" if seg["max_score"] > 0.7 else
                                 "medium" if seg["max_score"] > 0.5 else "low")

        print(f"  Long-clip segments: {len(fused_long)}")

        # Intersection: keep segments detected in BOTH runs
        fused_intersection = []
        for seg_s in fused:
            for seg_l in fused_long:
                inter = max(0, min(seg_s["end_frame"], seg_l["end_frame"]) -
                               max(seg_s["start_frame"], seg_l["start_frame"]))
                union = max(seg_s["end_frame"], seg_l["end_frame"]) - \
                        min(seg_s["start_frame"], seg_l["start_frame"])
                iou = inter / (union + 1e-6)
                if iou > 0.3:
                    # Merge: use the intersection bounds
                    merged_seg = {
                        "start_frame": max(seg_s["start_frame"], seg_l["start_frame"]),
                        "end_frame": min(seg_s["end_frame"], seg_l["end_frame"]),
                        "max_score": max(seg_s["max_score"], seg_l["max_score"]),
                        "mean_score": (seg_s["mean_score"] + seg_l["mean_score"]) / 2,
                        "timestamp_start": seg_s["timestamp_start"],
                        "timestamp_end": seg_s["timestamp_end"],
                        "source": "multi_scale",
                        "confidence": seg_s["confidence"],
                    }
                    fused_intersection.append(merged_seg)
                    break

        print(f"  Multi-scale intersection: {len(fused_intersection)} segments")
        fused = fused_intersection

    # ════════════════════════════════════════════════════════════════
    #  Phase 4: 可视化 + 报告
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[Phase 4] 可视化 + 报告")
    print("=" * 60)

    frames_dir = os.path.join(args.output, "frames")
    vjepa_paths, ijepa_paths, combined_paths = generate_all_annotations(
        args.video, vjepa_result, ijepa_result,
        vjepa_heatmaps,
        None,  # ijepa_patch_relative (no longer computed)
        ijepa_result.get("ijepa_frame_indices", []),
        fused,
        frames_dir
    )
    print(f"  V-JEPA heatmaps:  {len(vjepa_paths)}")
    print(f"  I-JEPA boxes:     {len(ijepa_paths)}")
    print(f"  Combined:         {len(combined_paths)}")

    # 为每个异常段生成全部帧
    segment_frames = generate_segment_all_frames(
        args.video, vjepa_heatmaps,
        vjepa_result["tubelet_to_frames"],
        fused, fps, frames_dir
    )
    print(f"  Segment frames:   {sum(s['frame_count'] for s in segment_frames)} total "
          f"({len(segment_frames)} segments)")

    # 保存 composite 分数曲线（供前端绘制）
    physics_raw = np.zeros(total_frames)
    for t, (s, e) in enumerate(vjepa_result["tubelet_to_frames"]):
        if t < len(physics_scores):
            s = max(0, min(s, total_frames - 1))
            e = max(0, min(e, total_frames - 1))
            physics_raw[s:e+1] = physics_scores[t]
    corruption_raw, _ = expand_sparse_frame_scores(
        ijepa_result.get("ijepa_frame_indices", []),
        ijepa_scores,
        total_frames,
    )

    timeseries_data = {
        "composite": composite.tolist(),
        "physics_sig": p_sig.tolist(),
        "physics_raw": physics_raw.tolist(),
        "corruption_sig": c_sig.tolist(),
        "corruption_raw": corruption_raw.tolist(),
        "tubelet_to_frames": vjepa_result["tubelet_to_frames"],
        "fps": fps,
        "total_frames": total_frames,
        "threshold": args.threshold,
        "jepa_usage": jepa_usage,
    }
    timeseries_path = os.path.join(args.output, "vjepa_timeseries.json")
    with open(timeseries_path, "w") as f:
        json.dump(timeseries_data, f, indent=2)

    # 保存 V-JEPA 预测热力图数据
    heatmap_path = os.path.join(args.output, "vjepa_heatmaps.npy")
    np.save(heatmap_path, vjepa_heatmaps)
    print(f"  Heatmaps saved: {heatmap_path} ({vjepa_heatmaps.shape})")
    print(f"  Timeseries saved: {timeseries_path}")

    # 报告
    report = {
        "video": args.video,
        "video_name": video_name,
        "total_frames": total_frames,
        "fps": fps,
        "threshold": args.threshold,
        "jepa_usage": jepa_usage,
        "fused_anomalies": fused,
        "stats": {
            "total_segments": len(fused),
            "high": sum(1 for s in fused if s["confidence"] == "high"),
            "medium": sum(1 for s in fused if s["confidence"] == "medium"),
            "low": sum(1 for s in fused if s["confidence"] == "low"),
        },
        "vjepa_heatmap_frames": vjepa_paths,
        "ijepa_box_frames": ijepa_paths,
        "combined_frames": combined_paths,
        "vjepa_timeseries": timeseries_path,
        "composite_mean": float(composite.mean()),
        "composite_max": float(composite.max()),
        "segment_frames": segment_frames,
    }

    report_path = os.path.join(args.output, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 摘要
    print("\n" + "=" * 60)
    print(f"报告: {report_path}")
    print(f"\n摘要:")
    print(f"  Composite 分数: mean={composite.mean():.3f} max={composite.max():.3f}")
    print(f"  异常段: {len(fused)} 段 "
          f"(高:{report['stats']['high']} 中:{report['stats']['medium']} 低:{report['stats']['low']})")
    for seg in fused:
        print(f"    [{seg['confidence']}] 帧{seg['start_frame']}-{seg['end_frame']} "
              f"(max={seg['max_score']:.3f})")

    # 裁剪异常段热力图视频片段
    if fused:
        import subprocess as sp
        clip_dir = os.path.join(args.output, "anomaly_clips")
        os.makedirs(clip_dir, exist_ok=True)
        print(f"\n[Phase 5] 生成异常段热力图视频片段...")
        sp.run([
            sys.executable, os.path.join(DATA_ROOT, "extract_anomaly_clips.py"),
            "--report", report_path,
            "--video", args.video,
            "--output-dir", clip_dir,
        ], check=False, timeout=600)
        print(f"  片段: {clip_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
