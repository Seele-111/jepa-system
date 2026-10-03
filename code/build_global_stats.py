#!/usr/bin/env python3
"""build_global_stats.py — 从210个正常视频建全局参照分布（使用与检测管线一致的分数）"""
import torch, glob, numpy as np, os
import torch.nn.functional as F

VJEPA_GRID = 24; IJEPA_GRID = 14; FEAT_DIM = 1408
DATA_DIR = "/home/zzy/jepa_data/features_cache_v2"
OUT_PATH = "/home/zzy/jepa_data/global_normal_stats.pt"

print("[1/3] Loading normal videos...")
files = sorted(glob.glob(os.path.join(DATA_DIR, "*_label0.pt")))
print(f"  Found {len(files)} normal files")

v_physics = []   # V-JEPA final composite (same as score_vjepa_multi)
i_composite = []  # I-JEPA final composite (same as score_ijepa_multi)

for fi, fpath in enumerate(files):
    try:
        data = torch.load(fpath, map_location="cpu", weights_only=False)
    except: continue

    # V-JEPA
    vfeat = data["vjepa_perframe"].float()
    if vfeat.shape[0] % (VJEPA_GRID * VJEPA_GRID) != 0: continue
    T = vfeat.shape[0] // (VJEPA_GRID * VJEPA_GRID)
    if T < 3: continue
    vfeat = vfeat.reshape(T, VJEPA_GRID * VJEPA_GRID, FEAT_DIM)

    # temporal (same weights as score_vjepa_multi)
    scales = [1, 3, 5]
    tm = np.zeros((T, len(scales)))
    for si, s in enumerate(scales):
        for t in range(T):
            if t + s < T:
                a = vfeat[t].mean(0); b = vfeat[t+s].mean(0)
                tm[t, si] = 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            else:
                tm[t, si] = tm[max(0,t-s), si] if t >= s else 0
    w = np.array([1.0, 0.7/3, 0.4/5])[:len(scales)]
    temporal = np.clip((tm * w).sum(1) / w.sum(), 0, 1)

    spatial = np.clip(vfeat.var(1).mean(1).numpy() / (vfeat.var(1).mean(1).mean().item() + 1e-6), 0, 3)
    en = vfeat.norm(dim=-1).mean(dim=1).numpy()  # per-frame mean energy
    energy = np.clip(np.abs(np.diff(en, prepend=en[0])) / (en.mean() + 1e-6), 0, 0.3)

    # adaptive weights
    def q(s): 
        t = s[s>np.percentile(s,75)]; return min(1.0, 1.0/(t.std()/(t.mean()+1e-6)+0.5)) if len(t)>1 else 0.3
    tq, sq, eq = q(temporal), q(spatial), q(energy)
    tw = tq/(tq+sq+eq+1e-6)*0.6; sw = sq/(tq+sq+eq+1e-6)*0.25; ew = 1-tw-sw
    physics = tw*temporal + sw*spatial + ew*energy
    v_physics.extend(physics.tolist())

    # I-JEPA
    ipatch = data["ijepa_patch"].float()
    if ipatch.dim()==2: ipatch = ipatch.reshape(ipatch.shape[0], IJEPA_GRID*IJEPA_GRID, -1)
    K,P,D = ipatch.shape
    fvar_i = ipatch.var(1).mean(1).numpy() / (ipatch.var(1).mean(1).mean().item()+1e-6)
    spinc_i = np.array([1.0 - (F.normalize(ipatch[k],dim=-1) @ F.normalize(ipatch[k],dim=-1).T).sum().item()/max(1,P*(P-1)) for k in range(K)])
    en_i = ipatch.norm(dim=-1).mean(dim=1).numpy()
    z_i = np.abs(en_i - en_i.mean()) / (en_i.std() + 1e-6)
    i_comp = 0.3*fvar_i/2.0 + 0.3*spinc_i + 0.4*z_i/3.0
    i_composite.extend(i_comp.tolist())

    if (fi+1)%50==0: print(f"  Processed {fi+1}/{len(files)}")

vp, ic = np.array(v_physics), np.array(i_composite)
print(f"\n[2/3] Statistics (composite scores):")
print(f"  V-JEPA physics: μ={vp.mean():.4f} σ={vp.std():.4f} (n={len(vp)})")
print(f"  I-JEPA composite: μ={ic.mean():.4f} σ={ic.std():.4f} (n={len(ic)})")

stats = {"v_physics_mu": float(vp.mean()), "v_physics_sigma": float(vp.std()),
         "i_composite_mu": float(ic.mean()), "i_composite_sigma": float(ic.std())}
torch.save(stats, OUT_PATH)
print(f"\n[3/3] Saved to {OUT_PATH}")
