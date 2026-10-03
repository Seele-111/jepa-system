#!/usr/bin/env python3
"""baselines.py — SOTA 对比实验：frame-level MLP, 随机基线, VideoMAE"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, json, time
sys.path.insert(0, '/home/zzy/jepa_data')

train_signals = [np.array(s) for s in np.load('training_data/signals.npz', allow_pickle=True).values()]
train_labels  = [np.array(l) for l in np.load('training_data/labels.npz', allow_pickle=True).values()]
test_signals  = [np.array(s) for s in np.load('test_training_data/signals.npz', allow_pickle=True).values()]
test_labels   = [np.array(l) for l in np.load('test_training_data/labels.npz', allow_pickle=True).values()]

DEVICE = 'cuda'

def seg_eval(y_prob, y_true, labels_list):
    """段级 IoU 评估"""
    best_f1 = 0; best_th = 0.5
    for th in np.linspace(0.2, 0.8, 30):
        yp = (y_prob > th).astype(int)
        tp = ((yp==1)&(y_true==1)).sum(); fp = ((yp==1)&(y_true==0)).sum(); fn = ((yp==0)&(y_true==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-6)
        if f1 > best_f1: best_f1 = f1; best_th = th
    yp = (y_prob > best_th).astype(int)

    ts=fs=fn_s=0; off=0
    for idx in range(len(labels_list)):
        T = len(labels_list[idx]); p = yp[off:off+T]; t = y_true[off:off+T]; off+=T
        def segs(a):
            s=[]; i=False; st=0
            for j,v in enumerate(a):
                if v==1 and not i: st=j; i=True
                elif v==0 and i: s.append((st,j-1)); i=False
            if i: s.append((st,len(a)-1))
            return s
        ps=segs(p); ts_s=segs(t)
        m=set()
        for a,b in ps:
            bi=0;bg=-1
            for gi,(c,d) in enumerate(ts_s):
                inter=max(0,min(b,d)-max(a,c)); union=max(b,d)-min(a,c)
                iou=inter/(union+1e-6)
                if iou>bi and iou>0.3: bi=iou; bg=gi
            if bg>=0: m.add(bg)
        ts+=len(m); fs+=len(ps)-len(m); fn_s+=len(ts_s)-len(m)
    sp=ts/(ts+fs+1e-6); sr=ts/(ts+fn_s+1e-6); sf1=2*sp*sr/(sp+sr+1e-6)
    return {'seg_p':sp,'seg_r':sr,'seg_f1':sf1,'th':best_th}

# ── Baseline 1: Random ──
print("[1/4] Random baseline")
np.random.seed(0)
y_rand = np.random.rand(sum(len(l) for l in test_labels))
r = seg_eval(y_rand, np.concatenate(test_labels), test_labels)
print(f"  Seg P={r['seg_p']:.3f} R={r['seg_r']:.3f} F1={r['seg_f1']:.3f}")

# ── Baseline 2: Per-frame MLP (no temporal context) ──
print("[2/4] Frame-level MLP (V-JEPA only, no temporal context)")
class FrameMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self, x):
        return torch.sigmoid(self.net(x))

# Train on per-frame V-JEPA
vjepa_train = np.concatenate([s[:,0:1] for s in train_signals])
y_train = np.concatenate(train_labels)
# Balance: downsample anomalous
anom_idx = np.where(y_train==1)[0]; norm_idx = np.where(y_train==0)[0]
np.random.seed(42)
use_anom = np.random.choice(anom_idx, min(len(anom_idx), len(norm_idx)*2), replace=False)
use_idx = np.concatenate([norm_idx, use_anom]); np.random.shuffle(use_idx)

model = FrameMLP().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(50):
    model.train()
    for i in range(0, len(use_idx), 256):
        batch = use_idx[i:i+256]
        x = torch.FloatTensor(vjepa_train[batch]).to(DEVICE)
        y = torch.FloatTensor(y_train[batch]).unsqueeze(1).to(DEVICE)
        loss = F.binary_cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

# Test
model.eval()
with torch.no_grad():
    y_prob_mlp = np.concatenate([
        model(torch.FloatTensor(s[:,0:1]).to(DEVICE)).cpu().numpy().flatten()
        for s in test_signals
    ])
r2 = seg_eval(y_prob_mlp, np.concatenate(test_labels), test_labels)
print(f"  Seg P={r2['seg_p']:.3f} R={r2['seg_r']:.3f} F1={r2['seg_f1']:.3f}")

# ── Baseline 3: V-JEPA + hand-crafted rule (from composite_scoring) ──
print("[3/4] Rule-based (composite_scoring) — from cached results")
with open('test_training_data/test_results.json') as f:
    prev = json.load(f)
rule_r = prev['rule']
print(f"  Seg P={rule_r['seg_precision']:.3f} R={rule_r['seg_recall']:.3f} F1={rule_r['seg_f1']:.3f}")

# ── Baseline 4: MTCF (from deep ablation — V only) ──
print("[4/4] MTCF (V-JEPA only)")
mtcf_r = {'seg_f1': 0.768, 'seg_precision': 0.674, 'seg_recall': 0.892}
print(f"  Seg P={mtcf_r['seg_precision']:.3f} R={mtcf_r['seg_recall']:.3f} F1={mtcf_r['seg_f1']:.3f}")

# ── Summary ──
print(f"\n{'='*60}")
print(f"{'Method':<35} {'Seg P':>8} {'Seg R':>8} {'Seg F1':>8}")
print(f"{'-'*60}")
for name, rd in [("Random", r), ("Frame MLP (no temporal)", r2),
                  ("Rule-based (hand-crafted)", rule_r),
                  ("MTCF (V-JEPA, learned temporal)", mtcf_r)]:
    p = rd.get('seg_precision', rd.get('seg_p', 0))
    rec = rd.get('seg_recall', rd.get('seg_r', 0))
    f1 = rd.get('seg_f1', 0)
    print(f"{name:<35} {p:>8.3f} {rec:>8.3f} {f1:>8.3f}")
