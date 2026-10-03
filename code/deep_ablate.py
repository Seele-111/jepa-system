#!/usr/bin/env python3
"""deep_ablate.py — 深层消融：测试信号组合而非单个信号"""
import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, '/home/zzy/jepa_data')
from mtcf_train import MultiScaleFusion, segment_aware_loss

train_signals = [np.array(s) for s in np.load('training_data/signals.npz', allow_pickle=True).values()]
train_labels  = [np.array(l) for l in np.load('training_data/labels.npz', allow_pickle=True).values()]
train_cats    = [np.array(c) for c in np.load('training_data/categories.npz', allow_pickle=True).values()]
test_signals  = [np.array(s) for s in np.load('test_training_data/signals.npz', allow_pickle=True).values()]
test_labels   = [np.array(l) for l in np.load('test_training_data/labels.npz', allow_pickle=True).values()]

DEVICE = 'cuda'
cat_weights = torch.FloatTensor([1.0, 0.45, 1.61, 5.0])

# Column indices: 0=V-JEPA, 1=I-JEPA, 2=Flow, 3=Freq, 4=Depth, 5=CLIP
EXPERIMENTS = [
    ("All-6",               [0,1,2,3,4,5]),
    ("V+I only (core)",     [0,1]),           # just the two JEPA models
    ("V only",              [0]),
    ("I only",              [1]),
    ("No JEPA (4 aux)",     [2,3,4,5]),       # no JEPA at all
    ("V+I+Freq (minimal)",  [0,1,3]),         # best minimal combo?
]

def train_and_eval(name, cols):
    n_sig = len(cols)
    t_sigs = [s[:, cols] for s in train_signals]
    e_sigs = [s[:, cols] for s in test_signals]

    model = MultiScaleFusion(n_signals=n_sig).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    np.random.seed(42)
    idxs = np.random.permutation(len(t_sigs))
    n_val = max(1, int(0.15*len(idxs)))
    ti = idxs[n_val:]; vi = idxs[:n_val]

    best_f1 = 0; best_st = None; ni = 0
    for ep in range(100):
        model.train()
        np.random.shuffle(ti)
        for idx in ti:
            s = torch.FloatTensor(t_sigs[idx]).unsqueeze(0).to(DEVICE)
            l = torch.LongTensor(train_labels[idx]).unsqueeze(0).to(DEVICE)
            c = torch.LongTensor(train_cats[idx]).unsqueeze(0).to(DEVICE)
            opt.zero_grad()
            loss = segment_aware_loss(model(s), l, c, cat_weights=cat_weights,
                                      lambda_smooth=0.15, lambda_cat=0.5, lambda_seg=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ep % 5 != 0 and ep > 0: continue
        model.eval()
        ap = []; al = []
        with torch.no_grad():
            for idx in vi:
                s = torch.FloatTensor(t_sigs[idx]).unsqueeze(0).to(DEVICE)
                ap.append(F.softmax(model(s), dim=-1).squeeze(0).cpu().numpy())
                al.append(train_labels[idx])
        yp = np.concatenate([(p[:,1:].sum(axis=1)>0.5).astype(int) for p in ap])
        yt = np.concatenate(al)
        tp = ((yp==1)&(yt==1)).sum(); fp = ((yp==1)&(yt==0)).sum(); fn = ((yp==0)&(yt==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-6)
        if f1 > best_f1: best_f1 = f1; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= 10: print(f"  {name}: stop@{ep}"); break

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        apt = [F.softmax(model(torch.FloatTensor(s).unsqueeze(0).to(DEVICE)),dim=-1).squeeze(0).cpu().numpy() for s in e_sigs]
    yp = np.concatenate([p[:,1:].sum(axis=1) for p in apt])
    yt = np.concatenate(test_labels)

    best_t = 0.5; best_f1_s = 0
    for th in np.linspace(0.2,0.8,30):
        ypr = (yp>th).astype(int)
        tp = ((ypr==1)&(yt==1)).sum(); fp = ((ypr==1)&(yt==0)).sum(); fn = ((ypr==0)&(yt==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-6)
        if f1 > best_f1_s: best_f1_s = f1; best_t = th
    ypr = (yp>best_t).astype(int)

    ts=fs=fn_s=0; off=0
    for idx in range(len(e_sigs)):
        T = len(test_labels[idx]); p = ypr[off:off+T]; t = yt[off:off+T]; off+=T
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
    return {'name':name, 'seg_p':sp, 'seg_r':sr, 'seg_f1':sf1}

# ── Run all experiments ──
results = []
for name, cols in EXPERIMENTS:
    print(f"\n{'='*50}")
    print(f"  {name} ({len(cols)} signals: {cols})")
    t0 = time.time()
    r = train_and_eval(name, cols)
    r['time'] = time.time() - t0
    results.append(r)
    print(f"  Seg P={r['seg_p']:.3f} R={r['seg_r']:.3f} F1={r['seg_f1']:.3f}")

# ── Summary ──
full_f1 = results[0]['seg_f1']
print(f"\n{'='*60}")
print(f"{'Experiment':<25} {'P':>8} {'R':>8} {'F1':>8} {'Δ':>8}")
print(f"{'-'*57}")
for r in results:
    d = r['seg_f1'] - full_f1
    print(f"{r['name']:<25} {r['seg_p']:>8.3f} {r['seg_r']:>8.3f} {r['seg_f1']:>8.3f} {d:>+8.3f}")
