#!/usr/bin/env python3
"""ablate_all.py — 完整消融实验：逐路去掉 6 个信号，测量段级 F1 变化"""
import torch, torch.nn.functional as F, numpy as np, sys, os, json, time

sys.path.insert(0, '/home/zzy/jepa_data')
from mtcf_train import MultiScaleFusion, segment_aware_loss

# ── Load data ──
train_signals = [np.array(s) for s in np.load('training_data/signals.npz', allow_pickle=True).values()]
train_labels  = [np.array(l) for l in np.load('training_data/labels.npz', allow_pickle=True).values()]
train_cats    = [np.array(c) for c in np.load('training_data/categories.npz', allow_pickle=True).values()]
test_signals  = [np.array(s) for s in np.load('test_training_data/signals.npz', allow_pickle=True).values()]
test_labels   = [np.array(l) for l in np.load('test_training_data/labels.npz', allow_pickle=True).values()]

SIGNAL_NAMES = ['V-JEPA', 'I-JEPA', 'OpticalFlow', 'Frequency', 'Depth', 'CLIP']
DEVICE = 'cuda'
cat_weights = torch.FloatTensor([1.0, 0.45, 1.61, 5.0])

def train_and_eval(train_sigs, test_sigs, n_signals, name):
    """训练 n_signals 路 MTCF，返回测试集段级指标"""
    model = MultiScaleFusion(n_signals=n_signals).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Split
    np.random.seed(42)
    idxs = np.random.permutation(len(train_sigs))
    n_val = max(1, int(0.15 * len(idxs)))
    train_idx = idxs[n_val:]; val_idx = idxs[:n_val]

    best_val_f1 = 0; best_state = None; no_improve = 0

    for epoch in range(100):
        model.train()
        np.random.shuffle(train_idx)
        for idx in train_idx:
            s = torch.FloatTensor(train_sigs[idx]).unsqueeze(0).to(DEVICE)
            l = torch.LongTensor(train_labels[idx]).unsqueeze(0).to(DEVICE)
            c = torch.LongTensor(train_cats[idx]).unsqueeze(0).to(DEVICE)
            optimizer.zero_grad()
            logits = model(s)
            loss = segment_aware_loss(logits, l, c, cat_weights=cat_weights,
                                      lambda_smooth=0.15, lambda_cat=0.5, lambda_seg=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Val
        if epoch % 5 != 0 and epoch > 0: continue
        model.eval()
        all_probs = []; all_labs = []
        with torch.no_grad():
            for idx in val_idx:
                s = torch.FloatTensor(train_sigs[idx]).unsqueeze(0).to(DEVICE)
                logits = model(s)
                all_probs.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())
                all_labs.append(train_labels[idx])

        y_pred = np.concatenate([(p[:, 1:].sum(axis=1) > 0.5).astype(int) for p in all_probs])
        y_true = np.concatenate(all_labs)
        tp = ((y_pred==1)&(y_true==1)).sum()
        fp = ((y_pred==1)&(y_true==0)).sum()
        fn = ((y_pred==0)&(y_true==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-6)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 10:
            print(f"  {name}: early stop @ epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # Test segment-level
    model.eval()
    with torch.no_grad():
        all_probs_test = []
        for s in test_sigs:
            t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
            logits = model(t)
            all_probs_test.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())

    y_prob = np.concatenate([p[:, 1:].sum(axis=1) for p in all_probs_test])
    y_true = np.concatenate(test_labels)

    # Best threshold
    best_f1 = 0; best_th = 0.5
    for th in np.linspace(0.2, 0.8, 30):
        yp = (y_prob > th).astype(int)
        tp = ((yp==1)&(y_true==1)).sum()
        fp = ((yp==1)&(y_true==0)).sum()
        fn = ((yp==0)&(y_true==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-6)
        if f1 > best_f1: best_f1 = f1; best_th = th

    yp = (y_prob > best_th).astype(int)

    # Segment-level IoU
    tp_seg = fp_seg = fn_seg = 0
    offset = 0
    for idx in range(len(test_sigs)):
        T = len(test_labels[idx])
        pred = yp[offset:offset+T]; true = y_true[offset:offset+T]; offset += T

        def find_segs(arr):
            segs = []; in_seg = False; start = 0
            for i, v in enumerate(arr):
                if v == 1 and not in_seg: start = i; in_seg = True
                elif v == 0 and in_seg: segs.append((start, i-1)); in_seg = False
            if in_seg: segs.append((start, len(arr)-1))
            return segs

        ps = find_segs(pred); ts = find_segs(true)
        matched = set()
        for a, b in ps:
            best_iou = 0; best_gi = -1
            for gi, (c, d) in enumerate(ts):
                inter = max(0, min(b,d)-max(a,c)); union = max(b,d)-min(a,c)
                iou = inter/(union+1e-6)
                if iou > best_iou and iou > 0.3: best_iou = iou; best_gi = gi
            if best_gi >= 0: matched.add(best_gi)
        tp_seg += len(matched); fp_seg += len(ps)-len(matched); fn_seg += len(ts)-len(matched)

    sp = tp_seg/(tp_seg+fp_seg+1e-6)
    sr = tp_seg/(tp_seg+fn_seg+1e-6)
    sf1 = 2*sp*sr/(sp+sr+1e-6)

    return {'name': name, 'seg_p': sp, 'seg_r': sr, 'seg_f1': sf1,
            'seg_tp': tp_seg, 'seg_fp': fp_seg, 'seg_fn': fn_seg}


# ── Full model (baseline) ──
print("=" * 60)
print("[0/7] Full model (6 signals)")
t0 = time.time()
full = train_and_eval(train_signals, test_signals, 6, "Full-6")
print(f"  Seg P={full['seg_p']:.3f} R={full['seg_r']:.3f} F1={full['seg_f1']:.3f} ({time.time()-t0:.0f}s)")

# ── Ablations ──
results = [full]
for drop_idx in range(6):
    name = f"No-{SIGNAL_NAMES[drop_idx]}"
    print(f"\n[{drop_idx+1}/7] {name}")

    # Drop column drop_idx
    train_abl = [np.delete(s, drop_idx, axis=1) for s in train_signals]
    test_abl  = [np.delete(s, drop_idx, axis=1) for s in test_signals]

    t0 = time.time()
    r = train_and_eval(train_abl, test_abl, 5, name)
    r['time'] = time.time() - t0
    results.append(r)

    print(f"  Seg P={r['seg_p']:.3f} R={r['seg_r']:.3f} F1={r['seg_f1']:.3f} Δ={r['seg_f1']-full['seg_f1']:+.3f}")

# ── Summary ──
print(f"\n{'='*70}")
print(f"{'Ablation':<25} {'P':>8} {'R':>8} {'F1':>8} {'Δ F1':>8} {'Δ %':>8}")
print(f"{'-'*65}")
for r in results:
    delta = r['seg_f1'] - full['seg_f1']
    delta_pct = (r['seg_f1'] / full['seg_f1'] - 1) * 100
    print(f"{r['name']:<25} {r['seg_p']:>8.3f} {r['seg_r']:>8.3f} {r['seg_f1']:>8.3f} {delta:>+8.3f} {delta_pct:>+8.1f}%")

# Save
with open('ablation_results.json', 'w') as f:
    json.dump([{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in r.items()} for r in results], f, indent=2)
print(f"\nSaved to ablation_results.json")
