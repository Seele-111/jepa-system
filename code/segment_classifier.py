#!/usr/bin/env python3
"""
segment_classifier.py — 段级二分类器，精筛 composite_scoring 的候选段

策略: composite_scoring 用低阈值(0.50)产生候选段（保召回）
      提取每段的 12 维特征，训练 XGBoost/MLP 区分 TP vs FP
      只用高分候选 → 高精度
"""
import json, os, glob, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
import pickle

ANNO_DIR = '/mnt/c/Users/admin/Desktop/测试/annotations'
CACHE_DIR = '/home/zzy/jepa_data/mtcf_filter_cache'

anno_files = sorted(glob.glob(os.path.join(ANNO_DIR, '*_annotations.json')))

# ── Step 1: Generate candidates (low threshold for high recall) ──
# Re-run detection on each video with threshold=0.50
import subprocess
VIDEO_DIR = '/mnt/c/Users/admin/Desktop/测试'
OUT_DIR = '/tmp/candidates_v2'
os.makedirs(OUT_DIR, exist_ok=True)

print("Step 1: Generating candidates (th=0.50)...")
for i, af in enumerate(anno_files):
    with open(af) as f: data = json.load(f)
    name = data['video_name']
    video_path = data.get('video', os.path.join(VIDEO_DIR, name))
    result_dir = os.path.join(OUT_DIR, name.replace('.mp4',''))
    report_path = os.path.join(result_dir, 'report.json')
    if os.path.exists(report_path): continue

    cmd = ['/home/zzy/vjepa2-main/vjepa-env/bin/python',
           '/home/zzy/jepa_data/detect_and_report_v4.py',
           '--video', video_path, '--output', result_dir,
           '--threshold', '0.50', '--min-gap', '1', '--min-length', '1',
           '--use-predictor-embed', '--use-optical-flow', '--use-frequency']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd='/home/zzy/jepa_data')
    if (i+1) % 5 == 0:
        print(f"  [{i+1}/{len(anno_files)}]")

# ── Step 2: Extract segment features ──
print("\nStep 2: Extracting segment features...")
X = []  # features
y = []  # 1=TP, 0=FP

for af in anno_files:
    if not os.path.exists(af): continue
    with open(af) as f: data = json.load(f)
    name = data['video_name'].replace('.mp4','')
    result_dir = os.path.join(OUT_DIR, name)
    report_path = os.path.join(result_dir, 'report.json')
    if not os.path.exists(report_path): continue

    with open(report_path) as f: report = json.load(f)
    gt_segs = [(a['start_frame'], a['end_frame']) for a in data.get('annotations',[])]
    total_frames = data['total_frames']

    # Load composite timeseries
    ts_path = os.path.join(result_dir, 'vjepa_timeseries.json')
    composite = None
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            ts = json.load(f)
        composite = np.array(ts.get('composite', ts.get('scores', [])))
        if len(composite) < total_frames:
            composite = np.pad(composite, (0, total_frames-len(composite)), 'edge')
        elif len(composite) > total_frames:
            composite = composite[:total_frames]

    candidates = report.get('fused_anomalies', [])

    for cand in candidates:
        sf, ef = cand['start_frame'], cand['end_frame']
        length = ef - sf + 1
        max_s = cand['max_score']
        mean_s = cand.get('mean_score', max_s)

        # Features
        feats = [
            max_s,                             # 0: max score
            mean_s,                            # 1: mean score
            length,                            # 2: duration
            np.log1p(length),                  # 3: log duration
            max_s * length,                    # 4: score × length
            mean_s / (max_s + 1e-6),           # 5: score flatness
        ]

        # Composite statistics within segment
        if composite is not None and sf < len(composite) and ef < len(composite):
            seg_comp = composite[sf:ef+1]
            feats.extend([
                seg_comp.std(),                # 6: score std within segment
                seg_comp.max() - seg_comp.min(), # 7: score range
            ])
            if len(seg_comp) >= 2:
                feats.extend([
                    np.gradient(seg_comp).mean(),  # 8: mean gradient
                    np.gradient(seg_comp).max(),   # 9: max positive gradient
                    (seg_comp[-1] - seg_comp[0]) / (length + 1e-6), # 10: score trend
                ])
            else:
                feats.extend([0, 0, 0])
        else:
            feats.extend([0]*5)

        # Position in video
        feats.append(sf / max(1, total_frames))  # 11: relative start position

        # Is it a TP?
        is_tp = 0
        for gs, ge in gt_segs:
            inter = max(0, min(ef, ge) - max(sf, gs))
            union = max(ef, ge) - min(sf, gs)
            if inter / (union + 1e-6) > 0.3:
                is_tp = 1; break

        X.append(feats)
        y.append(is_tp)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# Handle NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Samples: {len(X)} ({y.sum()} TP, {(1-y).sum()} FP)")

# ── Step 3: Train classifier ──
print("\nStep 3: Training segment classifier...")

clf = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    random_state=42, subsample=0.8
)
clf.fit(X, y)

# Feature importance
importances = clf.feature_importances_
feat_names = ['max_score', 'mean_score', 'length', 'log_len', 'score*len',
              'flatness', 'std', 'range', 'grad_mean', 'grad_max', 'trend', 'pos']
print("Feature importances:")
for fn, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    print(f"  {fn:<15s}: {imp:.3f}")

# ── Step 4: Evaluate on train + cross-val ──
print("\nStep 4: Evaluation...")

train_pred = clf.predict(X)
train_prob = clf.predict_proba(X)[:, 1]

# Find best threshold
best_f1 = 0; best_th = 0.5
for th in np.linspace(0.1, 0.9, 40):
    pred = (train_prob > th).astype(int)
    f1 = f1_score(y, pred)
    if f1 > best_f1: best_f1 = f1; best_th = th

pred = (train_prob > best_th).astype(int)
tp = ((pred==1)&(y==1)).sum(); fp = ((pred==1)&(y==0)).sum(); fn = ((pred==0)&(y==1)).sum()
p = tp/(tp+fp+1e-6); r = tp/(tp+fn+1e-6); f1 = 2*p*r/(p+r+1e-6)
print(f"  Train (th={best_th:.2f}): P={p:.3f} R={r:.3f} F1={f1:.3f} TP={tp} FP={fp} FN={fn}")

# Leave-one-video-out cross-val
print("\n  Leave-one-video-out:")
video_ids = []
for af in anno_files:
    name = os.path.basename(af).replace('_annotations.json','')
    video_ids.append(name)

# Map each sample to its video
sample_videos = []
for af in anno_files:
    if not os.path.exists(af): continue
    with open(af) as f: data = json.load(f)
    name = data['video_name'].replace('.mp4','')
    result_dir = os.path.join(OUT_DIR, name)
    report_path = os.path.join(result_dir, 'report.json')
    if not os.path.exists(report_path): continue
    with open(report_path) as f:
        n_cands = len(json.load(f).get('fused_anomalies', []))
    sample_videos.extend([name] * n_cands)

unique_vids = sorted(set(sample_videos))
loo_tp = loo_fp = loo_fn = 0

for test_vid in unique_vids:
    test_mask = np.array([v == test_vid for v in sample_videos])
    train_mask = ~test_mask
    if test_mask.sum() == 0 or train_mask.sum() == 0: continue

    clf_loo = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    clf_loo.fit(X[train_mask], y[train_mask])
    probs = clf_loo.predict_proba(X[test_mask])[:, 1]
    preds = (probs > best_th).astype(int)

    for pi, yi in zip(preds, y[test_mask]):
        if pi == 1 and yi == 1: loo_tp += 1
        elif pi == 1 and yi == 0: loo_fp += 1
        elif pi == 0 and yi == 1: loo_fn += 1

lp = loo_tp/(loo_tp+loo_fp+1e-6); lr = loo_tp/(loo_tp+loo_fn+1e-6); lf1 = 2*lp*lr/(lp+lr+1e-6)
print(f"  LOO: P={lp:.3f} R={lr:.3f} F1={lf1:.3f} TP={loo_tp} FP={loo_fp} FN={loo_fn}")

# Compare with raw threshold approach
print(f"\nStep 5: Segment-level F1 comparison...")
# Raw: same candidates, fixed threshold
for th, ml, label in [(0.50, 1, 'Raw th=0.50'), (0.65, 2, 'Raw th=0.65'),
                        (0.70, 5, 'Raw th=0.70+len5')]:
    tp=0; fp=0; fn=0; all_gt=0
    for af in anno_files:
        with open(af) as f: data = json.load(f)
        name = data['video_name'].replace('.mp4','')
        result_dir = os.path.join(OUT_DIR, name)
        report_path = os.path.join(result_dir, 'report.json')
        if not os.path.exists(report_path): continue
        with open(report_path) as f: report = json.load(f)
        gt_segs = [(a['start_frame'], a['end_frame']) for a in data.get('annotations',[])]
        all_gt += len(gt_segs)
        pred_segs = [(s['start_frame'], s['end_frame']) for s in report.get('fused_anomalies',[])
                     if s['max_score'] > th and (s['end_frame']-s['start_frame']+1) >= ml]
        matched = set()
        for ps, pe in pred_segs:
            best_iou = 0; best_gt = -1
            for gi, (gs, ge) in enumerate(gt_segs):
                inter = max(0, min(pe, ge) - max(ps, gs)); union = max(pe, ge) - min(ps, gs)
                iou = inter / (union + 1e-6)
                if iou > best_iou and iou > 0.3: best_iou = iou; best_gt = gi
            if best_gt >= 0: matched.add(best_gt)
        tp += len(matched); fp += len(pred_segs) - len(matched); fn += len(gt_segs) - len(matched)
    p = tp/(tp+fp+1e-6); r = tp/(tp+fn+1e-6); f = 2*p*r/(p+r+1e-6)
    print(f"  {label:<22s}: P={p:.3f} R={r:.3f} F1={f:.3f}")

# Our learned classifier applied to candidates
# Apply to all candidates: keep if classifier says TP
tp=0; fp=0; fn=0; all_gt=0
sample_idx = 0
for af in anno_files:
    if not os.path.exists(af): continue
    with open(af) as f: data = json.load(f)
    name = data['video_name'].replace('.mp4','')
    result_dir = os.path.join(OUT_DIR, name)
    report_path = os.path.join(result_dir, 'report.json')
    if not os.path.exists(report_path): continue
    with open(report_path) as f: report = json.load(f)
    gt_segs = [(a['start_frame'], a['end_frame']) for a in data.get('annotations',[])]
    all_gt += len(gt_segs)
    candidates = report.get('fused_anomalies', [])
    pred_segs = []
    for ci, cand in enumerate(candidates):
        if sample_idx < len(X) and train_prob[sample_idx] > best_th:
            pred_segs.append((cand['start_frame'], cand['end_frame']))
        sample_idx += 1
    matched = set()
    for ps, pe in pred_segs:
        best_iou = 0; best_gt = -1
        for gi, (gs, ge) in enumerate(gt_segs):
            inter = max(0, min(pe, ge) - max(ps, gs)); union = max(pe, ge) - min(ps, gs)
            iou = inter / (union + 1e-6)
            if iou > best_iou and iou > 0.3: best_iou = iou; best_gt = gi
        if best_gt >= 0: matched.add(best_gt)
    tp += len(matched); fp += len(pred_segs) - len(matched); fn += len(gt_segs) - len(matched)
p = tp/(tp+fp+1e-6); r = tp/(tp+fn+1e-6); f = 2*p*r/(p+r+1e-6)
print(f"  {'Learned (ours)':<22s}: P={p:.3f} R={r:.3f} F1={f:.3f}")

# Save
with open('/home/zzy/jepa_data/segment_clf.pkl', 'wb') as f:
    pickle.dump({'clf': clf, 'threshold': best_th, 'feature_names': feat_names}, f)
print(f"\nSaved segment classifier to segment_clf.pkl")
