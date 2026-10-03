#!/usr/bin/env python3
"""deep_analyze.py — 逐视频分析 composite_scoring 的错误模式"""
import json, os, glob, numpy as np

ANNO_DIR = '/mnt/c/Users/admin/Desktop/测试/annotations'
CACHE_DIR = '/home/zzy/jepa_data/mtcf_filter_cache'

anno_files = sorted(glob.glob(os.path.join(ANNO_DIR, '*_annotations.json')))

tp_total = fp_total = fn_total = 0
per_video = []

for af in anno_files:
    with open(af) as f: data = json.load(f)
    name = data['video_name'].replace('.mp4', '')
    result_dir = os.path.join(CACHE_DIR, name)
    report_path = os.path.join(result_dir, 'report.json')
    if not os.path.exists(report_path): continue

    with open(report_path) as f: report = json.load(f)
    gt_segs = [(a['start_frame'], a['end_frame']) for a in data.get('annotations', [])]
    pred_segs_full = report.get('fused_anomalies', [])
    pred_segs = [(s['start_frame'], s['end_frame'], s['max_score'], s.get('confidence','?')) for s in pred_segs_full]

    total_frames = data['total_frames']

    # Per-prediction analysis
    tp = []; fp = []; fn = []
    matched_gt = set()

    for ps, pe, score, conf in pred_segs:
        best_iou = 0; best_gt = -1
        for gi, (gs, ge) in enumerate(gt_segs):
            inter = max(0, min(pe, ge) - max(ps, gs)); union = max(pe, ge) - min(ps, gs)
            iou = inter / (union + 1e-6)
            if iou > best_iou and iou > 0.3: best_iou = iou; best_gt = gi
        if best_gt >= 0:
            tp.append({'start': ps, 'end': pe, 'score': score, 'conf': conf, 'iou': best_iou, 'length': pe-ps+1})
            matched_gt.add(best_gt)
        else:
            fp.append({'start': ps, 'end': pe, 'score': score, 'conf': conf, 'length': pe-ps+1})

    for gi, (gs, ge) in enumerate(gt_segs):
        if gi not in matched_gt:
            fn.append({'start': gs, 'end': ge, 'length': ge-gs+1})

    tp_total += len(tp); fp_total += len(fp); fn_total += len(fn)
    per_video.append({'name': name, 'tp': len(tp), 'fp': len(fp), 'fn': len(fn),
                      'tp_scores': [t['score'] for t in tp],
                      'fp_scores': [f['score'] for f in fp],
                      'tp_lengths': [t['length'] for t in tp],
                      'fp_lengths': [f['length'] for f in fp],
                      'fn_lengths': [f['length'] for f in fn],
                      'tp_ious': [t['iou'] for t in tp],
                      'total_frames': total_frames})

# Aggregate stats
all_tp_scores = [s for v in per_video for s in v['tp_scores']]
all_fp_scores = [s for v in per_video for s in v['fp_scores']]
all_tp_len = [l for v in per_video for l in v['tp_lengths']]
all_fp_len = [l for v in per_video for l in v['fp_lengths']]
all_fn_len = [l for v in per_video for l in v['fn_lengths']]
all_tp_ious = [i for v in per_video for i in v['tp_ious']]

print(f"Total: TP={tp_total} FP={fp_total} FN={fn_total}")
print()

if all_tp_scores and all_fp_scores:
    print(f"TP max_score: mean={np.mean(all_tp_scores):.3f} median={np.median(all_tp_scores):.3f}")
    print(f"FP max_score: mean={np.mean(all_fp_scores):.3f} median={np.median(all_fp_scores):.3f}")
    # Can we separate TP from FP by score?
    for th in [0.55, 0.60, 0.65, 0.70, 0.75]:
        tp_kept = sum(1 for s in all_tp_scores if s > th)
        fp_removed = sum(1 for s in all_fp_scores if s <= th)
        print(f"  Score > {th}: keep {tp_kept}/{len(all_tp_scores)} TP, remove {fp_removed}/{len(all_fp_scores)} FP")

print()
if all_tp_len and all_fp_len:
    print(f"TP length: mean={np.mean(all_tp_len):.1f} median={np.median(all_tp_len):.1f} frames")
    print(f"FP length: mean={np.mean(all_fp_len):.1f} median={np.median(all_fp_len):.1f} frames")
    print(f"FN length: mean={np.mean(all_fn_len):.1f} median={np.median(all_fn_len):.1f} frames")

print()
if all_tp_ious:
    print(f"TP IoU distribution: mean={np.mean(all_tp_ious):.3f} min={np.min(all_tp_ious):.3f} max={np.max(all_tp_ious):.3f}")
    for th in [0.5, 0.7, 0.9]:
        pct = sum(1 for i in all_tp_ious if i > th) / len(all_tp_ious) * 100
        print(f"  IoU > {th}: {pct:.0f}%")

# Per-video stats
print(f"\nVideos with FP > TP: {sum(1 for v in per_video if v['fp'] > v['tp'])}/{len(per_video)}")
print(f"Videos with FN > 0:   {sum(1 for v in per_video if v['fn'] > 0)}/{len(per_video)}")

# Best threshold analysis
print(f"\nIf we filter by min_score:")
for min_s in [0.55, 0.60, 0.65, 0.70, 0.75]:
    kept_tp = sum(1 for s in all_tp_scores if s > min_s)
    kept_fp = sum(1 for s in all_fp_scores if s > min_s)
    new_tp = sum(1 for v in per_video for s in v['tp_scores'] if s > min_s)
    new_fp = sum(1 for v in per_video for s in v['fp_scores'] if s > min_s)
    new_fn = fn_total + (tp_total - new_tp)
    p = new_tp/(new_tp+new_fp+1e-6); r = new_tp/(new_tp+new_fn+1e-6)
    f1 = 2*p*r/(p+r+1e-6)
    print(f"  min_score > {min_s}: P={p:.3f} R={r:.3f} F1={f1:.3f} (kept {new_tp}/{tp_total} TP, removed {fp_total-new_fp}/{fp_total} FP)")
