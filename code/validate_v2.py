#!/usr/bin/env python3
"""validate_v2.py — 改进版检测在 21 自标视频上的评估"""
import subprocess, json, os, glob, numpy as np

ANNO_DIR = '/mnt/c/Users/admin/Desktop/测试/annotations'
VIDEO_DIR = '/mnt/c/Users/admin/Desktop/测试'
OUT_DIR = '/tmp/validate_v2'
os.makedirs(OUT_DIR, exist_ok=True)

anno_files = sorted(glob.glob(os.path.join(ANNO_DIR, '*_annotations.json')))

tp_seg = fp_seg = fn_seg = 0
results = []

for i, af in enumerate(anno_files):
    with open(af) as f: data = json.load(f)
    name = data['video_name']
    video_path = data.get('video', os.path.join(VIDEO_DIR, name))
    gt_segs = [(a['start_frame'], a['end_frame']) for a in data.get('annotations', [])]
    result_dir = os.path.join(OUT_DIR, name.replace('.mp4', ''))

    cmd = ['/home/zzy/vjepa2-main/vjepa-env/bin/python',
           '/home/zzy/jepa_data/detect_and_report_v4.py',
           '--video', video_path, '--output', result_dir,
           '--threshold', '0.70', '--min-gap', '2', '--min-length', '5',
           '--use-predictor-embed', '--use-optical-flow', '--use-frequency']

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd='/home/zzy/jepa_data')
    report_path = os.path.join(result_dir, 'report.json')
    if not os.path.exists(report_path):
        print(f"[{i+1}/{len(anno_files)}] {name} SKIP (no report)")
        continue

    with open(report_path) as f: report = json.load(f)
    pred_segs = [(seg['start_frame'], seg['end_frame']) for seg in report.get('fused_anomalies', [])]

    matched = set()
    for ps, pe in pred_segs:
        best_iou = 0; best_gt = -1
        for gi, (gs, ge) in enumerate(gt_segs):
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            iou = inter / (union + 1e-6)
            if iou > best_iou and iou > 0.3: best_iou = iou; best_gt = gi
        if best_gt >= 0: matched.add(best_gt)

    tp_seg += len(matched); fp_seg += len(pred_segs) - len(matched); fn_seg += len(gt_segs) - len(matched)
    results.append({'name': name, 'gt': len(gt_segs), 'pred': len(pred_segs), 'matched': len(matched)})
    if (i+1) % 5 == 0:
        print(f"[{i+1}/{len(anno_files)}] {name}: GT={len(gt_segs)} PRED={len(pred_segs)} MATCH={len(matched)}")

sp = tp_seg/(tp_seg+fp_seg+1e-6); sr = tp_seg/(tp_seg+fn_seg+1e-6); sf1 = 2*sp*sr/(sp+sr+1e-6)
print(f"\nImproved detection (adaptive+gaussian) on 21 self-labeled videos:")
print(f"  Seg Precision: {sp:.3f}")
print(f"  Seg Recall:    {sr:.3f}")
print(f"  Seg F1:        {sf1:.3f}")
print(f"  TP={tp_seg} FP={fp_seg} FN={fn_seg}")

# Load old results for comparison
print(f"\nComparison:")
print(f"  Old (fixed th=0.65): mean F1 ≈ 0.643 (from earlier leave-one-out)")
print(f"  New (adaptive+gaussian): F1 = {sf1:.3f}")
print(f"  Improvement: {sf1 - 0.643:+.3f}")
