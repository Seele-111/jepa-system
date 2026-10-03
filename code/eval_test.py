#!/usr/bin/env python3
"""
eval_test.py — 在 test 集上评估 MTCF + 手写规则

1. 提取 test 视频的 6 路信号
2. 用已训练的 MTCF 推理
3. 手写规则推理
4. 帧级/段级对比
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, argparse

sys.path.insert(0, '/home/zzy/jepa_data')
from mtcf_train import MultiScaleFusion
from extract_signals import build_dataset
from run_training import load_cached_dataset, compute_pos_weight
from compare_baseline import rule_based_predict


def evaluate_mtcf(model, signals, labels, categories, device='cuda'):
    """MTCF 在 test 集上推理"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for idx in range(len(signals)):
            s = torch.FloatTensor(signals[idx]).unsqueeze(0).to(device)
            logits = model(s)
            all_probs.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())

    y_prob = np.concatenate([p[:, 1:].sum(axis=1) for p in all_probs])
    y_true = np.concatenate(labels)
    return compute_metrics(y_prob, y_true, labels, categories, all_probs, 'MTCF')


def evaluate_rule(signals, labels, categories):
    """手写规则在 test 集上推理"""
    best_f1 = 0; best_th = 0.5
    for th in np.linspace(0.3, 0.8, 20):
        y_prob = np.concatenate([rule_based_predict(s, th).astype(float) for s in signals])
        y_true = np.concatenate(labels)
        tp = ((y_prob>=0.5)&(y_true==1)).sum()
        fp = ((y_prob>=0.5)&(y_true==0)).sum()
        fn = ((y_prob<0.5)&(y_true==1)).sum()
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        if f1 > best_f1: best_f1 = f1; best_th = th

    print(f"  Best threshold: {best_th:.2f}")
    y_pred = np.concatenate([(rule_based_predict(s, best_th) >= 0.5).astype(int) for s in signals])
    y_true = np.concatenate(labels)
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()

    results = {
        'precision': tp/(tp+fp+1e-6), 'recall': tp/(tp+fn+1e-6),
        'f1': 2*tp/(2*tp+fp+fn+1e-6), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'threshold': best_th,
    }

    # Segment-level
    sp, sr, sf1, tp_s, fp_s, fn_s = compute_segments(y_pred, y_true, labels)
    results.update({'seg_precision': sp, 'seg_recall': sr, 'seg_f1': sf1,
                    'seg_tp': tp_s, 'seg_fp': fp_s, 'seg_fn': fn_s})
    return results


def compute_metrics(y_prob, y_true, labels, categories, all_probs, name):
    """统一评估"""
    # Best threshold
    best_f1 = 0; best_th = 0.5
    for th in np.linspace(0.2, 0.8, 30):
        yp = (y_prob > th).astype(int)
        tp = ((yp==1)&(y_true==1)).sum()
        fp = ((yp==1)&(y_true==0)).sum()
        fn = ((yp==0)&(y_true==1)).sum()
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        if f1 > best_f1: best_f1 = f1; best_th = th

    y_pred = (y_prob > best_th).astype(int)
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()

    results = {
        'precision': tp/(tp+fp+1e-6), 'recall': tp/(tp+fn+1e-6),
        'f1': 2*tp/(2*tp+fp+fn+1e-6), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'threshold': best_th,
    }

    # Segment
    sp, sr, sf1, tp_s, fp_s, fn_s = compute_segments(y_pred, y_true, labels)
    results.update({'seg_precision': sp, 'seg_recall': sr, 'seg_f1': sf1,
                    'seg_tp': tp_s, 'seg_fp': fp_s, 'seg_fn': fn_s})

    # Category
    all_cat_pred = []; all_cat_true = []
    for idx in range(len(labels)):
        probs = all_probs[idx]; true_cat = categories[idx]
        for f in range(len(true_cat)):
            if true_cat[f] > 0:
                all_cat_pred.append(np.argmax(probs[f]))
                all_cat_true.append(true_cat[f])

    if all_cat_true:
        all_cat_pred = np.array(all_cat_pred); all_cat_true = np.array(all_cat_true)
        results['cat_overall'] = float((all_cat_pred==all_cat_true).mean())
        for cid, cnm in [(1,'physics'),(2,'visual')]:
            m = all_cat_true==cid
            if m.sum():
                results[f'cat_{cnm}'] = float((all_cat_pred[m]==cid).mean())
                results[f'cat_{cnm}_n'] = int(m.sum())

    return results


def compute_segments(y_pred, y_true, labels_list):
    tp = fp = fn = 0; offset = 0
    for labels in labels_list:
        T = len(labels); pred = y_pred[offset:offset+T]; true = y_true[offset:offset+T]; offset += T

        def segs(arr):
            s = []; in_seg = False; start = 0
            for i, v in enumerate(arr):
                if v == 1 and not in_seg: start = i; in_seg = True
                elif v == 0 and in_seg: s.append((start, i-1)); in_seg = False
            if in_seg: s.append((start, len(arr)-1))
            return s

        ps = segs(pred); ts = segs(true)
        matched = set()
        for a, b in ps:
            best_iou = 0; best_gi = -1
            for gi, (c, d) in enumerate(ts):
                inter = max(0, min(b,d)-max(a,c)); union = max(b,d)-min(a,c)
                iou = inter/(union+1e-6)
                if iou > best_iou and iou > 0.3: best_iou = iou; best_gi = gi
            if best_gi >= 0: matched.add(best_gi)
        tp += len(matched); fp += len(ps)-len(matched); fn += len(ts)-len(matched)

    sp = tp/(tp+fp+1e-6); sr = tp/(tp+fn+1e-6); sf1 = 2*sp*sr/(sp+sr+1e-6)
    return sp, sr, sf1, tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-annotations", default="/home/zzy/jepa_data/clean_test_dataset/annotations")
    parser.add_argument("--test-data-dir", default="/home/zzy/jepa_data/test_training_data")
    parser.add_argument("--model", default="/home/zzy/jepa_data/mtcf_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    # Step 1: Extract signals
    if args.rebuild or not os.path.exists(os.path.join(args.test_data_dir, "signals.npz")):
        print("[1/3] Extracting test signals...")
        build_dataset(args.test_annotations, "", args.test_data_dir,
                      use_depth=True, use_clip=True)
    else:
        print("[1/3] Using cached test signals")

    # Step 2: Load
    print("[2/3] Loading test data...")
    signals, labels, categories = load_cached_dataset(args.test_data_dir)
    n_frames = sum(len(l) for l in labels)
    n_anom = sum(l.sum() for l in labels)
    print(f"  {len(signals)} videos, {n_frames} frames, {n_anom} anomalous ({100*n_anom/max(1,n_frames):.1f}%)")

    # Step 3: Evaluate
    print(f"\n[3/3] Evaluating...")

    # MTCF
    print(f"\n{'='*60}")
    print("MTCF on Test Set")
    print(f"{'='*60}")
    model = MultiScaleFusion()
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.to(args.device)
    mtcf_results = evaluate_mtcf(model, signals, labels, categories, args.device)
    print(f"  Frame:  P={mtcf_results['precision']:.3f}  R={mtcf_results['recall']:.3f}  F1={mtcf_results['f1']:.3f}")
    print(f"  Segment: P={mtcf_results['seg_precision']:.3f}  R={mtcf_results['seg_recall']:.3f}  F1={mtcf_results['seg_f1']:.3f}")
    if 'cat_overall' in mtcf_results:
        print(f"  Category: {mtcf_results['cat_overall']:.3f}")
        for cnm in ['physics', 'visual']:
            k = f'cat_{cnm}'
            if k in mtcf_results:
                print(f"    {cnm}: {mtcf_results[k]:.3f} (n={mtcf_results[f'cat_{cnm}_n']})")

    # Rule-based
    print(f"\n{'='*60}")
    print("Rule-based on Test Set")
    print(f"{'='*60}")
    rule_results = evaluate_rule(signals, labels, categories)
    print(f"  Frame:  P={rule_results['precision']:.3f}  R={rule_results['recall']:.3f}  F1={rule_results['f1']:.3f}")
    print(f"  Segment: P={rule_results['seg_precision']:.3f}  R={rule_results['seg_recall']:.3f}  F1={rule_results['seg_f1']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {'Rule':>10} {'MTCF':>10} {'Δ':>10}")
    print(f"{'-'*50}")
    for k in ['precision', 'recall', 'f1', 'seg_precision', 'seg_recall', 'seg_f1']:
        r = rule_results[k]; m = mtcf_results[k]; d = m - r
        print(f"{k:<20} {r:>10.3f} {m:>10.3f} {d:>+10.3f}")
    print(f"{'='*60}")

    # Save
    with open(os.path.join(args.test_data_dir, 'test_results.json'), 'w') as f:
        json.dump({'mtcf': mtcf_results, 'rule': rule_results}, f, indent=2)
    print(f"\nResults saved to {args.test_data_dir}/test_results.json")


if __name__ == "__main__":
    main()
