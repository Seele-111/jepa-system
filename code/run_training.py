#!/usr/bin/env python3
"""
run_training.py — MTCF 完整训练+评估（对接清洗数据集）

用法:
  python run_training.py
  python run_training.py --annotations /home/zzy/jepa_data/clean_dataset/annotations \
                         --data-dir /home/zzy/jepa_data/training_data --rebuild
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, argparse, glob
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, '/home/zzy/jepa_data')
from mtcf_train import MultiScaleFusion, segment_aware_loss


def load_cached_dataset(data_dir):
    signals = [np.array(s) for s in np.load(os.path.join(data_dir, "signals.npz"),
                                              allow_pickle=True).values()]
    labels = [np.array(l) for l in np.load(os.path.join(data_dir, "labels.npz"),
                                            allow_pickle=True).values()]
    categories = [np.array(c) for c in np.load(os.path.join(data_dir, "categories.npz"),
                                                 allow_pickle=True).values()]
    return signals, labels, categories


def run_true_jepa_training(args):
    """Run the current binary JEPA-Loc training path.

    This path is intentionally separate from the legacy MTCF/category path:
    extract_true_jepa_segment_signals.py creates [T, 3] true-JEPA signals,
    build_jepa_event_dataset.py expands them to event tokens, and
    train_segment_locator.py trains the binary temporal locator.
    """
    from build_jepa_event_dataset import build_event_dataset
    from extract_true_jepa_segment_signals import build_dataset as build_true_jepa_dataset
    from train_segment_locator import (
        load_signal_dataset,
        save_checkpoint,
        train_model as train_binary_locator,
        train_val_split as binary_train_val_split,
    )

    raw_dir = Path(args.true_jepa_raw_dir or f"{args.data_dir}_raw_true_jepa")
    event_dir = Path(args.data_dir)

    if args.rebuild or not (raw_dir / "signals.npz").exists():
        print("[1/4] Extracting true V/I-JEPA scalar signals...")
        extract_args = SimpleNamespace(
            annotations=args.annotations,
            output=str(raw_dir),
            limit=args.limit,
            max_frames=args.max_frames,
            max_keyframes=args.max_keyframes,
            timeout=args.timeout,
            rebuild=args.rebuild,
            keep_runs=args.keep_runs,
        )
        rc = build_true_jepa_dataset(extract_args)
        if rc != 0:
            raise RuntimeError(f"true-JEPA signal extraction failed with exit code {rc}")
    else:
        print(f"[1/4] Using cached true-JEPA scalar signals: {raw_dir}")

    if args.rebuild or not (event_dir / "signals.npz").exists():
        print("[2/4] Building multi-scale event-token dataset...")
        windows = [int(item) for item in args.event_windows.split(",") if item.strip()]
        summary = build_event_dataset(raw_dir, event_dir, windows)
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(f"[2/4] Using cached event-token dataset: {event_dir}")

    print("[3/4] Loading event-token records...")
    records = load_signal_dataset(event_dir)
    n_frames = sum(len(record.labels) for record in records)
    n_pos = sum(int(record.labels.sum()) for record in records)
    print(f"  {len(records)} videos, {n_frames} frames, {n_pos} positives ({100*n_pos/max(1,n_frames):.1f}%)")

    train_idx, val_idx = binary_train_val_split(records, args.val_ratio, args.seed)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    print(f"\n[4/4] Training binary JEPA-Loc temporal locator...")
    model, metrics, mean, std = train_binary_locator(
        records,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        lambda_dice=args.lambda_dice,
        lambda_boundary=args.lambda_boundary,
        architecture=args.architecture,
    )
    save_checkpoint(args.output, model, mean, std, train_idx, val_idx, metrics, args)
    print(f"  Model saved to {args.output}")
    print(json.dumps({"validation": metrics}, indent=2), flush=True)
    return metrics


def train_test_split(signals, labels, categories, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    n = len(signals)
    has_anom = np.array([l.sum() > 0 for l in labels])
    anom_idx = np.where(has_anom)[0]
    norm_idx = np.where(~has_anom)[0]
    np.random.shuffle(anom_idx); np.random.shuffle(norm_idx)
    n_val_anom = max(1, int(len(anom_idx) * val_ratio))
    n_val_norm = max(1, int(len(norm_idx) * val_ratio)) if len(norm_idx) > 0 else 0
    val_idx = np.concatenate([anom_idx[:n_val_anom], norm_idx[:n_val_norm]])
    train_idx = np.array([i for i in range(n) if i not in val_idx])
    return train_idx, val_idx


def compute_pos_weight(labels):
    n_pos = sum(l.sum() for l in labels)
    n_neg = sum(len(l) - l.sum() for l in labels)
    return n_neg / max(1, n_pos)


def compute_cat_weights(categories, labels):
    """全局类别权重: 1/freq，clamp到[0.2, 5.0]"""
    counts = np.zeros(4, dtype=np.float64)
    for c, l in zip(categories, labels):
        for cls_id in range(1, 4):
            counts[cls_id] += ((c == cls_id) & (l == 1)).sum()

    # Inverse frequency, normalized
    weights = np.ones(4, dtype=np.float64)
    for i in range(1, 4):
        if counts[i] > 0:
            weights[i] = counts.sum() / (3 * counts[i])
            weights[i] = np.clip(weights[i], 0.2, 5.0)

    print(f"  Cat weights: physics={weights[1]:.2f}, visual={weights[2]:.2f}, other={weights[3]:.2f}")
    return torch.FloatTensor(weights)


def train_model(signals, labels, categories, train_idx, val_idx,
                pos_weight=1.0, cat_weights=None, epochs=300, lr=1e-3,
                device='cuda', patience=50):
    model = MultiScaleFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0; best_state = None; no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        np.random.shuffle(train_idx)
        for idx in train_idx:
            s = torch.FloatTensor(signals[idx]).unsqueeze(0).to(device)
            l = torch.LongTensor(labels[idx]).unsqueeze(0).to(device)
            c = torch.LongTensor(categories[idx]).unsqueeze(0).to(device)
            mask = (l == 1)
            optimizer.zero_grad()
            logits = model(s)
            loss = segment_aware_loss(logits, l, c, cat_weights=cat_weights,
                                      lambda_smooth=0.15, lambda_cat=0.5, lambda_seg=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step(); train_loss /= len(train_idx)

        model.eval()
        all_probs = []; all_labs = []
        with torch.no_grad():
            for idx in val_idx:
                s = torch.FloatTensor(signals[idx]).unsqueeze(0).to(device)
                l = torch.LongTensor(labels[idx]).unsqueeze(0).to(device)
                logits = model(s)
                all_probs.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())
                all_labs.append(labels[idx])

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

        if epoch % 30 == 0 or epoch < 5 or no_improve >= patience:
            print(f"Epoch {epoch:3d}  loss={train_loss:.4f}  F1={f1:.3f}  best={best_val_f1:.3f}")
        if no_improve >= patience: break

    model.load_state_dict(best_state)
    return model, best_val_f1


def evaluate(model, signals, labels, categories, device='cuda'):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for idx in range(len(signals)):
            s = torch.FloatTensor(signals[idx]).unsqueeze(0).to(device)
            logits = model(s)
            all_probs.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())

    y_prob = np.concatenate([p[:, 1:].sum(axis=1) for p in all_probs])
    y_true = np.concatenate(labels)

    # Best threshold search
    best_f1 = 0; best_th = 0.5; best = {}
    for th in np.linspace(0.3, 0.8, 20):
        yp = (y_prob > th).astype(int)
        tp = ((yp==1)&(y_true==1)).sum()
        fp = ((yp==1)&(y_true==0)).sum()
        fn = ((yp==0)&(y_true==1)).sum()
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        if f1 > best_f1: best_f1 = f1; best_th = th; best = {'precision':prec,'recall':rec,'f1':f1,'tp':int(tp),'fp':int(fp),'fn':int(fn)}

    print(f"\n{'='*60}")
    print(f"Frame-level (threshold={best_th:.2f})")
    print(f"{'='*60}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall:    {best['recall']:.3f}")
    print(f"  F1:        {best['f1']:.3f}")
    print(f"  TP={best['tp']}  FP={best['fp']}  FN={best['fn']}")

    # Segment-level
    yp = (y_prob > best_th).astype(int)
    tp_seg = fp_seg = fn_seg = 0
    offset = 0
    for idx in range(len(signals)):
        T = len(labels[idx]); pred = yp[offset:offset+T]; true = y_true[offset:offset+T]; offset += T

        def find_segs(arr):
            segs = []; in_seg = False; start = 0
            for i, v in enumerate(arr):
                if v == 1 and not in_seg: start = i; in_seg = True
                elif v == 0 and in_seg: segs.append((start, i-1)); in_seg = False
            if in_seg: segs.append((start, len(arr)-1))
            return segs

        pred_segs = find_segs(pred); true_segs = find_segs(true)
        matched = set()
        for ps, pe in pred_segs:
            best_iou = 0; best_gi = -1
            for gi, (gs, ge) in enumerate(true_segs):
                inter = max(0, min(pe, ge) - max(ps, gs))
                union = max(pe, ge) - min(ps, gs)
                iou = inter/(union+1e-6)
                if iou > best_iou and iou > 0.3: best_iou = iou; best_gi = gi
            if best_gi >= 0: matched.add(best_gi)
        tp_seg += len(matched); fp_seg += len(pred_segs)-len(matched); fn_seg += len(true_segs)-len(matched)

    sp = tp_seg/(tp_seg+fp_seg+1e-6); sr = tp_seg/(tp_seg+fn_seg+1e-6); sf1 = 2*sp*sr/(sp+sr+1e-6)
    print(f"\nSegment-level (IoU>0.3)")
    print(f"  Precision: {sp:.3f}  Recall: {sr:.3f}  F1: {sf1:.3f}")

    # Category accuracy
    all_cat_pred = []; all_cat_true = []
    for idx in range(len(signals)):
        probs = all_probs[idx]; true_cat = categories[idx]
        for f in range(len(true_cat)):
            if true_cat[f] > 0:
                all_cat_pred.append(np.argmax(probs[f]))
                all_cat_true.append(true_cat[f])

    if all_cat_true:
        all_cat_pred = np.array(all_cat_pred); all_cat_true = np.array(all_cat_true)
        acc = (all_cat_pred==all_cat_true).mean()
        print(f"\nCategory Classification (anomalous frames only)")
        print(f"  Overall: {acc:.3f}")
        for cid, cnm in [(1,'physics'),(2,'visual'),(3,'other')]:
            m = all_cat_true==cid
            if m.sum(): print(f"  {cnm}: {(all_cat_pred[m]==cid).mean():.3f} (n={m.sum()})")

    return best, sf1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="/home/zzy/jepa_data/clean_dataset/annotations")
    parser.add_argument("--videos", default="")
    parser.add_argument("--data-dir", default="/home/zzy/jepa_data/training_data")
    parser.add_argument("--output", default="/home/zzy/jepa_data/mtcf_best.pt")
    parser.add_argument(
        "--true-jepa",
        action="store_true",
        help="Use the current binary JEPA-Loc pipeline: extract_true_jepa_segment_signals -> build_jepa_event_dataset -> train_segment_locator.",
    )
    parser.add_argument("--true-jepa-raw-dir", default="", help="Cache/output dir for raw [T,3] true-JEPA scalar signals.")
    parser.add_argument("--event-windows", default="3,5,9,15", help="Temporal windows for build_jepa_event_dataset.py.")
    parser.add_argument("--limit", type=int, default=None, help="Optional annotation limit for true-JEPA extraction.")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--architecture", choices=["tcn", "attn_tcn", "bilstm"], default="tcn")
    parser.add_argument("--lambda-dice", type=float, default=0.0)
    parser.add_argument("--lambda-boundary", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--pos-weight", type=float, default=None, help="BCE pos_weight (auto if not set)")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    if args.true_jepa:
        return run_true_jepa_training(args)

    # Step 1: Extract signals
    if args.rebuild or not os.path.exists(os.path.join(args.data_dir, "signals.npz")):
        print("[1/4] Extracting 6-channel signals...")
        try:
            from extract_signals import build_dataset as build_legacy_dataset
        except ImportError as exc:
            raise ImportError(
                "Legacy MTCF mode requires extract_signals.py. "
                "For the current JEPA-Loc binary pipeline, rerun with --true-jepa."
            ) from exc
        build_legacy_dataset(args.annotations, args.videos, args.data_dir,
                             use_depth=not args.no_depth, use_clip=not args.no_clip)
    else:
        print("[1/4] Using cached signals")

    # Step 2: Load
    print("[2/4] Loading dataset...")
    signals, labels, categories = load_cached_dataset(args.data_dir)
    n_frames = sum(len(l) for l in labels)
    n_anom = sum(l.sum() for l in labels)
    print(f"  {len(signals)} videos, {n_frames} frames, {n_anom} anomalous ({100*n_anom/max(1,n_frames):.1f}%)")

    # pos_weight + cat_weights
    pos_weight = compute_pos_weight(labels) if args.pos_weight is None else args.pos_weight
    cat_weights = compute_cat_weights(categories, labels)
    print(f"  pos_weight = {pos_weight:.2f}")

    # Step 3: Split + Train
    train_idx, val_idx = train_test_split(signals, labels, categories, args.val_ratio)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    print(f"\n[3/4] Training MTCF ({sum(p.numel() for p in MultiScaleFusion().parameters()):,} params)...")
    model, val_f1 = train_model(signals, labels, categories, train_idx, val_idx,
                                pos_weight=pos_weight, cat_weights=cat_weights,
                                epochs=args.epochs, lr=args.lr,
                                device=args.device, patience=args.patience)
    print(f"  Best val F1: {val_f1:.3f}")

    torch.save(model.state_dict(), args.output)
    print(f"  Model saved to {args.output}")

    # Step 4: Evaluate on all data
    print(f"\n[4/4] Evaluation...")
    evaluate(model, signals, labels, categories, device=args.device)


if __name__ == "__main__":
    main()
