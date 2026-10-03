#!/usr/bin/env python3
"""
extract_signals_v3.py — 修复版信号提取

V1/V2 问题: 全局基线(μ=0.124)让测试视频的 V-JEPA 全部饱和到 1.0
V3 修复: 每视频内部 z-score + sigmoid → 保留视频内对比信息

这样 MTCF 拿到的信号跟 composite_scoring 的预处理逻辑一致，
只是把共识融合/权重分配替换为可学习。
"""
import torch, numpy as np, cv2, json, os, sys, glob, argparse
sys.path.insert(0, '/home/zzy/jepa_data')

from detect_and_report_v4 import (
    extract_vjepa_tokens, score_vjepa_multi,
    extract_ijepa_all, score_ijepa_multi,
)


def per_video_zscore_sigmoid(values):
    """每视频内部 z-score + sigmoid → [0,1]"""
    mu, std = values.mean(), values.std()
    if std < 1e-6:
        return np.full_like(values, 0.5)
    z = (values - mu) / std
    return 1.0 / (1.0 + np.exp(-z))


class SignalExtractorV3:
    def __init__(self, use_depth=True, use_clip=True):
        self.use_depth = use_depth
        self.use_clip = use_clip
        self._depth_scorer = None
        self._clip_scorer = None

    def extract(self, video_path, annotations_data, max_frames=64, max_keyframes=32):
        total_frames = annotations_data.get('total_frames', 0)

        # ── V-JEPA ──
        print("    [V-JEPA]", flush=True)
        vjepa_result = extract_vjepa_tokens(video_path, use_cache=True, max_frames=max_frames)
        if vjepa_result is None:
            vjepa_processed = np.full(total_frames, 0.5)
        else:
            if total_frames == 0:
                total_frames = vjepa_result.get('total_frames', 1)
            ttf = vjepa_result['tubelet_to_frames']
            physics = score_vjepa_multi(vjepa_result['vjepa_tokens'], ttf, total_frames, predictor_embed=None)
            vjepa_raw = np.zeros(total_frames)
            for t, (s, e) in enumerate(ttf):
                if t < len(physics):
                    s, e = max(0, s), min(total_frames - 1, e)
                    vjepa_raw[s:e+1] = physics[t]
            vjepa_processed = per_video_zscore_sigmoid(vjepa_raw)

        # ── I-JEPA ──
        print("    [I-JEPA]", flush=True)
        ijepa_result = extract_ijepa_all(video_path, use_cache=True, max_keyframes=max_keyframes)
        if ijepa_result is None or ijepa_result.get('ijepa_patch') is None:
            ijepa_processed = np.full(total_frames, 0.5)
        else:
            ijepa_patch = ijepa_result['ijepa_patch']
            ijepa_indices = ijepa_result.get('ijepa_frame_indices',
                np.linspace(0, total_frames-1, len(ijepa_patch)).astype(int))
            corruption = score_ijepa_multi(ijepa_patch)
            if isinstance(corruption, np.ndarray) and corruption.ndim > 1:
                corruption = corruption.flatten()
            ijepa_raw = np.zeros(total_frames)
            for k, fid in enumerate(ijepa_indices):
                if k < len(corruption) and 0 <= fid < total_frames:
                    ijepa_raw[fid] = corruption[k]
            ijepa_processed = per_video_zscore_sigmoid(ijepa_raw)

        # ── Flow ──
        print("    [Flow]", flush=True)
        try:
            from optical_flow_scorer import OpticalFlowScorer
            of = OpticalFlowScorer(scale=0.5)
            raw_f, _ = of.score_video(video_path, 0, total_frames-1, step=1)
            flow_raw = np.interp(np.arange(total_frames), np.linspace(0, total_frames-1, len(raw_f)), raw_f)
            flow_processed = per_video_zscore_sigmoid(flow_raw)
        except Exception as e:
            print(f"    [Flow] FAILED: {e}", flush=True)
            flow_processed = np.full(total_frames, 0.5)

        # ── Freq ──
        print("    [Freq]", flush=True)
        try:
            from frequency_scorer import FrequencyScorer
            fq = FrequencyScorer(scale=0.5, block_size=8)
            raw_f, _ = fq.score_video(video_path, 0, total_frames-1, step=2)
            freq_raw = np.interp(np.arange(total_frames), np.linspace(0, total_frames-1, len(raw_f)), raw_f)
            freq_processed = per_video_zscore_sigmoid(freq_raw)
        except Exception as e:
            print(f"    [Freq] FAILED: {e}", flush=True)
            freq_processed = np.full(total_frames, 0.5)

        # ── Depth ──
        depth_processed = np.full(total_frames, 0.5)
        if self.use_depth:
            try:
                print("    [Depth]", flush=True)
                if self._depth_scorer is None:
                    from depth_consistency_scorer import DepthConsistencyScorer
                    self._depth_scorer = DepthConsistencyScorer()
                kf = self._extract_keyframes(video_path, max_keyframes)
                raw_d = self._depth_scorer.score_video(kf, list(range(len(kf))))
                depth_raw = np.interp(np.arange(total_frames), np.linspace(0, total_frames-1, len(raw_d)), raw_d)
                depth_processed = per_video_zscore_sigmoid(depth_raw)
            except Exception as e:
                print(f"    [Depth] FAILED: {e}", flush=True)

        # ── CLIP ──
        clip_processed = np.full(total_frames, 0.5)
        if self.use_clip:
            try:
                print("    [CLIP]", flush=True)
                if self._clip_scorer is None:
                    from clip_semantic_scorer import CLIPSemanticScorer
                    self._clip_scorer = CLIPSemanticScorer()
                kf = self._extract_keyframes(video_path, max_keyframes)
                raw_c = self._clip_scorer.score_video(kf, list(range(len(kf))))
                clip_raw = np.interp(np.arange(total_frames), np.linspace(0, total_frames-1, len(raw_c)), raw_c)
                clip_processed = per_video_zscore_sigmoid(clip_raw)
            except Exception as e:
                print(f"    [CLIP] FAILED: {e}", flush=True)

        # Assemble
        signals = np.stack([vjepa_processed, ijepa_processed, flow_processed,
                            freq_processed, depth_processed, clip_processed], axis=1).astype(np.float32)

        # Labels
        labels = np.zeros(total_frames, dtype=np.int64)
        categories = np.zeros(total_frames, dtype=np.int64)
        cat_map = {'physics': 1, 'visual': 2, 'other': 3}
        for ann in annotations_data.get('annotations', []):
            sf = max(0, ann['start_frame'])
            ef = min(total_frames - 1, ann['end_frame'])
            cat = cat_map.get(ann.get('category', ''), 3)
            for f in range(sf, ef + 1):
                labels[f] = 1
                categories[f] = cat

        return signals, labels, categories

    def _extract_keyframes(self, video_path, max_kf):
        cap = cv2.VideoCapture(video_path)
        frames = []; idx = 0
        while cap.isOpened() and len(frames) < max_kf:
            ret, frame = cap.read()
            if not ret: break
            if idx % 2 == 0: frames.append(frame)
            idx += 1
        cap.release()
        return np.array(frames) if frames else np.zeros((1, 224, 224, 3), dtype=np.uint8)


def build_dataset(annotations_dir, output_dir, use_depth=True, use_clip=True):
    os.makedirs(output_dir, exist_ok=True)
    anno_files = sorted(glob.glob(os.path.join(annotations_dir, "*_annotations.json")))
    print(f"Found {len(anno_files)} annotated videos")

    extractor = SignalExtractorV3(use_depth=use_depth, use_clip=use_clip)
    all_signals = []; all_labels = []; all_categories = []; video_names = []

    for i, af in enumerate(anno_files):
        with open(af) as f: data = json.load(f)
        name = data.get('video_name', os.path.basename(af))
        video_path = data.get('video_path', '')
        if not os.path.exists(video_path):
            print(f"  [{i+1}/{len(anno_files)}] SKIP {name}")
            continue

        n_ann = len(data.get('annotations', []))
        print(f"  [{i+1}/{len(anno_files)}] {name} ({n_ann} segments)")

        try:
            signals, labels, categories = extractor.extract(video_path, data)
            all_signals.append(signals)
            all_labels.append(labels)
            all_categories.append(categories)
            video_names.append(name)
            print(f"    → [{signals.shape[0]} frames, {labels.sum()} anomalous, "
                  f"V range=[{signals[:,0].min():.3f},{signals[:,0].max():.3f}]]")
        except Exception as e:
            import traceback
            print(f"    → FAILED: {e}")
            traceback.print_exc()

    if not all_signals:
        print("ERROR: No signals!")
        return

    np.savez_compressed(os.path.join(output_dir, "signals.npz"), *all_signals, allow_pickle=True)
    np.savez_compressed(os.path.join(output_dir, "labels.npz"), *all_labels, allow_pickle=True)
    np.savez_compressed(os.path.join(output_dir, "categories.npz"), *all_categories, allow_pickle=True)
    with open(os.path.join(output_dir, "video_names.json"), 'w') as f:
        json.dump(video_names, f, indent=2)

    total_frames = sum(len(l) for l in all_labels)
    total_anom = sum(l.sum() for l in all_labels)
    all_cat = np.concatenate(all_categories)
    print(f"\nDataset: {len(all_signals)} videos, {total_frames} frames")
    print(f"  Anomalous: {total_anom} ({100*total_anom/max(1,total_frames):.1f}%)")
    print(f"  Categories: physics={(all_cat==1).sum()}, visual={(all_cat==2).sum()}, other={(all_cat==3).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="/home/zzy/jepa_data/clean_dataset/annotations")
    parser.add_argument("--output", default="/home/zzy/jepa_data/training_data_v3")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()
    build_dataset(args.annotations, args.output,
                  use_depth=not args.no_depth, use_clip=not args.no_clip)
