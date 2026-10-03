#!/usr/bin/env python3
"""
extract_signals.py — 从标注视频提取 6 路帧级信号用于 MTCF 训练

与 detect_and_report_v4 主流程完全一致的调用方式，确保信号语义匹配。
"""
import torch
import numpy as np
import cv2
import json, os, sys, glob, argparse

sys.path.insert(0, '/home/zzy/jepa_data')


class SignalExtractor:
    """复用 detect_and_report_v4 的特征提取，直接返回 6 路原始帧级信号"""

    def __init__(self, use_depth=True, use_clip=True):
        self.use_depth = use_depth
        self.use_clip = use_clip
        self._depth_scorer = None
        self._clip_scorer = None

    def extract(self, video_path, annotations_data, max_frames=64, max_keyframes=32):
        from detect_and_report_v4 import (
            extract_vjepa_tokens, score_vjepa_multi,
            extract_ijepa_all, score_ijepa_multi,
        )

        # Step 1: V-JEPA
        print("    [V-JEPA]")
        vjepa_result = extract_vjepa_tokens(video_path, use_cache=True, max_frames=max_frames)
        if vjepa_result is None:
            raise RuntimeError("V-JEPA extraction failed")

        total_frames = vjepa_result.get("total_frames", 1)
        tubelet_to_frames = vjepa_result["tubelet_to_frames"]
        vjepa_tokens = vjepa_result["vjepa_tokens"]

        # Use same scoring as main pipeline (without predictor_embed for speed)
        physics_scores = score_vjepa_multi(vjepa_tokens, tubelet_to_frames, total_frames,
                                           predictor_embed=None)

        # Upsample V-JEPA tubelet scores to frame level
        vjepa_frame = np.zeros(total_frames)
        for t, (s, e) in enumerate(tubelet_to_frames):
            if t < len(physics_scores):
                s, e = max(0, s), min(total_frames - 1, e)
                vjepa_frame[s:e+1] = physics_scores[t]

        # Step 2: I-JEPA
        print("    [I-JEPA]")
        ijepa_result = extract_ijepa_all(video_path, use_cache=True, max_keyframes=max_keyframes)
        if ijepa_result is None or ijepa_result.get("ijepa_patch") is None:
            ijepa_frame = np.zeros(total_frames)
        else:
            ijepa_patch = ijepa_result["ijepa_patch"]
            ijepa_indices = ijepa_result.get("ijepa_frame_indices",
                                              np.linspace(0, total_frames-1,
                                                          len(ijepa_patch) if hasattr(ijepa_patch,'__len__') else 1).astype(int))
            corruption_scores = score_ijepa_multi(ijepa_patch, vjepa_attention=None)

            # Flatten if 2D
            if isinstance(corruption_scores, np.ndarray) and corruption_scores.ndim > 1:
                corruption_scores = corruption_scores.flatten()

            ijepa_frame = np.zeros(total_frames)
            for k, fid in enumerate(ijepa_indices):
                if k < len(corruption_scores) and 0 <= fid < total_frames:
                    ijepa_frame[fid] = corruption_scores[k]

        # Step 3: Optical Flow
        flow_scores = np.zeros(total_frames)
        try:
            print("    [Flow]")
            from optical_flow_scorer import OpticalFlowScorer
            of = OpticalFlowScorer(scale=0.5)
            raw_flow, _ = of.score_video(video_path, start_frame=0, end_frame=total_frames-1, step=1)
            flow_scores = np.interp(np.arange(total_frames),
                                    np.linspace(0, total_frames-1, len(raw_flow)), raw_flow)
        except Exception as e:
            print(f"    [Flow] FAILED: {e}")

        # Step 4: Frequency
        freq_scores = np.zeros(total_frames)
        try:
            print("    [Freq]")
            from frequency_scorer import FrequencyScorer
            fq = FrequencyScorer(scale=0.5, block_size=8)
            raw_freq, _ = fq.score_video(video_path, start_frame=0, end_frame=total_frames-1, step=2)
            freq_scores = np.interp(np.arange(total_frames),
                                    np.linspace(0, total_frames-1, len(raw_freq)), raw_freq)
        except Exception as e:
            print(f"    [Freq] FAILED: {e}")

        # Step 5: Depth
        depth_scores = np.zeros(total_frames)
        if self.use_depth:
            try:
                print("    [Depth]")
                if self._depth_scorer is None:
                    from depth_consistency_scorer import DepthConsistencyScorer
                    self._depth_scorer = DepthConsistencyScorer()
                keyframes = self._extract_keyframes(video_path, max_keyframes)
                raw_d = self._depth_scorer.score_video(keyframes, list(range(len(keyframes))))
                depth_scores = np.interp(np.arange(total_frames),
                                         np.linspace(0, total_frames-1, len(raw_d)), raw_d)
            except Exception as e:
                print(f"    [Depth] FAILED: {e}, using zeros")

        # Step 6: CLIP
        clip_scores = np.zeros(total_frames)
        if self.use_clip:
            try:
                print("    [CLIP]")
                if self._clip_scorer is None:
                    from clip_semantic_scorer import CLIPSemanticScorer
                    self._clip_scorer = CLIPSemanticScorer()
                keyframes = self._extract_keyframes(video_path, max_keyframes)
                raw_c = self._clip_scorer.score_video(keyframes, list(range(len(keyframes))))
                clip_scores = np.interp(np.arange(total_frames),
                                        np.linspace(0, total_frames-1, len(raw_c)), raw_c)
            except Exception as e:
                print(f"    [CLIP] FAILED: {e}, using zeros")

        # Assemble [T, 6]
        signals = np.stack([vjepa_frame, ijepa_frame, flow_scores,
                            freq_scores, depth_scores, clip_scores], axis=1).astype(np.float32)

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
        frames = []
        idx = 0
        while cap.isOpened() and len(frames) < max_kf:
            ret, frame = cap.read()
            if not ret: break
            if idx % 2 == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        return np.array(frames) if frames else np.zeros((1, 224, 224, 3), dtype=np.uint8)


def build_dataset(annotations_dir, videos_dir, output_dir, use_depth=True, use_clip=True):
    os.makedirs(output_dir, exist_ok=True)
    anno_files = sorted(glob.glob(os.path.join(annotations_dir, "*_annotations.json")))
    print(f"Found {len(anno_files)} annotated videos")

    extractor = SignalExtractor(use_depth=use_depth, use_clip=use_clip)
    all_signals = []; all_labels = []; all_categories = []; video_names = []

    for i, af in enumerate(anno_files):
        with open(af) as f:
            data = json.load(f)
        name = data.get('video_name', os.path.basename(af))
        video_path = data.get('video_path', os.path.join(videos_dir, name))
        if not os.path.exists(video_path):
            print(f"  [{i+1}/{len(anno_files)}] SKIP {name} (video not found)")
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
                  f"cat: {np.bincount(categories[categories>0], minlength=4)[1:]}]")
        except Exception as e:
            print(f"    → FAILED: {e}")
            continue

    if not all_signals:
        print("ERROR: No signals extracted!")
        return

    # Save as single arrays (padded to same length for batching)
    np.savez_compressed(os.path.join(output_dir, "signals.npz"),
                        *all_signals, allow_pickle=True)
    np.savez_compressed(os.path.join(output_dir, "labels.npz"),
                        *all_labels, allow_pickle=True)
    np.savez_compressed(os.path.join(output_dir, "categories.npz"),
                        *all_categories, allow_pickle=True)
    with open(os.path.join(output_dir, "video_names.json"), 'w') as f:
        json.dump(video_names, f, indent=2)

    total_frames = sum(len(l) for l in all_labels)
    total_anom = sum(l.sum() for l in all_labels)
    all_cat = np.concatenate(all_categories)
    print(f"\n{'='*50}")
    print(f"Dataset: {len(all_signals)} videos, {total_frames} frames")
    print(f"  Anomalous: {total_anom} ({100*total_anom/max(1,total_frames):.1f}%)")
    print(f"  Categories: physics={(all_cat==1).sum()}, visual={(all_cat==2).sum()}, other={(all_cat==3).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="/home/zzy/jepa_data/clean_dataset/annotations")
    parser.add_argument("--videos", default="")
    parser.add_argument("--output", default="/home/zzy/jepa_data/training_data")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()
    build_dataset(args.annotations, args.videos, args.output,
                  use_depth=not args.no_depth, use_clip=not args.no_clip)
