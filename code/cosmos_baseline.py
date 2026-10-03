#!/usr/bin/env python3
"""
cosmos_baseline.py — Cosmos-Embed1-448p-anomaly-detection baseline

用法：
1. 从正常视频提取 baseline embeddings
2. 对 query 视频计算 embedding 距离
3. 距离越大 → 越可能异常

Cosmos-Embed1 是视频-文本联合 embedder，用 normal/anomalous text prompt
做 zero-shot 分类，或用 kNN 距离做 anomaly scoring。
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os, sys, json, glob, argparse
import logging

logger = logging.getLogger("cosmos")

MODEL_DIR = "/home/zzy/jepa_data/external_models/cosmos-anomaly"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chunking_patch():
    """Monkey-patch apply_chunking_to_forward for transformers 5.x compat"""
    import torch as _t

    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size > 0 and all(
            isinstance(t, _t.Tensor) and t.shape[chunk_dim] > chunk_size
            for t in input_tensors if isinstance(t, _t.Tensor)
        ):
            max_len = max(
                t.shape[chunk_dim] for t in input_tensors if isinstance(t, _t.Tensor)
            )
            outputs = []
            for i in range(0, max_len, chunk_size):
                chunk_inputs = []
                for t in input_tensors:
                    if isinstance(t, _t.Tensor) and t.shape[chunk_dim] > 0:
                        start = i
                        end = min(i + chunk_size, t.shape[chunk_dim])
                        chunk_inputs.append(t.narrow(chunk_dim, start, end - start))
                    else:
                        chunk_inputs.append(t)
                outputs.append(forward_fn(*chunk_inputs))
            if isinstance(outputs[0], tuple):
                return tuple(_t.cat([o[i] for o in outputs], dim=chunk_dim)
                           for i in range(len(outputs[0])))
            return _t.cat(outputs, dim=chunk_dim)
        return forward_fn(*input_tensors)

    return apply_chunking_to_forward


class CosmosAnomalyScorer:
    """
    Cosmos-Embed1 异常评分器

    策略：Compute video embedding, measure cosine distance from
    a centroid of normal reference videos.
    """

    def __init__(self, model_dir=MODEL_DIR, device=None):
        self.device = device or DEVICE
        self.model_dir = model_dir
        self.model = None
        self.processor = None

    def _ensure_loaded(self):
        if self.model is not None:
            return

        # Use compatible transformers (4.47) for Cosmos
        import sys, os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        COSMOS_DEPS = "/home/zzy/cosmos-deps"
        if COSMOS_DEPS not in sys.path:
            sys.path.insert(0, COSMOS_DEPS)

        logger.info("[Cosmos] Loading Cosmos-Embed1-448p-anomaly-detection...")

        # Add local dir to sys.path for custom modeling code
        sys.path.insert(0, self.model_dir)

        # Import AFTER path change to use compatible transformers
        from transformers import AutoModel, AutoProcessor, AutoConfig

        # Manually load weights to workaround safetensors metadata issue
        config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Load state dict manually from safetensors
        import safetensors.torch
        import json as _json
        with open(os.path.join(self.model_dir, "model.safetensors.index.json")) as f:
            index = _json.load(f)
        state_dict = {}
        loaded_files = set()
        for key, filename in index["weight_map"].items():
            if filename not in loaded_files:
                shard = safetensors.torch.load_file(
                    os.path.join(self.model_dir, filename)
                )
                for k, v in shard.items():
                    state_dict[k] = v
                loaded_files.add(filename)

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.half().to(self.device)
        self.model.eval()

        # Load processor separately
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        logger.info("[Cosmos] Loaded.")

    @torch.no_grad()
    def encode_video(self, video_path, num_frames=8, resolution=448):
        """
        编码视频为 768-dim embedding

        Cosmos-Embed1 takes 8 frames at 448×448.
        Returns: numpy array [768]
        """
        self._ensure_loaded()

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            cap.release()
            return None

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (resolution, resolution))
            frames.append(frame_rgb)
        cap.release()

        if len(frames) < num_frames:
            return None

        # Process through model — Cosmos expects (videos, input_ids, attention_mask)
        try:
            inputs = self.processor(videos=frames, return_tensors="pt")
        except Exception as e:
            logger.warning(f"[Cosmos] Processor failed: {e}, trying manual...")
            img_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [1, 8, 3, 448, 448]
            # Dummy text input
            tokenizer = getattr(self.processor, 'tokenizer', None)
            if tokenizer:
                text_inputs = tokenizer("", return_tensors="pt")
                inputs = {
                    "videos": img_tensor,
                    "input_ids": text_inputs["input_ids"].to(self.device),
                    "attention_mask": text_inputs["attention_mask"].to(self.device),
                }
            else:
                inputs = {
                    "videos": img_tensor,
                    "input_ids": torch.zeros(1, 1, dtype=torch.long, device=self.device),
                    "attention_mask": torch.ones(1, 1, device=self.device),
                }

        if isinstance(inputs, dict):
            inputs = {
                k: (v.to(self.device).half() if v.is_floating_point() else v.to(self.device))
                if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        outputs = self.model(**inputs)

        # Get embedding — depends on model output format
        if hasattr(outputs, 'video_embeds'):
            emb = outputs.video_embeds
        elif hasattr(outputs, 'image_embeds'):
            emb = outputs.image_embeds
        elif hasattr(outputs, 'last_hidden_state'):
            emb = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(outputs, torch.Tensor):
            emb = outputs
        else:
            # Try first tensor attribute, then pool
            for attr in dir(outputs):
                val = getattr(outputs, attr)
                if isinstance(val, torch.Tensor) and val.dim() >= 2:
                    emb = val
                    break
            else:
                raise ValueError(f"Cannot extract embedding from {type(outputs)}")

        emb = emb.cpu().float().numpy()
        # Pool across all dimensions except the last (embedding dim)
        if emb.ndim > 1:
            emb = emb.reshape(-1, emb.shape[-1]).mean(axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        return emb.astype(np.float32)

    @torch.no_grad()
    def score_video(self, video_path, normal_centroid=None):
        """
        异常分数 = 1 - cosine_sim(video_emb, normal_centroid)
        如果没有 normal_centroid，返回 embedding 本身用于建 baseline
        """
        emb = self.encode_video(video_path)
        if emb is None:
            return None

        if normal_centroid is not None:
            sim = np.dot(emb, normal_centroid)
            return 1.0 - max(0, float(sim))
        return emb

    def build_normal_centroid(self, video_paths):
        """从多个正常视频建 baseline centroid"""
        embs = []
        for vp in video_paths:
            emb = self.encode_video(vp)
            if emb is not None:
                embs.append(emb)
        if not embs:
            return None
        centroid = np.mean(embs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
        return centroid


def evaluate_cosmos(test_annotations_dir, model_dir, output_path):
    """在 test 集上评估 Cosmos"""
    scorer = CosmosAnomalyScorer(model_dir=model_dir)

    # Step 1: 从 test 集的正常视频建 baseline
    print("[1/3] Building normal centroid from test normal videos...")
    anno_files = sorted(glob.glob(os.path.join(test_annotations_dir, "*_annotations.json")))
    normal_videos = []
    all_videos = []

    for af in anno_files:
        with open(af) as f:
            data = json.load(f)
        video_path = data['video_path']
        if not os.path.exists(video_path):
            continue

        n_ann = len(data.get('annotations', []))
        if n_ann == 0:
            normal_videos.append(video_path)
        all_videos.append(data)

    print(f"  Found {len(normal_videos)} normal videos for centroid")
    centroid = scorer.build_normal_centroid(normal_videos[:30])  # use up to 30
    if centroid is None:
        print("  ERROR: Could not build centroid")
        return
    print(f"  Centroid built (dim={len(centroid)})")

    # Step 2: Score all test videos
    print(f"[2/3] Scoring {len(all_videos)} test videos...")
    scores = []
    labels = []
    for i, data in enumerate(all_videos):
        video_path = data['video_path']
        total_frames = data.get('total_frames', 1)
        anns = data.get('annotations', [])

        # Video-level label: 1 if any anomalous segment
        has_anomaly = 1 if len(anns) > 0 else 0

        score = scorer.score_video(video_path, centroid)
        if score is None:
            print(f"  [{i+1}/{len(all_videos)}] SKIP {data['video_name']}")
            continue

        scores.append(score)
        labels.append(has_anomaly)

        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(all_videos)}]")

    scores = np.array(scores)
    labels = np.array(labels)

    # Step 3: Evaluate (video-level)
    print(f"\n[3/3] Video-level evaluation...")
    best_f1 = 0; best_th = 0.5
    for th in np.linspace(scores.min(), scores.max(), 50):
        y_pred = (scores > th).astype(int)
        tp = ((y_pred==1) & (labels==1)).sum()
        fp = ((y_pred==1) & (labels==0)).sum()
        fn = ((y_pred==0) & (labels==1)).sum()
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        if f1 > best_f1: best_f1 = f1; best_th = th

    y_pred = (scores > best_th).astype(int)
    tp = ((y_pred==1)&(labels==1)).sum()
    fp = ((y_pred==1)&(labels==0)).sum()
    fn = ((y_pred==0)&(labels==1)).sum()

    results = {
        'model': 'Cosmos-Embed1-448p-anomaly-detection',
        'level': 'video',
        'precision': tp/(tp+fp+1e-6),
        'recall': tp/(tp+fn+1e-6),
        'f1': 2*tp/(2*tp+fp+fn+1e-6),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'threshold': float(best_th),
        'n_videos': len(scores),
    }

    print(f"\n{'='*50}")
    print("Cosmos-Embed1 on Test Set (Video-level)")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall:    {results['recall']:.3f}")
    print(f"  F1:        {results['f1']:.3f}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-annotations", default="/home/zzy/jepa_data/clean_test_dataset/annotations")
    parser.add_argument("--model-dir", default="/home/zzy/jepa_data/external_models/cosmos-anomaly")
    parser.add_argument("--output", default="/home/zzy/jepa_data/cosmos_results.json")
    args = parser.parse_args()

    evaluate_cosmos(args.test_annotations, args.model_dir, args.output)
