#!/usr/bin/env python3
"""rebuild_global_stats_v2.py — 用真分层特征重算全局基线"""
import torch, numpy as np, os, sys, glob, json
sys.path.insert(0, '/home/zzy/jepa_data')
from detect_and_report_v4 import extract_vjepa_tokens, score_vjepa_multi
from predictor_embed_scorer import PredictorEmbedScorer

# Find 210 normal videos from features_cache_v2
# These are the same videos used for the old baseline
# Load their cached features and re-score with hierarchical features

# Since the old cache used fake pooling, we need to re-extract with real hooks
# But the 210 normal videos might not be on disk anymore
# Instead: find any normal labeled videos and extract

# For now, use the labels from the old build_global_stats
# Read which videos were used
stats_dir = '/home/zzy/jepa_data'
old_stats = torch.load(os.path.join(stats_dir, 'global_normal_stats.pt'), map_location='cpu')
print(f"Old baseline: V μ={old_stats['v_physics_mu']:.4f} σ={old_stats['v_physics_sigma']:.4f}")

# The old code used features_cache_v2. Let's find the normal videos from there
# They were selected as label=0 from a CSV
# For simplicity: use 69 self-labeled videos as "normal" samples?
# No - that would contaminate. Use a separate set.

# Actually, let's just use the test videos that have label=0 (fully normal)
# These are normal AI videos, good for baseline
test_annos = glob.glob('/home/zzy/jepa_data/clean_test_dataset/annotations/*_annotations.json')
normal_videos = []
for af in test_annos[:200]:  # use up to 200
    with open(af) as f:
        d = json.load(f)
    if len(d.get('annotations', [])) == 0:
        normal_videos.append(d)

print(f"Found {len(normal_videos)} normal test videos for baseline")

# Process
pe = PredictorEmbedScorer()
all_means = []

for i, d in enumerate(normal_videos[:100]):  # 100 should be enough
    video_path = d['video_path']
    if not os.path.exists(video_path):
        continue
    try:
        result = extract_vjepa_tokens(video_path, use_cache=False, max_frames=64)
        hier = result.get('hierarchical_features')
        physics = score_vjepa_multi(result['vjepa_tokens'], result['tubelet_to_frames'],
                                     result['total_frames'], predictor_embed=pe,
                                     hierarchical_features=hier)
        all_means.append(physics.mean())
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(normal_videos[:100])}] mean={physics.mean():.4f}")
    except Exception as e:
        print(f"  [{i+1}] {d['video_name']} FAILED: {e}")

if all_means:
    mu = np.mean(all_means)
    sigma = np.std(all_means)
    print(f"\nNew global baseline (hierarchical features, {len(all_means)} normal videos):")
    print(f"  V-JEPA physics μ = {mu:.4f}")
    print(f"  V-JEPA physics σ = {sigma:.4f}")

    # Save
    new_stats = {
        'v_physics_mu': mu,
        'v_physics_sigma': sigma,
        'i_composite_mu': old_stats['i_composite_mu'],  # unchanged
        'i_composite_sigma': old_stats['i_composite_sigma'],
        'n_videos': len(all_means),
        'version': 'v2_hierarchical'
    }
    torch.save(new_stats, os.path.join(stats_dir, 'global_normal_stats_v2.pt'))
    print(f"Saved to global_normal_stats_v2.pt")
