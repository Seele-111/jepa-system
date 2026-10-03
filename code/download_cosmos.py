#!/usr/bin/env python3
"""download_cosmos.py — 下载 Cosmos-Embed1-448p-anomaly-detection"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

model_id = "nvidia/Cosmos-Embed1-448p-anomaly-detection"
target = "/home/zzy/jepa_data/external_models/cosmos-anomaly"

if os.path.exists(target) and os.listdir(target):
    print(f"Already exists: {target}")
else:
    print(f"Downloading {model_id}...")
    path = snapshot_download(model_id, local_dir=target)
    print(f"Done: {path}")
