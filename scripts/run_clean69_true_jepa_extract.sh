#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code

CUDA_VISIBLE_DEVICES=0 /home/zzy/vjepa2-main/vjepa-env/bin/python -u extract_true_jepa_segment_signals.py \
  --annotations "/mnt/c/Users/admin/Desktop/测试/annotations" \
  --output /home/zzy/jepa_data/clean69_true_jepa \
  --max-frames 32 \
  --max-keyframes 8 \
  --timeout 900 \
  --keep-runs
