#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs

CUDA_VISIBLE_DEVICES=0 /home/zzy/vjepa2-main/vjepa-env/bin/python -u extract_true_jepa_segment_signals.py \
  --annotations /home/zzy/jepa_data/clean_anchor_extra10_annotations \
  --output /home/zzy/jepa_data/clean_anchor_extra10_true_jepa \
  --max-frames 32 \
  --max-keyframes 8 \
  --timeout 900 \
  --keep-runs
