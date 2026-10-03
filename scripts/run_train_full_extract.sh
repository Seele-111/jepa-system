#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mnt/e/jepa-system/data/logs
mkdir -p /home/zzy/jepa_data/segment_train_full_true_jepa

exec /home/zzy/vjepa2-main/vjepa-env/bin/python \
  /mnt/e/jepa-system/code/extract_true_jepa_segment_signals.py \
  --annotations /mnt/e/jepa-label/JEPA-data/trainingdata/annotations-train \
  --output /home/zzy/jepa_data/segment_train_full_true_jepa \
  --max-frames 32 \
  --max-keyframes 8
