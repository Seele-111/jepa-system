#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs

CUDA_VISIBLE_DEVICES=0 /home/zzy/vjepa2-main/vjepa-env/bin/python -u train_jepa_temporal_set_predictor.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_jepa_set_predictor_smoke_cv2.summary.json \
  --folds 2 \
  --epochs 20 \
  --hidden 96 \
  --num-queries 8 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.1 \
  --batch-size 16 \
  --lr 0.0003 \
  --max-target-segments 8 \
  --object-weight 1.0 \
  --l1-weight 5.0 \
  --iou-weight 2.0 \
  --no-object-weight 0.25 \
  --decode-thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
  --decode-nms-ious 0.1,0.3,none \
  --device cuda
