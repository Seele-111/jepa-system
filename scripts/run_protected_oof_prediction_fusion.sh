#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code

/home/zzy/vjepa2-main/vjepa-env/bin/python -u jepa_protected_oof_prediction_fusion.py \
  --base-predictions /home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_reranker_protected_cv5.predictions.json \
  --recall-predictions /home/zzy/jepa_data/segment_train_full_event_v2_clean_anchor_soft_weights_graph_cv5.predictions.json \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_protected_oof_prediction_fusion.summary.json \
  --max-fp-increase 0 \
  --max-base-ious 0,0.05,0.1,0.25,none \
  --max-additions-per-video 1,2,3 \
  --nms-ious none,0.1,0.3 \
  --iou-threshold 0.3
