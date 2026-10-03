#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs

export CUDA_VISIBLE_DEVICES=0

/home/zzy/vjepa2-main/vjepa-env/bin/python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_fn_aware_v2_graph_fast_cv2.summary.json \
  --folds 2 \
  --epochs 30 \
  --hidden 32 \
  --patience 8 \
  --device cuda \
  --blend-aux-name true_vjepa_raw_rank \
  --selector-model extratrees \
  --selector-target binary \
  --graph-reranker \
  --graph-reranker-thresholds 0.5,0.6,0.7 \
  --graph-reranker-support-ious 0.2,0.3 \
  --graph-reranker-support-weights 0,0.1 \
  --graph-reranker-support-count-weights 0,0.05 \
  --graph-reranker-split-penalties 0,0.1 \
  --graph-reranker-mainline-overlap-penalties 0,0.2 \
  --graph-reranker-length-penalties 0,0.005 \
  --rescuer-model none \
  --fn-aware-reranker \
  --fn-aware-model extratrees \
  --fn-aware-train-max-per-video 120 \
  --fn-aware-thresholds 0.3,0.4,0.5,0.6,0.7,0.8 \
  --fn-aware-max-base-ious 0,0.05,0.1,0.25,0.5,none \
  --fn-aware-nms-ious none,0.1,0.3,0.5 \
  --fn-aware-length-penalties 0,0.005,0.01 \
  --fn-aware-max-per-video 1,2,3 \
  --fn-aware-max-fp-increase 0
