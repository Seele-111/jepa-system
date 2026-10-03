#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs
export CUDA_VISIBLE_DEVICES=0

/home/zzy/vjepa2-main/vjepa-env/bin/python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_clean_anchor_rectify_graph_fast_cv2.summary.json \
  --folds 2 \
  --epochs 30 \
  --hidden 32 \
  --patience 8 \
  --device cuda \
  --blend-aux-name true_vjepa_raw_rank \
  --selector-model extratrees \
  --selector-target binary \
  --clean-anchor-rectify \
  --clean-anchor-data-dir /home/zzy/jepa_data/clean79_event_jepa_v2 \
  --clean-anchor-evidence-names true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank \
  --clean-anchor-add-threshold 0.9 \
  --clean-anchor-remove-threshold 0.15 \
  --clean-anchor-min-add-length 2 \
  --clean-anchor-min-keep-length 2 \
  --clean-anchor-smooth-window 3 \
  --clean-anchor-max-added-fraction 0.03 \
  --graph-reranker \
  --graph-reranker-thresholds 0.5,0.6,0.7 \
  --graph-reranker-support-ious 0.2,0.3 \
  --graph-reranker-support-weights 0,0.1 \
  --graph-reranker-support-count-weights 0,0.05 \
  --graph-reranker-split-penalties 0,0.1 \
  --graph-reranker-mainline-overlap-penalties 0,0.2 \
  --graph-reranker-length-penalties 0,0.005 \
  --rescuer-model none
