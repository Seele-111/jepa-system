#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs

export CUDA_VISIBLE_DEVICES=0

/home/zzy/vjepa2-main/vjepa-env/bin/python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_ot_event_set_graph_fast_cv2.summary.json \
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
  --ot-event-set-matcher \
  --ot-score-thresholds 0.8,1.0 \
  --ot-base-keep-scores 0.25,0.5,0.75 \
  --ot-evidence-thresholds 0.45,0.55 \
  --ot-selector-weights 0,0.1 \
  --ot-contrast-weights 0.5 \
  --ot-active-fraction-weights 0.5 \
  --ot-agreement-weights 0.5 \
  --ot-island-coverage-weights 1 \
  --ot-peak-alignment-weights 0.5 \
  --ot-base-overlap-penalties 0,0.25 \
  --ot-length-penalties 0,0.005 \
  --ot-smooth-windows 1 \
  --ot-min-evidence-island-lengths 1 \
  --ot-max-fp-increase 0
