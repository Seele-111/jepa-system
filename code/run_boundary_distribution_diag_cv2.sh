#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mnt/e/jepa-system/data/logs
cd /mnt/e/jepa-system/code
source /home/zzy/vjepa2-main/vjepa-env/bin/activate

python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_boundary_distribution_diag_cv2.summary.json \
  --folds 2 \
  --epochs 80 \
  --patience 15 \
  --hidden 32 \
  --device cuda \
  --selector-model extratrees \
  --selector-target binary \
  --rescuer-model none \
  --proposal-replacement \
  --proposal-replacement-thresholds 0.5 \
  --proposal-replacement-parent-min-lengths 72 \
  --proposal-replacement-min-replacements 3 \
  --proposal-replacement-max-per-parent 3 \
  --proposal-replacement-coverages 0.8,1.0 \
  --proposal-replacement-max-ratios 0.6 \
  --proposal-replacement-length-penalties 0.01 \
  --graph-reranker \
  --graph-reranker-thresholds 0.75 \
  --graph-reranker-support-ious 0.2,0.3 \
  --graph-reranker-support-weights 0,0.2 \
  --graph-reranker-support-count-weights 0,0.05 \
  --graph-reranker-split-penalties 0 \
  --graph-reranker-mainline-overlap-penalties 0,0.2 \
  --graph-reranker-length-penalties 0 \
  --boundary-distribution-refine \
  --boundary-distribution-thresholds 0.5,0.6,0.7,0.8 \
  --boundary-distribution-base-coverages 0.5,0.75,0.9 \
  --boundary-distribution-max-ratios 2,3,4 \
  --boundary-distribution-start-quantiles 0.25,0.5 \
  --boundary-distribution-end-quantiles 0.5,0.75 \
  --boundary-distribution-base-weights 0,0.25,0.5 \
  --boundary-distribution-max-shift-ratios 1,2,4
