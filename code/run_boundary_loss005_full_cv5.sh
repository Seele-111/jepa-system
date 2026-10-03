#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mnt/e/jepa-system/data/logs
cd /mnt/e/jepa-system/code
source /home/zzy/vjepa2-main/vjepa-env/bin/activate

python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_boundary_loss005_cv5.summary.json \
  --folds 5 \
  --epochs 80 \
  --patience 15 \
  --hidden 32 \
  --device cuda \
  --lambda-boundary 0.05 \
  --selector-model extratrees \
  --selector-target binary \
  --rescuer-model none \
  --proposal-replacement \
  --graph-reranker \
  --graph-reranker-thresholds 0.55,0.6,0.65,0.7,0.75 \
  --graph-reranker-support-ious 0.2,0.3 \
  --graph-reranker-support-weights 0,0.1,0.2 \
  --graph-reranker-support-count-weights 0,0.05 \
  --graph-reranker-split-penalties 0,0.1,0.2 \
  --graph-reranker-mainline-overlap-penalties 0,0.2 \
  --graph-reranker-length-penalties 0,0.005
