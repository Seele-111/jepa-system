#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mnt/e/jepa-system/data/logs
cd /mnt/e/jepa-system/code
source /home/zzy/vjepa2-main/vjepa-env/bin/activate

python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_scale_completeness_diag_cv2.summary.json \
  --folds 2 \
  --epochs 80 \
  --patience 15 \
  --hidden 32 \
  --device cuda \
  --selector-model extratrees \
  --selector-target binary \
  --rescuer-model none \
  --proposal-replacement \
  --graph-reranker \
  --graph-reranker-thresholds 0.75 \
  --graph-reranker-support-ious 0.2,0.3 \
  --graph-reranker-support-weights 0,0.2 \
  --graph-reranker-support-count-weights 0,0.05 \
  --graph-reranker-split-penalties 0 \
  --graph-reranker-mainline-overlap-penalties 0,0.2 \
  --graph-reranker-length-penalties 0 \
  --scale-completeness-rescue \
  --scale-completeness-prob-thresholds 0.35,0.4 \
  --scale-completeness-evidence-thresholds 0.55,0.65 \
  --scale-completeness-min-active-fractions 0.5,0.67 \
  --scale-completeness-min-means 0.5,0.6 \
  --scale-completeness-min-contrasts 0.05,0.1,0.2 \
  --scale-completeness-max-base-ious 0 \
  --scale-completeness-length-penalties 0 \
  --scale-completeness-nms-ious none \
  --scale-completeness-max-per-video 1
