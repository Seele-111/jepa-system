#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs

CUDA_VISIBLE_DEVICES=0 /home/zzy/vjepa2-main/vjepa-env/bin/python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_h32_graph_clean_calibrated_tiny_cv2.summary.json \
  --folds 2 \
  --epochs 20 \
  --hidden 32 \
  --device cuda \
  --blend-aux-name true_vjepa_raw_rank \
  --selector-model extratrees \
  --selector-target binary \
  --graph-reranker \
  --rescuer-model none \
  --clean-calibrated-selector \
  --clean-calibration-data-dir /home/zzy/jepa_data/clean69_event_jepa_v2 \
  --clean-calibration-model extratrees \
  --clean-calibration-target quality \
  --clean-calibrated-thresholds 0.35,0.5 \
  --clean-calibrated-raw-thresholds 0 \
  --clean-calibrated-clean-weights 0.7 \
  --clean-calibrated-raw-weights 0 \
  --clean-calibrated-max-base-ious 0.25,none \
  --clean-calibrated-length-penalties 0 \
  --clean-calibrated-max-rescues-per-video 1 \
  --clean-calibrated-rescue-nms-ious 0.3 \
  --clean-calibrated-max-fp-increase 0
