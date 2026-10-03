#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs
export CUDA_VISIBLE_DEVICES=0

/home/zzy/vjepa2-main/vjepa-env/bin/python -u cross_validate_selector_fusion.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_clean_anchor_soft_weights_topology_matching_graph_fast_cv2.summary.json \
  --folds 2 \
  --epochs 30 \
  --hidden 32 \
  --patience 8 \
  --device cuda \
  --blend-aux-name true_vjepa_raw_rank \
  --selector-model extratrees \
  --selector-target binary \
  --clean-anchor-soft-weights \
  --clean-anchor-data-dir /home/zzy/jepa_data/clean79_event_jepa_v2 \
  --clean-anchor-evidence-names true_vjepa_raw,true_ijepa_dense_raw,dual_jepa_composite,true_vjepa_raw_rank,true_ijepa_dense_raw_rank,dual_jepa_composite_rank,vi_min,vi_max,vi_agreement,vi_abs_gap,vi_product,composite_times_vi_agreement,true_vjepa_raw_abs_delta,true_ijepa_dense_raw_abs_delta,dual_jepa_composite_abs_delta \
  --clean-anchor-teacher-model extratrees \
  --clean-anchor-positive-min-weight 0.55 \
  --clean-anchor-negative-min-weight 0.55 \
  --clean-anchor-smooth-window 3 \
  --graph-reranker \
  --graph-reranker-thresholds 0.5,0.6,0.7 \
  --graph-reranker-support-ious 0.2,0.3 \
  --graph-reranker-support-weights 0,0.1 \
  --graph-reranker-support-count-weights 0,0.05 \
  --graph-reranker-split-penalties 0,0.1 \
  --graph-reranker-mainline-overlap-penalties 0,0.2 \
  --graph-reranker-length-penalties 0,0.005 \
  --rescuer-model none \
  --topology-matching-network \
  --topology-matching-model extratrees \
  --topology-matching-risk-aware \
  --topology-matching-thresholds 0.15,0.25,0.35,0.45,0.55 \
  --topology-matching-parent-min-lengths 12,24,48 \
  --topology-matching-min-child-parent-coverages 0.5,0.7 \
  --topology-matching-max-child-parent-ratios 0.5,0.7 \
  --topology-matching-max-children-per-parent 2,3 \
  --topology-matching-max-set-proposals-per-parent 8,16 \
  --topology-matching-max-child-pool-per-parent 12,20 \
  --topology-matching-max-replaced-parents-per-video 1,2 \
  --topology-matching-risk-penalties 0.25,0.5,0.75,1.0 \
  --topology-matching-safety-thresholds 0.5,0.6,0.7,0.8 \
  --topology-matching-safety-weight 0.35 \
  --topology-matching-max-fp-increase 0
