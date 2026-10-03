#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system/code
mkdir -p /mnt/e/jepa-system/data/logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

/home/zzy/vjepa2-main/vjepa-env/bin/python -u run_strict_parent_topology_demand_oof.py \
  --data-dir /home/zzy/jepa_data/segment_train_full_event_jepa_v2 \
  --base-predictions /home/zzy/jepa_data/segment_train_full_event_v2_prediction_set_switcher.predictions.json \
  --summary /home/zzy/jepa_data/segment_train_full_event_v2_parent_topology_demand_proto_single_cv5.summary.json \
  --selector-model extratrees \
  --selector-target binary \
  --selector-thresholds 0.3,0.5,0.7 \
  --selector-min-gaps 0,2 \
  --selector-min-lengths 1,4 \
  --selector-epochs 40 \
  --selector-batch-size 512 \
  --evidence-reducer stack \
  --model prototype \
  --parent-min-lengths 12 \
  --min-child-parent-coverages 0.5 \
  --max-child-parent-ratios 0.7 \
  --max-children-per-parent 2 \
  --max-set-proposals-per-parent 4 \
  --max-child-pool-per-parent 8 \
  --thresholds 0 \
  --risk-penalties 0.25 \
  --safety-thresholds 0 \
  --safety-weight 0.25 \
  --max-replaced-parents-per-video 1 \
  --max-fp-increase 0 \
  --device cuda \
  --seed 42 \
  --epochs 40 \
  --batch-size 512
