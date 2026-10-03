#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system
mkdir -p data/logs

if pgrep -af "cross_validate_selector_fusion.py" >/dev/null; then
  echo "existing cross_validate_selector_fusion.py process:"
  pgrep -af "cross_validate_selector_fusion.py"
  exit 0
fi

nohup bash code/run_graph_reranker_full_cv5.sh \
  > data/logs/run_graph_reranker_full_cv5.export.out.log \
  2> data/logs/run_graph_reranker_full_cv5.export.err.log &
echo "$!" > data/logs/run_graph_reranker_full_cv5.export.pid
echo "started graph export pid=$(cat data/logs/run_graph_reranker_full_cv5.export.pid)"

echo "soft-weight export not started; run after graph export finishes:"
echo "  bash scripts/run_clean_anchor_soft_weights_graph_full_cv5.sh"
