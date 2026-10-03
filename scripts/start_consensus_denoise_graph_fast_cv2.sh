#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/jepa-system
mkdir -p /mnt/e/jepa-system/data/logs

log=/mnt/e/jepa-system/data/logs/consensus_denoise_graph_fast_cv2.detached.log
pid_file=/mnt/e/jepa-system/data/logs/consensus_denoise_graph_fast_cv2.pid

nohup bash /mnt/e/jepa-system/scripts/run_consensus_denoise_graph_fast_cv2.sh > "$log" 2>&1 < /dev/null &
echo "$!" > "$pid_file"
echo "$!"
