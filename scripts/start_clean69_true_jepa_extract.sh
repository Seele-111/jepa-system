#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/mnt/e/jepa-system/data/logs
RUN_SCRIPT=/mnt/e/jepa-system/scripts/run_clean69_true_jepa_extract.sh
PID_FILE="$LOG_DIR/clean69_true_jepa_extract.pid"
OUT_LOG="$LOG_DIR/clean69_true_jepa_extract.out.log"
ERR_LOG="$LOG_DIR/clean69_true_jepa_extract.err.log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "already running pid=$old_pid"
    exit 0
  fi
fi

nohup "$RUN_SCRIPT" > "$OUT_LOG" 2> "$ERR_LOG" &
echo "$!" > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")"
