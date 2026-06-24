#!/usr/bin/env bash
# Stall watchdog for self-training runs.
#
# A frozen CUDA process under WSL2 keeps its PID and burns wall-clock without
# making progress (observed 2026-06-12: ~11h and ~2.2h freezes mid-epoch). This
# watchdog watches the newest file mtime anywhere under a run directory -- any
# real progress (epoch checkpoint, pseudo-label write, dataset merge) touches a
# file -- and alerts loudly when nothing has been written for too long.
#
# Usage:
#   bash scripts/stall_watchdog.sh <run_dir> [--threshold SECONDS] [--interval SECONDS]
#
# Example:
#   bash scripts/stall_watchdog.sh experiments/yolov8_st_noisy_adaptive_v2_42
#
# Defaults: --threshold 900 (15 min, > 4 epochs of headroom), --interval 60.
# Runs in the foreground; Ctrl-C to stop. Safe to leave in a tmux/screen pane.

set -euo pipefail

RUN_DIR="${1:-}"
THRESHOLD=900
INTERVAL=60

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --interval)  INTERVAL="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "usage: bash scripts/stall_watchdog.sh <run_dir> [--threshold SEC] [--interval SEC]" >&2
  exit 2
fi

LOG="$RUN_DIR/stall_watchdog.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Best-effort loud alert: terminal bell, console beep + non-blocking Windows
# message box (both via WSL interop; failures are ignored on non-WSL hosts).
alert() {
  local msg="$1"
  printf '\a'
  echo "[$(ts)] ALERT: $msg" | tee -a "$LOG" >&2
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command "[console]::beep(800,500)" >/dev/null 2>&1 || true
    powershell.exe -NoProfile -Command \
      "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('$msg','Training stall watchdog')" \
      >/dev/null 2>&1 &
  fi
}

echo "[$(ts)] watchdog started on $RUN_DIR (threshold=${THRESHOLD}s interval=${INTERVAL}s)" | tee -a "$LOG"

alerted=0
while true; do
  # Newest mtime (epoch seconds) of any file under the run dir.
  latest=$(find "$RUN_DIR" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
  now=$(date +%s)
  if [[ -z "$latest" ]]; then
    sleep "$INTERVAL"; continue
  fi
  age=$(( now - ${latest%.*} ))

  # Current epoch from the most recently modified results.csv, for the heartbeat.
  csv=$(find "$RUN_DIR" -path '*/train/results.csv' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  epoch="?"
  [[ -n "$csv" ]] && epoch=$(($(wc -l < "$csv") - 1))

  if (( age > THRESHOLD )); then
    if (( alerted == 0 )); then
      alert "No file written in $RUN_DIR for ${age}s (last epoch=${epoch}). Process likely frozen."
      alerted=1
    fi
  else
    if (( alerted == 1 )); then
      echo "[$(ts)] recovered: progress resumed (age=${age}s, epoch=${epoch})" | tee -a "$LOG"
    fi
    alerted=0
    echo "[$(ts)] ok: epoch=${epoch} last_write=${age}s ago"
  fi
  sleep "$INTERVAL"
done
