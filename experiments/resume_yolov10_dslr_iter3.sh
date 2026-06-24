#!/usr/bin/env bash
# Salvage-resume for the frozen YOLOv10 PCB-DSLR static run.
#
# Context (2026-06-17): the run experiments/yolov10_st_dslr_static_v3_42 hard-froze
# mid-iter3 at epoch 23/50 — the documented WSL2 CUDA freeze (GPU pegged 100% util /
# ~3% mem-util, no forward progress, last.pt last written at epoch 22). The process
# keeps its PID and burns wall-clock without progressing; it does not self-recover.
#
# Why this script instead of just re-running run_yolov10_dslr_static_v3.sh:
# Ultralytics writes best.pt continuously from an early epoch, so iter3/train/weights/
# best.pt already exists (from ~epoch 15). SelfTrainer treats "best.pt exists" as
# "iteration complete" (src/labeling/self_trainer.py:70) — so a naive re-run would
# SKIP the remaining ~28 epochs and silently finalize iter3 on a half-trained model.
#
# Kill the frozen process tree YOURSELF before running this (it must be gone so the
# GPU is free, otherwise the resume OOMs). This script then:
#   1. Resumes the SAME Ultralytics run from last.pt (epoch 22) via resume=True,
#      training through to epoch 50 / patience and writing the FINAL best.pt.
#   2. Writes iter3's run_snapshot.yaml (the frozen run never reached the code that
#      writes it — reproducibility rule in CLAUDE.md).
#   3. Re-runs self_train.py so SelfTrainer sees the completed best.pt, runs the
#      final eval, plots the curve, and closes the loop the same way it normally would.
# A stall watchdog runs alongside the whole thing so a recurrence pages you in ~15 min.
#
# Run it yourself from the repo root, AFTER killing the frozen run (multi-hour run):
#   bash experiments/resume_yolov10_dslr_iter3.sh

set -euo pipefail

MODEL="yolov10"
CONFIG="configs/yolov10/self_train_dslr.yaml"
OUTPUT="experiments/yolov10_st_dslr_static_v3_42"
LABELED_IMAGES="data/PCB_DSLR_CROPS_512/orig_distribution/images"
LABELED_LABELS="data/PCB_DSLR_CROPS_512/orig_distribution/labels"
LAST="$OUTPUT/iter3/train/weights/last.pt"
BEST="$OUTPUT/iter3/train/weights/best.pt"
RESUME_LOG="experiments/run_yolov10_dslr_static_v3_resume.log"
UV="$HOME/.local/bin/uv"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESUME_LOG"; }

: > "$RESUME_LOG"
say "=== Resume YOLOv10 PCB-DSLR static iter3 (salvage from epoch-23 freeze) ==="
say "Output:  $OUTPUT"
say "Git SHA: $(git rev-parse HEAD)"

if [[ ! -f "$LAST" ]]; then
  say "ERROR: $LAST not found — cannot resume. Aborting."
  exit 1
fi

# --------------------------------------------------------------------------- #
# 0. Pre-flight: the frozen run must already be dead (you kill it). Abort if it
#    is still alive, since it holds VRAM and the resume would OOM.
# --------------------------------------------------------------------------- #
ST_PAT="scripts/self_train.py.*yolov10_st_dslr_static_v3_42"
if pgrep -f "$ST_PAT" >/dev/null; then
  say "ERROR: the frozen run is still alive:"
  pgrep -af "$ST_PAT" | tee -a "$RESUME_LOG"
  say "Kill it first (e.g. 'pkill -KILL -f \"$ST_PAT\"'), then re-run this script."
  exit 1
fi
say "Frozen run is dead. GPU state:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | tee -a "$RESUME_LOG"

# --------------------------------------------------------------------------- #
# 1. Start the stall watchdog in the background (auto-stopped on exit).
# --------------------------------------------------------------------------- #
say "Starting stall watchdog on $OUTPUT (threshold 900s)..."
bash scripts/stall_watchdog.sh "$OUTPUT" --threshold 900 --interval 60 &
WD_PID=$!
trap 'kill "$WD_PID" 2>/dev/null || true' EXIT

# --------------------------------------------------------------------------- #
# 2. Resume the Ultralytics run from last.pt (epoch 23 -> 50 / patience).
#    resume=True reads epochs/data/project/name from the checkpoint itself; no
#    hyperparameters are passed here. Writes the final best.pt into iter3/train.
# --------------------------------------------------------------------------- #
say "Resuming iter3 training from $LAST ..."
PYTHONPATH=. "$UV" run python -c \
  "from ultralytics import YOLO; YOLO('$LAST').train(resume=True)" \
  2>&1 | tee -a "$RESUME_LOG"

if [[ ! -f "$BEST" ]]; then
  say "ERROR: resume finished but $BEST is missing. Aborting before finalize."
  exit 1
fi
say "Resume complete — final best.pt in place."

# --------------------------------------------------------------------------- #
# 3. Write iter3's run_snapshot.yaml (the frozen run never reached the code path
#    that emits it — self_trainer.py:113 runs only in the training branch).
# --------------------------------------------------------------------------- #
say "Writing iter3 run_snapshot.yaml ..."
PYTHONPATH=. "$UV" run python -c "
from pathlib import Path
from src.utils.io import load_config, save_run_snapshot
cfg = load_config('$CONFIG')
cfg.setdefault('pseudo_label', {})['adaptive_threshold'] = False  # mirror --no-adaptive
save_run_snapshot(cfg, Path('$BEST'), 3)
print('wrote', Path('$BEST').parent / 'run_snapshot.yaml')
" 2>&1 | tee -a "$RESUME_LOG"

# --------------------------------------------------------------------------- #
# 4. Close the loop: re-run self_train.py. iter1/2/3 best.pt all exist now, so
#    SelfTrainer skips training, runs the final eval, and plots the curve.
# --------------------------------------------------------------------------- #
say "Finalizing loop via self_train.py (eval + curve; no retraining) ..."
PYTHONPATH=. "$UV" run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --no-adaptive \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$RESUME_LOG"

say "=== Done. Resume + finalize complete. ==="
