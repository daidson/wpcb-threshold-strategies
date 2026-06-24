#!/usr/bin/env bash
# YOLO12m noisy GEN0 baseline
# Architecture: Area Attention (A2) — local self-attention (Tian et al., 2025)
# Training set: data/FICS_PCB_REMAP_NOISE/ (75% annotations corrupted)
# Checkpoint lands at: runs/yolo12/baseline_noisy/weights/best.pt
# Prerequisite: yolo12m.pt must be present in repo root.
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/base_noisy.yaml"
LOG="experiments/run_yolo12_noisy_gen0.log"

echo "=== YOLO12m noisy GEN0 baseline training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONPATH=. ~/.local/bin/uv run python scripts/train.py --model "$MODEL" --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Checkpoint expected at: runs/yolo12/baseline_noisy/weights/best.pt" | tee -a "$LOG"
