#!/usr/bin/env bash
# PCB-DSLR target domain (v3) — yolo12 PATH A iter1 PROBE (plan step 2.5), adaptive.
# Cheap (~1 hr) gate before committing to the full 3-iteration Path A run.
#
# Runs ONE self-training iteration (iterations: 1 in the probe config) with the FICS
# support base retained, then computes the detection-growth curve over Gen0 + iter1.
#
# GATE (read off the printed detection-growth summary / detection_growth.csv):
#   on the PCB-DSLR domain the 7 non-IC classes must be NONZERO at iter1 AND improve
#   vs Gen0. If iter1 already collapses to IC-only, STOP — same lesson, ~1 hr spent —
#   and try the clean (data/train_clean) support base to disambiguate mechanism-vs-noise
#   before abandoning Path A. Only proceed to run_yolo12_dslr_pathA_adaptive_v3.sh
#   (iterations: 3) if the gate passes.
#
# Support base (NOISY): data/FICS_PCB_REMAP_NOISE_TRAIN.
# Teacher: runs/yolo12/baseline_noisy_v3/weights/best.pt (warm_start in config).
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_dslr_pathA_probe.yaml"
OUTPUT="experiments/yolo12_st_dslr_pathA_probe_v3_42"
LOG="experiments/run_yolo12_dslr_pathA_probe_v3.log"
LABELED_IMAGES="data/FICS_PCB_REMAP_NOISE_TRAIN/images"
LABELED_LABELS="data/FICS_PCB_REMAP_NOISE_TRAIN/labels"
GEN0_WEIGHTS="runs/yolo12/baseline_noisy_v3/weights/best.pt"

echo "=== yolo12 PCB-DSLR PATH A iter1 PROBE (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Support base: $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- self-training loop (1 iteration, adaptive, support base retained) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/self_train.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --labeled-images "$LABELED_IMAGES" \
    --labeled-labels "$LABELED_LABELS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- detection-growth gate (Gen0 + iter1 on PCB-DSLR, conf=0.25) ---" | tee -a "$LOG"
PYTHONPATH=. ~/.local/bin/uv run python scripts/evaluate_pcb_dslr.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --experiment "$OUTPUT" \
    --gen0-weights "$GEN0_WEIGHTS" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "GATE: inspect $OUTPUT/detection_growth.csv — non-IC classes must be nonzero AND > Gen0." | tee -a "$LOG"
