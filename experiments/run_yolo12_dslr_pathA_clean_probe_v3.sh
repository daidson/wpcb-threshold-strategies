#!/usr/bin/env bash
# PCB-DSLR target domain (v3) — yolo12 PATH A iter1 CLEAN-BASE PROBE (plan fallback).
# Identical to run_yolo12_dslr_pathA_probe_v3.sh EXCEPT the support base is the CLEAN
# (pre-corruption) FICS TRAIN set data/train_clean instead of FICS_PCB_REMAP_NOISE_TRAIN.
# Same 3324 images, same stems — only the labels differ (no class-swap/removal/distortion).
#
# WHY: the noisy-base iter1 probe FAILED the gate (2026-06-21) — 6/8 non-IC classes detect
# ZERO on PCB-DSLR even at conf 0.05 (true class-loss, not a threshold artifact), while the
# same checkpoint scores resistor 0.97 / ceramic 0.98 mAP50 on FICS DEV. Confounds ruled out:
# merge artifact (combined set = 3324 FICS + 2927 DSLR, all 8 classes present), calibration
# drift (0 at conf 0.05), best.pt-on-FICS selection (last.pt identical). The one remaining
# question is NOISE vs DOMAIN-DISCRIMINATION. This clean-base probe disambiguates:
#   * non-IC classes recover on DSLR with clean labels -> noise was the cause; proceed Path A clean.
#   * still collapse to IC-only -> domain-discrimination (Honorato merged-model mode);
#     Path A cannot give YOLO12 a generational multi-class DSLR student -> keep Gen0 assessor.
#
# Teacher: runs/yolo12/baseline_noisy_v3/weights/best.pt (warm_start in config).
set -euo pipefail

MODEL="yolo12"
CONFIG="configs/yolo12/self_train_dslr_pathA_probe.yaml"
OUTPUT="experiments/yolo12_st_dslr_pathA_clean_probe_v3_42"
LOG="experiments/run_yolo12_dslr_pathA_clean_probe_v3.log"
LABELED_IMAGES="data/train_clean/images"
LABELED_LABELS="data/train_clean/labels"
GEN0_WEIGHTS="runs/yolo12/baseline_noisy_v3/weights/best.pt"

echo "=== yolo12 PCB-DSLR PATH A iter1 CLEAN-BASE PROBE (v3) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "Config: $CONFIG" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "Support base (CLEAN): $LABELED_IMAGES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- self-training loop (1 iteration, adaptive, CLEAN support base retained) ---" | tee -a "$LOG"
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
echo "GATE: inspect $OUTPUT/detection_growth.csv — non-IC nonzero AND > Gen0 means noise was the cause." | tee -a "$LOG"
echo "Also worth a low-conf cross-check: rerun evaluate_pcb_dslr.py with --conf 0.05 (true loss vs drift)." | tee -a "$LOG"
