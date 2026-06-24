#!/usr/bin/env bash
# WPCB-EFAv2+ multi-class composition (MVP tier) on the PCB-DSLR target domain.
#
# Runs scripts/wpcb_efa.py with the Gen0 noisy teacher of each architecture. The
# Gen0 teacher is the multi-class assessor on purpose: on PCB-DSLR the self-trained
# students specialise to IC (the other 7 classes collapse to 0 from iter1 onward),
# so only the teacher gives an 8-class composition. Conf = config conf_threshold
# (0.25), same regime as the detection-growth eval, so the IC totals cross-check.
#
# Per model it writes three CSVs to results/efa/<model>/:
#   efa_per_capture.csv  efa_per_board.csv  efa_composition.csv
#
# Single inference pass per model over 2,927 tiles (~2-3 min each on the 4060).
#
# Prereq: Gen0 v3 teachers at runs/<model>/baseline_noisy_v3/weights/best.pt and
# the PCB-DSLR tiles at data/PCB_DSLR_CROPS_512/orig_distribution/images.
#
# Run from repo root:
#   bash experiments/run_wpcb_efa.sh 2>&1 | tee experiments/run_wpcb_efa.log
set -euo pipefail
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
UV="$HOME/.local/bin/uv"

echo "=== WPCB-EFAv2+ composition — $(date -u +%Y-%m-%dT%H:%MZ) ==="

efa () {  # $1=model
  local model="$1"
  local cfg="configs/$model/self_train_dslr.yaml"
  local gen0="runs/$model/baseline_noisy_v3/weights/best.pt"
  local dg="experiments/${model}_st_dslr_adaptive_v3_42/detection_growth.csv"
  echo "--- ${model} (Gen0 teacher) ---"
  if [[ ! -f "$gen0" ]]; then
    echo "MISSING Gen0 teacher ($gen0) — skipping" >&2
    return 0
  fi
  $UV run python scripts/wpcb_efa.py \
    --model "$model" --config "$cfg" --weights "$gen0" \
    --rec-policy average \
    --output-dir "results/efa/$model" \
    --detection-growth-csv "$dg"
}

for model in yolov8 yolov10 yolo12; do
  efa "$model"
done

echo "=== WPCB-EFAv2+ composition complete — $(date -u +%Y-%m-%dT%H:%MZ) ==="
