#!/usr/bin/env bash
# PCB-DSLR qualitative assessment (v3) — detection-growth curves, Gen0 -> Gen3.
#
# Runs scripts/evaluate_pcb_dslr.py at a fixed conf (config conf_threshold = 0.25)
# across the teacher (Gen0) and the three student generations on all PCB-DSLR images.
# Per run it writes two CSVs: <experiment>/detection_growth.csv (per-class detection
# counts per generation) and <experiment>/ic_quality.csv (IC TP/FP/FN/precision/recall
# vs the IC GT at IoU>=0.5 — only IC has GT, so cross-class mAP is N/A).
#
# Full matrix: 3 strategies x 3 architectures = 9 runs (~60-70 min on the 4060).
#
# Prereq: all 9 run_<model>_dslr_<strategy>_v3.sh runs have completed, and the Gen0
# v3 teachers exist at runs/<model>/baseline_noisy_v3/weights/best.pt.
#
# Run from repo root:
#   bash experiments/run_dslr_evals_v3.sh 2>&1 | tee experiments/run_dslr_evals_v3.log
set -euo pipefail
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
UV="$HOME/.local/bin/uv"

echo "=== PCB-DSLR detection-growth eval v3 — $(date -u +%Y-%m-%dT%H:%MZ) ==="

dslr_eval () {  # $1=model  $2=strategy
  local model="$1" strategy="$2"
  local cfg="configs/$model/self_train_dslr.yaml"
  [[ "$strategy" == "progressive" ]] && cfg="configs/$model/self_train_dslr_progressive.yaml"
  local exp="experiments/${model}_st_dslr_${strategy}_v3_42"
  local gen0="runs/$model/baseline_noisy_v3/weights/best.pt"
  echo "--- ${model} dslr ${strategy} ---"
  if [[ ! -d "$exp" || ! -f "$gen0" ]]; then
    echo "MISSING run dir or Gen0 teacher ($exp / $gen0) — skipping" >&2
    return 0
  fi
  $UV run python scripts/evaluate_pcb_dslr.py \
    --model "$model" --config "$cfg" --experiment "$exp" --gen0-weights "$gen0"
}

for model in yolov8 yolov10 yolo12; do
  dslr_eval "$model" static
  dslr_eval "$model" adaptive
  dslr_eval "$model" progressive
done

echo "=== PCB-DSLR detection-growth eval v3 complete — $(date -u +%Y-%m-%dT%H:%MZ) ==="
