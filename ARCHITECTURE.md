# Architecture

## Pipeline

### Data splits (v3 — board-grouped, leak-free; `scripts/make_splits.py`)

The 41-prefix / 24-board (s-number) clean FICS training set is partitioned at **s-number**
granularity (so no physical board straddles splits) into:

| Split | Path | Labels | ~Images | Used for |
|-------|------|--------|:-------:|----------|
| **TRAIN** | `data/FICS_PCB_REMAP_NOISE_TRAIN` | noisy (25/25/25) | 3,324 | training only |
| **DEV**   | `data/dev` | clean | 870 | `best.pt` selection (**all** strategies) + adaptive precision-target calibration |
| **TEST**  | `data/test` → `data/val` | clean | 906 | final `evaluate.py --val-images data/test/images` only |

DEV is the val/monitoring set every run trains against; TEST is read by nothing but the final
eval. This removes the historical calibration leak (adaptive thresholds were calibrated on the
same `data/val` used to report mAP). Split recorded in `data/split_manifest.json`.

### Loop 1 — FICS NOISE intra-dataset (active, `pseudo_only_mode: true`)

```
data/FICS_PCB_REMAP_NOISE_TRAIN/   (33 boards, 75% of labels corrupted)
        │
        ▼
  scripts/train.py ──► Generation 0 (noisy teacher)
        │                  └─► runs/<model>/baseline_noisy_v3/weights/best.pt
        ▼
  SelfTrainer: iter1 seeded from the teacher with the strategy threshold
    (static 0.25 / adaptive precision-target on DEV / progressive 0.35)
        │
        ▼
  PseudoLabeler ──► pseudo-labels for the FICS TRAIN images
        │
        ▼
  _merge_datasets()   [pseudo_only_mode = true]
    combined = pseudo-labeled FICS images only (noisy originals excluded)
        │
        ▼
  model.train() ──► Generation 1        (best.pt selected on clean DEV)
        │
        ▼
  model.val() on data/dev               (monitoring + adaptive calibration)
        │
        └──────► repeat for Gen 2, Gen 3   (each student seeds the next iteration)

  data/test (= data/val): read only by the final scripts/evaluate.py ──► TEST mAP (reported)
```

### Loop 2 — PCB-DSLR target-domain self-training (v3, Gen0 → Gen3)

Teacher: each model's noisy-FICS Gen0 (`baseline_noisy_v3`). IC ground-truth merged every
iteration via `partial_labels_dir`. Adaptive calibrates on the held-out DEV split (`data/dev`);
3 student generations (Gen0 → Gen3). Evaluation: detection-growth curve at fixed C=0.25 via
`scripts/evaluate_pcb_dslr.py` (mAP50 not possible — IC GT is consumed in training; no held-out
GT for other classes). The static run of each architecture (yolov8/yolov10/yolo12) is the
qualitative comparison; `experiments/run_dslr_evals_v3.sh` writes `detection_growth.csv`.

```
runs/<model>/baseline_noisy_v3/weights/best.pt   (FICS-noisy teacher = Gen0)
        │
        ▼
  data/PCB_DSLR_CROPS_512/   (2,927 crops; IC GT = class 2 only)
        │
        ▼
  PseudoLabeler ──► pseudo-labels for the 7 non-IC classes
        │              (conf: static / adaptive / progressive)
        ▼
  _merge_datasets()   [pseudo_only_mode = true  +  partial_labels_dir]
    combined = DSLR IC ground-truth  +  pseudo-labels (other 7 classes)
        │
        ▼
  model.train() ──► Generation 1
        │              (adaptive: per-class precision-target from the DEV PR curve)
        │
        └──────► repeat for Gen 2, Gen 3
        │
        ▼
  scripts/evaluate_pcb_dslr.py ──► detection-growth curve at fixed C=0.25
        └─► experiments/<run>/detection_growth.csv
            (every non-IC class collapses to 0 by iter2 — see note below)
```

**Collapse and the Gen0 assessor.** On PCB-DSLR every non-IC class falls to zero detections by
iter2 (only IC carries target-domain GT). This is structural, not a noise artefact: a clean-teacher
control and a **Path A** variant (`pseudo_only_mode: false`, retaining the full multi-class FICS
support set every iteration; `configs/yolo12/self_train_dslr_pathA*.yaml`) both still collapse to
IC-only. The downstream **WPCB-EFAv2+** assessment (`scripts/wpcb_efa.py`) therefore uses the
**Gen0 teacher** as its multi-class assessor for all models, not a later student.

---

## Modules

| Path | Responsibility |
|------|---------------|
| `scripts/train.py` | Train a single model on labeled data (Generation 0 / teacher) |
| `scripts/generate_pseudo_labels.py` | Run a trained model over unlabeled images and write YOLO `.txt` labels |
| `scripts/self_train.py` | Orchestrate the full N-iteration self-training loop |
| `scripts/make_splits.py` | Carve the board-grouped (s-number) held-out DEV split from clean TRAIN; materialize `data/{dev,train_clean}` + `data/test` alias; write `data/split_manifest.json` |
| `scripts/evaluate.py` | Evaluate a checkpoint on a val set; `--val-images` overrides the config (e.g. `data/test/images` for the held-out TEST set) |
| `scripts/evaluate_pcb_dslr.py` | PCB-DSLR detection-growth curve: load Gen0 + iter1–3 checkpoints, infer at fixed C=0.25, count per class, write long-format CSV |
| `scripts/count_pseudo_labels.py` / `eval_pseudo_label_quality.py` / `eval_pseudo_label_recovery.py` | Pseudo-label analysis (no GPU): volume, TP/FP/FN vs clean GT, and noise-recovery vs the corruption manifest → `results/analysis/pseudo_label_*.csv` |
| `scripts/wpcb_efa.py` | WPCB-EFAv2+ economic-feasibility assessment: per-board component composition from the Gen0 teacher on PCB-DSLR, IC anchor cross-check → `results/efa/<model>/` |
| `scripts/verify_crop_geometry.py` | Template-match PCB-DSLR crops back into the full CVL images to recover crop positions; tests grid/overlap/content-centring → `results/analysis/crop_geometry.csv` |
| `src/models/base.py` | `BaseDetector` — abstract interface: `train()`, `predict()`, `load_weights()`, `val()` |
| `src/models/__init__.py` | `build_model(model_type, config)` — single entry point; `MODEL_REGISTRY` |
| `src/models/yolo.py` | `YOLODetector` — wraps YOLOv8, YOLOv10, and YOLO12 via Ultralytics |
| `src/models/rtdetr.py` | `RTDETRDetector` — wraps RT-DETR-l and RT-DETR-x via Ultralytics |
| `src/labeling/pseudo_labeler.py` | `PseudoLabeler.generate()` — infers at `candidate_floor`, post-filters per class by confidence, writes YOLO labels |
| `src/labeling/self_trainer.py` | `SelfTrainer.run()` — seeds iter1 from the teacher via `_seed_thresholds()`, then merge → retrain → eval loop; adaptive/progressive threshold dispatch; crash recovery (skips completed iterations, renames partial train dirs) |
| `src/utils/io.py` | `load_config()`, `make_data_yaml()`, `save_run_snapshot()` — config loading, Ultralytics data file generation, reproducibility snapshots |
| `src/utils/metrics.py` | `compute_adaptive_thresholds()` — per-class precision-target threshold from the DEV PR curve (`box.p_curve`, `target_precision`); `save_metrics()`, `compare_runs()` — result persistence and cross-run comparison |
| `src/utils/labels.py` | Shared YOLO-label IO and geometry helpers (read/write boxes, IoU, per-class matching) used by the pseudo-label analysis scripts; tested in `tests/utils/test_labels.py` |
| `src/utils/visualization.py` | `plot_training_curves()` — training plots and debug overlays |

---

## Key design decisions

| Decision | Rationale |
|---|---|
| All models implement `BaseDetector` | Scripts are model-agnostic; swap model with `--model` flag only |
| Ultralytics as sole backend | Covers YOLOv8, YOLOv10, YOLO12, RT-DETR-l, RT-DETR-x with a unified API; avoids custom PyTorch implementations |
| Pseudo-labels written as YOLO `.txt` | Same format as hand-annotated labels; no conversion step needed |
| `pseudo_only_mode: true` for FICS NOISE loop | Student trains on pseudo-labels only; noisy originals excluded from combined set. Replaces the merge strategy for the intra-dataset loop, consistent with Honorato Section 4.3 |
| Partial GT labels merged at each iteration (PCB-DSLR loop) | When `pseudo_only_mode: false` with `partial_labels_dir` set: DSLR IC ground-truth is appended to pseudo-labels for non-IC classes, matching Honorato Section 4.4 |
| Iteration 1 is seeded inside the loop from the teacher | `SelfTrainer` pseudo-labels iter1 using the warm-started teacher and the strategy's own threshold (static `conf_threshold`, adaptive per-class precision-target from the teacher's DEV PR curve, progressive `conf_schedule[0]`). This makes Gen1 diverge across strategies. If `iter1/pseudo/labels` already exists (e.g. an external pre-seed or a resumed run) the teacher seed is skipped. |
| Calibration on a held-out DEV split, not on `data/val` (v3) | Adaptive thresholds and `best.pt` selection read `data/dev` (clean, board-disjoint); final mAP is reported on `data/test` (= `data/val`). Removes the calibration leak whereby adaptive was tuned on the same set used to report results. See `scripts/make_splits.py`, `data/split_manifest.json`. |
| Final iteration skips pseudo-label generation | Avoids wasting compute on labels that will never be consumed |
| Image discovery is case-insensitive | Linux filesystems are case-sensitive; dataset mixes `.jpg` and `.JPG` |
| `val/` is never merged or pseudo-labeled | Guarantees an uncontaminated evaluation set throughout all iterations |
| Three threshold strategies are mutually exclusive | **Static** (`conf_threshold`): fixed global value each iteration (0.25). **Adaptive** (`adaptive_threshold: true`): per-class precision-target on the DEV PR curve — lowest confidence reaching `target_precision` (default 0.9), recomputed each round (including the teacher seed for iter1). **Progressive** (`conf_schedule: [...]`): scheduled values applied from iter1 onward (e.g. `[0.35, 0.45, 0.55]` — index 0 seeds iter1, index *i* seeds iter*i*+1), independent of model performance. |
| `candidate_floor` is only set for adaptive mode | Adaptive thresholds are computed after inference, so the labeler must capture all candidates first; `candidate_floor` sets the inference confidence for this pass. Static and progressive know their threshold upfront and infer at `conf_threshold` directly. |
| Crash recovery is automatic | If `iter{i}/train/weights/best.pt` exists when `SelfTrainer.run()` starts, that iteration is skipped. Partial train dirs (crash with no `best.pt`) are renamed to `train.failed-<timestamp>` so Ultralytics uses the canonical directory name on the next attempt. |
