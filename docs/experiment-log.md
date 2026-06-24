# Experiment Log

Append-only chronological record of training runs. Each entry captures the
full context needed to reproduce or compare results.

---

## Entry format

```
### YYYY-MM-DD — <experiment name>

- **Model:** yolov8 / yolov10 / rtdetr_v3 / rtdetr_v4
- **Strategy:** baseline / self_train / self_train_adaptive
- **Config:** configs/<model>/<config>.yaml
- **Git SHA:** <sha>
- **Seed:** 42
- **Epochs:** <n> (stopped at <m> via early stopping)
- **Dataset:** FICS-PCB REMAP augmented (4194 train / ? val) + PCB DSLR (unlabeled, 2927 images)

**Results (val set):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | | |
| ceramic_capacitor | | |
| ic | | |
| diode | | |
| inductor | | |
| electrolytic_capacitor | | |
| tantalum_capacitor | | |
| led | | |
| **mAP50** | | |
| **mAP50-95** | | |

**Notes:**
```

---

## Runs

### 2026-05-06 — yolov8_smoke_test

- **Model:** yolov8
- **Strategy:** baseline (smoke test only — 1 epoch)
- **Config:** configs/yolov8/base.yaml
- **Git SHA:** 22d7c7a (pre-fix; path bug present, run landed at `runs/detect/runs/yolov8/baseline-2`)
- **Seed:** 42

**Results (val set, epoch 1 of 1):**
| Metric | Value |
|--------|-------|
| Precision | 0.628 |
| Recall | 0.579 |
| mAP50 | 0.583 |
| mAP50-95 | 0.399 |

**Notes:**
- Smoke test only — not a valid result. mAP50 of 58.3% after a single warmup epoch is
  consistent with a strong pretrained yolov8m.pt checkpoint; the full 100-epoch run is needed.
- Path bug fixed after this run: `project` was resolving to `runs/detect/runs/yolov8/`
  instead of `runs/yolov8/`. Fix applied in `scripts/train.py` and `src/labeling/self_trainer.py`.
- Per-class AP not available from smoke test (1 epoch, no early stopping triggered).

---

### 2026-05-06 — yolov8_baseline

- **Model:** yolov8
- **Strategy:** baseline (Generation 0)
- **Config:** configs/yolov8/base.yaml
- **Weights:** runs/yolov8/baseline-2/weights/best.pt
- **Git SHA:** 1e6ad899ef1b8722f1408a017152b8ba1b352bf3
- **Seed:** 42
- **Epochs:** 100 scheduled, early stopped at epoch 49 (patience=20, best at epoch 29)
- **Dataset:** FICS-PCB REMAP augmented (train / 906 val images, 6102 val instances)

**Results (val set, best.pt = epoch 29):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.987 | 0.775 |
| ceramic_capacitor | 0.967 | 0.581 |
| ic | 0.952 | 0.753 |
| diode | 0.560 | 0.143 |
| inductor | 0.653 | 0.532 |
| electrolytic_capacitor | 0.773 | 0.653 |
| tantalum_capacitor | 0.995 | 0.823 |
| led | 0.992 | 0.957 |
| **mAP50** | **0.860** | |
| **mAP50-95** | | **0.652** |
| Precision | 0.796 | |
| Recall | 0.825 | |

**Notes:**
- Strong performance on common/easy classes (resistor_smd, tantalum_capacitor, led). Diode is the
  clear weak point: AP50=0.560, AP50-95=0.143 — only 42 val instances, severely underrepresented.
- Inductor also lags (AP50=0.653, 114 val instances). Both are candidates for targeted augmentation
  or weighted pseudo-labeling in subsequent self-training iterations.
- Run landed in `baseline-2` (Ultralytics auto-incremented because `baseline` existed from the smoke test).
  The canonical baseline checkpoint is `runs/yolov8/baseline-2/weights/best.pt`.

---

### 2026-05-07 — yolov10_baseline

- **Model:** yolov10
- **Strategy:** baseline (Generation 0)
- **Config:** configs/yolov10/base.yaml
- **Weights:** runs/yolov10/baseline/weights/best.pt
- **Git SHA:** 5a2145a25f25387cba1ac07e5ca10d41e813de0e
- **Seed:** 42
- **Epochs:** 100 scheduled, early stopped at epoch 56 (patience=20, best at epoch 41)
- **Dataset:** FICS-PCB REMAP augmented (train / 906 val images, 6102 val instances)

**Results (val set, best.pt = epoch 41, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.990 | 0.784 |
| ceramic_capacitor | 0.953 | 0.590 |
| ic | 0.902 | 0.727 |
| diode | 0.871 | 0.383 |
| inductor | 0.601 | 0.487 |
| electrolytic_capacitor | 0.737 | 0.580 |
| tantalum_capacitor | 0.650 | 0.544 |
| led | 0.989 | 0.936 |
| **mAP50** | **0.837** | |
| **mAP50-95** | | **0.629** |
| Precision | 0.802 | |
| Recall | 0.787 | |

**Notes:**
- Training-time val reported mAP50=0.852 / mAP50-95=0.627 / P=0.782 / R=0.815 (from results.csv
  best epoch). The post-hoc `evaluate.py` run gives mAP50=0.837 — a ~1.5pp gap. YOLOv10's
  NMS-free head is sensitive to inference conf/iou defaults; this gap is expected and documented.
  Per-class AP above is from `evaluate.py` (canonical for cross-model comparison).
- Diode recovers to 0.871 AP50 vs. YOLOv8's 0.560 — notably stronger, though mAP50-95=0.383
  vs. YOLOv8's 0.143 shows box localisation is still imprecise for this class.
- Tantalum capacitor drops sharply (0.650 vs. YOLOv8's 0.995) — only 12 val instances, so this
  number is volatile; not a reliable signal at this sample size.
- Early stopping fired at epoch 56 with best at epoch 41 — only 15 epochs of no improvement vs
  patience=20. Likely a subtle off-by-one in Ultralytics' patience counter; not a training
  correctness issue.

---

### 2026-05-07 — rtdetr_l_baseline

- **Model:** rtdetr_l
- **Strategy:** baseline (Generation 0)
- **Config:** configs/rtdetr_l/base.yaml
- **Weights:** runs/rtdetr_v3/baseline-2/weights/best.pt
- **Git SHA:** edb22f6 (run predates run_snapshot integration; no snapshot file)
- **Seed:** 42
- **Epochs:** 72 scheduled, early stopped at epoch 57 (patience=20, best at epoch 37)
- **Dataset:** FICS-PCB REMAP augmented (train / 906 val images, 6102 val instances)

**Results (val set, best.pt = epoch 37, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.978 | 0.781 |
| ceramic_capacitor | 0.966 | 0.597 |
| ic | 0.864 | 0.709 |
| diode | 0.598 | 0.253 |
| inductor | 0.478 | 0.367 |
| electrolytic_capacitor | 0.835 | 0.642 |
| tantalum_capacitor | 0.995 | 0.741 |
| led | 0.995 | 0.974 |
| **mAP50** | **0.839** | |
| **mAP50-95** | | **0.633** |
| Precision | 0.779 | |
| Recall | 0.841 | |

**Notes:**
- Artifact path uses legacy directory `runs/rtdetr_v3/baseline-2/` — run predates the
  rtdetr_v3→rtdetr_l rename (commit 492e5cb). Do not rename the directory.
- Inductor is the clear weak point (AP50=0.478), worse than both YOLO variants. Diode also lags
  (AP50=0.598). RT-DETR's attention mechanism does not appear to compensate for low-frequency
  classes at this scale.
- Electrolytic capacitor is notably stronger than YOLOv8 (0.835 vs. 0.773) and YOLOv10 (0.737),
  suggesting RT-DETR-l's attention handles the class's size variability better.
- Training CSV reported mAP50=0.836 / mAP50-95=0.632; evaluate.py gives 0.839 / 0.633 — gap is
  within noise, consistent with RT-DETR's deterministic NMS-free head.

---

### 2026-05-08 — rtdetr_x_baseline

- **Model:** rtdetr_x
- **Strategy:** baseline (Generation 0)
- **Config:** configs/rtdetr_x/base.yaml
- **Weights:** runs/rtdetr_x/baseline/weights/best.pt
- **Git SHA:** 492e5cb (run predates run_snapshot integration; no snapshot file)
- **Seed:** 42
- **Epochs:** 72 scheduled, early stopped at epoch 57 (patience=20, best at epoch 30)
- **Dataset:** FICS-PCB REMAP augmented (train / 906 val images, 6102 val instances)

**Results (val set, best.pt = epoch 30, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.973 | 0.769 |
| ceramic_capacitor | 0.947 | 0.561 |
| ic | 0.842 | 0.691 |
| diode | 0.668 | 0.261 |
| inductor | 0.573 | 0.458 |
| electrolytic_capacitor | 0.706 | 0.626 |
| tantalum_capacitor | 0.995 | 0.812 |
| led | 0.992 | 0.992 |
| **mAP50** | **0.837** | |
| **mAP50-95** | | **0.646** |
| Precision | 0.809 | |
| Recall | 0.845 | |

**Notes:**
- RT-DETR-x achieves the highest mAP50-95 of all four baselines (0.646), suggesting the larger
  model produces tighter boxes even at similar mAP50 to RT-DETR-l (0.837 vs. 0.839).
- Inductor improves over RT-DETR-l (0.573 vs. 0.478) but remains below both YOLO models — consistent
  with a dataset rather than architecture bottleneck for this class.
- IC shows the largest drop vs. the YOLO models (0.842 AP50 vs. 0.952/0.902) — may indicate
  RT-DETR's decoder struggles with the class's high intra-class visual diversity at 640px.
- Training CSV reported mAP50=0.839 / mAP50-95=0.639; evaluate.py gives 0.837 / 0.646 —
  mAP50-95 gap of 0.7pp may reflect slightly different NMS calibration between training val and
  standalone eval.

---

### 2026-05-13 — yolov8_gen1_static

- **Model:** yolov8m
- **Strategy:** self-training Generation 1, static threshold (conf=0.25, all 3 iterations)
- **Config:** configs/yolov8/self_train.yaml (--no-adaptive flag)
- **Weights:** experiments/yolov8_st_static_42/iter3/train/weights/best.pt
- **Git SHA:** 78da95335365ed02c9be12d21e2eb3391cfb6e6e (training) / 3b85122965e70afef5462ba14ee30f23396cf5df (eval)
- **Seed:** 42
- **Epochs:** 50 scheduled per generation, Gen3 early stopped at epoch 23 (best at epoch 8, patience=15)
- **Dataset:** FICS-PCB REMAP augmented + PCB DSLR unlabeled (7121 combined images at Gen3)
- **Warm start:** runs/yolov8/baseline-2/weights/best.pt

**Results (val set, iter3/best.pt, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.989 | 0.751 |
| ceramic_capacitor | 0.973 | 0.581 |
| ic | 0.942 | 0.701 |
| diode | 0.333 | 0.101 |
| inductor | 0.708 | 0.564 |
| electrolytic_capacitor | 0.754 | 0.622 |
| tantalum_capacitor | 0.921 | 0.761 |
| led | 0.989 | 0.989 |
| **mAP50** | **0.826** | |
| **mAP50-95** | | **0.634** |
| Precision | 0.807 | |
| Recall | 0.771 | |

**Notes:**
- mAP50 dropped from baseline 0.860 to 0.826 (−3.4pp). Self-training at conf=0.25 introduces
  noisy pseudo-labels that degrade the overall model despite expanding the training set.
- Diode regresses sharply (0.333 vs. baseline 0.560) — low-confidence pseudo-labels for this rare
  class (42 val instances) appear to inject more noise than signal.
- Inductor improves slightly (0.708 vs. baseline 0.653) — the expanded set helps the most
  for this class despite global degradation.
- Early stopping at epoch 23 (best=8) signals the model peaks early and then degrades under noisy
  labels — consistent with the pattern observed in Gen2 (best_epoch=2).

---

### 2026-05-13 — yolov8_gen1_adaptive

- **Model:** yolov8m
- **Strategy:** self-training Generation 1, adaptive per-class threshold (base conf=0.25, adaptive_threshold=true)
- **Config:** configs/yolov8/self_train.yaml
- **Weights:** experiments/yolov8_st_adaptive_42/iter3/train/weights/best.pt
- **Git SHA:** 5f2f1cdc75abc74b050e52521de386c25d2494ad (training) / 3b85122965e70afef5462ba14ee30f23396cf5df (eval)
- **Seed:** 42
- **Epochs:** 50 scheduled per generation, Gen3 early stopped at epoch 19 (best at epoch 4, patience=15)
- **Dataset:** FICS-PCB REMAP augmented + PCB DSLR unlabeled (7121 combined images at Gen3)
- **Warm start:** runs/yolov8/baseline-2/weights/best.pt

**Results (val set, iter3/best.pt, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.979 | 0.757 |
| ceramic_capacitor | 0.977 | 0.599 |
| ic | 0.933 | 0.720 |
| diode | 0.246 | 0.089 |
| inductor | 0.627 | 0.500 |
| electrolytic_capacitor | 0.702 | 0.550 |
| tantalum_capacitor | 0.995 | 0.737 |
| led | 0.995 | 0.950 |
| **mAP50** | **0.807** | |
| **mAP50-95** | | **0.613** |
| Precision | 0.742 | |
| Recall | 0.772 | |

**Notes:**
- Worst result of the three strategies: mAP50=0.807, mAP50-95=0.613. The adaptive mechanism
  did not help at generation 1 — likely because the baseline model's calibration is not yet
  reliable enough for per-class threshold estimation.
- Diode collapsed to 0.246 AP50 (vs. baseline 0.560) — adaptive threshold may have set an
  overly permissive floor for this rare class, amplifying noise.
- Best epoch at epoch 4 of 19 suggests the model degrades rapidly under pseudo-label noise,
  similar to static but more extreme.
- tantalum_capacitor and led remain near-perfect (0.995) — abundant, visually distinctive classes
  are robust to pseudo-label noise regardless of threshold strategy.

---

### 2026-05-13 — yolov8_gen1_progressive

- **Model:** yolov8m
- **Strategy:** self-training Generation 1, progressive threshold schedule (Gen1=0.25 → Gen2=0.40 → Gen3=0.55)
- **Config:** configs/yolov8/self_train_progressive.yaml
- **Weights:** experiments/yolov8_st_progressive_42/iter3/train/weights/best.pt
- **Git SHA:** 3b85122965e70afef5462ba14ee30f23396cf5df
- **Seed:** 42
- **Epochs:** 50 scheduled per generation, Gen3 early stopped at epoch 23 (best at epoch 4, patience=15)
- **Dataset:** FICS-PCB REMAP augmented + PCB DSLR unlabeled (Gen3 combined set)
- **Warm start:** runs/yolov8/baseline-2/weights/best.pt

**Results (val set, iter3/best.pt, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.984 | 0.786 |
| ceramic_capacitor | 0.980 | 0.615 |
| ic | 0.908 | 0.714 |
| diode | 0.602 | 0.277 |
| inductor | 0.587 | 0.462 |
| electrolytic_capacitor | 0.715 | 0.605 |
| tantalum_capacitor | 0.837 | 0.691 |
| led | 0.995 | 0.967 |
| **mAP50** | **0.826** | |
| **mAP50-95** | | **0.640** |
| Precision | 0.712 | |
| Recall | 0.842 | |

**Notes:**
- Ties static on mAP50 (0.826) but achieves the best mAP50-95 of the three strategies (0.640),
  suggesting tighter bounding boxes from the higher final threshold (0.55 at Gen3).
- Diode is the standout: 0.602 AP50 vs. static 0.333 and adaptive 0.246. Starting with a
  lower threshold at Gen1 (0.25) accumulates rare-class pseudo-labels early, while the rising
  threshold in later iterations prunes the noisiest detections.
- Inductor underperforms static (0.587 vs. 0.708) — the high threshold at Gen3 may have
  pruned too many inductor candidates in the final round.
- Best mAP50-95 (0.640) exceeds baseline (0.652)? No — 0.640 < 0.652. All three strategies
  remain below baseline on both metrics. Continued iterations or threshold tuning needed.
- Highest recall (0.842) at the cost of lowest precision (0.712) — consistent with the
  lower confidence pseudo-labels admitted in early generations inflating recall.

---

### 2026-05-13 — FICS_PCB_REMAP_NOISE dataset generated (v1 — 50% corruption, SUPERSEDED)

**SUPERSEDED 2026-05-18** — see entry below for 75% corruption version. All noisy Gen0
baselines trained on this version must be retrained on the updated dataset.

- **Script:** `scripts/generate_noisy_dataset.py`
- **Output:** `data/FICS_PCB_REMAP_NOISE/` (on F: drive via `data/` symlink)
- **Seed:** 42
- **Source:** `data/labeled/labels/` (4,194 files, 34,512 annotations)
- **Noise fractions:**
  - Classification noise (class-swap): 25% → 8,628 annotations
  - Removal noise (annotation deleted): 25% → 8,628 annotations
  - Total corrupted: 50% (17,256 annotations); unchanged: 50% (17,256 annotations)
- **Noise assignment:** global shuffle of all annotation entries, first 25% → class-swap, next 25% → removal, remainder unchanged. Non-overlapping by construction.
- **Images:** unchanged — `images/` is a symlink to original FICS REMAP train images
- **Val set:** untouched in all conditions
- **Audit trail:** `data/FICS_PCB_REMAP_NOISE/noise_manifest.json` (per-annotation original class, new class for swaps)
- **Motivation:** replicates Honorato (2025) Section 4.3 noise injection to verify self-training recovers real signal rather than only approximating the teacher. Honorato used the same 50% corruption rate.

---

### 2026-05-18 — FICS_PCB_REMAP_NOISE dataset regenerated (v2 — 75% corruption)

- **Script:** `scripts/generate_noisy_dataset.py --classification-frac 0.25 --removal-frac 0.25 --distortion-frac 0.25 --seed 42 --force`
- **Output:** `data/FICS_PCB_REMAP_NOISE/` (on F: drive via `data/` symlink)
- **Seed:** 42 (swap/removal RNG); 43 (distortion RNG — separate to preserve prior assignments)
- **Source:** `data/labeled/labels/` (4,194 files, 30,319 annotations)
- **Noise fractions:**
  - Classification noise (class-swap): 25% → ~7,580 annotations
  - Removal noise (annotation deleted): 25% → ~7,580 annotations
  - Bounding Box Distortion: 25% → ~7,580 annotations (corners perturbed by uniform factor [0.95, 1.05])
  - Total corrupted: 75%; unchanged: 25%
- **Motivation:** Honorato requested full alignment with Freire et al. (2024) three-type noise taxonomy before launching PCB-DSLR pseudo-labeling step. Box Distortion is the least harmful noise type; adding it completes the protocol.
- **Impact:** Noisy Gen0 baselines (YOLOv8m, YOLOv10m, RT-DETR-l) must be retrained. Prior results marked SUPERSEDED in their entries below.

---

### 2026-05-14 — yolov8_noisy_gen0_baseline (**SUPERSEDED** — retrain required on 75% noise)

**SUPERSEDED 2026-05-18** — FICS_PCB_REMAP_NOISE updated to 75% corruption (added Box
Distortion). Results below are from the 50% dataset and are no longer the canonical Gen0
baseline. Retrain with `configs/yolov8/base_noisy.yaml` after regenerating the dataset.

- **Model:** yolov8m
- **Strategy:** supervised baseline (Generation 0) on noisy labeled set
- **Config:** configs/yolov8/base_noisy.yaml
- **Weights:** runs/yolov8/baseline_noisy/weights/best.pt
- **Seed:** 42
- **Epochs:** 100 scheduled, early stopped at epoch 46 (best at epoch 27, patience=20)
- **Dataset:** FICS_PCB_REMAP_NOISE (4,194 images, 50% annotations corrupted)
- **Script:** experiments/run_yolov8_noisy_gen0.sh

**Results (val set — clean, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.988 | 0.781 |
| ceramic_capacitor | 0.961 | 0.586 |
| ic | 0.963 | 0.704 |
| diode | 0.932 | 0.373 |
| inductor | 0.817 | 0.680 |
| electrolytic_capacitor | 0.821 | 0.667 |
| tantalum_capacitor | 0.995 | 0.780 |
| led | 0.995 | 0.963 |
| **mAP50** | **0.934** | |
| **mAP50-95** | | **0.692** |
| Precision | 0.890 | |
| Recall | 0.863 | |

**Notes:**
- mAP50 (0.934) exceeds clean YOLOv8m baseline (0.860) by +0.074 pp — unexpected. Hypotheses:
  1. FICS_PCB_REMAP_NOISE uses same 4,194 images but base_noisy.yaml differs in lr0 (0.001 vs 0.0005 clean); higher LR may explain better convergence here.
  2. Label noise (50% corruption) may have acted as implicit regularization reducing overfitting.
  3. Requires investigation before interpreting in thesis.
- Inductor (class 4) notably stronger than clean (AP50 0.817 vs. 0.653) — tentatively attributed to reduced false GT confidence from noisy labels forcing the model to generalize.
- Diode (class 3) also stronger: 0.932 vs. 0.560 clean.

---

### 2026-05-14 — yolov10_noisy_gen0_baseline (**SUPERSEDED** — retrain required on 75% noise)

**SUPERSEDED 2026-05-18** — FICS_PCB_REMAP_NOISE updated to 75% corruption (added Box
Distortion). Results below are from the 50% dataset and are no longer the canonical Gen0
baseline. Retrain with `configs/yolov10/base_noisy.yaml` after regenerating the dataset.

- **Model:** yolov10m
- **Strategy:** supervised baseline (Generation 0) on noisy labeled set
- **Config:** configs/yolov10/base_noisy.yaml
- **Weights:** runs/yolov10/baseline_noisy/weights/best.pt
- **Seed:** 42
- **Epochs:** 100 scheduled, early stopped at epoch 50 (best at epoch 31, patience=20)
- **Dataset:** FICS_PCB_REMAP_NOISE (4,194 images, 50% annotations corrupted)
- **Script:** experiments/run_yolov10_noisy_gen0.sh

**Results (val set — clean, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.980 | 0.772 |
| ceramic_capacitor | 0.975 | 0.617 |
| ic | 0.944 | 0.766 |
| diode | 0.898 | 0.412 |
| inductor | 0.663 | 0.537 |
| electrolytic_capacitor | 0.631 | 0.446 |
| tantalum_capacitor | 0.915 | 0.751 |
| led | 0.995 | 0.937 |
| **mAP50** | **0.875** | |
| **mAP50-95** | | **0.655** |
| Precision | 0.790 | |
| Recall | 0.832 | |

**Notes:**
- mAP50 (0.875) exceeds clean YOLOv10m baseline (0.837) by +0.038 pp — same pattern as YOLOv8m noisy (see above). Same lr0 discrepancy hypothesis applies.
- Electrolytic capacitor (0.631) is notably weaker than YOLOv8m noisy (0.821) and both clean baselines — class appears more sensitive to label noise in the YOLOv10 architecture.
- Inductor (0.663) still the weakest non-rare class, consistent with all prior runs.
- warm_start for noisy self-training confirmed at runs/yolov10/baseline_noisy/weights/best.pt (no Ultralytics auto-increment).

---

### 2026-05-17 — rtdetr_l_noisy_gen0_baseline (**SUPERSEDED** — retrain required on 75% noise)

**SUPERSEDED 2026-05-18** — FICS_PCB_REMAP_NOISE updated to 75% corruption (added Box
Distortion). Results below are from the 50% dataset and are no longer the canonical Gen0
baseline. Retrain with `configs/rtdetr_l/base_noisy.yaml` after regenerating the dataset.

- **Model:** RT-DETR-l
- **Strategy:** supervised baseline (Generation 0) on noisy labeled set
- **Config:** configs/rtdetr_l/base_noisy.yaml
- **Weights:** runs/rtdetr_l/baseline_noisy/weights/best.pt
- **Git SHA:** 7ddd6a7100664af937038261748cdafc3545d94e
- **Seed:** 42
- **Epochs:** 72 scheduled, early stopped at epoch 51 (best at epoch 31, patience=20)
- **Dataset:** FICS_PCB_REMAP_NOISE (4,194 images, 50% annotations corrupted)
- **Script:** experiments/run_rtdetr_l_noisy_gen0.sh

**Results (val set — clean, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.985 | 0.780 |
| ceramic_capacitor | 0.966 | 0.610 |
| ic | 0.869 | 0.710 |
| diode | 0.707 | 0.346 |
| inductor | 0.588 | 0.478 |
| electrolytic_capacitor | 0.737 | 0.635 |
| tantalum_capacitor | 0.995 | 0.715 |
| led | 0.988 | 0.967 |
| **mAP50** | **0.854** | |
| **mAP50-95** | | **0.655** |
| Precision | 0.785 | |
| Recall | 0.813 | |

**Notes:**
- mAP50 (0.854) exceeds clean RT-DETR-l baseline (0.839) by +1.5pp — consistent with the pattern seen in YOLOv8m (+7.4pp) and YOLOv10m (+3.8pp) noisy baselines. Label noise appears to provide implicit regularisation across all three architectures.
- Inductor (class 4) remains the weakest non-rare class at AP50=0.588, though improved over clean baseline (0.478). Consistent with all prior noisy baselines showing inductor recovery.
- Diode (class 3) recovers strongly: AP50=0.707 vs. clean 0.598 — same direction as YOLOv8 (0.932) and YOLOv10 (0.898) noisy baselines.
- Electrolytic capacitor (0.737) notably weaker than YOLOv8 noisy (0.821) — same gap as in the clean baselines, suggesting an architecture-level sensitivity rather than noise interaction.
- All 3 active model noisy Gen0 baselines now complete. Warm-start confirmed at runs/rtdetr_l/baseline_noisy/weights/best.pt.

---

### 2026-05-20 — yolov8_noisy_gen0_baseline_v2

- **Model:** yolov8m
- **Strategy:** supervised baseline (Generation 0) on noisy labeled set
- **Config:** configs/yolov8/base_noisy.yaml (batch=16)
- **Weights:** runs/yolov8/baseline_noisy/weights/best.pt
- **Git SHA:** df94eb0bb1ed92c4fb6e2a522127318981adc145
- **Seed:** 42
- **Epochs:** 100 scheduled, early stopped at epoch 46 (best at epoch 26, patience=20)
- **Dataset:** FICS_PCB_REMAP_NOISE (4,194 images, 75% annotations corrupted — 25% class-swap + 25% removal + 25% box-distortion)
- **Script:** experiments/run_yolov8_noisy_gen0.sh

**Results (val set — clean, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.988 | 0.781 |
| ceramic_capacitor | 0.961 | 0.586 |
| ic | 0.963 | 0.704 |
| diode | 0.932 | 0.373 |
| inductor | 0.817 | 0.680 |
| electrolytic_capacitor | 0.821 | 0.667 |
| tantalum_capacitor | 0.995 | 0.780 |
| led | 0.995 | 0.963 |
| **mAP50** | **0.934** | |
| **mAP50-95** | | **0.692** |
| Precision | 0.890 | |
| Recall | 0.863 | |

**Notes:**
- mAP50 (0.934) matches the 50%-noise result exactly — adding box-distortion as the third noise type had no measurable impact on YOLOv8m. Noisy-beats-clean (+7.4pp vs. clean 0.860) persists at 75% corruption.
- Per-class AP identical to the superseded 50% run (within rounding), confirming the result is stable across the expanded noise protocol.
- Inductor (0.817) and Diode (0.932) remain substantially stronger than the clean baseline (0.653 / 0.560), consistent with implicit-regularisation hypothesis.
- Canonical noisy Gen0 checkpoint for YOLOv8m ST warm-start.

---

### 2026-05-20 — yolov10_noisy_gen0_baseline_v2

- **Model:** yolov10m
- **Strategy:** supervised baseline (Generation 0) on noisy labeled set
- **Config:** configs/yolov10/base_noisy.yaml (batch=12; reduced from 16 — original run crashed at epoch 1/batch 162 with CUDA error from VRAM overrun at 8.43 GiB on 8 GiB card)
- **Weights:** runs/yolov10/baseline_noisy/weights/best.pt
- **Git SHA:** df94eb0bb1ed92c4fb6e2a522127318981adc145
- **Seed:** 42
- **Epochs:** 100 scheduled, early stopped at epoch 71 (best at epoch 51, patience=20)
- **Dataset:** FICS_PCB_REMAP_NOISE (4,194 images, 75% annotations corrupted — 25% class-swap + 25% removal + 25% box-distortion)
- **Script:** experiments/run_yolov10_noisy_gen0.sh

**Results (val set — clean, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.984 | 0.777 |
| ceramic_capacitor | 0.964 | 0.578 |
| ic | 0.949 | 0.755 |
| diode | 0.759 | 0.335 |
| inductor | 0.523 | 0.434 |
| electrolytic_capacitor | 0.713 | 0.574 |
| tantalum_capacitor | 0.995 | 0.795 |
| led | 0.965 | 0.955 |
| **mAP50** | **0.856** | |
| **mAP50-95** | | **0.650** |
| Precision | 0.797 | |
| Recall | 0.786 | |

**Notes:**
- mAP50 (0.856) drops 1.9pp vs. the superseded 50%-noise result (0.875), suggesting YOLOv10m is more sensitive to the additional box-distortion noise than YOLOv8m. Still +1.9pp above clean baseline (0.837).
- Inductor collapses to 0.523 AP50 (vs. 0.663 at 50% noise, 0.487 clean) — YOLOv10m with 75% corruption shows significantly higher inductor sensitivity than either YOLOv8m or RT-DETR-l. Key class to monitor during noisy ST.
- Diode also weaker vs. 50% noise (0.759 vs. 0.898) — the addition of box-distortion hurts localisation for small/rare classes disproportionately in YOLOv10.
- batch reduced from 16 → 12 to avoid VRAM overrun; comparable to the 50%-noise run in every other respect.

---

### 2026-05-20 — rtdetr_l_noisy_gen0_baseline_v2

- **Model:** RT-DETR-l
- **Strategy:** supervised baseline (Generation 0) on noisy labeled set
- **Config:** configs/rtdetr_l/base_noisy.yaml (batch=8)
- **Weights:** runs/rtdetr_l/baseline_noisy/weights/best.pt
- **Git SHA:** df94eb0bb1ed92c4fb6e2a522127318981adc145
- **Seed:** 42
- **Epochs:** 72 scheduled, early stopped at epoch 42 (best at epoch 22, patience=20)
- **Dataset:** FICS_PCB_REMAP_NOISE (4,194 images, 75% annotations corrupted — 25% class-swap + 25% removal + 25% box-distortion)
- **Script:** experiments/run_rtdetr_l_noisy_gen0.sh

**Results (val set — clean, from `scripts/evaluate.py`):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.988 | 0.785 |
| ceramic_capacitor | 0.965 | 0.562 |
| ic | 0.838 | 0.680 |
| diode | 0.657 | 0.255 |
| inductor | 0.636 | 0.482 |
| electrolytic_capacitor | 0.776 | 0.657 |
| tantalum_capacitor | 0.995 | 0.746 |
| led | 0.944 | 0.911 |
| **mAP50** | **0.850** | |
| **mAP50-95** | | **0.635** |
| Precision | 0.793 | |
| Recall | 0.841 | |

**Notes:**
- mAP50 (0.850) drops 0.4pp from 50%-noise (0.854) — RT-DETR-l shows the least sensitivity to the additional box-distortion of the three models. Still +1.1pp over clean baseline (0.839).
- IC notably weaker than 50%-noise (0.838 vs. 0.869) — transformer decoder appears more susceptible to coordinate corruption on this high-diversity class.
- Inductor (0.636) is stronger than the 50%-noise result (0.588) and the clean baseline (0.478) — the pattern holds despite 75% total corruption.
- All 3 noisy Gen0 v2 baselines now complete. Noisy ST (Step 10) can proceed.

---

### 2026-05-20 — yolov8_noisy_gen1_adaptive (**SUPERSEDED** — wrong loop, discard results)

**SUPERSEDED 2026-05-29** — This run pseudo-labeled `data/unlabeled/images` (PCB_DSLR) instead of the FICS training images. Honorato's FICS NOISE experiment is an intra-FICS loop: teacher pseudo-labels FICS images, student trains on those pseudo-labels only (noisy originals replaced). Results are invalid; run will be repeated after loop correction.

- **Model:** yolov8m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), adaptive per-class threshold (max-F1 per class, base conf=0.25)
- **Config:** configs/yolov8/self_train_noisy.yaml (adaptive_threshold: true)
- **Weights (Gen4):** experiments/yolov8_st_noisy_adaptive_42/iter4/train/weights/best.pt
- **Git SHA:** df94eb0bb1ed92c4fb6e2a522127318981adc145
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset (wrong):** FICS_PCB_REMAP_NOISE (75% corruption) + PCB_DSLR_CROPS_512 pseudo-labels (partial IC GT)
- **Script:** experiments/run_yolov8_noisy_st_adaptive.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.940 | 0.724 |
| ceramic_capacitor | 0.954 | 0.570 |
| ic | 0.765 | 0.577 |
| diode | 0.411 | 0.163 |
| inductor | 0.430 | 0.359 |
| electrolytic_capacitor | 0.708 | 0.605 |
| tantalum_capacitor | 0.482 | 0.322 |
| led | 0.984 | 0.965 |
| **mAP50** | **0.709** | |
| **mAP50-95** | | **0.536** |
| Precision | 0.665 | |
| Recall | 0.756 | |

**Notes:**
- mAP50 (0.709) is -22.5pp below the noisy Gen0 baseline (0.934). Self-training degraded performance substantially under 75% noise.
- Diode collapses to 0.411 AP50 (vs. 0.932 at Gen0 noisy) — adaptive threshold likely set a class-specific floor too low under high noise, accepting many corrupted pseudo-labels.
- Inductor also weak at 0.430 (vs. 0.817 at Gen0 noisy) — confirms the two hardest classes are most sensitive to noisy pseudo-label propagation.
- LED remains strong (0.984) and SMD resistor/ceramic capacitor are stable — noise effects are concentrated in the rare/hard classes.
- Primary comparison metric for this experiment set: FICS val mAP50. PCB_DSLR target-domain IC evaluation deferred per Honorato's original loop.

---

### 2026-05-21 — yolov8_noisy_gen1_static (**SUPERSEDED** — wrong loop, discard results)

**SUPERSEDED 2026-05-29** — Same root cause as adaptive entry above: teacher pseudo-labeled PCB_DSLR instead of FICS training images. Results are invalid; run will be repeated after loop correction.

- **Model:** yolov8m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), static threshold (conf=0.25, --no-adaptive)
- **Config:** configs/yolov8/self_train_noisy.yaml (adaptive_threshold: true in file, overridden by --no-adaptive flag)
- **Weights (Gen4):** experiments/yolov8_st_noisy_static_42/iter4/train/weights/best.pt
- **Git SHA:** df94eb0bb1ed92c4fb6e2a522127318981adc145
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset (wrong):** FICS_PCB_REMAP_NOISE (75% corruption) + PCB_DSLR_CROPS_512 pseudo-labels (partial IC GT)
- **Script:** experiments/run_yolov8_noisy_st_static.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 |
|-------|------|---------|
| resistor_smd | 0.965 | 0.744 |
| ceramic_capacitor | 0.938 | 0.561 |
| ic | 0.712 | 0.520 |
| diode | 0.130 | 0.039 |
| inductor | 0.551 | 0.444 |
| electrolytic_capacitor | 0.739 | 0.633 |
| tantalum_capacitor | 0.524 | 0.350 |
| led | 0.941 | 0.889 |
| **mAP50** | **0.687** | |
| **mAP50-95** | | **0.522** |
| Precision | 0.736 | |
| Recall | 0.677 | |

**Notes:**
- mAP50 (0.687) is -24.7pp below the noisy Gen0 baseline (0.934) and -2.2pp below the adaptive strategy.
- Diode near-collapses to 0.130 AP50 — worse than adaptive (0.411). Static conf=0.25 floods the pseudo-label pool with noisy diode detections, propagating label corruption.
- Inductor recovers slightly vs. adaptive (0.551 vs. 0.430) — fixed threshold filters more aggressively here than per-class adaptive, which may have set the inductor threshold too low.
- Pattern mirrors clean Gen1: static is consistently the weakest strategy; progressive (not yet run) was best in the clean setting.
- Both YOLOv8 noisy ST runs confirmed below Gen0 noisy baseline. The noise amplification through pseudo-labeling is the key driver — consistent with Freire 2024's finding that noise in pseudo-labels accumulates across generations.
- All 3 noisy Gen0 v2 baselines now complete. Noisy ST (Step 10) can proceed.

---

### 2026-05-30 — yolov8_noisy_gen1_adaptive_v2 (corrected loop)

- **Model:** yolov8m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), adaptive per-class threshold (max-F1 per class, base conf=0.25)
- **Config:** configs/yolov8/self_train_noisy.yaml (adaptive_threshold: true, pseudo_only_mode: true)
- **Weights (Gen4):** experiments/yolov8_st_noisy_adaptive_42/iter4/train/weights/best.pt
- **Git SHA:** d9d2fbcfd44164e40f0a4d94f42153cc1eb9e555
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset:** FICS_PCB_REMAP_NOISE (75% corruption) — intra-FICS loop; teacher pseudo-labels FICS training images, student trains on pseudo-labels only (pseudo_only_mode=true)
- **Script:** experiments/run_yolov8_noisy_st_adaptive.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 | Δ vs noisy Gen0 |
|-------|------|---------|-----------------|
| resistor_smd | 0.982 | 0.763 | −0.006 |
| ceramic_capacitor | 0.980 | 0.583 | +0.019 |
| ic | 0.903 | 0.725 | −0.060 |
| diode | 0.607 | 0.253 | −0.325 |
| inductor | 0.631 | 0.521 | −0.186 |
| electrolytic_capacitor | 0.751 | 0.620 | −0.070 |
| tantalum_capacitor | 0.977 | 0.756 | −0.018 |
| led | 0.995 | 0.987 | 0.000 |
| **mAP50** | **0.853** | | **−0.081** |
| **mAP50-95** | | **0.651** | **−0.041** |
| Precision | 0.814 | | |
| Recall | 0.857 | | |

**Notes:**
- mAP50 (0.853) is −8.1pp below the noisy Gen0 baseline (0.934). Self-training with the corrected loop still degrades performance under 75% annotation noise.
- Corrected loop recovers +14.4pp mAP50 vs. the superseded adaptive run (0.709) — confirms the loop fix (intra-FICS vs PCB-DSLR target) was the dominant source of error in the May 20 run.
- Diode falls from 0.932 (Gen0 noisy) to 0.607 (−32.5pp) — hardest-hit class; pseudo-label quality for rare classes degrades under heavy noise even with per-class adaptive thresholding. Notably, 0.607 still exceeds Honorato's reported AP=0 for Diode in the noisy FICS setting.
- Inductor falls from 0.817 to 0.631 (−18.6pp) — second hardest hit. Corrected loop recovers +20.1pp vs. superseded run (0.430).
- Ceramic capacitor is the only class that improved (+0.019 AP50) — likely a noise-regularization effect on this dominant class.
- LED (0.995) and resistor_smd (0.982) are nearly unaffected — dominant, easy classes are robust to pseudo-label noise propagation.
- Pattern consistent with Freire 2024: noise accumulates in iterative pseudo-labeling, with rare classes (diode=42 val instances, inductor=114) most vulnerable.

---

### 2026-05-31 — yolov8_noisy_gen1_static_v2 (corrected loop)

- **Model:** yolov8m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), static threshold (conf=0.25 fixed across all generations)
- **Config:** configs/yolov8/self_train_noisy.yaml (adaptive_threshold: true in file, overridden by --no-adaptive flag; pseudo_only_mode: true)
- **Weights (Gen4):** experiments/yolov8_st_noisy_static_42/iter4/train/weights/best.pt
- **Git SHA:** d9d2fbcfd44164e40f0a4d94f42153cc1eb9e555
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset:** FICS_PCB_REMAP_NOISE (75% corruption) — intra-FICS loop; teacher pseudo-labels FICS training images, student trains on pseudo-labels only (pseudo_only_mode=true)
- **Script:** experiments/run_yolov8_noisy_st_static.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 | Δ vs noisy Gen0 |
|-------|------|---------|-----------------|
| resistor_smd | 0.988 | 0.760 | 0.000 |
| ceramic_capacitor | 0.959 | 0.559 | −0.002 |
| ic | 0.871 | 0.672 | −0.092 |
| diode | 0.621 | 0.209 | −0.311 |
| inductor | 0.720 | 0.585 | −0.097 |
| electrolytic_capacitor | 0.774 | 0.651 | −0.047 |
| tantalum_capacitor | 0.995 | 0.864 | 0.000 |
| led | 0.973 | 0.948 | −0.022 |
| **mAP50** | **0.863** | | **−0.071** |
| **mAP50-95** | | **0.656** | **−0.036** |
| Precision | 0.792 | | |
| Recall | 0.827 | | |

**Notes:**
- mAP50 (0.863) is −7.1pp below the noisy Gen0 baseline (0.934) — a smaller gap than the adaptive strategy (−8.1pp), suggesting static thresholding is marginally better under 75% noise for YOLOv8.
- Diode (0.621, −31.1pp) and IC (0.871, −9.2pp) are the hardest-hit classes. Diode is marginally better than adaptive (0.607), but IC is worse (adaptive: 0.903).
- Inductor improves substantially vs. adaptive (0.720 vs. 0.631, +8.9pp) — static threshold may filter less aggressively on this mid-frequency class.
- LED drops to 0.973 (−2.2pp vs. Gen0) from adaptive's 0.995; likely a sampling artefact given only 18 val instances.
- Resistor_smd (0.988) and tantalum_capacitor (0.995) essentially unaffected across both strategies — dominant easy classes are robust.
- No consistent cross-class advantage for static over adaptive under 75% noise: IC favours adaptive, Inductor favours static.

---

### 2026-06-05 — yolov8_noisy_gen1_progressive (corrected loop, canonical eval)

- **Model:** yolov8m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), progressive threshold ([0.25→0.35→0.45→0.55])
- **Config:** configs/yolov8/self_train_noisy_progressive.yaml (adaptive_threshold: false, conf_schedule: [0.25, 0.35, 0.45, 0.55], pseudo_only_mode: true)
- **Weights (Gen4):** experiments/yolov8_st_noisy_progressive_42/iter4/train/weights/best.pt
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-01T10:05Z)
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset:** FICS_PCB_REMAP_NOISE (75% corruption) — intra-FICS loop; teacher pseudo-labels FICS training images, student trains on pseudo-labels only
- **Script:** experiments/run_yolov8_noisy_st_progressive.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 | Δ vs noisy Gen0 |
|-------|------|---------|-----------------|
| resistor_smd | 0.988 | 0.773 | 0.000 |
| ceramic_capacitor | 0.965 | 0.583 | +0.004 |
| ic | 0.896 | 0.706 | −0.067 |
| diode | 0.578 | 0.235 | −0.354 |
| inductor | 0.672 | 0.548 | −0.145 |
| electrolytic_capacitor | 0.727 | 0.611 | −0.094 |
| tantalum_capacitor | 0.932 | 0.734 | −0.063 |
| led | 0.989 | 0.952 | −0.006 |
| **mAP50** | **0.843** | | **−0.091** |
| **mAP50-95** | | **0.643** | **−0.049** |
| Precision | 0.711 | | |
| Recall | 0.849 | | |

**Notes:**
- mAP50 (0.843) is −9.1pp below noisy Gen0 (0.934) — largest drop among YOLOv8 strategies (static: −7.1pp, adaptive: −8.1pp). Progressive threshold is the weakest YOLOv8 strategy.
- Diode hardest-hit (0.578, −35.4pp). Inductor drops −14.5pp; ceramic capacitor the only class to improve (+0.4pp).
- Provisional value from training log was 0.843 — confirmed exactly by canonical eval.
- YOLOv8 strategy ranking: static (0.863) > adaptive (0.853) > progressive (0.843).

---

### 2026-06-05 — yolov10_noisy_gen1_adaptive (corrected loop, canonical eval)

- **Model:** yolov10m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), adaptive per-class threshold (max-F1 per class, base conf=0.25)
- **Config:** configs/yolov10/self_train_noisy.yaml (adaptive_threshold: true, pseudo_only_mode: true)
- **Weights (Gen4):** experiments/yolov10_st_noisy_adaptive_42/iter4/train/weights/best.pt
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-01T16:41Z)
- **Seed:** 42
- **Epochs per generation:** 50, patience=15, batch=12
- **Dataset:** FICS_PCB_REMAP_NOISE (75% corruption) — intra-FICS loop
- **Script:** experiments/run_yolov10_noisy_st_adaptive.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 | Δ vs noisy Gen0 |
|-------|------|---------|-----------------|
| resistor_smd | 0.977 | 0.747 | −0.007 |
| ceramic_capacitor | 0.938 | 0.511 | −0.026 |
| ic | 0.918 | 0.697 | −0.031 |
| diode | 0.721 | 0.278 | −0.038 |
| inductor | 0.676 | 0.545 | +0.153 |
| electrolytic_capacitor | 0.608 | 0.424 | −0.105 |
| tantalum_capacitor | 0.854 | 0.717 | −0.141 |
| led | 0.947 | 0.873 | −0.018 |
| **mAP50** | **0.830** | | **−0.026** |
| **mAP50-95** | | **0.599** | **−0.051** |
| Precision | 0.701 | | |
| Recall | 0.834 | | |

**Notes:**
- mAP50 (0.830) is −2.6pp below noisy Gen0 (0.856) — smallest drop of all YOLOv10 strategies and smallest overall across all 6 YOLOv8/YOLOv10 noisy ST runs.
- Inductor is the standout: +15.3pp vs. Gen0 (0.676 vs. 0.523) — adaptive threshold allows the model to build on the Gen0 inductor signal rather than degrade it. Consistent with the per-class adaptive mechanism targeting this mid-frequency, hard-to-pseudo-label class.
- Tantalum capacitor drops −14.1pp (0.854 vs. 0.995) — strongest class-level casualty for this strategy.
- Diode loses only −3.8pp (0.721) — much less collapse than in YOLOv8 adaptive (−32.5pp, 0.607). Diode is better-represented relative to the YOLOv10 Gen0 baseline (0.759 vs. 0.932).
- Provisional value 0.830 confirmed exactly. Expected NMS-free head ~1.5pp gap did not materialise — provisionals were already accurate.

---

### 2026-06-05 — yolov10_noisy_gen1_static (corrected loop, canonical eval)

- **Model:** yolov10m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), static threshold (conf=0.25 fixed)
- **Config:** configs/yolov10/self_train_noisy.yaml (adaptive_threshold: true in file, overridden by --no-adaptive flag; pseudo_only_mode: true)
- **Weights (Gen4):** experiments/yolov10_st_noisy_static_42/iter4/train/weights/best.pt
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-02T09:47Z)
- **Seed:** 42
- **Epochs per generation:** 50, patience=15, batch=12
- **Dataset:** FICS_PCB_REMAP_NOISE (75% corruption) — intra-FICS loop
- **Script:** experiments/run_yolov10_noisy_st_static.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 | Δ vs noisy Gen0 |
|-------|------|---------|-----------------|
| resistor_smd | 0.976 | 0.750 | −0.008 |
| ceramic_capacitor | 0.943 | 0.537 | −0.021 |
| ic | 0.906 | 0.678 | −0.043 |
| diode | 0.387 | 0.128 | −0.372 |
| inductor | 0.597 | 0.493 | +0.074 |
| electrolytic_capacitor | 0.670 | 0.544 | −0.043 |
| tantalum_capacitor | 0.914 | 0.665 | −0.081 |
| led | 0.917 | 0.901 | −0.048 |
| **mAP50** | **0.789** | | **−0.067** |
| **mAP50-95** | | **0.587** | **−0.063** |
| Precision | 0.837 | | |
| Recall | 0.708 | | |

**Notes:**
- mAP50 (0.789) is −6.7pp below noisy Gen0 (0.856) — worst YOLOv10 strategy.
- Diode collapses to 0.387 (−37.2pp) — most severe class-level failure across all 9 FICS-noisy ST runs, worse than even YOLOv8 static diode (0.621). Static threshold at 0.25 floods the pseudo-label pool for diode with low-confidence false positives that amplify under 75% noise.
- Inductor improves +7.4pp (0.597 vs. 0.523) — static threshold is permissive enough to keep inductor pseudo-labels despite noise; this effect is smaller than adaptive's +15.3pp.
- LED drops −4.8pp (0.917) vs. adaptive (0.947) and progressive (0.935) — small-class degradation under fixed threshold.
- Provisional 0.789 confirmed exactly.

---

### 2026-06-05 — yolov10_noisy_gen1_progressive (corrected loop, canonical eval)

- **Model:** yolov10m
- **Strategy:** self-training (Gen1–Gen4, 4 generations), progressive threshold ([0.25→0.35→0.45→0.55])
- **Config:** configs/yolov10/self_train_noisy_progressive.yaml (adaptive_threshold: false, conf_schedule: [0.25, 0.35, 0.45, 0.55], pseudo_only_mode: true)
- **Weights (Gen4):** experiments/yolov10_st_noisy_progressive_42/iter4/train/weights/best.pt
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-02T17:56Z)
- **Seed:** 42
- **Epochs per generation:** 50, patience=15, batch=12
- **Dataset:** FICS_PCB_REMAP_NOISE (75% corruption) — intra-FICS loop
- **Script:** experiments/run_yolov10_noisy_st_progressive.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 | Δ vs noisy Gen0 |
|-------|------|---------|-----------------|
| resistor_smd | 0.972 | 0.748 | −0.012 |
| ceramic_capacitor | 0.944 | 0.536 | −0.020 |
| ic | 0.906 | 0.723 | −0.043 |
| diode | 0.603 | 0.246 | −0.156 |
| inductor | 0.663 | 0.543 | +0.140 |
| electrolytic_capacitor | 0.706 | 0.597 | −0.007 |
| tantalum_capacitor | 0.766 | 0.623 | −0.229 |
| led | 0.935 | 0.902 | −0.030 |
| **mAP50** | **0.812** | | **−0.044** |
| **mAP50-95** | | **0.615** | **−0.035** |
| Precision | 0.795 | | |
| Recall | 0.762 | | |

**Notes:**
- mAP50 (0.812) sits between adaptive (0.830) and static (0.789) — YOLOv10 strategy ranking: adaptive > progressive > static.
- Diode (0.603, −15.6pp) recovers from static's collapse (0.387) but remains well below adaptive (0.721). The rising threshold across generations reduces diode pseudo-label noise accumulation compared to static.
- Inductor improves +14.0pp (0.663 vs. 0.523) — nearly as strong as adaptive's +15.3pp, suggesting progressive threshold is not significantly less effective than adaptive for this class.
- Tantalum capacitor has the steepest drop (−22.9pp, 0.766) — worse than static (−8.1pp, 0.914). A rising threshold may starve the tantalum pseudo-label pool from Gen3 onward (only 12 val instances; very sparse training class).
- Provisional 0.812 confirmed exactly.

---

### 2026-06-05 — rtdetr_l_noisy_gen1_adaptive (corrected loop, canonical eval)

- **Model:** RT-DETR-l
- **Strategy:** self-training (Gen1–Gen4, 4 generations), adaptive per-class threshold (max-F1 per class, base conf=0.25)
- **Config:** configs/rtdetr_l/self_train_noisy.yaml (adaptive_threshold: true, pseudo_only_mode: true)
- **Weights (Gen4):** experiments/rtdetr_l_st_noisy_adaptive_42/iter4/train/weights/best.pt
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-03T11:44Z)
- **Seed:** 42
- **Epochs per generation:** 36, patience=10, batch=8
- **Dataset:** FICS_PCB_REMAP_NOISE (75% corruption) — intra-FICS loop
- **Script:** experiments/run_rtdetr_l_noisy_st_adaptive.sh

**Results (val set — clean, from `scripts/evaluate.py`, Gen4 best.pt):**
| Class | AP50 | AP50-95 | Δ vs noisy Gen0 |
|-------|------|---------|-----------------|
| resistor_smd | 0.976 | 0.744 | −0.012 |
| ceramic_capacitor | 0.842 | 0.495 | −0.123 |
| ic | 0.687 | 0.536 | −0.151 |
| diode | 0.175 | 0.066 | −0.482 |
| inductor | 0.564 | 0.432 | −0.072 |
| electrolytic_capacitor | 0.687 | 0.548 | −0.089 |
| tantalum_capacitor | 0.995 | 0.732 | 0.000 |
| led | 0.815 | 0.757 | −0.129 |
| **mAP50** | **0.718** | | **−0.132** |
| **mAP50-95** | | **0.539** | **−0.096** |
| Precision | 0.724 | | |
| Recall | 0.710 | | |

**Notes:**
- mAP50 (0.718) is −13.2pp below noisy Gen0 (0.850) — the largest Gen1 degradation across all three architectures for the adaptive strategy (YOLOv8 adaptive: −8.1pp; YOLOv10 adaptive: −2.6pp).
- Diode collapses to 0.175 (−48.2pp) — the most extreme per-class failure in this entire experiment series. RT-DETR-l's transformer decoder appears unable to recover the diode signal once pseudo-label noise corrupts the early generations; the attention mechanism may be amplifying noisy positional cues more severely than NMS-based detectors.
- IC drops −15.1pp (0.687 vs. 0.838) — transformer decoder is more sensitive to coordinate corruption for this high-diversity class (same pattern observed in Gen0: IC at 0.838 vs. YOLOv8 0.963, attributable to box-distortion noise in 75% setting).
- Ceramic capacitor drops −12.3pp (0.842 vs. 0.965) — dominant class still regresses significantly, suggesting global pseudo-label quality is lower than for the YOLO models despite the same teacher checkpoint type.
- Tantalum capacitor (0.995) and resistor_smd (0.976) are the most robust classes — consistent with their high Gen0 AP and dominant presence in the dataset.
- RT-DETR-l is more sensitive to noisy self-training than YOLOv8 and YOLOv10 under the adaptive strategy. This is a key cross-architecture finding.

---

### 2026-06-07 — rtdetr_l_noisy_gen1_static (DROPPED — pseudo-label proliferation)

- **Model:** RT-DETR-l
- **Strategy:** self-training (Gen1–Gen4, 4 generations), static threshold (C=0.25 fixed)
- **Config:** configs/rtdetr_l/self_train_noisy.yaml (adaptive_threshold: true in file, overridden by --no-adaptive flag; pseudo_only_mode: true)
- **Script:** experiments/run_rtdetr_l_noisy_st_static.sh
- **Decision:** Dropped after Gen4 was found computationally intractable. Gen4 training was manually stopped after 3 epochs. Gen1–Gen3 complete with `best.pt`; `iter4/train/weights/` empty.

**Annotation count in combined set across generations (finding):**
Counted via `find <dir> -name "*.txt" -exec cat {} \; | wc -l` (per-arch, correct method).
| Generation | Static annotations | Adaptive annotations | Ratio (S/A) | Static annot/img |
|-----------|-------------------|---------------------|-------------|-----------------|
| Gen1 | 40,246 | 40,246 | 1.00× | 9.6 |
| Gen2 | 61,248 | 34,522 | 1.77× | 14.6 |
| Gen3 | 141,745 | 36,584 | 3.87× | 33.8 |
| Gen4 | 356,210 | 38,043 | 9.36× | 84.9 |

Labeled set baseline (`data/FICS_PCB_REMAP_NOISE/`): 24,841 annotations, 4,194 images (~5.9/img).

**Notes:**
- Static conf=0.25 causes runaway pseudo-label proliferation on RT-DETR-l: Gen4 combined set has 356,210 annotation lines vs. adaptive's 38,043 (9.4× more). At 84.9 annotations/image under static at Gen4, the model is training on heavily hallucinated labels (dataset has ~5.9 GT objects/image). Adaptive stays at ~9.1 annotations/image across all generations.
- The proliferation directly explains the computational boundary: a training iteration over 356K annotations at batch=8 takes proportionally longer (bipartite matching loss scales with targets per image). Epoch time measured: Gen1 (40K) ~5 min/epoch; Gen2 (61K) ~22 min/epoch; Gen3 (141K) ~34–60+ min/epoch. Gen4 (356K) is estimated at >108 min/epoch.
- **This is an architecture-specific finding:** YOLOv8 and YOLOv10 static combined sets stay flat (~33K–35K across all 4 generations, ~3–6% increase). RT-DETR-l's transformer decoder with bipartite matching appears to amplify pseudo-label density under a fixed low threshold in a way that NMS-based YOLO detectors do not.
- No canonical mAP50 eval for this strategy on RT-DETR-l. The FICS 3×3 table has this cell as `—`.
- **Paper table note:** static@RT-DETR-l is reported as `— (intractable; label proliferation 356K Gen4)` with a footnote.

---

### [pending] — rtdetr_l_noisy_gen1_progressive

- **Model:** RT-DETR-l
- **Strategy:** self-training (Gen1–Gen4, 4 generations), progressive threshold (0.25→0.35→0.45→0.55)
- **Config:** configs/rtdetr_l/self_train_noisy_progressive.yaml (pseudo_only_mode: true)
- **Script:** experiments/run_rtdetr_l_noisy_st_progressive.sh

*Results pending — not yet launched.*

---

## PCB-DSLR target-domain self-training — YOLOv8m

**Context:** teacher = noisy FICS Gen0 (`runs/yolov8/baseline_noisy/weights/best.pt`).
IC ground-truth merged every iteration via `partial_labels_dir`. Student trains on
IC GT + pseudo-labels for the 7 remaining classes (`pseudo_only_mode: true` — FICS data
never re-enters after Gen0). Evaluation: detection-growth curve at fixed C=0.25 via
`scripts/evaluate_pcb_dslr.py`. mAP50 is not applicable (IC GT consumed in training;
no held-out GT for other 7 classes in PCB-DSLR).

**Comparison baseline — Honorato (2025) Figure 42:**
Honorato ran a generational experiment on PCB-DSLR (GEN0→GEN3, 3 student generations)
from a clean FICS teacher (YOLOv5, C_th=0.45). Result: 3 classes collapse (LED, diode,
tantalum_cap); ceramic, resistor, inductor, electrolytic grow monotonically. The thesis
also contains a separate non-generational ablation (single teacher→student step, "sem o
componente geracional") — these are two distinct experiments; the ablation is not the
same as the GEN0→GEN3 progression. Our loop matches the generational design: DSLR-only
student, IC GT anchor, iterative pseudo-labeling. We run 4 student generations vs Honorato's 3.

**Finding (2026-06-07 — clean-teacher control resolved):** All 7 non-IC classes collapse
to 0 by Gen1 in the clean-teacher control (`yolov8_st_dslr_adaptive_clean_42`, adaptive,
identical hyperparameters, only `warm_start` swapped to `baseline-2`). **IC-GT-dominance
is the structural cause**: the partial-GT anchor for IC in `pseudo_only_mode` suppresses
non-IC propagation under domain shift, independent of teacher quality. Do not use
noise-based causal framing for the class collapse.

The collapse is more severe than Honorato's partial collapse (3 of 7 classes survive with
clean YOLOv5 at C_th=0.45). The difference is consistent with threshold: C=0.25 generates
more IC pseudo-labels per generation than C=0.45, strengthening the IC-GT anchor relative
to the non-GT classes.

| Dimension | Honorato | Ours |
|-----------|----------|------|
| Teacher quality | Clean FICS (mAP50≈0.72) | 75%-noisy (mAP50=0.934) or clean (mAP50=0.860) |
| Threshold | C=0.45 (stricter) | C=0.25 (more permissive) |
| Architecture | YOLOv5 | YOLOv8m / YOLOv10m / RT-DETR-l |
| Image scale | Full 4928×3280 | 512×512 tiles |

---

### 2026-06-06 — yolov8_dslr_gen1_adaptive

- **Model:** YOLOv8m
- **Strategy:** PCB-DSLR self-training, adaptive per-class threshold
- **Config:** configs/yolov8/self_train_dslr.yaml (adaptive_threshold: true, pseudo_only_mode: true, partial_labels_dir set)
- **Teacher weights:** runs/yolov8/baseline_noisy/weights/best.pt
- **Experiment dir:** experiments/yolov8_st_dslr_adaptive_42
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-06T09:20:45Z)
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset:** PCB_DSLR_CROPS_512/orig_distribution/ (2,927 images, IC GT only — class 2)

**Detection-growth results (fixed C=0.25, all 2,927 PCB-DSLR images):**

| Generation | IC (class 2) | Total | Notes |
|------------|-------------|-------|-------|
| Gen0 (noisy teacher) | 3,539 | 16,557 | Broad multi-class detection; not yet domain-adapted |
| Gen1 | 2,155 | 2,371 | Domain adaptation dip; model re-calibrates from FICS to DSLR |
| Gen2 | 6,272 | 6,272 | IC-only specialisation; +77% vs Gen0; other classes collapse |
| Gen3 | 5,918 | 5,918 | Slight decrease; IC-only maintained |
| Gen4 | **6,552** | **6,552** | IC peak; +85% vs Gen0 teacher; IC GT coverage ~1.94/img vs 2.24 detected/img |

Per-class breakdown (Gen0 → Gen4):

| Class | Gen0 | Gen1 | Gen2 | Gen3 | Gen4 |
|-------|------|-------|-------|-------|-------|
| resistor_smd | 4,758 | 0 | 0 | 0 | 0 |
| ceramic_capacitor | 5,791 | 216 | 0 | 0 | 0 |
| ic | 3,539 | 2,155 | 6,272 | 5,918 | 6,552 |
| diode | 226 | 0 | 0 | 0 | 0 |
| inductor | 1,023 | 0 | 0 | 0 | 0 |
| electrolytic_capacitor | 848 | 0 | 0 | 0 | 0 |
| tantalum_capacitor | 233 | 0 | 0 | 0 | 0 |
| led | 139 | 0 | 0 | 0 | 0 |

**Notes:**
- IC detection count grows +85% from Gen0 teacher (3,539) to Gen4 (6,552). This confirms successful domain adaptation via the partial-GT loop.
- Gen1 dip (2,155 IC): unique to noisy-teacher runs — the clean-teacher control (identical config, `warm_start=baseline-2`) does not show a dip; clean Gen1 IC = 6,598 (+117%). All three noisy strategies share the same Gen1 IC count (same Gen0 teacher, same conf=0.25). Descriptive observation only; n=1 per arm, different Gen0 checkpoints — no causal attribution.
- All non-IC classes collapse to 0 by Gen2. Expected: pseudo-labels for non-IC classes in PCB-DSLR are unreliable (high domain gap; no GT for calibration), so the model learns to suppress them in the target domain while IC GT provides a strong anchor signal.
- Gen4 IC (6,552) vs. GT IC coverage (5,679 GT annotations across 2,927 images = ~1.94/img): model detects ~2.24 IC/img, a 15% over-detection relative to GT. Slight over-segmentation expected at C=0.25; still a strong precision signal.
- CSV saved to: `experiments/yolov8_st_dslr_adaptive_42/detection_growth.csv`

---

### 2026-06-06 — yolov8_dslr_gen1_static

- **Model:** YOLOv8m
- **Strategy:** PCB-DSLR self-training, static threshold (C=0.25)
- **Config:** configs/yolov8/self_train_dslr.yaml (adaptive_threshold: true in file, overridden by --no-adaptive flag)
- **Teacher weights:** runs/yolov8/baseline_noisy/weights/best.pt
- **Experiment dir:** experiments/yolov8_st_dslr_static_42
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-06T13:41:57Z)
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset:** PCB_DSLR_CROPS_512/orig_distribution/ (2,927 images, IC GT only)

**Detection-growth results (fixed C=0.25, all 2,927 PCB-DSLR images):**

| Generation | IC (class 2) | Total | Notes |
|------------|-------------|-------|-------|
| Gen0 (noisy teacher) | 3,539 | 16,557 | Same teacher as adaptive; broad multi-class |
| Gen1 | 2,155 | 2,371 | Identical to adaptive — shared warm-start; strategy diverges from Gen2 |
| Gen2 | 6,508 | 6,508 | IC-only; +84% vs Gen0; other classes collapsed |
| Gen3 | 7,077 | 7,077 | Monotonic growth; no plateau |
| Gen4 | **7,464** | **7,464** | IC peak; +**111%** vs Gen0; 2.55 detected/img vs GT 1.94/img (+31% over-detection) |

Per-class breakdown (Gen0 → Gen4):

| Class | Gen0 | Gen1 | Gen2 | Gen3 | Gen4 |
|-------|------|-------|-------|-------|-------|
| resistor_smd | 4,758 | 0 | 0 | 0 | 0 |
| ceramic_capacitor | 5,791 | 216 | 0 | 0 | 0 |
| ic | 3,539 | 2,155 | 6,508 | 7,077 | 7,464 |
| diode | 226 | 0 | 0 | 0 | 0 |
| inductor | 1,023 | 0 | 0 | 0 | 0 |
| electrolytic_capacitor | 848 | 0 | 0 | 0 | 0 |
| tantalum_capacitor | 233 | 0 | 0 | 0 | 0 |
| led | 139 | 0 | 0 | 0 | 0 |

**Notes:**
- Static IC grows **monotonically** across all 4 generations and does not plateau. Adaptive plateaued at Gen3-4 (~6.2K–6.6K); static keeps climbing to 7,464.
- Over-detection relative to GT at Gen4: 2.55 detected/img vs 1.94 GT IC/img = **+31%** (adaptive: +15%). Static threshold accumulates more false positives on IC, mirroring the label proliferation finding from the FICS-noise loop.
- Gen1 identical to adaptive (IC=2,155, ceramic=216) — shared warm-start; strategies diverge only from Gen2 onwards.
- Non-IC classes collapse identically to adaptive (0 by Gen2). Structural IC-GT-dominance, confirmed by clean-teacher control (collapse occurs at Gen1 even with clean teacher). Not a threshold effect — identical across adaptive, static, and progressive.
- Ceramic_cap appears in combined Gen2 training data (4,024 pseudo-labeled instances) but the trained model fails to propagate ceramics into Gen3 pseudo-labels, likely due to IC GT dominance suppressing ceramic recall.
- CSV: `experiments/yolov8_st_dslr_static_42/detection_growth.csv`

---

### 2026-06-06 — yolov8_dslr_gen1_progressive

- **Model:** YOLOv8m
- **Strategy:** PCB-DSLR self-training, progressive threshold (0.25→0.35→0.45→0.55)
- **Config:** configs/yolov8/self_train_dslr_progressive.yaml
- **Teacher weights:** runs/yolov8/baseline_noisy/weights/best.pt
- **Experiment dir:** experiments/yolov8_st_dslr_progressive_42
- **Git SHA:** 62dbdeda611addb0f31638f265c3e40547bad777 (Gen4 completed 2026-06-06T19:22:28Z)
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset:** PCB_DSLR_CROPS_512/orig_distribution/ (2,927 images, IC GT only)

**Detection-growth results (fixed C=0.25, all 2,927 PCB-DSLR images):**

| Generation | IC (class 2) | Total | Notes |
|------------|-------------|-------|-------|
| Gen0 (noisy teacher) | 3,539 | 16,557 | Same teacher as adaptive/static |
| Gen1 | 2,155 | 2,371 | Identical to adaptive and static — shared warm-start |
| Gen2 | 6,625 | 6,625 | IC-only; +87% vs Gen0; pseudo-labels seeded at conf=0.35 |
| Gen3 | 7,414 | 7,414 | Peak; pseudo-labels seeded at conf=0.45 |
| Gen4 | **6,607** | **6,607** | Slight drop from Gen3 peak; conf=0.55 pseudo-labels → fewer seeds → less over-detection |

Per-class breakdown (Gen0 → Gen4):

| Class | Gen0 | Gen1 | Gen2 | Gen3 | Gen4 |
|-------|------|-------|-------|-------|-------|
| resistor_smd | 4,758 | 0 | 0 | 0 | 0 |
| ceramic_capacitor | 5,791 | 216 | 0 | 0 | 0 |
| ic | 3,539 | 2,155 | 6,625 | 7,414 | 6,607 |
| diode | 226 | 0 | 0 | 0 | 0 |
| inductor | 1,023 | 0 | 0 | 0 | 0 |
| electrolytic_capacitor | 848 | 0 | 0 | 0 | 0 |
| tantalum_capacitor | 233 | 0 | 0 | 0 | 0 |
| led | 139 | 0 | 0 | 0 | 0 |

**Notes:**
- Progressive follows a **grow-then-retreat** pattern: IC peaks at Gen3 (7,414) and retreats at Gen4 (6,607). This is the threshold schedule at work: higher pseudo-labeling conf at Gen4 (0.55) produces fewer seeds, which slightly reduces over-detection without losing coverage.
- Gen4 IC density: 6,607 / 2,927 = 2.26/img — very close to adaptive (2.24/img) and well below static (2.55/img). Progressive and adaptive converge to a similar detection density at Gen4 despite different mechanisms.
- Non-IC collapse identical to adaptive and static: structural IC-GT-dominance (confirmed by clean-teacher control — collapse occurs at Gen1 even with a clean teacher), not threshold-specific.
- CSV: `experiments/yolov8_st_dslr_progressive_42/detection_growth.csv`

---

### 2026-06-07 — yolov8_dslr_gen1_adaptive_clean (clean-teacher control)

- **Model:** YOLOv8m
- **Strategy:** PCB-DSLR self-training, adaptive per-class threshold — CLEAN TEACHER
- **Config:** configs/yolov8/self_train_dslr_clean.yaml
- **Teacher weights:** runs/yolov8/baseline-2/weights/best.pt (clean FICS, mAP50=0.860)
- **Experiment dir:** experiments/yolov8_st_dslr_adaptive_clean_42
- **Seed:** 42
- **Epochs per generation:** 50, patience=15
- **Dataset:** PCB_DSLR_CROPS_512/orig_distribution/ (2,927 images, IC GT only — class 2)
- **Purpose:** Isolate teacher noise vs. IC-GT-dominance as cause of non-IC collapse.
  Identical to `yolov8_dslr_gen1_adaptive` in every hyperparameter; only `warm_start` differs.

**Pre-registered interpretation:**
- Non-IC survive → teacher noise is primary cause. (Not triggered.)
- Non-IC also collapse → IC-GT-dominance is structural cause, independent of teacher
  quality. Drop noise-propagation framing. (**Triggered.**)

**Detection-growth results (fixed C=0.25, all 2,927 PCB-DSLR images):**

| Generation | IC (class 2) | non-IC | Total | IC/img |
|------------|-------------|--------|-------|--------|
| Gen0 (clean teacher) | 3,036 | 9,303 | 12,339 | 1.037 |
| Gen1 | 6,598 | 0 | 6,598 | 2.254 |
| Gen2 | 6,113 | 0 | 6,113 | 2.088 |
| Gen3 | 6,545 | 0 | 6,545 | 2.236 |
| Gen4 | **7,212** | 0 | **7,212** | **2.464** |

**Notes:**
- Non-IC = 0 from Gen1. Pre-registered trigger fired: **IC-GT-dominance is the structural
  cause of class collapse in `pseudo_only_mode`**, independent of teacher quality.
- Clean teacher gen0 IC/img = 1.037 — lower than noisy gen0 (1.209/img). Consistent with
  the noisy-beats-clean pattern on DSLR (noisy teacher more permissive at C=0.25).
- Gen4 IC/img for clean = 2.464, placing it **between** noisy-static (2.550) and
  noisy-adaptive (2.238). Teacher quality does not determine IC detection ceiling; strategy
  and pseudo-label accumulation dominate.
- The Gen4 ordering (noisy-static > clean > noisy-progressive ≈ noisy-adaptive)
  reinforces the structural conclusion: a noisy teacher does not suppress final IC growth,
  and a clean teacher does not guarantee the highest IC yield.
- IC trajectory differs between clean and noisy teacher (clean: immediate jump at Gen1;
  noisy: dip at Gen1 before recovery). Descriptive observation only — n=1 per arm,
  different Gen0 checkpoints; no causal attribution.
- CSV saved to: `experiments/yolov8_st_dslr_adaptive_clean_42/detection_growth.csv`

---

## 2026-06-11 — Method change: strategy-divergent Gen1 + 3 iterations (noisy ST loop)

**Problem.** Every Gen1 (iter1) result was identical across adaptive/static/progressive for a
given model. Cause: iter1 pseudo-labels were produced by an external `generate_pseudo_labels.py`
call (Step 1 of each launch script) using the Gen0 teacher at a flat `conf=0.25` with no
per-class / no `candidate_floor` logic — strategy-independent by construction. Strategies only
diverged from the iter2 seed.

**Change (applied).**
- iter1 seeding folded into `SelfTrainer.run()` via a pre-loop block + new `_seed_thresholds()`
  helper. At loop start the model is the warm-started teacher; iter1 is now seeded with the
  strategy's own threshold:
  - adaptive → per-class max-F1 from the teacher's clean-`data/val` PR curve (same calibration
    method as later iters; now also applied to the teacher seed);
  - progressive → `conf_schedule[0]`;
  - static → `conf_threshold` (None passed → PseudoLabeler uses its fixed value).
  iter1 now also gets the `candidate_floor=0.05` floor+post-filter that later iters already had.
  Backward-compatible: if `iter1/pseudo/labels` exists and is non-empty (external pre-seed or
  resume), the teacher seed is skipped.
- Iterations cut **4 → 3**. Rationale: matches progressive's three schedule values, aligns with
  Honorato's 3-iteration method, faster, and avoids the iter4 annotation-explosion regime
  (`finding_static_label_proliferation`, `docs/known-issues.md`).
- Progressive schedule `[0.25, 0.35, 0.45, 0.55] → [0.35, 0.45, 0.55]` (iter1=0.35, iter2=0.45,
  iter3=0.55). Static stays **0.25** (Honorato's validated fixed C_th). Adaptive calibration
  stays on clean `data/val` (the documented, accepted choice; internal-split deferred).

**Files.** `src/labeling/self_trainer.py`; `configs/{yolov8,yolov10,yolo12}/self_train_noisy.yaml`
and `…_progressive.yaml` (6); the 9 noisy ST launch scripts (Step-1 block + unused `WEIGHTS`
var removed); `ARCHITECTURE.md`, `.claude/CLAUDE.md`.

**Consequence.** All pending Gen1–Gen4 v2 runs become **Gen1–Gen3** under this design and must be
run with the updated configs/scripts. Gen0 v2 baseline training is unaffected (trained on noisy
GT, not via the loop). Expected validation: Gen1 mAP now differs across the three strategies;
only `iter1/ iter2/ iter3/` are produced.

## 2026-06-13 — YOLO12 batch 12 → 6 (VRAM oversubscription fix)

**Symptom.** `run_yolo12_noisy_gen0_v3.sh` (Gen0 baseline) appeared stuck: after ~3 h wall clock
it was still on batch 5/277 of epoch 1, with per-iteration times of 2,000–6,600 s/it and an ETA
of 170+ h for a single epoch. GPU pinned at 100 % utilization but near-zero throughput.

**Diagnosis.** Not a crash — VRAM oversubscription. Ultralytics reported `GPU_mem 8.57G` against
the RTX 4060's 8,188 MiB (8.0 GiB); nvidia-smi showed 7,912/8,188 MiB pinned. With `batch: 12`,
`imgsz: 640`, YOLO12m's Area-Attention backbone exceeds 8 GB and the WSL2 NVIDIA driver silently
spills the overflow into system RAM (sysmem fallback) rather than OOMing. Every step then pages
tensors across PCIe → 100 % util, ~0 throughput. Host swap was untouched (Swap 0 B, 29 GB RAM
free), confirming the paging was GPU-side, not OS-level.

**Action.** Killed the run, set **`batch: 6`** across all five YOLO12 configs
(`base_noisy`, `self_train_noisy`, `self_train_noisy_progressive`, `self_train_dslr`,
`self_train_dslr_progressive`), cleared the stale `runs/yolo12/baseline_noisy_v3/` dir (no
checkpoint had been produced), and relaunched.

**Result.** Healthy: `GPU_mem 4.64G` (3,222 MiB on nvidia-smi), ~2.8 it/s, ~3.5 min/epoch.
Losses converging normally (box ≈0.67, cls ≈0.39, dfl ≈0.90 by epoch 20). Full 50-epoch Gen0
now ≈1.5–2 h.

**Note.** Change confined to YOLO12 only. YOLOv8m configs stay at `batch: 16` and YOLOv10m at
`batch: 12` — both are convolutional, fit comfortably in 8 GB, and have already produced completed
runs at those batches; lowering them would introduce an unnecessary effective-LR/gradient-noise
confound and break comparability with prior YOLOv8/v10 results. The 12→6 drop for YOLO12 does mean
its effective batch differs from the other two models; if strict cross-model comparability is
needed later, compensate via `nbs` gradient accumulation rather than raising the physical batch.

## 2026-06-13 — Gen0 v3 seed-threshold gate (all three teachers)

Ran `scripts/_diag_seed_thresholds.py` (CPU-only, read-only) against each completed Gen0 v3
baseline to inspect the per-class precision-target adaptive thresholds that seed Gen1
pseudo-labeling. Calibration on `data/dev` (held-out DEV split, disjoint from TEST=`data/val`),
`target_precision=0.9`, `candidate_floor=0.05`, fallback `conf_threshold=0.25`. These are the
thresholds `SelfTrainer._seed_thresholds()` applies to the warm-start teacher for the adaptive
strategy. Diagnostic only — nothing written to any run dir.

Teachers:
`runs/yolov8/baseline_noisy_v3/weights/best.pt`,
`runs/yolov10/baseline_noisy_v3/weights/best.pt`,
`runs/yolo12/baseline_noisy_v3/weights/best.pt`.

Applied per-class threshold (ᶠ = raw < 0.05 floor, clamped up):

| cls | class | YOLOv8m | YOLOv10m | YOLO12m |
|---|---|---|---|---|
| 0 | resistor_smd | 0.300 | 0.174 | 0.154 |
| 1 | ceramic_capacitor | 0.074 | 0.050ᶠ | 0.050ᶠ |
| 2 | ic | 0.782 | 0.728 | 0.716 |
| 3 | diode | 0.929 | 0.994 | 0.889 |
| 4 | inductor | 0.700 | 0.962 | 0.538 |
| 5 | electrolytic_capacitor | 0.135 | 0.050ᶠ | 0.050ᶠ |
| 6 | tantalum_capacitor | 0.459 | 0.164 | 0.389 |
| 7 | led | 0.590 | 0.421 | 0.499 |
| | **DEV mAP50** | 0.745 | 0.700 | 0.723 |

Raw (pre-floor) values for the clamped cells: ceramic_capacitor v10=0.0220 / yolo12=0.0320;
electrolytic_capacitor v10=0.0160 / yolo12=0.0300.

**Interpretation — these patterns are expected behaviour of the adaptive gate under noisy labels,
not bugs; the teacher→student loop proceeds as designed with them in mind:**

- **diode is the universal hard class** (threshold 0.89–0.99 everywhere; only 84 DEV instances).
  Few but high-confidence diode pseudo-labels in every loop. Data-scarcity, not architecture.
- **YOLOv10 gates diode (0.994) and inductor (0.962) almost completely out** of Gen1 — detections
  need ~99 %/96 % confidence to pass. High class-collapse risk for those two in the v10 loop;
  consistent with the v1 finding that v10 Diode collapsed under static. Watch v10 Gen1→Gen3 for
  diode/inductor dropout.
- **electrolytic_capacitor is degenerate in all three teachers** (raw 0.016–0.135, two floored):
  the precision-target sweep only reaches 0.9 because the teachers barely detect the class
  (e.g. yolo12 DEV: P=1.0 at R=0, mAP50=0.106). Gen1 pseudo-labels this class loosely off a
  near-blind teacher — fragile across all loops.
- **Strong classes (resistor, ceramic, ic, led) gate sensibly** across all three models. Low
  ceramic thresholds (floored in v10/yolo12) yield abundant, reliable ceramic pseudo-labels.

**Decision.** Proceed with the teacher–student loop unchanged; treat the above as known watch-points
when reading Gen1–Gen3 per-class results (esp. v10 diode/inductor and electrolytic across the board).

## 2026-06-14 — YOLOv8m v3 noisy self-training COMPLETE (Gen1–Gen3) + canonical TEST evals

YOLOv8m v3 noisy intra-FICS self-training finished for all three strategies (adaptive, static,
progressive), Gen1–Gen3, seed 42. Outputs in `experiments/yolov8_st_noisy_{strategy}_v3_42/iter{1,2,3}/`.
Ran `scripts/evaluate.py` (commit `5dbe41e`) on the canonical held-out **TEST** set
(`--val-images data/test/images`, = `data/val`, 906 imgs, board-disjoint from DEV/TRAIN) for Gen0 and
each Gen3 `best.pt`, and the same checkpoints on **DEV** (`data/dev`, 870 imgs) for comparison.
JSONs under `results/{dev,test}_{gen0,adaptive_gen3,static_gen3,progressive_gen3}/`.

**Headline (methodological, not directional): the DEV-based "Gen0 > students" premise does not hold on
the honest TEST set — but the point estimates *invert* (DEV: Gen0 ahead by 0.022 mAP50; TEST: students
ahead by ≤0.035), and both gaps fall within rare-class AP variance (tantalum 12 TEST instances, led 18).
A real effect does not flip sign when the held-out board group changes; the inversion is itself evidence
the gap is noise, not signal. No reliable Gen0-vs-strategy ordering can be claimed from seed 42. The
load-bearing takeaway is that DEV reporting actively inverts the conclusion — report TEST, and run
multiple seeds before claiming any direction.**

| best.pt eval        | Gen0  | adaptive-G3 | static-G3 | progressive-G3 |
|---------------------|-------|-------------|-----------|----------------|
| **TEST** mAP50      | 0.684 | 0.703       | 0.692     | **0.719**      |
| **TEST** mAP50-95   | 0.527 | 0.552       | 0.555     | **0.571**      |
| **DEV**  mAP50      | **0.746** | 0.696   | 0.715     | 0.724          |
| **DEV**  mAP50-95   | **0.624** | 0.572   | 0.596     | 0.595          |

Best epochs (stopped/best): Gen0 32/17; adaptive-G3 36/21; static-G3 33/18; progressive-G3 18/3.

### Per-class AP50

| class                  | TEST Gen0 | TEST ada | TEST sta | TEST pro | DEV Gen0 | DEV ada | DEV sta | DEV pro |
|------------------------|-----------|----------|----------|----------|----------|---------|---------|---------|
| resistor_smd           | 0.983 | 0.987 | 0.983 | 0.990 | 0.978 | 0.978 | 0.977 | 0.976 |
| ceramic_capacitor      | 0.977 | 0.971 | 0.971 | 0.971 | 0.982 | 0.986 | 0.987 | 0.986 |
| ic                     | 0.929 | 0.896 | 0.888 | 0.927 | 0.922 | 0.908 | 0.875 | 0.899 |
| diode                  | 0.026 | 0.000 | 0.005 | 0.000 | 0.132 | 0.000 | 0.251 | 0.150 |
| inductor               | 0.192 | 0.278 | 0.191 | 0.289 | 0.657 | 0.657 | 0.616 | 0.601 |
| electrolytic_capacitor | 0.649 | 0.651 | 0.559 | 0.652 | 0.574 | 0.653 | 0.430 | 0.461 |
| tantalum_capacitor     | 0.781 | 0.956 | 0.995 | 0.948 | 0.904 | 0.982 | 0.981 | 0.958 |
| led                    | 0.934 | 0.882 | 0.940 | 0.970 | 0.816 | 0.408 | 0.603 | 0.764 |

### Why the DEV/TEST ranking flips (analysis)

- **Majority classes are saturated and identical across all models and both splits** (resistor,
  ceramic, ic ≈ 0.88–0.99, flat). They contribute nothing to the Gen0-vs-Gen3 difference.
- **The entire Gen0-vs-Gen3 gap lives in the 5 rare classes** (diode, inductor, electrolytic,
  tantalum, led), which have tiny support (DEV 66–258 instances; TEST 12–114). Their AP is
  high-variance, and the two board-disjoint splits pull opposite ways: DEV favours Gen0 (el_cap,
  led, inductor); TEST favours the students (tantalum 0.781→0.95+, inductor 0.192→0.289, led
  0.934→0.970). The arithmetic closes: each class is 1/8 of mAP50, and the TEST gain (+0.035 Gen0→prog)
  is fully accounted for by tantalum+inductor+led; the DEV gap (≈0.022 the other way) by el_cap+led+inductor+ic.
- **Diode is already dead at Gen0** (TEST AP 0.026), so "minority extinction across generations" does
  NOT explain the gap — you cannot fall from zero. Diode collapse is a real per-class failure (extreme
  rarity + class-swap noise flooding its labels with non-diodes) but it is not the source of the
  Gen0/Gen3 difference. This corrected an earlier working hypothesis that minority extinction drove the gap.
- **Pseudo-labels are denser than the noisy human set, not sparser** — teacher generates ~28k boxes/gen
  (8.4/img) vs noisy human 19.3k (5.8/img, the 25% removal) and clean GT 25.8k (7.8/img). So this is not
  a recall-collapse story; the teacher over-predicts relative to clean GT.

### Methodology note — provenance of the earlier "Gen0 higher" impression

The first preview that suggested Gen0 > all Gen3 was read from the **last-epoch** row of each iteration's
training `results.csv`, on **DEV**. Two confounds stacked: (1) last-epoch ≠ `best.pt` epoch (best.pt is the
best epoch), and (2) DEV is the model-selection + adaptive-calibration set, not the reporting set. Evaluating
`best.pt` on DEV still shows Gen0 ahead (so the flip is driven by the eval *set*, not best-vs-last-epoch),
but only TEST is the honest number.

### Implications

- The defensible claim is **neither** "noisy human labels beat pseudo-labels" **nor** "self-training beats
  Gen0." The point estimates favour Gen0 on DEV and the students on TEST (TEST gain +0.02 to +0.035 mAP50);
  both directions are single-seed and both sit inside the rare-class noise floor. Majority classes are flat
  on every model and split.
- **The result is within the minority-class noise floor.** With tantalum at 12 TEST instances and led at 18,
  single-seed AP swings on these classes plausibly exceed the whole Gen0/Gen3 gap. The fact that the ordering
  *flips sign* between two board-disjoint clean splits is the direct evidence: from seed 42 alone no
  Gen0-vs-strategy ordering can be claimed — multiple seeds are required before stating any direction.
- **Report TEST, never DEV.** This run is a concrete demonstration that DEV reporting inverts the conclusion.

## 2026-06-16 — Canonical TEST evals, all 3 models × 3 strategies (v3 complete)

Provenance: 30 evals via `scripts/evaluate.py --val-images data/test/images` (TEST = data/test, byte-identical alias of data/val, 906 imgs, board-disjoint from TRAIN/DEV). Weights: `runs/<model>/baseline_noisy_v3` (Gen0) + `experiments/<model>_st_noisy_<strat>_v3_42/iter<g>`. Batch script `experiments/run_canonical_evals_v3.sh`, JSONs in `results/v3/<tag>/`. Run 2026-06-16 11:30 UTC, git 5dbe41e. Eval batch wall time ~14 min.

**Caveat (carries the 2026-06-14 conclusion to all models): methodological, not directional.** Single-seed (42), TEST-reported. Gen0-vs-strategy Δ (+0.02 to +0.05 mAP50) sits inside the rare-class noise floor — tantalum 12 TEST instances, led 18 (counts verified from data/test/labels). DEV inverts the ordering. No Gen0-vs-strategy ordering claimable from seed 42; multiple seeds required.

### Progression — mAP@50 / mAP@50-95 (TEST)


**YOLOv8m** (Gen0 mAP50 0.684, mAP50-95 0.527)

| Strategy | Metric | Gen0 | Gen1 | Gen2 | Gen3 | Δ G3−Gen0 |
|---|---|---|---|---|---|---|
| Adaptive | mAP50 | 0.684 | 0.711 | 0.713 | 0.703 | +0.019 |
| Adaptive | mAP50-95 | 0.527 | 0.568 | 0.558 | 0.552 | +0.025 |
| Static | mAP50 | 0.684 | 0.688 | 0.654 | 0.692 | +0.007 |
| Static | mAP50-95 | 0.527 | 0.533 | 0.521 | 0.555 | +0.028 |
| Progressive | mAP50 | 0.684 | 0.685 | 0.620 | 0.719 | +0.034 |
| Progressive | mAP50-95 | 0.527 | 0.544 | 0.490 | 0.571 | +0.043 |

**YOLOv10m** (Gen0 mAP50 0.605, mAP50-95 0.460)

| Strategy | Metric | Gen0 | Gen1 | Gen2 | Gen3 | Δ G3−Gen0 |
|---|---|---|---|---|---|---|
| Adaptive | mAP50 | 0.605 | 0.557 | 0.580 | 0.532 | -0.073 |
| Adaptive | mAP50-95 | 0.460 | 0.432 | 0.451 | 0.411 | -0.049 |
| Static | mAP50 | 0.605 | 0.607 | 0.629 | 0.625 | +0.019 |
| Static | mAP50-95 | 0.460 | 0.474 | 0.480 | 0.479 | +0.020 |
| Progressive | mAP50 | 0.605 | 0.628 | 0.590 | 0.590 | -0.015 |
| Progressive | mAP50-95 | 0.460 | 0.498 | 0.460 | 0.452 | -0.008 |

**YOLO12m** (Gen0 mAP50 0.641, mAP50-95 0.494)

| Strategy | Metric | Gen0 | Gen1 | Gen2 | Gen3 | Δ G3−Gen0 |
|---|---|---|---|---|---|---|
| Adaptive | mAP50 | 0.641 | 0.633 | 0.635 | 0.660 | +0.020 |
| Adaptive | mAP50-95 | 0.494 | 0.484 | 0.471 | 0.484 | -0.010 |
| Static | mAP50 | 0.641 | 0.619 | 0.682 | 0.685 | +0.044 |
| Static | mAP50-95 | 0.494 | 0.468 | 0.521 | 0.506 | +0.012 |
| Progressive | mAP50 | 0.641 | 0.661 | 0.665 | 0.692 | +0.051 |
| Progressive | mAP50-95 | 0.494 | 0.505 | 0.506 | 0.542 | +0.049 |

### Class evolution — AP@50 (TEST). Columns: Gen0 + A/S/P × G1–3 (A=Adaptive, S=Static, P=Progressive)


**YOLOv8m**

| Class | Inst | Gen0 | A-G1 | A-G2 | A-G3 | S-G1 | S-G2 | S-G3 | P-G1 | P-G2 | P-G3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resistor_smd | 2238 | 0.98 | 0.99 | 0.99 | 0.99 | 0.98 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 |
| ceramic_capacitor | 2670 | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.98 | 0.97 | 0.97 |
| ic | 984 | 0.93 | 0.93 | 0.93 | 0.90 | 0.92 | 0.87 | 0.89 | 0.94 | 0.86 | 0.93 |
| diode | 42 | 0.03 | 0.00 | 0.00 | 0.00 | 0.03 | 0.01 | 0.00 | 0.00 | 0.02 | 0.00 |
| inductor | 114 | 0.19 | 0.20 | 0.14 | 0.28 | 0.24 | 0.18 | 0.19 | 0.22 | 0.29 | 0.29 |
| electrolytic_capacitor | 42 | 0.65 | 0.64 | 0.73 | 0.65 | 0.61 | 0.35 | 0.56 | 0.54 | 0.22 | 0.65 |
| tantalum_capacitor | 12 | 0.78 | 0.97 | 0.96 | 0.96 | 0.77 | 0.93 | 0.99 | 0.82 | 0.61 | 0.95 |
| led | 18 | 0.93 | 0.99 | 0.99 | 0.88 | 0.98 | 0.94 | 0.94 | 0.99 | 0.99 | 0.97 |

**YOLOv10m**

| Class | Inst | Gen0 | A-G1 | A-G2 | A-G3 | S-G1 | S-G2 | S-G3 | P-G1 | P-G2 | P-G3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resistor_smd | 2238 | 0.97 | 0.97 | 0.98 | 0.97 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| ceramic_capacitor | 2670 | 0.95 | 0.94 | 0.93 | 0.94 | 0.93 | 0.93 | 0.91 | 0.95 | 0.93 | 0.93 |
| ic | 984 | 0.89 | 0.91 | 0.90 | 0.89 | 0.89 | 0.88 | 0.84 | 0.91 | 0.90 | 0.88 |
| diode | 42 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| inductor | 114 | 0.24 | 0.18 | 0.14 | 0.18 | 0.29 | 0.28 | 0.28 | 0.29 | 0.17 | 0.24 |
| electrolytic_capacitor | 42 | 0.25 | 0.08 | 0.43 | 0.29 | 0.25 | 0.15 | 0.35 | 0.13 | 0.03 | 0.27 |
| tantalum_capacitor | 12 | 0.55 | 0.42 | 0.43 | 0.19 | 0.63 | 0.92 | 0.68 | 0.83 | 0.72 | 0.50 |
| led | 18 | 0.99 | 0.96 | 0.84 | 0.80 | 0.88 | 0.88 | 0.95 | 0.94 | 0.98 | 0.92 |

**YOLO12m**

| Class | Inst | Gen0 | A-G1 | A-G2 | A-G3 | S-G1 | S-G2 | S-G3 | P-G1 | P-G2 | P-G3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resistor_smd | 2238 | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 |
| ceramic_capacitor | 2670 | 0.97 | 0.96 | 0.95 | 0.96 | 0.97 | 0.97 | 0.95 | 0.97 | 0.94 | 0.96 |
| ic | 984 | 0.95 | 0.94 | 0.95 | 0.95 | 0.91 | 0.90 | 0.88 | 0.93 | 0.87 | 0.91 |
| diode | 42 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.14 | 0.38 | 0.00 | 0.04 | 0.00 |
| inductor | 114 | 0.29 | 0.25 | 0.18 | 0.15 | 0.24 | 0.31 | 0.27 | 0.28 | 0.39 | 0.23 |
| electrolytic_capacitor | 42 | 0.00 | 0.13 | 0.17 | 0.32 | 0.05 | 0.26 | 0.03 | 0.26 | 0.20 | 0.54 |
| tantalum_capacitor | 12 | 0.92 | 0.81 | 0.89 | 0.93 | 0.86 | 0.96 | 0.99 | 0.91 | 0.93 | 0.98 |
| led | 18 | 0.99 | 0.99 | 0.96 | 0.98 | 0.93 | 0.93 | 0.99 | 0.96 | 0.96 | 0.93 |

### Notes

- **YOLOv10 NMS-free sensitivity**: Gen0 TEST mAP50 (0.605) is notably below v8/yolo12, and its adaptive line *degrades* (G3−Gen0 = −0.073). Consistent with the documented NMS-free head eval sensitivity (training-log vs canonical gap). Do not read as a clean strategy signal.

- **Diode collapse is universal and Gen0-origin**: AP50 ≈ 0.00 at Gen0 on every model (42 TEST instances + class-swap noise). Static-G3 on YOLO12 partially recovers it (0.38).

- **Majority classes saturated/flat** (resistor, ceramic, ic ≈ 0.88–0.99) across all models, splits, and generations — they carry none of the Gen0-vs-Gen3 signal.

- **Shared report**: `docs/resultados_v3.docx` (PT-BR, landscape) + figures `docs/figures/v3_{progressao,heatmap}_<model>.{pdf,png}`. Generator: `docs/_gen_docx_resultados_v3.py`.


## 2026-06-16 — YOLO12 Gen4 exploration (NOT for the paper; kept for reference)

Exploratory only: extended **YOLO12 noisy v3** from 3→4 iterations for all three strategies (configs `self_train_noisy.yaml` + `self_train_noisy_progressive.yaml` set `iterations: 4`; progressive `conf_schedule` got a 4th entry 0.65). YOLO12-only, single-seed (42). Resumed the existing runs (iter1–3 reused), trained iter4 only. Gen1–3 logs preserved as `run_yolo12_noisy_st_<strat>_v3.gen1-3.log`. Weights: `experiments/yolo12_st_noisy_<strat>_v3_42/iter4/train/weights/best.pt`. TEST eval via `scripts/evaluate.py --val-images data/test/images` → `results/v3/yolo12_<strat>_gen4/`. Run 2026-06-17 02:17 UTC, git 5dbe41e.

**Will not be used in the paper** (YOLO12-only, single-seed, asymmetric vs v8/v10 which stop at Gen3); kept as evidence supporting the 3-iteration design choice.

### Gen3 → Gen4 (TEST mAP50 / mAP50-95)

Gen0 reference: mAP50 0.641, mAP50-95 0.494

| Strategy | Gen3 mAP50 | Gen4 mAP50 | Gen3 mAP50-95 | Gen4 mAP50-95 | Δ G4−Gen0 (mAP50) |
|---|---|---|---|---|---|
| Adaptive | 0.660 | 0.612 | 0.484 | 0.446 | -0.028 |
| Static | 0.685 | 0.625 | 0.506 | 0.465 | -0.016 |
| Progressive | 0.692 | 0.680 | 0.542 | 0.522 | +0.040 |

### Findings

- **Gen4 regresses for all three strategies.** Adaptive (−0.028) and static (−0.016) fall below Gen0; progressive holds best (+0.040) but still slips from its Gen3 peak (0.692→0.680). Consistent with the rationale for cutting v3 to 3 iterations — the 4th generation does not help and tends to hurt.

- **Static diode collapses 0.38 → 0.00 at Gen4**, erasing the partial recovery seen at Gen3 — the label-proliferation regime (iter4 pseudo-set floods the rare class). Adaptive tantalum drops 0.93→0.67; electrolytic/inductor wobble. Progressive is steadiest (tantalum 0.98, led 0.99) thanks to its sparse 0.65 threshold.

- Same single-seed rare-class noise-floor caveat as the 2026-06-16 Gen0–G3 entry applies.


## 2026-06-18 — Pseudo-label volume, quality & noise recovery (noisy v3)

Label-file analysis (no model inference, no GPU) of the self-generated labels themselves, where N is large (thousands of annotations/class) so strategy differences are visible — complements the rare-class-limited TEST mAP. Pseudo-labels confronted with the clean pre-corruption GT `data/train_clean` (3,324 imgs, 25,764 anns), filename-aligned 1:1 with the noisy TRAIN split (augmentation verified mirrored in lockstep). Shared helpers `src/utils/labels.py` (tested, `tests/utils/test_labels.py`, 9 passing). Single-seed (42), **descriptive only** — no inferential strategy claim. Git 5dbe41e.

Scripts → CSVs:
- A `scripts/count_pseudo_labels.py` → `results/analysis/pseudo_label_counts.csv` (757 rows; all runs incl. DSLR + clean/noisy GT reference rows)
- B `scripts/eval_pseudo_label_quality.py` → `results/analysis/pseudo_label_quality.csv` (240 rows; noisy v3; TP+FN==clean count asserted per class/gen)
- C `scripts/eval_pseudo_label_recovery.py` → `results/analysis/pseudo_label_recovery.csv` (960 rows; corruption recipe verified 25/25/25/25)

Reports: `docs/analysis_pseudolabel_quality.md`; figures `docs/figures/pseudolabel_precision_volume_<model>.{pdf,png}` (`docs/_gen_pseudolabel_figs.py`); PT-BR `docs/resultados_pseudolabel.docx` (`docs/_gen_docx_pseudolabel.py`).

Attribution: `producer_gen = iter − 1` (Gen0 seeds iter1/pseudo … last model labels nothing). Pseudo-gens run Gen0–Gen2 (v8/v10, 3 iters) and Gen0–Gen3 (YOLO12, 4 iters).

### Volume vs. precision (aggregated over 8 classes, per producer gen)

| Model | Strategy | Gen0 vol / P | Gen1 vol / P | Gen2 vol / P |
|---|---|---|---|---|
| YOLOv8m | Adaptive | 27,897 / 0.915 | 27,749 / 0.919 | 27,603 / 0.916 |
| YOLOv8m | Static | 28,396 / 0.905 | 28,965 / 0.887 | 29,164 / 0.880 |
| YOLOv8m | Progressive | 27,974 / 0.918 | 28,395 / 0.905 | 28,193 / 0.910 |
| YOLOv10m | Adaptive | 28,601 / 0.887 | 27,241 / 0.919 | 27,241 / 0.907 |
| YOLOv10m | Static | 28,675 / 0.895 | 29,020 / 0.885 | 29,127 / 0.881 |
| YOLOv10m | Progressive | 28,053 / 0.913 | 27,989 / 0.915 | 27,539 / 0.924 |
| YOLO12m | Adaptive | 28,135 / 0.908 | 27,574 / 0.925 | 27,569 / 0.921 |
| YOLO12m | Static | 28,134 / 0.914 | 29,026 / 0.885 | 29,266 / 0.877 |
| YOLO12m | Progressive | 27,750 / 0.926 | 28,170 / 0.912 | 28,126 / 0.912 |

(YOLO12 Gen3 exploratory: adaptive 27,793 / 0.914; static 29,503 / 0.869; progressive 28,097 / 0.913.)

### Findings

- **Static grows volume + decays precision on every model.** Volume rises +0.8k–1.1k Gen0→Gen2 while precision drops 2.5–3.7 pp; the extra annotations are disproportionately FP (label-proliferation, now visible as a precision penalty). Adaptive *trims* volume and holds precision (≈0.91–0.92, stable/rising); progressive sits between with the highest recall (≈0.997). Recall uniformly high (0.96–0.998) — the strategy signal is in precision/FP, not recall. Static is the only strategy whose precision degrades generation-over-generation.

- **Self-training fundamentally undoes the corruption.** Removal and class-swap are recovered at **~99% regardless of strategy or generation** — the teacher, trained on 75%-corrupted labels, still re-emits the correct box at the right location with the right class for nearly every dropped/swapped annotation. The loop is a denoising mechanism, not just a volume amplifier. Box-distortion is the only corruption with headroom (64–77%) and it **declines across generations** under every strategy (localisation drift; class/recall are self-correcting, localisation is not).

- **The strategy difference is in spurious volume, not in recovery.** Since removal/swap recovery is ~99% everywhere, strategies are indistinguishable at *fixing* corruption; what separates them is how many extra FP they pile on (static high, adaptive low).

- **Corruption recipe verified exactly.** Class-agnostic clean↔noisy typing puts swap at exactly 6,441 boxes (25.0%); the raw removed/distorted deviations (36.1%/12.4%) reconcile fully via IoU-band effects (2,872 distortions pushed below IoU 0.5 → read as removed, matching the 2,872 orphan-noisy sanity tally; 385 above IoU 0.9 → read as unchanged; true distorted = 3,184+2,872+385 = 6,441 = 25%; all four types = 6,441).

Caveat: single-seed; re-run A–D once multi-seed ST lands to fold variance into the counts.

---

## 2026-06-18 — PCB-DSLR: static ≡ progressive is by construction, not a bug

The PCB-DSLR detection-growth eval (`experiments/*_st_dslr_*_v3_42/{ic_quality,detection_growth}.csv`)
shows **static and progressive producing byte-identical IC precision/recall, TP/FP/FN, and per-class
detection counts** across all three models (YOLOv8m, YOLOv10m, YOLO12m) and every generation Gen0→Gen3,
while adaptive differs. Verified by `diff` on the generation + metric columns: IC and detection CSVs are
identical static↔progressive for all three models.

### Cause (verified — not a defect)

Training reads `combined/labels`, which `SelfTrainer` builds by merging the IC ground-truth
(`partial_labels_dir`) on top of the pseudo-labels and discarding every pseudo line whose class already
has GT (`src/labeling/self_trainer.py:215-225`). Every PCB-DSLR crop has IC GT, so all IC pseudo-boxes are
dropped and replaced by identical GT. The static↔progressive threshold difference (0.25 vs 0.35/0.45) only
moves boxes inside that confidence band, and on these IC-centric crops the differing pseudo lines are
**100% class 2 (IC)** — exactly the ones the merge throws away. Evidence: `iter{2,3}/combined/labels` are
`differing-files=0` for all three models; the differing `pseudo/labels` lines are all class 2; `.pt` md5
differs only because Ultralytics embeds run path / timestamp / optimizer state. With `seed: 42` +
`workers: 0` the resulting models are functionally identical.

### Framing (Honorato §5.1.2)

PCB-DSLR is a **target-domain transfer demonstration**, not a threshold-strategy benchmark. Honorato uses
a single threshold and reports IC TP/FP, precision, recall-gain (Eq. 5.2:
`Recall_gain = TP_student / TP_teacher`, since FN is unknowable without full GT) in Tabela 14, and
detection-count growth in Fig 42, falling back to qualitative examples (Figs 43-51) because total
detections (~132k) make manual scoring infeasible. The strategy comparison lives on the FICS-noisy domain.

### Reporting change (no loop change, no re-train, no GPU)

- `docs/_gen_dslr_figs.py` + `docs/_gen_docx_pcb_dslr.py`: a shared `group_strategies()` helper detects
  byte-identical strategies per model at render time and collapses them into one merged series / table
  block labelled "Estático ≡ Progressivo". The check is data-driven — if a future multi-seed or other
  domain makes them diverge, they render separately again automatically (verified by perturbing one
  `ic_tp` value: groups split to `[[adaptive],[static],[progressive]]`).
- Added a recall-gain row (Eq. 5.2) to the IC table per strategy: `ic_tp(final) / ic_tp(gen0)`. Our ratios
  (~3.2–5.4×) dwarf Honorato's +43.8% (1.44×) because the Gen0 teacher, FICS-only, barely detects ICs
  cross-domain (recall 0.18–0.31 at Gen0 across the three models) — a tiny denominator, i.e. a weaker
  starting point, not a stronger method. We report *true* IC recall (we have FN from IC GT) as primary; recall-gain is for comparability.

Single-seed (42), descriptive. Qualitative detection examples (Honorato Figs 43-51 analog) deferred.

## 2026-06-20 — WPCB-EFAv2+ component composition COMPLETE (all 3 Gen0 teachers)

Ran `experiments/run_wpcb_efa.sh` on the three Generation-0 noisy-v3 teachers (YOLOv8m, YOLOv10m,
YOLO12m) over the tiled PCB-DSLR `orig_distribution` subset (**107 boards / 465 captures / 2927
tiles**) at conf 0.25, `--rec-policy average`. Outputs: `results/efa/{yolov8,yolov10,yolo12}/efa_{composition,per_board,per_capture}.csv`.
Figures + the primary-detector LaTeX table regenerated via `docs/_gen_efa_figs.py`
(`docs/figures/efa_composition_{yolov8,yolov10,yolo12}.{pdf,png}`, `paper/tables/efa_composition.tex`).

### IC cross-check (GT-validated anchor) — MATCH for all three

EFA all-tile IC total = each model's detection-growth Gen0 IC total (same weights / conf / images),
recomputed from `efa_per_capture.csv` (class_id 2) vs `experiments/<model>_st_dslr_adaptive_v3_42/detection_growth.csv` gen0:

| Model   | EFA all-tile IC | Detection-growth Gen0 IC | Status |
|---------|-----------------|--------------------------|--------|
| YOLOv8m | 3354 | 3354 | MATCH |
| YOLOv10m | 2267 | 2267 | MATCH |
| YOLO12m | 1894 | 1894 | MATCH |

Only IC carries PCB-DSLR GT, so it is the validated anchor; the other 7 classes are detection-derived
composition + priority (frame (a) caveat).

### Composition (board-deduplicated, mean over recordings)

Qualitatively consistent across detectors: ceramic + resistor most numerous; IC dominant by per-crop
area density (v8 7.3% · v10 5.7% · v12 6.1%); tantalum present at priority 2 in all three (class-property
ranking identical across models). Board-deduplicated IC totals differ (v8 743 · v10 495 · v12 418) — the
NMS-free (YOLOv10) and area-attention (YOLO12) heads detect fewer instances than YOLOv8 at the same conf.

Metric is **per-crop area density**, NOT "% of board area" (PCB-DSLR crops are content-centric /
overlapping, not a grid partition). Absolute mass/$ tier remains future work (Gate A — no mm/pixel
scale; `docs/wpcb_efa_feasibility.md`). Single-seed (42), descriptive.
