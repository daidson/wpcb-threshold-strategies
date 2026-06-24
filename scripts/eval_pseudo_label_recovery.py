"""Noise-recovery decomposition: does self-training undo the label corruption?

Pure label-file operation (no model, no GPU). Two stages:

1. **Corruption typing (generation-independent).** Each clean GT box is matched
   class-agnostically to the noisy training label
   (``data/FICS_PCB_REMAP_NOISE_TRAIN``) at IoU >= 0.5 and typed as:
     * ``removed``       — no noisy box matches (the annotation was dropped);
     * ``swapped``       — matched, but the noisy class differs (class swap);
     * ``box_distorted`` — matched, same class, IoU in ``[0.5, distort_band)``;
     * ``unchanged``     — matched, same class, IoU >= ``distort_band``.
   The corruption recipe is 25% swap + 25% removal + 25% distortion + 25% kept,
   so the typed distribution should land near 25/25/25/25 — this is a built-in
   validator on the classifier (logged; a distorted box that crossed IoU 0.5
   reads as ``removed``, so expect *approximately*, not exactly).

2. **Recovery per generation.** Each generation's pseudo-labels are matched
   class-agnostically to the clean GT. A corrupted clean box counts as
   *recovered* when the pseudo-label fixes the specific corruption:
     * ``removed``       -> a pseudo box restores it with the correct class;
     * ``swapped``       -> the pseudo box carries the correct (clean) class;
     * ``box_distorted`` -> correct class and IoU >= ``distort_band`` (tightened);
     * ``unchanged``     -> correct class and IoU >= ``distort_band`` (preserved;
                           reported as a baseline, not a "correction").

Generation attribution: ``producer_gen = iter - 1``.

Output: ``results/analysis/pseudo_label_recovery.csv``.

Usage (from repo root):
    PYTHONPATH=. uv run python scripts/eval_pseudo_label_recovery.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.io import load_config
from src.utils.labels import (
    box_iou,
    greedy_match,
    parse_run_name,
    read_yolo_label,
    yolo_to_xyxy,
)

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
CLEAN_GT_DIR = Path("data/train_clean/labels")
NOISY_GT_DIR = Path("data/FICS_PCB_REMAP_NOISE_TRAIN/labels")
NAMES_CONFIG = Path("configs/yolov8/base_noisy.yaml")
OUTPUT_CSV = Path("results/analysis/pseudo_label_recovery.csv")

IOU_THR = 0.5
CORRUPTION_TYPES = ["removed", "swapped", "box_distorted", "unchanged"]

FIELDS = [
    "run",
    "model",
    "strategy",
    "iter",
    "producer_gen",
    "class_id",
    "class_name",
    "corruption_type",
    "n_corrupted",
    "n_recovered",
]


def type_corruption(
    clean_gt: dict[str, list],
    distort_band: float,
) -> tuple[dict[str, list[str]], int]:
    """Type every clean box by its corruption, via class-agnostic clean<->noisy match.

    Returns ``({stem: [type per clean box]}, n_orphan_noisy)`` where the orphan
    count is the sanity tally of noisy boxes matching no clean box.
    """
    types: dict[str, list[str]] = {}
    n_orphan = 0
    for stem, gt_boxes in clean_gt.items():
        noisy_boxes = read_yolo_label(NOISY_GT_DIR / f"{stem}.txt")
        gt_cls = [b[0] for b in gt_boxes]
        noisy_cls = [b[0] for b in noisy_boxes]
        iou = box_iou(yolo_to_xyxy(gt_boxes), yolo_to_xyxy(noisy_boxes))
        matches = greedy_match(
            iou, gt_cls, noisy_cls, iou_thr=IOU_THR, class_constrained=False
        )
        match_of = {i: j for i, j in matches}
        box_types: list[str] = []
        for i, c in enumerate(gt_cls):
            j = match_of.get(i)
            if j is None:
                box_types.append("removed")
            elif noisy_cls[j] != c:
                box_types.append("swapped")
            elif iou[i, j] >= distort_band:
                box_types.append("unchanged")
            else:
                box_types.append("box_distorted")
        types[stem] = box_types
        n_orphan += len(noisy_boxes) - len(matches)
    return types, n_orphan


def recover_iteration(
    clean_gt: dict[str, list],
    corruption: dict[str, list[str]],
    pseudo_dir: Path,
    distort_band: float,
) -> dict[tuple[int, str], dict[str, int]]:
    """Count recovered boxes per (class_id, corruption_type) for one generation."""
    stats: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: {"n_corrupted": 0, "n_recovered": 0}
    )
    for stem, gt_boxes in clean_gt.items():
        pred_boxes = read_yolo_label(pseudo_dir / f"{stem}.txt")
        gt_cls = [b[0] for b in gt_boxes]
        pred_cls = [b[0] for b in pred_boxes]
        iou = box_iou(yolo_to_xyxy(gt_boxes), yolo_to_xyxy(pred_boxes))
        matches = greedy_match(
            iou, gt_cls, pred_cls, iou_thr=IOU_THR, class_constrained=False
        )
        match_of = {i: j for i, j in matches}
        for i, c in enumerate(gt_cls):
            ctype = corruption[stem][i]
            key = (c, ctype)
            stats[key]["n_corrupted"] += 1  # count of clean boxes of this type/class
            j = match_of.get(i)
            if j is None:
                continue
            correct_class = pred_cls[j] == c
            if ctype in ("removed", "swapped"):
                recovered = correct_class
            else:  # box_distorted, unchanged
                recovered = correct_class and iou[i, j] >= distort_band
            if recovered:
                stats[key]["n_recovered"] += 1
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    ap.add_argument(
        "--distort-band",
        type=float,
        default=0.9,
        help="IoU at/above which a same-class match is 'unchanged' / recovered.",
    )
    args = ap.parse_args()

    cfg = load_config(NAMES_CONFIG)
    names = cfg["data"]["names"]
    name_of = {i: n for i, n in enumerate(names)}

    clean_gt = {p.stem: read_yolo_label(p) for p in CLEAN_GT_DIR.glob("*.txt")}
    n_clean = sum(len(v) for v in clean_gt.values())

    # Stage 1: corruption typing (once) + validator.
    corruption, n_orphan = type_corruption(clean_gt, args.distort_band)
    dist = Counter(t for boxes in corruption.values() for t in boxes)
    logger.info("Corruption typing over %d clean boxes (distort_band=%.2f):", n_clean, args.distort_band)
    for t in CORRUPTION_TYPES:
        logger.info("  %-13s %6d  (%.1f%%)", t, dist[t], 100 * dist[t] / n_clean)
    logger.info("  orphan noisy boxes (matched no clean): %d", n_orphan)
    # Soft validator: recipe is 25/25/25/25. Removal absorbs far-pushed distortions.
    for t in CORRUPTION_TYPES:
        frac = dist[t] / n_clean
        if not 0.15 <= frac <= 0.35:
            logger.warning("  WARNING: %s fraction %.3f outside [0.15,0.35]", t, frac)

    # Stage 2: recovery per generation, noisy v3 runs only.
    runs = sorted(d for d in EXPERIMENTS_DIR.glob("*_st_noisy_*_v3_42") if d.is_dir())
    rows: list[dict] = []
    for run in runs:
        meta = parse_run_name(run.name)
        if meta is None or meta["track"] != "noisy" or meta["version"] != "v3":
            continue
        iter_dirs = sorted(
            run.glob("iter*/pseudo/labels"),
            key=lambda p: int(re.search(r"iter(\d+)", str(p)).group(1)),
        )
        for pseudo_dir in iter_dirs:
            it = int(re.search(r"iter(\d+)", str(pseudo_dir)).group(1))
            stats = recover_iteration(clean_gt, corruption, pseudo_dir, args.distort_band)
            for c in sorted(name_of):
                for ctype in CORRUPTION_TYPES:
                    s = stats[(c, ctype)]
                    rows.append(
                        {
                            "run": run.name,
                            "model": meta["model"],
                            "strategy": meta["strategy"],
                            "iter": it,
                            "producer_gen": it - 1,
                            "class_id": c,
                            "class_name": name_of[c],
                            "corruption_type": ctype,
                            "n_corrupted": s["n_corrupted"],
                            "n_recovered": s["n_recovered"],
                        }
                    )
            # Per-generation summary over the three corrupted types.
            for ctype in ("removed", "swapped", "box_distorted"):
                nc = sum(stats[(c, ctype)]["n_corrupted"] for c in name_of)
                nr = sum(stats[(c, ctype)]["n_recovered"] for c in name_of)
                logger.info(
                    "%s iter%d (Gen%d) %-13s recovered %d / %d (%.1f%%)",
                    run.name, it, it - 1, ctype, nr, nc, 100 * nr / nc if nc else 0.0,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
