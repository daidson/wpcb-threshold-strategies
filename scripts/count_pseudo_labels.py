"""Count pseudo-label annotations per class, per generation, per strategy.

Pure label-file operation (no model, no GPU): walks every self-training run
under ``experiments/*_st_*/iter*/pseudo/labels`` and counts one annotation per
non-empty line, bucketed by class id. Appends reference rows counting the clean
pre-corruption GT (``data/train_clean/labels``) and the noisy training labels
(``data/FICS_PCB_REMAP_NOISE_TRAIN/labels``) so volume can be read against both
ground truths.

Generation attribution: ``producer_gen = iter - 1`` (the Gen0 teacher seeds
iter1/pseudo, Gen1 produces iter2/pseudo, ...). Both columns are written so the
off-by-one is explicit.

Output: ``results/analysis/pseudo_label_counts.csv`` (long format).

Usage (from repo root):
    PYTHONPATH=. uv run python scripts/count_pseudo_labels.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path

from src.utils.io import load_config
from src.utils.labels import parse_run_name, read_yolo_label

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
CLEAN_GT_DIR = Path("data/train_clean/labels")
NOISY_GT_DIR = Path("data/FICS_PCB_REMAP_NOISE_TRAIN/labels")
NAMES_CONFIG = Path("configs/yolov8/base_noisy.yaml")
OUTPUT_CSV = Path("results/analysis/pseudo_label_counts.csv")

FIELDS = [
    "source",
    "run",
    "model",
    "track",
    "strategy",
    "version",
    "seed",
    "iter",
    "producer_gen",
    "class_id",
    "class_name",
    "count",
]


def count_dir(labels_dir: Path) -> Counter:
    """Count annotations per class id over all .txt files in a labels dir."""
    counts: Counter = Counter()
    for txt in labels_dir.glob("*.txt"):
        for cls, *_ in read_yolo_label(txt):
            counts[cls] += 1
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = ap.parse_args()

    cfg = load_config(NAMES_CONFIG)
    names = cfg["data"]["names"]
    name_of = {i: n for i, n in enumerate(names)}

    rows: list[dict] = []

    # Pseudo-label rows, every run dir with an iter*/pseudo/labels.
    run_dirs = sorted(d for d in EXPERIMENTS_DIR.glob("*_st_*") if d.is_dir())
    for run in run_dirs:
        meta = parse_run_name(run.name)
        if meta is None:
            continue
        iter_dirs = sorted(
            run.glob("iter*/pseudo/labels"),
            key=lambda p: int(re.search(r"iter(\d+)", str(p)).group(1)),
        )
        for labels_dir in iter_dirs:
            it = int(re.search(r"iter(\d+)", str(labels_dir)).group(1))
            counts = count_dir(labels_dir)
            for cls, count in sorted(counts.items()):
                rows.append(
                    {
                        "source": "pseudo",
                        **meta,
                        "iter": it,
                        "producer_gen": it - 1,
                        "class_id": cls,
                        "class_name": name_of.get(cls, f"class_{cls}"),
                        "count": count,
                    }
                )
            logger.info(
                "%s iter%d: %d annotations across %d classes",
                run.name, it, sum(counts.values()), len(counts),
            )

    # Reference rows: clean GT and noisy GT over the training split.
    for source, gt_dir in (("clean_gt", CLEAN_GT_DIR), ("noisy_gt", NOISY_GT_DIR)):
        counts = count_dir(gt_dir)
        for cls, count in sorted(counts.items()):
            rows.append(
                {
                    "source": source,
                    "run": "",
                    "model": "",
                    "track": "",
                    "strategy": "",
                    "version": "",
                    "seed": "",
                    "iter": "",
                    "producer_gen": "",
                    "class_id": cls,
                    "class_name": name_of.get(cls, f"class_{cls}"),
                    "count": count,
                }
            )
        logger.info("%s: %d annotations total", source, sum(counts.values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
