#!/usr/bin/env python3
"""Carve a held-out DEV split from the clean training boards for leak-free calibration.

Motivation
----------
The adaptive per-class threshold used to be calibrated on ``data/val`` — the same
set used to report final mAP. That is a calibration leak that biases the very
comparison the thesis makes. This script builds a board-grouped held-out DEV set
(Honorato's *subconjunto validado*) so that:

  * TRAIN  — noisy labels, training only.
  * DEV    — clean labels, ``best.pt`` monitoring (all runs) AND adaptive
             precision-target calibration.
  * TEST   — the existing ``data/val`` (clean), read only by the final eval.

Split unit
----------
The grouping unit is the **s-number** (e.g. ``s14``), NOT the ``_crop_`` prefix.
The original train/val split was made at s-number granularity (no s-number
straddles train and val), and several s-numbers have multiple prefix variants
(``s14_back1``, ``s14_back2``, ...) that are the same physical board captured
differently. Carving DEV on prefixes would split one physical board across
DEV and TRAIN — reintroducing the leak inside the calibration set.

Selection
---------
Seeded class-stratified search over subsets of the train s-numbers. A DEV
candidate is valid when:
  1. DEV image count is within ``[--min-images, --max-images]`` (≈816 default).
  2. Every one of the 8 classes appears in DEV with count >= ``--min-dev-per-class``.
  3. For every class, TRAIN retains >= ``--train-retain-frac`` of the global
     instance count (so a rare class — electrolytic_capacitor lives in only 4
     boards — is never drained into DEV).
Among valid candidates, the one maximizing the minimum per-class DEV fraction
(most balanced coverage) is chosen; ties broken by closeness to the target image
count, then lexicographically for determinism.

Materializes (symlinks by default; ``--copy`` to copy):
  data/dev/{images,labels}          -- clean images + clean labels, DEV s-numbers
  data/train_clean/{images,labels}  -- clean images + clean labels, TRAIN s-numbers
                                       (intermediate; feed to generate_noisy_dataset.py)
  data/split_manifest.json          -- board->split, seed, per-class counts

``data/val`` (= TEST) is never touched.

Usage (from repo root):
    uv run python scripts/make_splits.py --seed 42
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
NUM_CLASSES = 8
CLASS_NAMES = [
    "resistor_smd", "ceramic_capacitor", "ic", "diode",
    "inductor", "electrolytic_capacitor", "tantalum_capacitor", "led",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
S_NUMBER_RE = re.compile(r"^(s\d+)_")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src-images", type=Path, default=ROOT / "data/labeled/images",
                   help="Clean train images (default: data/labeled/images)")
    p.add_argument("--src-labels", type=Path, default=ROOT / "data/labeled/labels",
                   help="Clean train labels (default: data/labeled/labels)")
    p.add_argument("--test-images", type=Path, default=ROOT / "data/val/images",
                   help="TEST images, used only for the disjointness assertion (default: data/val/images)")
    p.add_argument("--dev-out", type=Path, default=ROOT / "data/dev")
    p.add_argument("--train-out", type=Path, default=ROOT / "data/train_clean")
    p.add_argument("--test-alias", type=Path, default=ROOT / "data/test",
                   help="Semantic TEST path; created as a symlink to the TEST images' parent "
                        "(data/val) so final evals can use --val-images data/test/images "
                        "while data/val stays byte-for-byte. (default: data/test)")
    p.add_argument("--manifest", type=Path, default=ROOT / "configs/split_manifest.json",
                   help="Tracked split record (default: configs/split_manifest.json; data/ is "
                        "gitignored + symlinked, so the manifest cannot be committed there)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-images", type=int, default=730, help="Min DEV image count")
    p.add_argument("--max-images", type=int, default=900, help="Max DEV image count")
    p.add_argument("--min-dev-per-class", type=int, default=10,
                   help="Minimum instances of every class required in DEV")
    p.add_argument("--train-retain-frac", type=float, default=0.5,
                   help="TRAIN must keep at least this fraction of every class's instances")
    p.add_argument("--trials", type=int, default=200_000, help="Seeded random search trials")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of symlinking (use if symlinks are unreadable on the mount)")
    p.add_argument("--force", action="store_true", help="Overwrite existing output dirs")
    return p.parse_args()


def _s_number(name: str) -> str:
    m = S_NUMBER_RE.match(name)
    if not m:
        raise ValueError(f"Cannot extract s-number from {name!r}")
    return m.group(1)


def _scan(src_images: Path, src_labels: Path):
    """Return (images_by_board, class_counts_by_board, global_class_counts)."""
    images_by_board: dict[str, list[Path]] = defaultdict(list)
    cls_by_board: dict[str, Counter] = defaultdict(Counter)
    global_cls: Counter = Counter()
    for img in sorted(src_images.iterdir()):
        if img.suffix not in IMAGE_EXTS:
            continue
        board = _s_number(img.name)
        images_by_board[board].append(img)
        lf = src_labels / (img.stem + ".txt")
        if lf.exists():
            for line in lf.read_text().splitlines():
                line = line.strip()
                if line:
                    c = int(line.split()[0])
                    cls_by_board[board][c] += 1
                    global_cls[c] += 1
    return images_by_board, cls_by_board, global_cls


def _select_dev(boards, imgs_by_board, cls_by_board, global_cls, args):
    """Seeded class-stratified search. Returns the chosen DEV board list."""
    img_count = {b: len(imgs_by_board[b]) for b in boards}
    target = (args.min_images + args.max_images) / 2
    rng = random.Random(args.seed)

    best = None  # (score, -img_dist, board_tuple)
    best_boards = None
    for _ in range(args.trials):
        shuffled = boards[:]
        rng.shuffle(shuffled)
        dev: list[str] = []
        total = 0
        for b in shuffled:
            if total >= args.min_images:
                break
            dev.append(b)
            total += img_count[b]
        if not (args.min_images <= total <= args.max_images):
            continue

        dev_cls = Counter()
        for b in dev:
            dev_cls.update(cls_by_board[b])
        # (2) every class present in DEV with min count
        if any(dev_cls[c] < args.min_dev_per_class for c in range(NUM_CLASSES)):
            continue
        # (3) TRAIN retains >= train_retain_frac of every class
        if any(
            (global_cls[c] - dev_cls[c]) < args.train_retain_frac * global_cls[c]
            for c in range(NUM_CLASSES)
        ):
            continue

        score = min(dev_cls[c] / global_cls[c] for c in range(NUM_CLASSES))
        key = (score, -abs(total - target), tuple(sorted(dev)))
        if best is None or key > best:
            best = key
            best_boards = sorted(dev)
    if best_boards is None:
        raise RuntimeError(
            "No valid DEV split found. Relax --min-dev-per-class / --train-retain-frac "
            "or widen the image window."
        )
    return best_boards


def _materialize(boards, imgs_by_board, src_labels, out_dir: Path, copy: bool):
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)
    n = 0
    for b in boards:
        for img in imgs_by_board[b]:
            dst_img = out_images / img.name
            src_lbl = src_labels / (img.stem + ".txt")
            dst_lbl = out_labels / (img.stem + ".txt")
            if copy:
                shutil.copy(img, dst_img)
                if src_lbl.exists():
                    shutil.copy(src_lbl, dst_lbl)
            else:
                os.symlink(img.resolve(), dst_img)
                if src_lbl.exists():
                    os.symlink(src_lbl.resolve(), dst_lbl)
            n += 1
    return n


def main() -> None:
    args = _parse_args()
    src_images = args.src_images.resolve()
    src_labels = args.src_labels.resolve()

    for out in (args.dev_out, args.train_out):
        if out.exists():
            if not args.force:
                raise FileExistsError(f"{out} exists. Pass --force to overwrite.")
            shutil.rmtree(out)

    log.info("Scanning clean train set: %s", src_images)
    imgs_by_board, cls_by_board, global_cls = _scan(src_images, src_labels)
    boards = sorted(imgs_by_board)
    total_images = sum(len(v) for v in imgs_by_board.values())
    log.info("Found %d s-numbers, %d images", len(boards), total_images)

    dev_boards = _select_dev(boards, imgs_by_board, cls_by_board, global_cls, args)
    train_boards = [b for b in boards if b not in set(dev_boards)]

    dev_cls = Counter()
    for b in dev_boards:
        dev_cls.update(cls_by_board[b])
    train_cls = Counter()
    for b in train_boards:
        train_cls.update(cls_by_board[b])

    dev_imgs = sum(len(imgs_by_board[b]) for b in dev_boards)
    train_imgs = sum(len(imgs_by_board[b]) for b in train_boards)

    log.info("DEV boards (%d, %d imgs): %s", len(dev_boards), dev_imgs, dev_boards)
    log.info("TRAIN boards (%d, %d imgs): %s", len(train_boards), train_imgs, train_boards)
    log.info("Per-class counts (class: DEV / TRAIN / global):")
    for c in range(NUM_CLASSES):
        log.info("  %d %-24s %5d / %5d / %5d", c, CLASS_NAMES[c],
                 dev_cls[c], train_cls[c], global_cls[c])

    # ── Disjointness assertions (board overlap) ──────────────────────────────
    assert not (set(dev_boards) & set(train_boards)), "DEV/TRAIN board overlap!"
    test_boards = set()
    if args.test_images.exists():
        test_boards = {_s_number(p.name) for p in args.test_images.iterdir()
                       if p.suffix in IMAGE_EXTS}
        assert not (set(dev_boards) & test_boards), "DEV/TEST board overlap!"
        assert not (set(train_boards) & test_boards), "TRAIN/TEST board overlap!"
    log.info("Board disjointness OK (TRAIN/DEV/TEST share no s-number).")

    # ── Materialize ──────────────────────────────────────────────────────────
    mode = "copying" if args.copy else "symlinking"
    log.info("Materializing DEV (%s)...", mode)
    n_dev = _materialize(dev_boards, imgs_by_board, src_labels, args.dev_out, args.copy)
    log.info("Materializing TRAIN_CLEAN (%s)...", mode)
    n_train = _materialize(train_boards, imgs_by_board, src_labels, args.train_out, args.copy)
    log.info("DEV: %d images -> %s", n_dev, args.dev_out)
    log.info("TRAIN_CLEAN: %d images -> %s", n_train, args.train_out)

    # ── TEST alias (data/test -> data/val) ───────────────────────────────────
    # The TEST set is the existing clean val split, kept byte-for-byte. We expose
    # it under the semantic name data/test (a directory symlink) so the final
    # evaluation can read data/test/{images,labels} without touching data/val.
    test_src = args.test_images.parent  # data/val
    if args.test_alias.is_symlink() or args.test_alias.exists():
        if args.test_alias.is_symlink():
            args.test_alias.unlink()
        else:
            log.warning("%s exists and is not a symlink — leaving it untouched", args.test_alias)
            test_src = None
    if test_src is not None and test_src.exists():
        os.symlink(test_src.resolve(), args.test_alias)
        log.info("TEST alias: %s -> %s", args.test_alias, test_src)

    # ── Manifest ─────────────────────────────────────────────────────────────
    board_to_split = {b: "dev" for b in dev_boards}
    board_to_split.update({b: "train" for b in train_boards})
    for b in sorted(test_boards):
        board_to_split[b] = "test"
    manifest = {
        "seed": args.seed,
        "split_unit": "s-number",
        "selection": {
            "min_images": args.min_images,
            "max_images": args.max_images,
            "min_dev_per_class": args.min_dev_per_class,
            "train_retain_frac": args.train_retain_frac,
            "trials": args.trials,
        },
        "source": {"images": str(src_images), "labels": str(src_labels)},
        "materialized_as": "copy" if args.copy else "symlink",
        "counts": {
            "dev_boards": len(dev_boards), "dev_images": dev_imgs,
            "train_boards": len(train_boards), "train_images": train_imgs,
            "test_boards": len(test_boards),
        },
        "dev_boards": dev_boards,
        "train_boards": train_boards,
        "test_boards": sorted(test_boards),
        "per_class": {
            CLASS_NAMES[c]: {"dev": dev_cls[c], "train": train_cls[c], "global": global_cls[c]}
            for c in range(NUM_CLASSES)
        },
        "board_to_split": board_to_split,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written: %s", args.manifest)


if __name__ == "__main__":
    main()
