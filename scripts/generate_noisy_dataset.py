#!/usr/bin/env python3
"""
Generate a noise-injected variant of data/labeled/ for noise-robustness experiments.

Replicates and extends Honorato et al. (2025) noise-injection protocol with three
noise types from Freire et al. (2024):
  - classification noise: annotations have their class ID swapped to a different class
    (simulates teacher mis-classification)
  - removal noise: annotation lines are deleted from the label file
    (simulates missed detections / false negatives)
  - distortion noise: bounding box coordinates perturbed by uniform factor in [0.95, 1.05]
    (simulates inaccurate localisation)

Default fractions match the 75% total corruption rate (Honorato request 2026-05-18):
  25% classification noise + 25% removal noise + 25% box distortion = 75% of all annotations
  corrupted. The distortion RNG uses seed+1 so existing swap/removal assignments are preserved.

val set (data/val/) is never touched.

Usage (from repo root):
    uv run python scripts/generate_noisy_dataset.py \\
        --classification-frac 0.25 --removal-frac 0.25 --distortion-frac 0.25 --seed 42 --force

Output:
    data/FICS_PCB_REMAP_NOISE/images/           -- symlink to original images (unchanged)
    data/FICS_PCB_REMAP_NOISE/labels/           -- noisy label .txt files
    data/FICS_PCB_REMAP_NOISE/noise_manifest.json -- full audit trail for reproducibility
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
NUM_CLASSES = 8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classification-frac", type=float, default=0.25,
                   help="Fraction of all annotations to corrupt with class-swap noise (default: 0.25)")
    p.add_argument("--removal-frac", type=float, default=0.25,
                   help="Fraction of all annotations to remove (default: 0.25)")
    p.add_argument("--distortion-frac", type=float, default=0.0,
                   help="Fraction of all annotations to corrupt with box distortion (default: 0.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--src-labels", type=Path, default=ROOT / "data/labeled/labels",
                   help="Source label directory (default: data/labeled/labels)")
    p.add_argument("--src-images", type=Path, default=ROOT / "data/labeled/images",
                   help="Source images directory (default: data/labeled/images)")
    p.add_argument("--output", type=Path, default=ROOT / "data/FICS_PCB_REMAP_NOISE",
                   help="Output directory (default: data/FICS_PCB_REMAP_NOISE)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output directory if it already exists")
    return p.parse_args()


def _swap_class(original: int, rng: random.Random, num_classes: int) -> int:
    """Return a class ID that differs from original, chosen uniformly."""
    candidates = [c for c in range(num_classes) if c != original]
    return rng.choice(candidates)


def _distort_box(line: str, rng: random.Random) -> str:
    """Apply uniform [0.95, 1.05] multiplicative perturbation to each box corner.

    Converts YOLO cx/cy/w/h to corners, perturbs each corner independently,
    clamps to [0,1], ensures valid box (x2>x1, y2>y1), converts back.
    Class ID is preserved unchanged.
    """
    parts = line.strip().split()
    cls = parts[0]
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    x1 *= rng.uniform(0.95, 1.05)
    y1 *= rng.uniform(0.95, 1.05)
    x2 *= rng.uniform(0.95, 1.05)
    y2 *= rng.uniform(0.95, 1.05)

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1:
        x1, x2 = x2, x1
    if y2 <= y1:
        y1, y2 = y2, y1

    # Guard against degenerate zero-area box after clamping
    if x2 == x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 == y1:
        y2 = min(1.0, y1 + 1e-4)

    new_cx = (x1 + x2) / 2
    new_cy = (y1 + y2) / 2
    new_w = x2 - x1
    new_h = y2 - y1

    return f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n"


def main() -> None:
    args = _parse_args()

    if args.classification_frac + args.removal_frac + args.distortion_frac > 1.0:
        raise ValueError(
            f"classification_frac ({args.classification_frac}) + "
            f"removal_frac ({args.removal_frac}) + "
            f"distortion_frac ({args.distortion_frac}) must be <= 1.0"
        )

    src_labels: Path = args.src_labels.resolve()
    src_images: Path = args.src_images.resolve()
    dst: Path = args.output.resolve()
    dst_labels = dst / "labels"
    dst_images = dst / "images"

    if dst.exists() and not args.force:
        raise FileExistsError(
            f"{dst} already exists. Pass --force to overwrite."
        )

    # ── Collect all (file, line_index, raw_line) annotation entries ──────────
    log.info("Scanning source labels: %s", src_labels)
    label_files = sorted(src_labels.glob("*.txt"))
    if not label_files:
        raise RuntimeError(f"No .txt files found in {src_labels}")

    # Each entry: (Path, line_idx, raw_line_content)
    all_annotations: list[tuple[Path, int, str]] = []
    file_lines: dict[Path, list[str]] = {}

    for lf in label_files:
        lines = lf.read_text().splitlines(keepends=True)
        file_lines[lf] = lines
        for idx, line in enumerate(lines):
            if line.strip():
                all_annotations.append((lf, idx, line))

    total = len(all_annotations)
    n_class = int(total * args.classification_frac)
    n_remove = int(total * args.removal_frac)
    n_distort = int(total * args.distortion_frac)
    log.info("Total annotations: %d", total)
    log.info("Classification noise: %d (%.1f%%)", n_class, 100 * n_class / total)
    log.info("Removal noise:        %d (%.1f%%)", n_remove, 100 * n_remove / total)
    log.info("Distortion noise:     %d (%.1f%%)", n_distort, 100 * n_distort / total)
    log.info("Unchanged:            %d (%.1f%%)", total - n_class - n_remove - n_distort,
             100 * (total - n_class - n_remove - n_distort) / total)

    # ── Assign noise types ────────────────────────────────────────────────────
    # Primary RNG (seed) governs swap/removal assignment — same as before this change.
    # Distortion RNG (seed+1) is separate to preserve existing swap/removal assignments.
    rng = random.Random(args.seed)
    shuffled_indices = list(range(total))
    rng.shuffle(shuffled_indices)

    classification_set = set(shuffled_indices[:n_class])
    removal_set = set(shuffled_indices[n_class: n_class + n_remove])
    distortion_set = set(shuffled_indices[n_class + n_remove: n_class + n_remove + n_distort])

    distortion_rng = random.Random(args.seed + 1)

    # ── Build noisy label content per file ───────────────────────────────────
    # Maps Path → list of noisy lines (may be shorter than original if lines removed)
    noisy_files: dict[Path, list[str]] = {lf: list(lines) for lf, lines in file_lines.items()}

    manifest_entries: list[dict] = []

    for entry_idx, (lf, line_idx, raw) in enumerate(all_annotations):
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        original_class = int(parts[0])

        if entry_idx in classification_set:
            new_class = _swap_class(original_class, rng, NUM_CLASSES)
            noisy_line = f"{new_class} " + " ".join(parts[1:]) + "\n"
            noisy_files[lf][line_idx] = noisy_line
            manifest_entries.append({
                "file": lf.name,
                "line_idx": line_idx,
                "type": "classification",
                "original_class": original_class,
                "new_class": new_class,
            })

        elif entry_idx in removal_set:
            # Replace with empty string; strip later when writing
            noisy_files[lf][line_idx] = ""
            manifest_entries.append({
                "file": lf.name,
                "line_idx": line_idx,
                "type": "removal",
                "original_class": original_class,
            })

        elif entry_idx in distortion_set:
            noisy_line = _distort_box(raw, distortion_rng)
            noisy_files[lf][line_idx] = noisy_line
            manifest_entries.append({
                "file": lf.name,
                "line_idx": line_idx,
                "type": "distortion",
                "original_class": original_class,
            })

    # ── Write output ──────────────────────────────────────────────────────────
    if dst.exists() and args.force:
        import shutil
        shutil.rmtree(dst)

    dst_labels.mkdir(parents=True, exist_ok=True)

    # Symlink images directory to the original (images unchanged)
    real_images = src_images.resolve()
    dst_images.symlink_to(real_images)
    log.info("Symlinked %s → %s", dst_images, real_images)

    for lf, lines in noisy_files.items():
        out_path = dst_labels / lf.name
        content = "".join(line for line in lines if line)  # drop removed annotations
        out_path.write_text(content)

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "seed": args.seed,
        "num_annotations_total": total,
        "num_classification_noise": n_class,
        "num_removal_noise": n_remove,
        "num_distortion_noise": n_distort,
        "classification_fraction": args.classification_frac,
        "removal_fraction": args.removal_frac,
        "distortion_fraction": args.distortion_frac,
        "total_noise_fraction": (n_class + n_remove + n_distort) / total,
        "source_labels": str(src_labels),
        "output": str(dst),
        "corrupted_annotations": manifest_entries,
    }
    manifest_path = dst / "noise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written: %s", manifest_path)
    log.info("Done. Noisy dataset at: %s", dst)

    # ── Summary ───────────────────────────────────────────────────────────────
    actual_total_corrupted = n_class + n_remove + n_distort
    log.info(
        "\nNoise summary\n"
        "  Source:               %s\n"
        "  Output:               %s\n"
        "  Files processed:      %d\n"
        "  Total annotations:    %d\n"
        "  Classification noise: %d  (%.1f%%)\n"
        "  Removal noise:        %d  (%.1f%%)\n"
        "  Distortion noise:     %d  (%.1f%%)\n"
        "  Total corrupted:      %d  (%.1f%%)\n"
        "  Unchanged:            %d  (%.1f%%)",
        src_labels, dst,
        len(label_files), total,
        n_class, 100 * n_class / total,
        n_remove, 100 * n_remove / total,
        n_distort, 100 * n_distort / total,
        actual_total_corrupted, 100 * actual_total_corrupted / total,
        total - actual_total_corrupted, 100 * (total - actual_total_corrupted) / total,
    )


if __name__ == "__main__":
    main()
