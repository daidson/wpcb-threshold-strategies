"""Visualize YOLO annotations on a few images to verify class ID → name mapping.

Usage:
    python scripts/verify_classes.py                    # random sample of 6 images
    python scripts/verify_classes.py --class-id 3      # only images containing class 3
    python scripts/verify_classes.py --class-id 4 -n 4
"""
import argparse
import random
from pathlib import Path

import matplotlib.axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

NAMES = [
    "resistor_smd",
    "ceramic_capacitor",
    "ic",
    "diode",
    "inductor",
    "electrolytic_capacitor",
    "tantalum_capacitor",
    "led",
]
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]

IMAGES_DIR = Path("data/labeled/images")
LABELS_DIR = Path("data/labeled/labels")


def draw_image(ax: matplotlib.axes.Axes, img_path: Path, label_path: Path) -> None:
    img = Image.open(img_path)
    w, h = img.size
    ax.imshow(img)
    ax.set_title(img_path.stem, fontsize=7)
    ax.axis("off")

    if not label_path.exists():
        return

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        x = (cx - bw / 2) * w
        y = (cy - bh / 2) * h
        rect = patches.Rectangle(
            (x, y), bw * w, bh * h,
            linewidth=1.5, edgecolor=COLORS[cls % len(COLORS)], facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x, y - 2, NAMES[cls] if cls < len(NAMES) else str(cls),
            fontsize=6, color=COLORS[cls % len(COLORS)],
            bbox={"boxstyle": "square,pad=0.1", "fc": "black", "alpha": 0.5},
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--class-id", type=int, default=None, help="Filter to images containing this class ID")
    p.add_argument("-n", type=int, default=6, help="Number of images to show")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    label_files = list(LABELS_DIR.glob("*.txt"))
    if args.class_id is not None:
        prefix = f"{args.class_id} "
        label_files = [
            f for f in label_files
            if any(line.startswith(prefix) for line in f.read_text().splitlines())
        ]

    sample = random.sample(label_files, min(args.n, len(label_files)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = [axes] if rows == 1 and cols == 1 else list(
        axes.flat if hasattr(axes, "flat") else axes
    )

    for ax, label_path in zip(axes, sample):
        stem = label_path.stem
        img_path = next(
            (IMAGES_DIR / f"{stem}{ext}" for ext in (".png", ".jpg", ".PNG", ".JPG", ".jpeg", ".JPEG")
             if (IMAGES_DIR / f"{stem}{ext}").exists()),
            None,
        )
        if img_path:
            draw_image(ax, img_path, label_path)
        else:
            ax.set_visible(False)

    for ax in axes[len(sample):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("class_verification.png", dpi=150, bbox_inches="tight")
    print("Saved class_verification.png — open it to inspect the annotations.")


if __name__ == "__main__":
    main()
