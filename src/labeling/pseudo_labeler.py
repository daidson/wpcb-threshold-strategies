import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.models.base import BaseDetector
from src.utils.io import IMAGE_EXTS

logger = logging.getLogger(__name__)


class PseudoLabeler:
    """Generates YOLO-format pseudo-labels for unlabeled images using a trained detector."""

    def __init__(
        self,
        model: BaseDetector,
        conf_threshold: float,
        iou_threshold: float,
        candidate_floor: float | None = None,
    ):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Used as inference floor when per-class thresholds are provided, so
        # adaptive thresholds below conf_threshold are not silently filtered out.
        # Stored as-is (None when not provided) so SelfTrainer can detect missing config.
        self.candidate_floor: float | None = candidate_floor

    def generate(
        self,
        images_dir: str,
        output_labels_dir: str,
        per_class_thresholds: dict[int, float] | None = None,
    ) -> dict:
        """Generate pseudo-labels for all images in images_dir.

        Args:
            images_dir: Directory of unlabeled images.
            output_labels_dir: Where to write YOLO .txt label files.
            per_class_thresholds: Optional per-class confidence thresholds.
                When provided, inference runs at candidate_floor and results
                are post-filtered per class. When None, conf_threshold is used
                for both inference and filtering.

        Returns:
            Stats dict with keys ``total``, ``labeled``, ``skipped``.
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_labels_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if per_class_thresholds is not None:
            inference_conf = self.candidate_floor if self.candidate_floor is not None else self.conf_threshold
        else:
            inference_conf = self.conf_threshold
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix in IMAGE_EXTS)
        stats = {"total": len(image_paths), "labeled": 0, "skipped": 0}
        per_class: dict[int, int] = defaultdict(int)

        for img_path in tqdm(image_paths, desc="Generating pseudo-labels"):
            results = self.model.predict(
                str(img_path),
                conf=inference_conf,
                iou=self.iou_threshold,
            )
            label_lines = self._results_to_yolo(results, per_class_thresholds)
            for line in label_lines:
                per_class[int(line.split()[0])] += 1
            if label_lines:
                label_path = output_dir / img_path.with_suffix(".txt").name
                label_path.write_text("\n".join(label_lines))
                stats["labeled"] += 1
            else:
                stats["skipped"] += 1

        stats["per_class"] = dict(per_class)
        return stats

    def _results_to_yolo(
        self,
        results,
        per_class_thresholds: dict[int, float] | None,
    ) -> list[str]:
        lines = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                threshold = (
                    per_class_thresholds.get(cls, self.conf_threshold)
                    if per_class_thresholds is not None
                    else self.conf_threshold
                )
                if conf < threshold:
                    continue
                cx, cy, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines
