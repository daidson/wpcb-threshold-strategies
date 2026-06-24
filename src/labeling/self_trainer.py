import logging
import shutil
from datetime import datetime
from pathlib import Path

from src.models.base import BaseDetector
from src.utils.io import IMAGE_EXTS, make_data_yaml, save_run_snapshot
from src.utils.metrics import compute_adaptive_thresholds
from .pseudo_labeler import PseudoLabeler

logger = logging.getLogger(__name__)


class SelfTrainer:
    """Iterative self-training loop: train → pseudo-label → retrain."""

    def __init__(self, model: BaseDetector, pseudo_labeler: PseudoLabeler, config: dict):
        self.model = model
        self.pseudo_labeler = pseudo_labeler
        self.config = config
        st_cfg = config.get("self_training", {})
        self.iterations = st_cfg.get("iterations", 3)
        self.partial_labels_dir: str | None = st_cfg.get("partial_labels_dir") or None
        self._pseudo_only_mode: bool = st_cfg.get("pseudo_only_mode", False)
        pl_cfg = config.get("pseudo_label", {})
        self._adaptive = pl_cfg.get("adaptive_threshold", False)
        self._target_precision = pl_cfg.get("target_precision", 0.9)
        if self._adaptive and self.pseudo_labeler.candidate_floor is None:
            raise ValueError("adaptive_threshold requires candidate_floor to be set")
        self._conf_schedule: list[float] | None = pl_cfg.get("conf_schedule")
        if self._adaptive and self._conf_schedule is not None:
            raise ValueError("adaptive_threshold and conf_schedule are mutually exclusive")
        if self._conf_schedule is not None and len(self._conf_schedule) < self.iterations:
            raise ValueError(
                f"conf_schedule has {len(self._conf_schedule)} entries but iterations={self.iterations}; "
                f"need at least {self.iterations} (index 0 seeds iter1 from the teacher, "
                f"indices 1–{self.iterations - 1} seed iter2–iter{self.iterations})"
            )
        self._nc = config["data"]["nc"]
        self._class_names: list[str] = config["data"]["names"]

    def run(
        self,
        labeled_images: str,
        labeled_labels: str,
        unlabeled_images: str,
        val_data_yaml: str,
        output_dir: str,
    ) -> list[dict]:
        output_dir = Path(output_dir).resolve()
        history = []

        # Seed iteration 1 from the warm-started teacher using the strategy threshold,
        # so the strategies diverge from Gen1 rather than sharing a flat teacher seed.
        # At this point self.model is the teacher (loaded from warm_start by self_train.py).
        iter1_pseudo = output_dir / "iter1/pseudo/labels"
        if iter1_pseudo.exists() and any(iter1_pseudo.iterdir()):
            logger.info("Pseudo-labels for iter1 already exist — skipping teacher seed.")
        else:
            per_class = self._seed_thresholds(val_data_yaml)
            self.pseudo_labeler.generate(
                unlabeled_images, str(iter1_pseudo), per_class_thresholds=per_class
            )

        for i in range(1, self.iterations + 1):
            logger.info("=== Self-training iteration %d/%d ===", i, self.iterations)

            best_weights = output_dir / f"iter{i}/train/weights/best.pt"

            if best_weights.exists():
                logger.info("Iteration %d already complete — resuming from existing weights.", i)
                self.model.load_weights(best_weights)
            else:
                # Rename any partial train dir left by a crash so Ultralytics uses the
                # expected name rather than auto-incrementing to train2/, train3/, etc.
                partial_train = output_dir / f"iter{i}/train"
                if partial_train.exists():
                    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                    partial_train.rename(partial_train.with_name(f"train.failed-{stamp}"))
                    logger.warning("Renamed incomplete train dir to train.failed-%s", stamp)

                # Merge labeled + pseudo-labeled (+ partial ground-truth labels) into combined set
                combined_images = output_dir / f"iter{i}/combined/images"
                combined_labels = output_dir / f"iter{i}/combined/labels"
                self._merge_datasets(
                    labeled_images, labeled_labels,
                    str(output_dir / f"iter{i}/pseudo/labels"),
                    unlabeled_images,
                    str(combined_images),
                    str(combined_labels),
                    partial_labels_dir=self.partial_labels_dir,
                    pseudo_only_mode=self._pseudo_only_mode,
                )

                # Retrain
                iter_data_yaml = make_data_yaml(
                    train_images=str(combined_images),
                    val_images=self.config["data"]["val"],
                    nc=self.config["data"]["nc"],
                    names=self.config["data"]["names"],
                    output_path=output_dir / f"iter{i}/data.yaml",
                )
                train_cfg = {
                    **self.config.get("train", {}),
                    "data": iter_data_yaml,
                    "project": str(output_dir / f"iter{i}"),
                    "name": "train",
                }
                self.model.train(train_cfg)
                if not best_weights.exists():
                    raise FileNotFoundError(f"Training did not produce weights at {best_weights}")
                self.model.load_weights(best_weights)
                save_run_snapshot(self.config, best_weights, i)

            # Evaluate
            metrics = self.model.val(val_data_yaml)
            history.append({"iteration": i, "metrics": metrics, "weights": str(best_weights)})
            logger.info("Iteration %d done — mAP50: %.4f", i, metrics.box.map50)

            # Generate pseudo-labels for next iteration (skip on the final iteration)
            if i < self.iterations:
                pseudo_label_dir = output_dir / f"iter{i+1}/pseudo/labels"
                if pseudo_label_dir.exists() and any(pseudo_label_dir.iterdir()):
                    logger.info("Pseudo-labels for iter%d already exist — skipping generation.", i + 1)
                else:
                    if self._adaptive:
                        per_class = self._compute_adaptive_thresholds(metrics)
                    elif self._conf_schedule is not None:
                        thr = self._conf_schedule[i]  # i is 1-indexed; schedule[1] seeds iter2, schedule[2] seeds iter3
                        per_class = {cls: thr for cls in range(self._nc)}
                        logger.info("Progressive threshold for iter%d: %.2f", i + 1, thr)
                    else:
                        per_class = None
                    self.pseudo_labeler.generate(unlabeled_images, str(pseudo_label_dir), per_class_thresholds=per_class)

        return history

    def _seed_thresholds(self, val_data_yaml: str) -> dict[int, float] | None:
        """Return per-class thresholds for the iter1 seed, derived from the teacher.

        Adaptive evaluates the teacher on the clean val set and takes per-class max-F1;
        progressive uses the first scheduled value; static returns None so the
        PseudoLabeler falls back to its fixed ``conf_threshold``.
        """
        if self._adaptive:
            teacher_metrics = self.model.val(val_data_yaml)
            return self._compute_adaptive_thresholds(teacher_metrics)
        if self._conf_schedule is not None:
            thr = self._conf_schedule[0]
            logger.info("Progressive threshold for iter1: %.2f", thr)
            return {cls: thr for cls in range(self._nc)}
        return None

    def _compute_adaptive_thresholds(self, metrics) -> dict[int, float]:
        """Return per-class precision-target confidence thresholds from val metrics."""
        fallback = self.config["pseudo_label"]["conf_threshold"]
        raw_thresholds = compute_adaptive_thresholds(
            metrics, self._nc, fallback, self._target_precision
        )
        floor = self.pseudo_labeler.candidate_floor
        thresholds = {
            cls_idx: max(thr, floor)
            for cls_idx, thr in raw_thresholds.items()
        }
        rows = "\n".join(
            f"  cls {cls_idx:2d} ({self._class_names[cls_idx]}): {raw_thresholds[cls_idx]:.4f}"
            + (f" → {thresholds[cls_idx]:.4f} (clamped)" if thresholds[cls_idx] != raw_thresholds[cls_idx] else "")
            for cls_idx in sorted(thresholds)
        )
        logger.info("Adaptive thresholds (precision≥%.2f):\n%s", self._target_precision, rows)
        return thresholds

    @staticmethod
    def _merge_datasets(
        labeled_images: str, labeled_labels: str,
        pseudo_labels: str, pseudo_images: str,
        out_images: str, out_labels: str,
        partial_labels_dir: str | None = None,
        pseudo_only_mode: bool = False,
    ) -> None:
        # Clear the entire combined directory (images/, labels/, and labels.cache) so
        # stale files from a previous or crashed run cannot accumulate on top.
        combined_dir = Path(out_images).parent
        if combined_dir == Path(out_labels).parent and combined_dir.exists():
            shutil.rmtree(combined_dir)
        for out in [Path(out_images), Path(out_labels)]:
            out.mkdir(parents=True)

        if not pseudo_only_mode:
            for img in Path(labeled_images).glob("*"):
                if img.suffix not in IMAGE_EXTS:
                    continue
                shutil.copy(img, Path(out_images) / img.name)
            for lbl in Path(labeled_labels).glob("*.txt"):
                shutil.copy(lbl, Path(out_labels) / lbl.name)

        # For each unlabeled image that has a pseudo-label, copy image + pseudo-label.
        # If a partial ground-truth label also exists for that image, merge it in by
        # appending its lines to the pseudo-label (the partial label takes precedence
        # only for the annotated class; pseudo-label lines for other classes are kept).
        pseudo_label_path = Path(pseudo_labels)
        partial_path = Path(partial_labels_dir) if partial_labels_dir else None
        if pseudo_label_path.exists():
            for lbl in pseudo_label_path.glob("*.txt"):
                img_src = None
                for ext in IMAGE_EXTS:
                    candidate = Path(pseudo_images) / lbl.with_suffix(ext).name
                    if candidate.exists():
                        img_src = candidate
                        break
                if img_src is None:
                    continue
                shutil.copy(img_src, Path(out_images) / img_src.name)
                dest_lbl = Path(out_labels) / lbl.name
                if partial_path is not None:
                    partial_lbl = partial_path / lbl.name
                    if partial_lbl.exists() and partial_lbl.stat().st_size > 0:
                        # Merge: partial ground-truth lines + pseudo-label lines
                        gt_lines = partial_lbl.read_text().splitlines(keepends=True)
                        gt_classes = {int(line.split()[0]) for line in gt_lines if line.strip()}
                        pseudo_lines = lbl.read_text().splitlines(keepends=True)
                        # Keep pseudo lines only for classes absent from ground truth
                        filtered = [l for l in pseudo_lines if l.strip() and int(l.split()[0]) not in gt_classes]
                        dest_lbl.write_text("".join(gt_lines + filtered))
                        continue
                shutil.copy(lbl, dest_lbl)

        # Copy images that have partial ground-truth labels but no pseudo-label
        if partial_path is not None:
            for partial_lbl in partial_path.glob("*.txt"):
                if partial_lbl.stat().st_size == 0:
                    continue
                dest_lbl = Path(out_labels) / partial_lbl.name
                if dest_lbl.exists():
                    continue  # already handled above
                img_src = None
                for ext in IMAGE_EXTS:
                    candidate = Path(pseudo_images) / partial_lbl.with_suffix(ext).name
                    if candidate.exists():
                        img_src = candidate
                        break
                if img_src is not None:
                    shutil.copy(img_src, Path(out_images) / img_src.name)
                    shutil.copy(partial_lbl, dest_lbl)
