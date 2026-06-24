import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.labeling.self_trainer import SelfTrainer
from src.labeling.pseudo_labeler import PseudoLabeler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_image(directory: Path, stem: str, ext: str = ".jpg") -> Path:
    """Write a 1-byte placeholder image so shutil.copy has a real source."""
    p = directory / f"{stem}{ext}"
    p.write_bytes(b"\x00")
    return p


def _fake_label(directory: Path, stem: str, lines: list[str]) -> Path:
    p = directory / f"{stem}.txt"
    # Trailing newline ensures splitlines(keepends=True) returns complete lines
    # and concatenation in the merge code doesn't fuse adjacent lines.
    p.write_text(("".join(l + "\n" for l in lines)) if lines else "")
    return p


def _base_config(nc: int = 8) -> dict:
    names = [f"cls{i}" for i in range(nc)]
    return {
        "data": {"nc": nc, "names": names, "val": "val/images"},
        "train": {"epochs": 1},
        "pseudo_label": {"conf_threshold": 0.5, "adaptive_threshold": False},
        "self_training": {"iterations": 3},
    }


def _make_trainer(config: dict | None = None) -> SelfTrainer:
    config = config or _base_config()
    pl_cfg = config.get("pseudo_label", {})
    model = MagicMock()
    labeler = PseudoLabeler(
        model=MagicMock(),
        conf_threshold=pl_cfg.get("conf_threshold", 0.5),
        iou_threshold=0.45,
        candidate_floor=pl_cfg.get("candidate_floor"),
    )
    return SelfTrainer(model=model, pseudo_labeler=labeler, config=config)


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestSelfTrainerInit:
    def test_adaptive_and_conf_schedule_raises(self):
        cfg = _base_config()
        cfg["pseudo_label"]["adaptive_threshold"] = True
        cfg["pseudo_label"]["conf_schedule"] = [0.25, 0.40, 0.55]
        cfg["pseudo_label"]["candidate_floor"] = 0.25
        with pytest.raises(ValueError, match="mutually exclusive"):
            _make_trainer(cfg)

    def test_adaptive_without_candidate_floor_raises(self):
        cfg = _base_config()
        cfg["pseudo_label"]["adaptive_threshold"] = True
        model = MagicMock()
        # candidate_floor not passed → PseudoLabeler stores None
        labeler = PseudoLabeler(model=MagicMock(), conf_threshold=0.5, iou_threshold=0.45)
        assert labeler.candidate_floor is None
        with pytest.raises(ValueError, match="candidate_floor"):
            SelfTrainer(model=model, pseudo_labeler=labeler, config=cfg)

    def test_valid_adaptive_config_does_not_raise(self):
        cfg = _base_config()
        cfg["pseudo_label"]["adaptive_threshold"] = True
        cfg["pseudo_label"]["candidate_floor"] = 0.25
        trainer = _make_trainer(cfg)
        assert trainer._adaptive is True

    def test_valid_conf_schedule_does_not_raise(self):
        cfg = _base_config()
        cfg["pseudo_label"]["conf_schedule"] = [0.25, 0.40, 0.55]
        trainer = _make_trainer(cfg)
        assert trainer._conf_schedule == [0.25, 0.40, 0.55]


# ---------------------------------------------------------------------------
# _merge_datasets
# ---------------------------------------------------------------------------

class TestMergeDatasets:
    def _run_merge(
        self,
        tmp_path: Path,
        *,
        labeled_images=(),
        labeled_labels=(),
        pseudo_labels=(),
        pseudo_images=(),
        partial_labels=None,
    ) -> tuple[Path, Path]:
        """Set up directories, populate them, run merge, return (out_images, out_labels)."""
        li_dir = tmp_path / "labeled/images"
        ll_dir = tmp_path / "labeled/labels"
        pl_dir = tmp_path / "pseudo/labels"
        pi_dir = tmp_path / "unlabeled/images"
        out_img = tmp_path / "combined/images"
        out_lbl = tmp_path / "combined/labels"
        for d in [li_dir, ll_dir, pl_dir, pi_dir]:
            d.mkdir(parents=True)

        for stem, ext in labeled_images:
            _fake_image(li_dir, stem, ext)
        for stem, lines in labeled_labels:
            _fake_label(ll_dir, stem, lines)
        for stem, lines in pseudo_labels:
            _fake_label(pl_dir, stem, lines)
        for stem, ext in pseudo_images:
            _fake_image(pi_dir, stem, ext)

        partial_dir = None
        if partial_labels is not None:
            partial_dir = tmp_path / "partial/labels"
            partial_dir.mkdir(parents=True)
            for stem, lines in partial_labels:
                _fake_label(partial_dir, stem, lines)

        SelfTrainer._merge_datasets(
            labeled_images=str(li_dir),
            labeled_labels=str(ll_dir),
            pseudo_labels=str(pl_dir),
            pseudo_images=str(pi_dir),
            out_images=str(out_img),
            out_labels=str(out_lbl),
            partial_labels_dir=str(partial_dir) if partial_dir else None,
        )
        return out_img, out_lbl

    # -- Basic merge --------------------------------------------------------

    def test_labeled_images_and_labels_copied(self, tmp_path):
        out_img, out_lbl = self._run_merge(
            tmp_path,
            labeled_images=[("img1", ".jpg")],
            labeled_labels=[("img1", ["0 0.5 0.5 0.2 0.2"])],
        )
        assert (out_img / "img1.jpg").exists()
        assert (out_lbl / "img1.txt").exists()

    def test_pseudo_labeled_pair_copied(self, tmp_path):
        out_img, out_lbl = self._run_merge(
            tmp_path,
            pseudo_labels=[("img2", ["1 0.5 0.5 0.1 0.1"])],
            pseudo_images=[("img2", ".jpg")],
        )
        assert (out_img / "img2.jpg").exists()
        assert (out_lbl / "img2.txt").exists()

    def test_pseudo_label_without_matching_image_skipped(self, tmp_path):
        out_img, out_lbl = self._run_merge(
            tmp_path,
            pseudo_labels=[("orphan", ["2 0.5 0.5 0.1 0.1"])],
            # no image for "orphan" in pseudo_images
        )
        assert not (out_img / "orphan.jpg").exists()
        assert not (out_lbl / "orphan.txt").exists()

    # -- Stale-file clearing ------------------------------------------------

    def test_stale_files_in_combined_are_removed(self, tmp_path):
        combined = tmp_path / "combined"
        combined_img = combined / "images"
        combined_lbl = combined / "labels"
        combined_img.mkdir(parents=True)
        combined_lbl.mkdir(parents=True)
        stale = combined_img / "stale_old.jpg"
        stale.write_bytes(b"\x00")

        out_img, _ = self._run_merge(
            tmp_path,
            labeled_images=[("new_img", ".jpg")],
            labeled_labels=[("new_img", ["0 0.5 0.5 0.2 0.2"])],
        )
        assert not stale.exists(), "stale file from previous run must be cleared"
        assert (out_img / "new_img.jpg").exists()

    # -- Case-insensitive image matching ------------------------------------

    def test_uppercase_extension_matched(self, tmp_path):
        # pseudo-label is foo.txt; image on disk is foo.JPG (uppercase)
        out_img, out_lbl = self._run_merge(
            tmp_path,
            pseudo_labels=[("foo", ["3 0.5 0.5 0.1 0.1"])],
            pseudo_images=[("foo", ".JPG")],
        )
        assert (out_img / "foo.JPG").exists()
        assert (out_lbl / "foo.txt").exists()

    def test_jpeg_extension_matched(self, tmp_path):
        out_img, out_lbl = self._run_merge(
            tmp_path,
            pseudo_labels=[("bar", ["1 0.4 0.4 0.1 0.1"])],
            pseudo_images=[("bar", ".jpeg")],
        )
        assert (out_img / "bar.jpeg").exists()

    # -- Partial-label merging ----------------------------------------------

    def test_partial_label_gt_class_replaces_pseudo_class(self, tmp_path):
        # GT annotates class 3; pseudo has both class 0 and class 3.
        # Result: GT line for class 3 + pseudo line for class 0; pseudo class 3 dropped.
        out_img, out_lbl = self._run_merge(
            tmp_path,
            pseudo_labels=[("img", ["0 0.5 0.5 0.1 0.1", "3 0.3 0.3 0.1 0.1"])],
            pseudo_images=[("img", ".jpg")],
            partial_labels=[("img", ["3 0.9 0.9 0.05 0.05"])],
        )
        content = (out_lbl / "img.txt").read_text().splitlines()
        classes = [int(line.split()[0]) for line in content if line.strip()]
        assert 3 in classes, "GT class 3 must be present"
        assert 0 in classes, "pseudo class 0 must be kept"
        # GT line for class 3 wins — pseudo line for class 3 must be absent
        class_3_lines = [l for l in content if l.strip() and int(l.split()[0]) == 3]
        assert len(class_3_lines) == 1
        assert "0.9" in class_3_lines[0], "surviving class-3 line must be the GT one"

    def test_empty_partial_label_file_skipped(self, tmp_path):
        # An empty partial label should not replace the pseudo-label
        out_img, out_lbl = self._run_merge(
            tmp_path,
            pseudo_labels=[("img", ["0 0.5 0.5 0.1 0.1"])],
            pseudo_images=[("img", ".jpg")],
            partial_labels=[("img", [])],   # empty file
        )
        content = (out_lbl / "img.txt").read_text()
        assert "0" in content

    def test_partial_label_without_pseudo_gets_image_copied(self, tmp_path):
        # Image has a partial GT label but no pseudo-label. It must still appear
        # in combined because it carries ground truth we don't want to lose.
        out_img, out_lbl = self._run_merge(
            tmp_path,
            pseudo_images=[("solo", ".jpg")],   # image exists
            partial_labels=[("solo", ["7 0.5 0.5 0.1 0.1"])],
            # no pseudo-label for "solo"
        )
        assert (out_img / "solo.jpg").exists()
        assert (out_lbl / "solo.txt").exists()


# ---------------------------------------------------------------------------
# _compute_adaptive_thresholds (floor clamping)
# ---------------------------------------------------------------------------

class TestComputeAdaptiveThresholds:
    def _make_metrics(self, p_curves, px=None):
        n = len(p_curves[0])
        mock = MagicMock()
        mock.box.p_curve = np.array(p_curves, dtype=float)
        mock.box.px = np.array(px if px is not None else np.linspace(0, 1, n))
        return mock

    def _make_adaptive_trainer(self, floor: float) -> SelfTrainer:
        cfg = _base_config(nc=2)
        cfg["pseudo_label"]["adaptive_threshold"] = True
        cfg["pseudo_label"]["candidate_floor"] = floor
        cfg["pseudo_label"]["conf_threshold"] = floor
        model = MagicMock()
        # candidate_floor must be passed explicitly so pseudo_labeler.candidate_floor
        # reflects the desired floor, not the hardcoded conf_threshold default.
        labeler = PseudoLabeler(
            model=MagicMock(),
            conf_threshold=floor,
            iou_threshold=0.45,
            candidate_floor=floor,
        )
        return SelfTrainer(model=model, pseudo_labeler=labeler, config=cfg)

    def test_threshold_below_floor_is_clamped_up(self):
        # cls 0 first reaches precision≥0.9 at conf=0.1, which is below floor=0.25
        px = [0.1, 0.5, 0.9]
        p = [[0.9, 0.95, 0.99], [0.1, 0.5, 0.95]]
        metrics = self._make_metrics(p, px)
        trainer = self._make_adaptive_trainer(floor=0.25)
        result = trainer._compute_adaptive_thresholds(metrics)
        assert result[0] == pytest.approx(0.25), "below-floor threshold must clamp to floor"

    def test_threshold_above_floor_passes_through(self):
        # cls 0 first reaches precision≥0.9 only at conf=0.9
        px = [0.1, 0.5, 0.9]
        p = [[0.1, 0.3, 0.9], [0.1, 0.5, 0.95]]
        metrics = self._make_metrics(p, px)
        trainer = self._make_adaptive_trainer(floor=0.25)
        result = trainer._compute_adaptive_thresholds(metrics)
        assert result[0] == pytest.approx(0.9), "above-floor threshold must not be clamped"

    def test_class_never_reaching_target_falls_back(self):
        # cls 0 never reaches precision≥0.9 → falls back to conf_threshold (=floor here)
        px = [0.1, 0.5, 0.9]
        p = [[0.1, 0.3, 0.5], [0.95, 0.96, 0.97]]
        metrics = self._make_metrics(p, px)
        trainer = self._make_adaptive_trainer(floor=0.25)
        result = trainer._compute_adaptive_thresholds(metrics)
        assert result[0] == pytest.approx(0.25), "no-target class must use fallback"

    def test_all_classes_have_thresholds(self):
        p = [[0.0, 0.95], [0.95, 0.2]]
        metrics = self._make_metrics(p)
        trainer = self._make_adaptive_trainer(floor=0.25)
        result = trainer._compute_adaptive_thresholds(metrics)
        assert set(result.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# _seed_thresholds (iteration-1 teacher seed)
# ---------------------------------------------------------------------------

class TestSeedThresholds:
    """The iter1 seed makes the three strategies diverge from Gen1.

    Static → None (PseudoLabeler uses its fixed conf_threshold); progressive →
    conf_schedule[0] for every class; adaptive → per-class precision-target
    threshold from the teacher's val PR curve (delegated to
    _compute_adaptive_thresholds).
    """

    def test_static_returns_none(self):
        trainer = _make_trainer()  # no adaptive, no schedule
        assert trainer._seed_thresholds("val.yaml") is None

    def test_progressive_seeds_schedule_index_zero(self):
        cfg = _base_config()
        cfg["pseudo_label"]["conf_schedule"] = [0.35, 0.45, 0.55]
        trainer = _make_trainer(cfg)
        seed = trainer._seed_thresholds("val.yaml")
        assert seed == {cls: 0.35 for cls in range(cfg["data"]["nc"])}

    def test_progressive_does_not_val_the_teacher(self):
        # Progressive/static thresholds are model-independent; no val() call.
        cfg = _base_config()
        cfg["pseudo_label"]["conf_schedule"] = [0.35, 0.45, 0.55]
        trainer = _make_trainer(cfg)
        trainer._seed_thresholds("val.yaml")
        trainer.model.val.assert_not_called()

    def test_adaptive_vals_teacher_once_and_delegates(self):
        cfg = _base_config(nc=2)
        cfg["pseudo_label"]["adaptive_threshold"] = True
        cfg["pseudo_label"]["candidate_floor"] = 0.05
        cfg["pseudo_label"]["conf_threshold"] = 0.05
        trainer = _make_trainer(cfg)
        metrics = MagicMock()
        # cls0 first reaches precision≥0.9 at px=0.5; cls1 at px=0.2 (both ≥ floor 0.05)
        metrics.box.p_curve = np.array([[0.1, 0.9, 0.95], [0.95, 0.3, 0.1]], dtype=float)
        metrics.box.px = np.array([0.2, 0.5, 0.9])
        trainer.model.val.return_value = metrics

        seed = trainer._seed_thresholds("val.yaml")

        trainer.model.val.assert_called_once_with("val.yaml")
        assert seed == {0: pytest.approx(0.5), 1: pytest.approx(0.2)}
