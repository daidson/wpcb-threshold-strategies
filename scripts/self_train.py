"""Run the full iterative self-training pipeline."""
import argparse
import logging

from pathlib import Path

from src.models import MODEL_REGISTRY, build_model
from src.labeling.pseudo_labeler import PseudoLabeler
from src.labeling.self_trainer import SelfTrainer
from src.utils.io import load_config, make_data_yaml
from src.utils.visualization import plot_training_curves

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--config", required=True)
    p.add_argument("--output", default="experiments/self_train")
    p.add_argument("--no-adaptive", dest="no_adaptive", action="store_true", default=False)
    p.add_argument("--labeled-images", default="data/labeled/images",
                   help="Path to labeled training images (default: data/labeled/images)")
    p.add_argument("--labeled-labels", default="data/labeled/labels",
                   help="Path to labeled training label files (default: data/labeled/labels)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.no_adaptive:
        cfg.setdefault("pseudo_label", {})["adaptive_threshold"] = False

    if "seed" not in cfg.get("train", {}):
        raise ValueError("No 'seed' in train config — add 'seed: 42' to the train block.")

    model = build_model(args.model, cfg)
    warm_start = cfg.get("self_training", {}).get("warm_start")
    if warm_start:
        model.load_weights(warm_start)

    pl_cfg = cfg["pseudo_label"]
    pseudo_labeler = PseudoLabeler(
        model,
        conf_threshold=pl_cfg["conf_threshold"],
        iou_threshold=pl_cfg["nms_iou"],
        candidate_floor=pl_cfg.get("candidate_floor"),
    )

    val_yaml = make_data_yaml(
        train_images=cfg["data"]["train"],
        val_images=cfg["data"]["val"],
        nc=cfg["data"]["nc"],
        names=cfg["data"]["names"],
        output_path=f"{args.output}/data.yaml",
    )

    unlabeled_images = cfg.get("self_training", {}).get("unlabeled_dir", "data/unlabeled/images")
    trainer = SelfTrainer(model, pseudo_labeler, cfg)
    history = trainer.run(
        labeled_images=args.labeled_images,
        labeled_labels=args.labeled_labels,
        unlabeled_images=unlabeled_images,
        val_data_yaml=val_yaml,
        output_dir=args.output,
    )

    plot_training_curves(history, f"{args.output}/training_curve.png")
    logger.info("Self-training complete.")


if __name__ == "__main__":
    main()
