"""Generate pseudo-labels for unlabeled images using a trained model."""
import argparse
import logging

from src.models import MODEL_REGISTRY, build_model
from src.labeling.pseudo_labeler import PseudoLabeler
from src.utils.io import load_config

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--config", required=True, help="Path to YAML config (pseudo_label block required)")
    p.add_argument("--weights", required=True, help="Path to trained weights (.pt)")
    p.add_argument("--images", default="data/unlabeled/images")
    p.add_argument("--output", default="data/pseudo_labeled/labels")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    pl_cfg = cfg["pseudo_label"]

    model = build_model(args.model, cfg)
    model.load_weights(args.weights)

    labeler = PseudoLabeler(
        model,
        conf_threshold=pl_cfg["conf_threshold"],
        iou_threshold=pl_cfg["nms_iou"],
    )
    stats = labeler.generate(args.images, args.output)

    logger.info(
        "Done: %d/%d images labeled, %d skipped (no detections)",
        stats["labeled"], stats["total"], stats["skipped"],
    )


if __name__ == "__main__":
    main()
