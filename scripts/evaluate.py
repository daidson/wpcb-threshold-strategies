"""Evaluate a trained model on the validation set."""
import argparse
import json
import logging
from pathlib import Path

from src.models import MODEL_REGISTRY, build_model
from src.utils.io import load_config, make_data_yaml

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", default="results")
    p.add_argument(
        "--val-images", default=None,
        help="Override the config's val image dir (e.g. data/test/images for the held-out TEST "
             "set). Labels are derived by Ultralytics from the sibling labels/ dir. "
             "Defaults to cfg['data']['val'] (= DEV under the v3 design).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    val_images = args.val_images or cfg["data"]["val"]
    val_yaml = make_data_yaml(
        train_images=cfg["data"]["train"],
        val_images=val_images,
        nc=cfg["data"]["nc"],
        names=cfg["data"]["names"],
        output_path=f"{args.output}/data.yaml",
    )

    model = build_model(args.model, cfg)
    model.load_weights(args.weights)
    metrics = model.val(val_yaml)

    per_class = {
        metrics.names[int(i)]: {"ap50": float(a50), "ap50_95": float(a)}
        for i, a50, a in zip(
            metrics.box.ap_class_index,
            metrics.box.ap50,
            metrics.box.ap,
        )
    }

    results = {
        "model": args.model,
        "weights": args.weights,
        "val_images": val_images,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
    }

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) / f"{args.model}_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", out_file)
    for k, v in results.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
