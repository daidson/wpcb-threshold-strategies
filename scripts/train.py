"""Train a baseline detector on labeled data only."""
import argparse
import logging
from pathlib import Path

from src.models import MODEL_REGISTRY, build_model
from src.utils.io import load_config, make_data_yaml

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--config", required=True, help="Path to YAML config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if "seed" not in cfg.get("train", {}):
        logger.warning("No 'seed' in train config — results will not be reproducible. Add 'seed: 42' to the train block.")

    data_yaml = make_data_yaml(
        train_images=cfg["data"]["train"],
        val_images=cfg["data"]["val"],
        nc=cfg["data"]["nc"],
        names=cfg["data"]["names"],
        output_path="configs/data.yaml",
    )

    model = build_model(args.model, cfg)
    train_cfg = {**cfg.get("train", {}), "data": data_yaml}
    if "project" in train_cfg:
        train_cfg["project"] = str(Path(train_cfg["project"]).resolve())
    model.train(train_cfg)


if __name__ == "__main__":
    main()
