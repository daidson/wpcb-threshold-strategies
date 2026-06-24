import logging
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_data_yaml(
    train_images: str,
    val_images: str,
    nc: int,
    names: list[str],
    output_path: str | Path,
) -> str:
    data = {
        "train": str(Path(train_images).resolve()),
        "val": str(Path(val_images).resolve()),
        "nc": nc,
        "names": names,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return str(output_path)


def save_run_snapshot(config: dict, weights_path: Path, iteration: int) -> None:
    """Write config + git SHA next to best.pt so the run is reproducible."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        logger.warning("Could not determine git SHA — snapshot will record 'unknown'")
        sha = "unknown"

    snapshot = {
        "_meta": {
            "git_sha": sha,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iteration": iteration,
            "weights": str(weights_path),
        },
        **config,
    }
    out = Path(weights_path).parent / "run_snapshot.yaml"
    with open(out, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    logger.info("Run snapshot saved to %s", out)
