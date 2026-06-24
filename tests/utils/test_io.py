import yaml
import pytest
from pathlib import Path

from src.utils.io import load_config, make_data_yaml
from src.utils.metrics import save_metrics, load_metrics


def test_load_config_returns_dict(tmp_path):
    cfg = {"train": {"epochs": 10}, "data": {"nc": 8}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    result = load_config(p)
    assert result == cfg


def test_make_data_yaml_roundtrip(tmp_path):
    out = tmp_path / "iter1" / "data.yaml"
    path = make_data_yaml(
        train_images=str(tmp_path / "combined/images"),
        val_images=str(tmp_path / "val/images"),
        nc=8,
        names=["a", "b", "c", "d", "e", "f", "g", "h"],
        output_path=out,
    )
    assert Path(path).exists()
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["nc"] == 8
    assert data["names"] == ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert "train" in data and "val" in data


def test_make_data_yaml_creates_parent_dirs(tmp_path):
    out = tmp_path / "deep" / "nested" / "data.yaml"
    make_data_yaml(
        train_images=str(tmp_path),
        val_images=str(tmp_path),
        nc=2,
        names=["x", "y"],
        output_path=out,
    )
    assert out.exists()


def test_save_and_load_metrics_roundtrip(tmp_path):
    metrics = {"map50": 0.860, "map50_95": 0.652, "classes": {"diode": 0.56}}
    out = tmp_path / "results" / "metrics.json"
    save_metrics(metrics, out)
    loaded = load_metrics(out)
    assert loaded == metrics


def test_load_metrics_preserves_float_precision(tmp_path):
    metrics = {"map50": 0.826123}
    out = tmp_path / "m.json"
    save_metrics(metrics, out)
    assert load_metrics(out)["map50"] == pytest.approx(0.826123)
