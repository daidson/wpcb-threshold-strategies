from pathlib import Path
from ultralytics import RTDETR
from .base import BaseDetector


class RTDETRDetector(BaseDetector):
    """Wrapper for RT-DETR-l and RT-DETR-x via Ultralytics."""

    def __init__(self, model_name: str = "rtdetr-l.pt"):
        self.model_name = model_name
        self.model = RTDETR(model_name)

    def train(self, config: dict) -> None:
        self.model.train(**config)

    def predict(self, source: str, conf: float, iou: float) -> list:
        return self.model.predict(source=source, conf=conf, iou=iou)

    def load_weights(self, weights_path: str | Path) -> None:
        self.model = RTDETR(str(weights_path))

    def val(self, data_yaml: str) -> object:
        return self.model.val(data=data_yaml)
