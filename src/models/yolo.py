from pathlib import Path
from ultralytics import YOLO
from .base import BaseDetector


class YOLODetector(BaseDetector):
    """Wrapper for YOLOv8 and YOLOv10 via Ultralytics."""

    def __init__(self, model_name: str = "yolov8m.pt"):
        self.model_name = model_name
        self.model = YOLO(model_name)

    def train(self, config: dict) -> None:
        self.model.train(**config)

    def predict(self, source: str, conf: float, iou: float) -> list:
        return self.model.predict(source=source, conf=conf, iou=iou)

    def load_weights(self, weights_path: str | Path) -> None:
        self.model = YOLO(str(weights_path))

    def val(self, data_yaml: str) -> object:
        return self.model.val(data=data_yaml)
