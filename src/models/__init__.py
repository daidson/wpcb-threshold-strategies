from .yolo import YOLODetector
from .rtdetr import RTDETRDetector

MODEL_REGISTRY = {
    "yolov8": lambda cfg: YOLODetector(cfg.get("model", "yolov8m.pt")),
    "yolov10": lambda cfg: YOLODetector(cfg.get("model", "yolov10m.pt")),
    "yolo12": lambda cfg: YOLODetector(cfg.get("model", "yolo12m.pt")),
    "rtdetr_l": lambda cfg: RTDETRDetector(cfg.get("model", "rtdetr-l.pt")),
    "rtdetr_x": lambda cfg: RTDETRDetector(cfg.get("model", "rtdetr-x.pt")),
}


def build_model(model_type: str, config: dict):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](config)
