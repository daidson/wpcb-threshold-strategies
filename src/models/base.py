from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDetector(ABC):
    """Common interface for all detection models."""

    @abstractmethod
    def train(self, config: dict) -> None: ...

    @abstractmethod
    def predict(self, source: str, conf: float, iou: float) -> list: ...

    @abstractmethod
    def load_weights(self, weights_path: str | Path) -> None: ...

    @abstractmethod
    def val(self, data_yaml: str) -> Any: ...
