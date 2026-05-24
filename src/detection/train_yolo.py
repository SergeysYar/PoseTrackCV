from __future__ import annotations

from dataclasses import dataclass

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class YOLOTrainConfig:
    model_name: str
    data_yaml: str
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"


def train_yolo(cfg: YOLOTrainConfig) -> None:
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed.")
    model = YOLO(cfg.model_name)
    model.train(data=cfg.data_yaml, epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device)

