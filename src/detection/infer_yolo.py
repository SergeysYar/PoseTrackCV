from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.pose.geometry_utils import bbox_center
from src.pose.pca_orientation import pca_orientation_from_points
from src.pose.pose_estimator import PosePrediction

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLODetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 640, device: str = "cpu") -> None:
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(weights) if YOLO is not None else None

    def _angle_from_roi(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pts = np.column_stack(np.where(th > 0))
        if pts.shape[0] < 2:
            return 0.0
        pts_xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)
        return pca_orientation_from_points(pts_xy)

    def predict(self, image: np.ndarray) -> list[PosePrediction]:
        if self.model is None:
            raise RuntimeError("Ultralytics not installed.")
        results = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        preds: list[PosePrediction] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cx, cy = bbox_center(x1, y1, x2, y2)
                angle = self._angle_from_roi(image, (x1, y1, x2, y2))
                preds.append(PosePrediction(cx, cy, angle, (x1, y1, x2, y2), float(b.conf[0]), "yolo"))
        return preds

    def predict_path(self, image_path: Path) -> list[PosePrediction]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        return self.predict(image)

