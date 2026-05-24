from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geometry_utils import bbox_center, normalize_angle_180
from .pca_orientation import pca_orientation_from_points


@dataclass
class ClassicalResult:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    angle_rect: float
    angle_pca: float
    mask: np.ndarray


class ClassicalPoseDetector:
    def __init__(
        self,
        hsv_lower: tuple[int, int, int] = (0, 0, 30),
        hsv_upper: tuple[int, int, int] = (180, 80, 255),
        morph_kernel: int = 5,
        min_area: int = 500,
    ) -> None:
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.morph_kernel = morph_kernel
        self.min_area = min_area

    def detect(self, image_bgr: np.ndarray) -> Optional[ClassicalResult]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        center = bbox_center(x, y, x + w, y + h)
        rect = cv2.minAreaRect(contour)
        rect_angle = rect[2] + (90.0 if rect[1][0] < rect[1][1] else 0.0)
        rect_angle = normalize_angle_180(float(rect_angle))
        pca_angle = pca_orientation_from_points(contour.reshape(-1, 2))

        return ClassicalResult((x, y, x + w, y + h), center, rect_angle, pca_angle, mask)

