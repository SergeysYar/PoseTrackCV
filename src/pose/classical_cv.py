"""Classical OpenCV toothbrush pose estimation pipeline for BrushPose AI."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ClassicalCVConfig:
    """Configuration for classical CV pose estimation."""

    hsv_lower: tuple[int, int, int] = (0, 40, 40)
    hsv_upper: tuple[int, int, int] = (180, 255, 255)
    blur_kernel: int = 5
    morph_kernel: int = 5
    min_contour_area: float = 500.0
    max_contour_area_ratio: float = 0.7
    use_pca_angle: bool = True
    use_min_area_rect_angle: bool = False
    draw_debug: bool = False
    arrow_length: int = 80


@dataclass
class PosePrediction:
    """Structured prediction output for one image."""

    filename: str
    success: bool
    x_min: int | None
    y_min: int | None
    x_max: int | None
    y_max: int | None
    x_center: float | None
    y_center: float | None
    angle_deg: float | None
    area: float | None
    confidence: float | None
    processing_time_ms: float
    message: str


class ClassicalCVPoseEstimator:
    """Classical OpenCV estimator for toothbrush localization and orientation."""

    def __init__(self, config: ClassicalCVConfig) -> None:
        self.config = config

    def predict_image(self, image_path: Path) -> tuple[PosePrediction, np.ndarray | None]:
        """Run prediction for a single image path."""
        start = time.perf_counter()
        if not image_path.exists():
            return (
                PosePrediction(
                    filename=image_path.name,
                    success=False,
                    x_min=None,
                    y_min=None,
                    x_max=None,
                    y_max=None,
                    x_center=None,
                    y_center=None,
                    angle_deg=None,
                    area=None,
                    confidence=None,
                    processing_time_ms=(time.perf_counter() - start) * 1000.0,
                    message="Image file does not exist.",
                ),
                None,
            )

        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return (
                PosePrediction(
                    filename=image_path.name,
                    success=False,
                    x_min=None,
                    y_min=None,
                    x_max=None,
                    y_max=None,
                    x_center=None,
                    y_center=None,
                    angle_deg=None,
                    area=None,
                    confidence=None,
                    processing_time_ms=(time.perf_counter() - start) * 1000.0,
                    message="Unsupported image extension.",
                ),
                None,
            )

        image = cv2.imread(str(image_path))
        if image is None:
            return (
                PosePrediction(
                    filename=image_path.name,
                    success=False,
                    x_min=None,
                    y_min=None,
                    x_max=None,
                    y_max=None,
                    x_center=None,
                    y_center=None,
                    angle_deg=None,
                    area=None,
                    confidence=None,
                    processing_time_ms=(time.perf_counter() - start) * 1000.0,
                    message="OpenCV cannot read image.",
                ),
                None,
            )
        return self.predict_array(image, filename=image_path.name)

    def predict_array(self, image: np.ndarray, filename: str = "") -> tuple[PosePrediction, np.ndarray | None]:
        """Run prediction for an already-loaded image array."""
        start = time.perf_counter()
        image_out = image.copy()
        try:
            pre = self.preprocess(image)
            mask = self.segment(pre)
            mask = self.postprocess_mask(mask)
            contours = self.find_candidate_contours(mask, image.shape)
            contour = self.select_best_contour(contours)
            if contour is None:
                pred = PosePrediction(
                    filename=filename,
                    success=False,
                    x_min=None,
                    y_min=None,
                    x_max=None,
                    y_max=None,
                    x_center=None,
                    y_center=None,
                    angle_deg=None,
                    area=None,
                    confidence=None,
                    processing_time_ms=(time.perf_counter() - start) * 1000.0,
                    message="No valid contour found.",
                )
                cv2.putText(image_out, "Detection failed: no valid contour", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return pred, image_out

            x, y, w, h = cv2.boundingRect(contour)
            x_center, y_center = self.estimate_center(contour)
            if self.config.use_pca_angle:
                angle = self.estimate_angle_pca(contour)
            elif self.config.use_min_area_rect_angle:
                angle = self.estimate_angle_min_area_rect(contour)
            else:
                angle = self.estimate_angle_min_area_rect(contour)
            angle = self.normalize_angle(angle)
            area = float(cv2.contourArea(contour))
            confidence = self.compute_confidence(contour, mask, image.shape)
            pred = PosePrediction(
                filename=filename,
                success=True,
                x_min=int(x),
                y_min=int(y),
                x_max=int(x + w),
                y_max=int(y + h),
                x_center=float(x_center),
                y_center=float(y_center),
                angle_deg=float(angle),
                area=area,
                confidence=confidence,
                processing_time_ms=(time.perf_counter() - start) * 1000.0,
                message="OK",
            )
            return pred, self.draw_prediction(image_out, contour, pred)
        except Exception as exc:
            pred = PosePrediction(
                filename=filename,
                success=False,
                x_min=None,
                y_min=None,
                x_max=None,
                y_max=None,
                x_center=None,
                y_center=None,
                angle_deg=None,
                area=None,
                confidence=None,
                processing_time_ms=(time.perf_counter() - start) * 1000.0,
                message=f"Processing error: {exc}",
            )
            cv2.putText(image_out, f"Detection failed: {exc}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return pred, image_out

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply optional blur before segmentation."""
        k = self.config.blur_kernel
        if k <= 1:
            return image
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), 0)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to HSV and threshold."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(self.config.hsv_lower, dtype=np.uint8)
        upper = np.array(self.config.hsv_upper, dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening and closing."""
        k = self.config.morph_kernel
        if k <= 1:
            return mask
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def find_candidate_contours(self, mask: np.ndarray, image_shape: tuple[int, int, int]) -> list[np.ndarray]:
        """Find and filter contours by area constraints."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = float(image_shape[0] * image_shape[1])
        max_area = img_area * self.config.max_contour_area_ratio
        valid: list[np.ndarray] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.config.min_contour_area:
                continue
            if area > max_area:
                continue
            valid.append(c)
        return valid

    def select_best_contour(self, contours: list[np.ndarray]) -> np.ndarray | None:
        """Select best contour (largest valid area)."""
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def estimate_center(self, contour: np.ndarray) -> tuple[float, float]:
        """Estimate contour center using moments with bbox fallback."""
        m = cv2.moments(contour)
        if abs(m["m00"]) > 1e-9:
            return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        return float(x + w / 2.0), float(y + h / 2.0)

    def estimate_angle_pca(self, contour: np.ndarray) -> float:
        """Estimate orientation using manual PCA over contour points."""
        pts = contour.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] < 2:
            raise ValueError("PCA failed: too few contour points.")
        mean = np.mean(pts, axis=0, keepdims=True)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        angle_rad = math.atan2(float(major[1]), float(major[0]))
        return math.degrees(angle_rad)

    def estimate_angle_min_area_rect(self, contour: np.ndarray) -> float:
        """Estimate orientation using OpenCV minAreaRect convention."""
        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]
        angle = float(rect[2])
        if w < h:
            angle += 90.0
        return angle

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [0, 180)."""
        v = angle % 180.0
        return v if v >= 0 else v + 180.0

    def compute_confidence(self, contour: np.ndarray, mask: np.ndarray, image_shape: tuple[int, int, int]) -> float:
        """Compute heuristic confidence in [0, 1].

        Uses a weighted blend:
        - area ratio quality (penalizes too small/too large regions)
        - solidity (area / convex hull area)
        - bbox fill ratio (contour area / bbox area)
        - mask coverage consistency (contour area / mask foreground area)
        """
        area = float(cv2.contourArea(contour))
        img_area = float(image_shape[0] * image_shape[1])
        area_ratio = area / max(img_area, 1.0)
        max_ratio = max(self.config.max_contour_area_ratio, 1e-6)
        area_quality = min(1.0, area_ratio / max_ratio)
        area_quality = max(0.0, area_quality)

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = float(max(w * h, 1))
        bbox_fill = area / rect_area

        fg_pixels = float(np.count_nonzero(mask))
        mask_consistency = area / fg_pixels if fg_pixels > 1e-6 else 0.0

        conf = 0.30 * area_quality + 0.30 * solidity + 0.20 * bbox_fill + 0.20 * mask_consistency
        return float(max(0.0, min(1.0, conf)))

    def draw_prediction(self, image: np.ndarray, contour: np.ndarray, prediction: PosePrediction) -> np.ndarray:
        """Draw bbox, center, orientation arrow, labels, and optional contour."""
        out = image.copy()
        if not prediction.success:
            cv2.putText(out, f"Failed: {prediction.message}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return out

        assert prediction.x_min is not None
        assert prediction.y_min is not None
        assert prediction.x_max is not None
        assert prediction.y_max is not None
        assert prediction.x_center is not None
        assert prediction.y_center is not None
        assert prediction.angle_deg is not None
        assert prediction.confidence is not None

        cv2.rectangle(out, (prediction.x_min, prediction.y_min), (prediction.x_max, prediction.y_max), (0, 255, 0), 2)
        cpt = (int(round(prediction.x_center)), int(round(prediction.y_center)))
        cv2.circle(out, cpt, 4, (0, 255, 255), -1)

        rad = math.radians(prediction.angle_deg)
        dx = int(round(self.config.arrow_length * math.cos(rad)))
        dy = int(round(self.config.arrow_length * math.sin(rad)))
        cv2.arrowedLine(out, cpt, (cpt[0] + dx, cpt[1] + dy), (0, 0, 255), 2, tipLength=0.25)

        label1 = f"ClassicalCV angle={prediction.angle_deg:.1f} deg"
        label2 = f"conf={prediction.confidence:.2f}"
        cv2.putText(out, label1, (prediction.x_min, max(20, prediction.y_min - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, label2, (prediction.x_min, max(38, prediction.y_min - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.config.draw_debug:
            cv2.drawContours(out, [contour], -1, (255, 0, 0), 2)
        return out

