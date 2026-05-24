"""Pure metric functions for BrushPose AI evaluation subsystem."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable

import numpy as np


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU for two boxes in (x_min, y_min, x_max, y_max) format."""
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
    except Exception:
        return 0.0

    if ax1 >= ax2 or ay1 >= ay2 or bx1 >= bx2 or by1 >= by2:
        return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(max(0.0, min(1.0, inter / union)))


def compute_center_error(pred_center: tuple[float, float], gt_center: tuple[float, float]) -> float:
    """Compute Euclidean distance between predicted and ground-truth centers."""
    px, py = pred_center
    gx, gy = gt_center
    return float(math.hypot(float(px) - float(gx), float(py) - float(gy)))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 180]."""
    a = float(angle) % 180.0
    return a if a >= 0 else a + 180.0


def compute_angle_error(pred_angle: float, gt_angle: float) -> float:
    """Compute minimal absolute angle error with 180-degree symmetry."""
    p = normalize_angle(pred_angle)
    g = normalize_angle(gt_angle)
    diff = abs(p - g)
    return float(min(diff, 180.0 - diff))


def compute_detection_accuracy(matches: Iterable[bool]) -> float:
    """Compute successful detections / total samples."""
    values = [bool(m) for m in matches]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_precision_recall(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and f1 score from TP/FP/FN counts."""
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_processing_stats(times_ms: Iterable[float]) -> dict[str, float]:
    """Compute processing-time statistics and FPS."""
    values = [float(v) for v in times_ms if v is not None and not math.isnan(float(v))]
    if not values:
        return {
            "mean_processing_time_ms": 0.0,
            "median_processing_time_ms": 0.0,
            "min_processing_time_ms": 0.0,
            "max_processing_time_ms": 0.0,
            "fps": 0.0,
        }
    mean_ms = float(mean(values))
    return {
        "mean_processing_time_ms": mean_ms,
        "median_processing_time_ms": float(median(values)),
        "min_processing_time_ms": float(min(values)),
        "max_processing_time_ms": float(max(values)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def summarize_numeric_metric(values: Iterable[float]) -> dict[str, float]:
    """Compute mean/median/std/min/max for numeric values."""
    arr = np.array([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# Backward-compatibility aliases used by legacy code
def iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Backward-compatible IoU helper."""
    return compute_iou(tuple(a), tuple(b))


def angle_diff_deg(a: float, b: float) -> float:
    """Backward-compatible angle error helper."""
    return compute_angle_error(a, b)

