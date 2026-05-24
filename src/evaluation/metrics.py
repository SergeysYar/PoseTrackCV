from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalSample:
    iou: float
    center_error: float
    angle_error: float
    inference_time_ms: float


def iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def angle_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def summarize(samples: list[EvalSample], iou_thr: float = 0.5) -> dict[str, float]:
    if not samples:
        return {}
    ious = np.array([s.iou for s in samples], dtype=float)
    center = np.array([s.center_error for s in samples], dtype=float)
    angle = np.array([s.angle_error for s in samples], dtype=float)
    times = np.array([s.inference_time_ms for s in samples], dtype=float)
    tp = np.sum(ious >= iou_thr)
    n = len(samples)
    mean_t = float(np.mean(times))
    return {
        "samples": float(n),
        "mean_iou": float(np.mean(ious)),
        "detection_accuracy": float(tp / n),
        "precision": float(tp / n),
        "recall": float(tp / n),
        "map50_placeholder": float(tp / n),
        "mean_center_error": float(np.mean(center)),
        "mean_angle_error": float(np.mean(angle)),
        "median_angle_error": float(np.median(angle)),
        "angle_error_lt_5deg_pct": float(np.mean(angle < 5.0) * 100.0),
        "mean_inference_time_ms": mean_t,
        "fps": float(1000.0 / mean_t) if mean_t > 0 else 0.0,
    }

