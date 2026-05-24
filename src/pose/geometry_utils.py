from __future__ import annotations

import numpy as np


def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def normalize_angle_180(angle_deg: float) -> float:
    angle = angle_deg % 180.0
    return angle if angle >= 0 else angle + 180.0


def vector_to_angle(vx: float, vy: float) -> float:
    angle = np.degrees(np.arctan2(vy, vx))
    return normalize_angle_180(float(angle))

