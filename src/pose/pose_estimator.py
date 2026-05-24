from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PosePrediction:
    x_center: float
    y_center: float
    angle_deg: float
    bbox: tuple[int, int, int, int]
    score: float = 1.0
    method: str = "unknown"

