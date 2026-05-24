from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def draw_pose(ax: plt.Axes, image_rgb: np.ndarray, bbox: tuple[int, int, int, int], center: tuple[float, float], angle_deg: float, label: str) -> None:
    ax.imshow(image_rgb)
    x1, y1, x2, y2 = bbox
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2))
    cx, cy = center
    ax.scatter([cx], [cy], c="yellow", s=40)
    length = max(20, (x2 - x1) * 0.4)
    rad = math.radians(angle_deg)
    ax.arrow(cx, cy, length * math.cos(rad), length * math.sin(rad), color="red", width=1.2, head_width=8)
    ax.text(x1, max(0, y1 - 8), f"{label} | {angle_deg:.1f} deg", color="white", fontsize=9, bbox={"facecolor": "black", "alpha": 0.5, "pad": 2})
    ax.axis("off")

