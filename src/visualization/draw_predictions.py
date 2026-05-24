from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.pose.pose_estimator import PosePrediction
from src.visualization.overlay_utils import draw_pose


def save_prediction_viz(image_bgr, pred: PosePrediction, out_path: Path, title: str = "Prediction") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_pose(ax, rgb, pred.bbox, (pred.x_center, pred.y_center), pred.angle_deg, pred.method)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

