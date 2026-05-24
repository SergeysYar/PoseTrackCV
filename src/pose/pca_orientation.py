from __future__ import annotations

import numpy as np

from .geometry_utils import normalize_angle_180


def pca_orientation_from_points(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    centered = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    return normalize_angle_180(float(angle))

