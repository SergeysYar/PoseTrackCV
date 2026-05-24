from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def list_images(folder: Path, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

