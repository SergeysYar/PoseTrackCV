from __future__ import annotations

from pathlib import Path

import cv2

from src.utils.io_utils import ensure_dir


def extract_frames(video_path: Path, output_dir: Path, every_n: int = 10) -> int:
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            cv2.imwrite(str(output_dir / f"frame_{idx:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

