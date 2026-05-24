from __future__ import annotations

from pathlib import Path

import pandas as pd


def csv_to_yolo(csv_path: Path, output_dir: Path, image_width: int, image_height: int) -> int:
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_name, group in df.groupby("image"):
        lines: list[str] = []
        for _, row in group.iterrows():
            x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
            cx = ((x1 + x2) / 2) / image_width
            cy = ((y1 + y2) / 2) / image_height
            w = (x2 - x1) / image_width
            h = (y2 - y1) / image_height
            cls = int(row.get("class_id", 0))
            angle = float(row.get("angle_deg", 0.0))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.3f}")
        (output_dir / f"{Path(image_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        count += 1
    return count

