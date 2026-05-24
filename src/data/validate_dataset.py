from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.image_utils import list_images


@dataclass
class DatasetValidationResult:
    total_images: int
    labeled_images: int
    missing_labels: int


def validate_dataset(images_dir: Path, labels_dir: Path, stats_csv: Path | None = None) -> DatasetValidationResult:
    images = list_images(images_dir)
    rows: list[dict[str, object]] = []
    labeled = 0
    for img in images:
        has_label = (labels_dir / f"{img.stem}.txt").exists()
        labeled += int(has_label)
        rows.append({"image": img.name, "label_exists": has_label})
    if stats_csv:
        stats_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(stats_csv, index=False)
    return DatasetValidationResult(len(images), labeled, len(images) - labeled)


def create_dataset_summary(result: DatasetValidationResult, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        f"# Dataset Summary\n\n- Total images: {result.total_images}\n- Labeled images: {result.labeled_images}\n- Missing labels: {result.missing_labels}\n",
        encoding="utf-8",
    )

