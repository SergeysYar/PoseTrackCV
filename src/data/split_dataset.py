from __future__ import annotations

import random
import shutil
from pathlib import Path

from src.utils.image_utils import list_images
from src.utils.io_utils import ensure_dir


def split_dataset(images_dir: Path, labels_dir: Path, out_root: Path, train_ratio: float = 0.7, val_ratio: float = 0.2, seed: int = 42) -> None:
    random.seed(seed)
    images = list_images(images_dir)
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {"train": images[:n_train], "val": images[n_train : n_train + n_val], "test": images[n_train + n_val :]}
    for split, split_imgs in splits.items():
        img_out = out_root / split / "images"
        lab_out = out_root / split / "labels"
        ensure_dir(img_out)
        ensure_dir(lab_out)
        for img in split_imgs:
            shutil.copy2(img, img_out / img.name)
            label = labels_dir / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, lab_out / label.name)

