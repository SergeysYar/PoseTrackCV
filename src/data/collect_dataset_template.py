"""Create a normalized image dataset and annotation template for BrushPose AI.

This script ingests raw images, normalizes filenames, optionally resizes images,
optionally converts them to JPG, and generates an initial annotation CSV template.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CollectSummary:
    found_files: int = 0
    processed_images: int = 0
    skipped_files: int = 0


def _iter_supported_files(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])


def _target_name(index: int, force_jpg: bool, source_suffix: str) -> str:
    ext = ".jpg" if force_jpg else source_suffix.lower()
    return f"image_{index:06d}{ext}"


def _prepare_image(image: "cv2.Mat", resize_width: int | None, resize_height: int | None) -> "cv2.Mat":
    if resize_width is None and resize_height is None:
        return image
    if resize_width is None or resize_height is None:
        raise ValueError("Both --resize-width and --resize-height must be provided together.")
    return cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def build_dataset_template(
    input_dir: Path,
    output_dir: Path,
    annotations_out: Path,
    do_copy: bool,
    do_move: bool,
    resize_width: int | None,
    resize_height: int | None,
    convert_to_jpg: bool,
    start_index: int,
    class_name: str,
) -> CollectSummary:
    """Create normalized dataset images and annotation template CSV."""
    if do_copy == do_move:
        raise ValueError("Choose exactly one mode: --copy or --move.")
    if not class_name.strip():
        raise ValueError("--class-name must not be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_out.parent.mkdir(parents=True, exist_ok=True)

    files = _iter_supported_files(input_dir)
    summary = CollectSummary(found_files=len(files))
    rows: list[dict[str, object]] = []

    current_index = start_index
    for src_path in files:
        image = cv2.imread(str(src_path))
        if image is None:
            summary.skipped_files += 1
            continue

        image = _prepare_image(image, resize_width, resize_height)
        out_name = _target_name(current_index, convert_to_jpg, src_path.suffix)
        out_path = output_dir / out_name

        if convert_to_jpg or resize_width is not None:
            if not cv2.imwrite(str(out_path), image):
                summary.skipped_files += 1
                continue
            if do_move:
                src_path.unlink(missing_ok=True)
        else:
            if do_copy:
                shutil.copy2(src_path, out_path)
            else:
                shutil.move(str(src_path), str(out_path))
            image = cv2.imread(str(out_path))
            if image is None:
                summary.skipped_files += 1
                continue

        h, w = image.shape[:2]
        rows.append(
            {
                "filename": out_name,
                "width": int(w),
                "height": int(h),
                "x_min": pd.NA,
                "y_min": pd.NA,
                "x_max": pd.NA,
                "y_max": pd.NA,
                "x_center": pd.NA,
                "y_center": pd.NA,
                "angle_deg": pd.NA,
                "class_name": class_name,
            }
        )
        summary.processed_images += 1
        current_index += 1

    pd.DataFrame(rows).to_csv(annotations_out, index=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build BrushPose dataset image template and annotation CSV.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations-out", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy raw images into dataset.")
    parser.add_argument("--move", action="store_true", help="Move raw images into dataset.")
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--convert-to-jpg", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--class-name", type=str, default="toothbrush")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = build_dataset_template(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            annotations_out=args.annotations_out,
            do_copy=args.copy,
            do_move=args.move,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
            convert_to_jpg=args.convert_to_jpg,
            start_index=args.start_index,
            class_name=args.class_name,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Dataset template collection finished.")
    print(f"- Found files: {summary.found_files}")
    print(f"- Processed images: {summary.processed_images}")
    print(f"- Skipped files: {summary.skipped_files}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Annotation CSV: {args.annotations_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

