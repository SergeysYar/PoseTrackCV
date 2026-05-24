"""Validate BrushPose annotations.csv and image integrity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pandas as pd

REQUIRED_COLUMNS = [
    "filename",
    "width",
    "height",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "x_center",
    "y_center",
    "angle_deg",
    "class_name",
]


@dataclass
class DatasetValidationResult:
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    missing_images: int = 0
    invalid_boxes: int = 0
    invalid_centers: int = 0
    invalid_angles: int = 0
    dimension_mismatches: int = 0
    duplicate_filenames: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class LegacyDatasetValidationResult:
    """Backward-compatible summary for existing project callers."""

    total_images: int
    labeled_images: int
    missing_labels: int


def _to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _validate_row(row: pd.Series, images_dir: Path, idx: int, strict: bool, result: DatasetValidationResult) -> bool:
    row_ok = True
    filename = str(row.get("filename", "")).strip()
    if not filename:
        result.errors.append(f"Row {idx}: empty filename.")
        return False

    image_path = images_dir / filename
    if not image_path.exists():
        result.missing_images += 1
        result.errors.append(f"Row {idx}: image not found '{filename}'.")
        return False

    image = cv2.imread(str(image_path))
    if image is None:
        result.missing_images += 1
        result.errors.append(f"Row {idx}: image is unreadable '{filename}'.")
        return False

    actual_h, actual_w = image.shape[:2]
    width = _to_float(row.get("width"))
    height = _to_float(row.get("height"))
    if width is None or height is None:
        if strict:
            result.dimension_mismatches += 1
            result.errors.append(f"Row {idx}: width/height missing in strict mode.")
            row_ok = False
        else:
            result.warnings.append(f"Row {idx}: width/height missing, can be inferred from image.")
    else:
        if int(width) != int(actual_w) or int(height) != int(actual_h):
            result.dimension_mismatches += 1
            result.errors.append(f"Row {idx}: dimension mismatch csv=({int(width)},{int(height)}) img=({actual_w},{actual_h}).")
            row_ok = False

    x_min = _to_float(row.get("x_min"))
    y_min = _to_float(row.get("y_min"))
    x_max = _to_float(row.get("x_max"))
    y_max = _to_float(row.get("y_max"))

    has_bbox = None not in (x_min, y_min, x_max, y_max)
    if has_bbox:
        if not (0 <= x_min < x_max <= actual_w - 1 and 0 <= y_min < y_max <= actual_h - 1):
            result.invalid_boxes += 1
            result.errors.append(f"Row {idx}: invalid bbox [{x_min},{y_min},{x_max},{y_max}] for image size {actual_w}x{actual_h}.")
            row_ok = False

        x_center = _to_float(row.get("x_center"))
        y_center = _to_float(row.get("y_center"))
        if x_center is None or y_center is None:
            if strict:
                result.invalid_centers += 1
                result.errors.append(f"Row {idx}: center is missing in strict mode.")
                row_ok = False
            else:
                result.warnings.append(f"Row {idx}: missing center, can be computed from bbox.")
        else:
            if not (x_min <= x_center <= x_max and y_min <= y_center <= y_max):
                result.invalid_centers += 1
                result.errors.append(f"Row {idx}: center ({x_center},{y_center}) outside bbox.")
                row_ok = False
    else:
        if strict:
            result.invalid_boxes += 1
            result.errors.append(f"Row {idx}: bbox fields are missing in strict mode.")
            row_ok = False
        else:
            result.warnings.append(f"Row {idx}: bbox fields missing.")

    angle = _to_float(row.get("angle_deg"))
    if angle is None:
        if strict:
            result.invalid_angles += 1
            result.errors.append(f"Row {idx}: angle_deg missing in strict mode.")
            row_ok = False
        else:
            result.warnings.append(f"Row {idx}: angle_deg missing.")
    else:
        if not (0.0 <= angle <= 180.0):
            result.invalid_angles += 1
            result.errors.append(f"Row {idx}: angle_deg={angle} out of [0, 180].")
            row_ok = False

    class_name = str(row.get("class_name", "")).strip()
    if not class_name:
        result.errors.append(f"Row {idx}: class_name is empty.")
        row_ok = False

    return row_ok


def validate_annotations(images_dir: Path, annotations_path: Path, strict: bool = False) -> DatasetValidationResult:
    result = DatasetValidationResult()
    if not annotations_path.exists():
        result.errors.append(f"Annotations file not found: {annotations_path}")
        return result

    df = pd.read_csv(annotations_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        result.errors.append(f"Missing required columns: {missing_columns}")
        return result

    result.total_rows = len(df)
    duplicates = df["filename"].astype(str).str.strip().value_counts()
    result.duplicate_filenames = int((duplicates > 1).sum())
    if result.duplicate_filenames:
        result.errors.append(f"Duplicate filenames detected: {result.duplicate_filenames}")

    for idx, row in df.iterrows():
        ok = _validate_row(row, images_dir, idx + 1, strict, result)
        if ok:
            result.valid_rows += 1
        else:
            result.invalid_rows += 1

    if result.invalid_rows == 0:
        result.invalid_rows = max(0, result.total_rows - result.valid_rows)
    return result


def _build_report(result: DatasetValidationResult, strict: bool) -> str:
    lines = [
        "# Dataset Validation Report",
        "",
        f"- Strict mode: `{strict}`",
        f"- Total annotation rows: {result.total_rows}",
        f"- Total valid rows: {result.valid_rows}",
        f"- Total invalid rows: {result.invalid_rows}",
        f"- Missing images: {result.missing_images}",
        f"- Invalid boxes: {result.invalid_boxes}",
        f"- Invalid centers: {result.invalid_centers}",
        f"- Invalid angles: {result.invalid_angles}",
        f"- Dimension mismatches: {result.dimension_mismatches}",
        f"- Duplicate filenames: {result.duplicate_filenames}",
        "",
        "## First 50 Errors",
    ]
    errors = result.errors[:50] if result.errors else ["No errors."]
    lines.extend([f"- {e}" for e in errors])
    lines.append("")
    lines.append("## First 50 Warnings")
    warnings = result.warnings[:50] if result.warnings else ["No warnings."]
    lines.extend([f"- {w}" for w in warnings])
    return "\n".join(lines) + "\n"


def write_validation_report(report_out: Path, result: DatasetValidationResult, strict: bool) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(_build_report(result, strict), encoding="utf-8")


def validate_dataset(images_dir: Path, labels_dir: Path, stats_csv: Path | None = None) -> LegacyDatasetValidationResult:
    """Legacy validation helper retained for compatibility with existing CLI/tests.

    This function checks presence of per-image TXT labels in `labels_dir`.
    """
    image_files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    rows: list[dict[str, object]] = []
    labeled = 0
    for img in image_files:
        has_label = (labels_dir / f"{img.stem}.txt").exists()
        labeled += int(has_label)
        rows.append({"image": img.name, "label_exists": has_label})
    if stats_csv:
        stats_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(stats_csv, index=False)
    return LegacyDatasetValidationResult(total_images=len(image_files), labeled_images=labeled, missing_labels=len(image_files) - labeled)


def create_dataset_summary(result: LegacyDatasetValidationResult, out_md: Path) -> None:
    """Write backward-compatible compact dataset summary markdown."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        f"# Dataset Summary\n\n- Total images: {result.total_images}\n- Labeled images: {result.labeled_images}\n- Missing labels: {result.missing_labels}\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate BrushPose dataset annotations and images.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_annotations(args.images_dir, args.annotations, strict=args.strict)
    write_validation_report(args.report_out, result, strict=args.strict)

    print("Dataset validation finished.")
    print(f"- Total rows: {result.total_rows}")
    print(f"- Valid rows: {result.valid_rows}")
    print(f"- Invalid rows: {result.invalid_rows}")
    print(f"- Missing images: {result.missing_images}")
    print(f"- Invalid boxes: {result.invalid_boxes}")
    print(f"- Invalid centers: {result.invalid_centers}")
    print(f"- Invalid angles: {result.invalid_angles}")
    print(f"- Dimension mismatches: {result.dimension_mismatches}")
    print(f"- Duplicate filenames: {result.duplicate_filenames}")
    print(f"- Report: {args.report_out}")

    has_errors = len(result.errors) > 0
    if args.strict and has_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
