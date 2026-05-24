"""Convert BrushPose annotations between csv/yolo/pose-csv formats."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

BRUSHPOSE_COLUMNS = [
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


def _read_image_size(images_dir: Path, filename: str) -> tuple[int, int]:
    image = cv2.imread(str(images_dir / filename))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {images_dir / filename}")
    h, w = image.shape[:2]
    return w, h


def _load_brushpose_csv(path: Path, images_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in BRUSHPOSE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for brushpose-csv: {missing}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df = df[df["filename"] != ""].reset_index(drop=True)

    for idx, row in df.iterrows():
        if pd.isna(row["width"]) or pd.isna(row["height"]):
            w, h = _read_image_size(images_dir, str(row["filename"]))
            df.at[idx, "width"] = w
            df.at[idx, "height"] = h
        if pd.isna(row["x_center"]) or pd.isna(row["y_center"]):
            df.at[idx, "x_center"] = (float(row["x_min"]) + float(row["x_max"])) / 2.0
            df.at[idx, "y_center"] = (float(row["y_min"]) + float(row["y_max"])) / 2.0
    return df


def brushpose_to_yolo(df: pd.DataFrame, output_dir: Path, class_name: str, class_id: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, row in df.iterrows():
        if class_name and str(row["class_name"]) != class_name:
            continue
        w = float(row["width"])
        h = float(row["height"])
        x_center = float(row["x_center"]) / w
        y_center = float(row["y_center"]) / h
        box_w = (float(row["x_max"]) - float(row["x_min"])) / w
        box_h = (float(row["y_max"]) - float(row["y_min"])) / h
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
        (output_dir / f"{Path(str(row['filename'])).stem}.txt").write_text(line, encoding="utf-8")
        count += 1
    return count


def brushpose_to_pose_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_df = df[["filename", "x_center", "y_center", "angle_deg", "class_name"]].copy()
    out_path = output_dir / "pose_annotations.csv"
    pose_df.to_csv(out_path, index=False)
    return out_path


def run_conversion(
    input_path: Path,
    images_dir: Path,
    output_dir: Path,
    from_format: str,
    to_format: str,
    class_name: str,
    class_id: int,
) -> int:
    if from_format != "brushpose-csv":
        raise ValueError("Only --from-format brushpose-csv is currently supported.")
    if to_format not in {"yolo", "pose-csv"}:
        raise ValueError("Supported --to-format values: yolo, pose-csv")

    df = _load_brushpose_csv(input_path, images_dir)
    if to_format == "yolo":
        converted = brushpose_to_yolo(df, output_dir, class_name=class_name, class_id=class_id)
        print(f"YOLO labels generated: {converted}")
        return converted
    pose_csv = brushpose_to_pose_csv(df, output_dir)
    print(f"Pose CSV generated: {pose_csv}")
    return len(df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert BrushPose annotations between formats.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-format", type=str, required=True)
    parser.add_argument("--to-format", type=str, required=True)
    parser.add_argument("--class-name", type=str, default="toothbrush")
    parser.add_argument("--class-id", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        converted = run_conversion(
            input_path=args.input,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            from_format=args.from_format,
            to_format=args.to_format,
            class_name=args.class_name,
            class_id=args.class_id,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Annotation conversion finished.")
    print(f"- Converted records/files: {converted}")
    print(f"- Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
