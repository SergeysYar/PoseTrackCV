"""Export BrushPose CSV annotations to YOLO dataset format."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
try:
    import yaml
except Exception:  # pragma: no cover - environment dependent
    yaml = None

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
class ExportConfig:
    images_dir: Path
    annotations: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    class_name: str = "toothbrush"
    class_id: int = 0
    copy_images: bool = False
    overwrite: bool = False


def _load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")


def _normalize_row(row: pd.Series) -> tuple[float, float, float, float]:
    width = float(row["width"])
    height = float(row["height"])
    x_center = float(row["x_center"]) / width
    y_center = float(row["y_center"]) / height
    box_w = (float(row["x_max"]) - float(row["x_min"])) / width
    box_h = (float(row["y_max"]) - float(row["y_min"])) / height
    return x_center, y_center, box_w, box_h


def _prepare_output_dirs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (output_dir / "angle_labels").mkdir(parents=True, exist_ok=True)


def _detect_existing_split(images_dir: Path, filename: str) -> str | None:
    """Detect split by existing directories data/train|val|test/images."""
    # Typical case: images_dir points to data/images, and split folders are siblings.
    data_root = images_dir.parent
    for split in ("train", "val", "test"):
        if (data_root / split / "images" / filename).exists():
            return split
    return None


def _compute_split(df: pd.DataFrame, cfg: ExportConfig) -> pd.DataFrame:
    split_values: list[str] = []
    unresolved_idx: list[int] = []
    for idx, row in df.iterrows():
        filename = str(row["filename"]).strip()
        split = _detect_existing_split(cfg.images_dir, filename)
        if split is None:
            split_values.append("")
            unresolved_idx.append(idx)
        else:
            split_values.append(split)
    df = df.copy()
    df["split"] = split_values

    if unresolved_idx:
        indices = unresolved_idx.copy()
        random.Random(cfg.seed).shuffle(indices)
        n = len(indices)
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)
        for i, df_idx in enumerate(indices):
            if i < n_train:
                df.at[df_idx, "split"] = "train"
            elif i < n_train + n_val:
                df.at[df_idx, "split"] = "val"
            else:
                df.at[df_idx, "split"] = "test"
    return df


def _write_dataset_yaml(output_dir: Path, class_id: int, class_name: str) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    if yaml is None:
        content = "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  {int(class_id)}: {class_name}",
                "",
            ]
        )
        yaml_path.write_text(content, encoding="utf-8")
        return yaml_path
    data = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(class_id): str(class_name)},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def _write_export_report(output_dir: Path, report: dict[str, Any]) -> Path:
    report_path = output_dir / "export_report.md"
    lines = [
        "# YOLO Dataset Export Report",
        "",
        f"- Total images in annotations: {report['total_images']}",
        f"- Train count: {report['train_count']}",
        f"- Val count: {report['val_count']}",
        f"- Test count: {report['test_count']}",
        f"- Skipped rows: {report['skipped_rows']}",
        f"- Class list: {report['class_list']}",
        f"- Output directory: `{output_dir}`",
        f"- Dataset YAML: `{output_dir / 'dataset.yaml'}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def export_yolo_dataset(cfg: ExportConfig) -> tuple[Path, Path]:
    """Convert BrushPose CSV annotations to YOLO dataset format."""
    _validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    if not cfg.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {cfg.images_dir}")
    if not cfg.class_name.strip():
        raise ValueError("--class-name must be non-empty")

    _prepare_output_dirs(cfg.output_dir, cfg.overwrite)
    df = _load_annotations(cfg.annotations)
    df = df.copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df = df[df["filename"] != ""].reset_index(drop=True)
    df = df[df["class_name"].astype(str) == cfg.class_name].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows with class_name='{cfg.class_name}' found.")

    for col in ["width", "height", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center", "angle_deg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _compute_split(df, cfg)

    skipped_rows = 0
    angle_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for _, row in df.iterrows():
        filename = str(row["filename"])
        image_path = cfg.images_dir / filename
        split = str(row["split"])
        if split not in split_counts:
            skipped_rows += 1
            continue
        if image_path.suffix.lower() not in SUPPORTED_EXTS:
            skipped_rows += 1
            continue
        if not image_path.exists():
            skipped_rows += 1
            continue
        if any(pd.isna(row[c]) for c in ["width", "height", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center"]):
            skipped_rows += 1
            continue

        width = float(row["width"])
        height = float(row["height"])
        x_min, y_min = float(row["x_min"]), float(row["y_min"])
        x_max, y_max = float(row["x_max"]), float(row["y_max"])
        if width <= 0 or height <= 0 or x_min >= x_max or y_min >= y_max:
            skipped_rows += 1
            continue

        xcn, ycn, bwn, bhn = _normalize_row(row)
        if not (0 <= xcn <= 1 and 0 <= ycn <= 1 and 0 < bwn <= 1 and 0 < bhn <= 1):
            skipped_rows += 1
            continue

        split_image_dir = cfg.output_dir / "images" / split
        split_label_dir = cfg.output_dir / "labels" / split
        if cfg.copy_images:
            shutil.copy2(image_path, split_image_dir / filename)

        label_file = split_label_dir / f"{Path(filename).stem}.txt"
        label_file.write_text(f"{cfg.class_id} {xcn:.6f} {ycn:.6f} {bwn:.6f} {bhn:.6f}\n", encoding="utf-8")

        angle_rows[split].append(
            {
                "filename": filename,
                "angle_deg": None if pd.isna(row["angle_deg"]) else float(row["angle_deg"]),
                "x_center": float(row["x_center"]),
                "y_center": float(row["y_center"]),
            }
        )
        split_counts[split] += 1

    for split in ("train", "val", "test"):
        angle_df = pd.DataFrame(angle_rows[split], columns=["filename", "angle_deg", "x_center", "y_center"])
        angle_df.to_csv(cfg.output_dir / "angle_labels" / f"{split}_angles.csv", index=False)

    dataset_yaml = _write_dataset_yaml(cfg.output_dir, cfg.class_id, cfg.class_name)
    report_path = _write_export_report(
        cfg.output_dir,
        {
            "total_images": int(len(df)),
            "train_count": split_counts["train"],
            "val_count": split_counts["val"],
            "test_count": split_counts["test"],
            "skipped_rows": skipped_rows,
            "class_list": f"{cfg.class_id}:{cfg.class_name}",
        },
    )
    return dataset_yaml, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export BrushPose CSV annotations to YOLO dataset format.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-name", type=str, default="toothbrush")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        yaml_path, report_path = export_yolo_dataset(
            ExportConfig(
                images_dir=args.images_dir,
                annotations=args.annotations,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                class_name=args.class_name,
                class_id=args.class_id,
                copy_images=args.copy_images,
                overwrite=args.overwrite,
            )
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("YOLO dataset export finished.")
    print(f"- Dataset YAML: {yaml_path}")
    print(f"- Export report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
