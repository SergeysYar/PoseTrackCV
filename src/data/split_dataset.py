"""Split BrushPose dataset into train/val/test and export labels/statistics."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ALLOWED_FORMATS = {"csv", "yolo", "both"}


@dataclass
class SplitConfig:
    images_dir: Path
    annotations: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    copy_images: bool = False
    export_format: str = "both"


def ensure_dir(path: Path) -> None:
    """Create directory recursively if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def _prepare_df(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    required = [
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df = df[df["filename"] != ""]
    df = df[df["filename"].map(lambda fn: (images_dir / fn).exists())]
    return df.reset_index(drop=True)


def _split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def _write_yolo_labels(split_df: pd.DataFrame, labels_dir: Path, class_map: dict[str, int]) -> None:
    for _, row in split_df.iterrows():
        w = float(row["width"])
        h = float(row["height"])
        x_min, y_min = float(row["x_min"]), float(row["y_min"])
        x_max, y_max = float(row["x_max"]), float(row["y_max"])
        x_center = float(row["x_center"])
        y_center = float(row["y_center"])
        box_w = x_max - x_min
        box_h = y_max - y_min
        cls = class_map[row["class_name"]]
        content = f"{cls} {x_center / w:.6f} {y_center / h:.6f} {box_w / w:.6f} {box_h / h:.6f}\n"
        (labels_dir / f"{Path(str(row['filename'])).stem}.txt").write_text(content, encoding="utf-8")


def _write_angle_labels(split_df: pd.DataFrame, labels_dir: Path) -> None:
    angle_df = split_df[["filename", "angle_deg", "x_center", "y_center"]].copy()
    angle_df.to_csv(labels_dir / "angle_labels.csv", index=False)


def _write_split(split_name: str, split_df: pd.DataFrame, cfg: SplitConfig, class_map: dict[str, int]) -> None:
    split_root = cfg.output_dir / split_name
    images_out = split_root / "images"
    labels_out = split_root / "labels"
    ensure_dir(images_out)
    ensure_dir(labels_out)

    if cfg.copy_images:
        for filename in split_df["filename"].tolist():
            src = cfg.images_dir / filename
            dst = images_out / filename
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)

    if cfg.export_format in {"csv", "both"}:
        split_df.to_csv(labels_out / "annotations.csv", index=False)
    if cfg.export_format in {"yolo", "both"}:
        _write_yolo_labels(split_df, labels_out, class_map)
        _write_angle_labels(split_df, labels_out)


def _generate_dataset_stats(df: pd.DataFrame, split_sizes: dict[str, int], output_root: Path) -> None:
    metrics_dir = output_root / "outputs" / "metrics"
    reports_dir = output_root / "outputs" / "reports"
    ensure_dir(metrics_dir)
    ensure_dir(reports_dir)

    bbox_area = (df["x_max"] - df["x_min"]) * (df["y_max"] - df["y_min"])
    class_dist = df["class_name"].value_counts().to_dict()
    class_dist_str = "; ".join([f"{k}:{v}" for k, v in class_dist.items()])

    stats_row = {
        "total_images": int(df["filename"].nunique()),
        "annotated_images": int(len(df)),
        "missing_annotations": int(max(0, df["filename"].nunique() - len(df))),
        "num_classes": int(df["class_name"].nunique()),
        "class_distribution": class_dist_str,
        "width_min": float(df["width"].min()),
        "width_max": float(df["width"].max()),
        "height_min": float(df["height"].min()),
        "height_max": float(df["height"].max()),
        "bbox_area_mean": float(bbox_area.mean()),
        "bbox_area_median": float(bbox_area.median()),
        "bbox_area_min": float(bbox_area.min()),
        "bbox_area_max": float(bbox_area.max()),
        "angle_mean": float(df["angle_deg"].mean()),
        "angle_std": float(df["angle_deg"].std(ddof=0)),
        "angle_min": float(df["angle_deg"].min()),
        "angle_max": float(df["angle_deg"].max()),
        "train_count": split_sizes["train"],
        "val_count": split_sizes["val"],
        "test_count": split_sizes["test"],
    }
    stats_df = pd.DataFrame([stats_row])
    stats_csv = metrics_dir / "dataset_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    md = [
        "# Dataset Statistics",
        "",
        "## Global Summary",
        "",
        stats_df.to_markdown(index=False),
        "",
        "## Class Distribution",
        "",
        pd.DataFrame(list(class_dist.items()), columns=["class_name", "count"]).to_markdown(index=False),
        "",
        "## Split Counts",
        "",
        pd.DataFrame(
            [{"split": "train", "count": split_sizes["train"]}, {"split": "val", "count": split_sizes["val"]}, {"split": "test", "count": split_sizes["test"]}]
        ).to_markdown(index=False),
        "",
    ]
    (reports_dir / "dataset_stats.md").write_text("\n".join(md), encoding="utf-8")


def run_split(cfg: SplitConfig) -> dict[str, int]:
    """Run train/val/test split and export labels."""
    if cfg.export_format not in ALLOWED_FORMATS:
        raise ValueError(f"--format must be one of {sorted(ALLOWED_FORMATS)}")
    _validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    df = pd.read_csv(cfg.annotations)
    df = _prepare_df(df, cfg.images_dir)
    for col in ["width", "height", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center", "angle_deg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["width", "height", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center", "angle_deg", "class_name"])
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid annotation rows available after filtering.")

    class_names = sorted(df["class_name"].astype(str).unique().tolist())
    class_map = {name: idx for idx, name in enumerate(class_names)}
    split_idx = _split_indices(len(df), cfg.train_ratio, cfg.val_ratio, cfg.seed)

    split_sizes: dict[str, int] = {}
    for split_name, idx_list in split_idx.items():
        split_df = df.iloc[idx_list].copy().reset_index(drop=True)
        _write_split(split_name, split_df, cfg, class_map)
        split_sizes[split_name] = len(split_df)

    _generate_dataset_stats(df, split_sizes, cfg.output_dir.parent)
    return split_sizes


def split_dataset(images_dir: Path, labels_dir: Path, out_root: Path, train_ratio: float = 0.7, val_ratio: float = 0.2, seed: int = 42) -> None:
    """Legacy wrapper retained for compatibility with existing CLI.

    Expects canonical annotations CSV at data/annotations/annotations.csv.
    """
    _ = labels_dir
    cfg = SplitConfig(
        images_dir=images_dir,
        annotations=out_root / "annotations" / "annotations.csv",
        output_dir=out_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=max(0.0, 1.0 - train_ratio - val_ratio),
        seed=seed,
        copy_images=True,
        export_format="both",
    )
    run_split(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split BrushPose dataset into train/val/test.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--format", dest="export_format", choices=sorted(ALLOWED_FORMATS), default="both")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        split_sizes = run_split(
            SplitConfig(
                images_dir=args.images_dir,
                annotations=args.annotations,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                copy_images=args.copy_images,
                export_format=args.export_format,
            )
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Dataset split completed.")
    print(f"- Train samples: {split_sizes['train']}")
    print(f"- Val samples: {split_sizes['val']}")
    print(f"- Test samples: {split_sizes['test']}")
    print(f"- Output dir: {args.output_dir}")
    print(f"- Stats: {args.output_dir.parent / 'outputs' / 'metrics' / 'dataset_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
