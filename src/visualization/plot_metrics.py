"""Metric plotting utilities and CLI for BrushPose AI (matplotlib only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _save_histogram(
    values: pd.Series,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    dpi: int = 150,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with `pip install matplotlib`.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values.dropna().astype(float), bins=25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    dpi: int = 150,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install with `pip install matplotlib`.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df[x_col].astype(str), df[y_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _require_columns(df: pd.DataFrame, cols: list[str], source: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in {source}: {missing}. Skipping this plot.")
        return False
    return True


def plot_angle_error_distribution(metrics_df: pd.DataFrame, out_dir: Path, dpi: int, ext: str) -> None:
    if not _require_columns(metrics_df, ["angle_error_deg"], "metrics CSV"):
        return
    out = out_dir / f"angle_error_distribution.{ext}"
    _save_histogram(metrics_df["angle_error_deg"], out, "Angle Error Distribution", "Angle Error (deg)", dpi=dpi)


def plot_center_error_distribution(metrics_df: pd.DataFrame, out_dir: Path, dpi: int, ext: str) -> None:
    if not _require_columns(metrics_df, ["center_error_px"], "metrics CSV"):
        return
    out = out_dir / f"center_error_distribution.{ext}"
    _save_histogram(metrics_df["center_error_px"], out, "Center Error Distribution", "Center Error (px)", dpi=dpi)


def plot_iou_distribution(metrics_df: pd.DataFrame, out_dir: Path, dpi: int, ext: str) -> None:
    if not _require_columns(metrics_df, ["iou"], "metrics CSV"):
        return
    out = out_dir / f"iou_distribution.{ext}"
    _save_histogram(metrics_df["iou"], out, "IoU Distribution", "IoU", dpi=dpi)


def plot_method_quality_comparison(benchmark_df: pd.DataFrame, out_dir: Path, dpi: int, ext: str) -> None:
    if not _require_columns(benchmark_df, ["method_name", "detection_accuracy"], "benchmark results CSV"):
        return
    data = benchmark_df.dropna(subset=["method_name", "detection_accuracy"]).copy()
    if data.empty:
        print("[WARN] No valid rows for method quality comparison.")
        return
    out = out_dir / f"method_quality_comparison.{ext}"
    _save_bar(data, "method_name", "detection_accuracy", out, "Method Quality Comparison (Detection Accuracy)", "Detection Accuracy", dpi=dpi)


def plot_fps_comparison(benchmark_df: pd.DataFrame, out_dir: Path, dpi: int, ext: str) -> None:
    if not _require_columns(benchmark_df, ["method_name", "fps"], "benchmark results CSV"):
        return
    data = benchmark_df.dropna(subset=["method_name", "fps"]).copy()
    if data.empty:
        print("[WARN] No valid rows for FPS comparison.")
        return
    out = out_dir / f"fps_comparison.{ext}"
    _save_bar(data, "method_name", "fps", out, "FPS Comparison", "FPS", dpi=dpi)


def plot_processing_time_comparison(benchmark_df: pd.DataFrame, out_dir: Path, dpi: int, ext: str) -> None:
    if not _require_columns(benchmark_df, ["method_name", "mean_processing_time_ms"], "benchmark results CSV"):
        return
    data = benchmark_df.dropna(subset=["method_name", "mean_processing_time_ms"]).copy()
    if data.empty:
        print("[WARN] No valid rows for processing-time comparison.")
        return
    out = out_dir / f"processing_time_comparison.{ext}"
    _save_bar(data, "method_name", "mean_processing_time_ms", out, "Processing Time Comparison", "Mean Processing Time (ms)", dpi=dpi)


# Backward-compatible wrappers used by existing project code
def plot_method_comparison(summary_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(summary_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if "method_name" not in df.columns and "method" in df.columns:
        df = df.rename(columns={"method": "method_name"})
    if "detection_accuracy" not in df.columns and "mean_angle_error" in df.columns:
        # Fallback for legacy files where only angle error existed: invert interpretation is not needed,
        # so just plot angle error as generic quality indicator.
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df["method_name"], df["mean_angle_error"].astype(float))
        ax.set_title("Method Comparison (Mean Angle Error)")
        ax.set_ylabel("Mean Angle Error (deg)")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["method_name"].astype(str), df["detection_accuracy"].astype(float))
    ax.set_title("Method Quality Comparison (Detection Accuracy)")
    ax.set_xlabel("method_name")
    ax.set_ylabel("Detection Accuracy")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_angle_histogram(results_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(results_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    col = "angle_error_deg" if "angle_error_deg" in df.columns else ("angle_error" if "angle_error" in df.columns else None)
    if col is None:
        print(f"[WARN] No angle error column in {results_csv}; histogram skipped.")
        return
    _save_histogram(df[col], out_png, "Angle Error Distribution", "Angle Error (deg)", dpi=150)


def _load_csv(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] {label} file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read {label} CSV {path}: {exc}")
        return None
    if df.empty:
        print(f"[WARN] {label} CSV is empty: {path}")
        return None
    return df


def run_plotting(metrics_path: Path | None, benchmark_path: Path | None, output_dir: Path, dpi: int, ext: str, plots: str) -> int:
    if plt is None:
        print("[ERROR] matplotlib is not installed. Install with `pip install matplotlib`.")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = _load_csv(metrics_path, "metrics")
    benchmark_df = _load_csv(benchmark_path, "benchmark results")

    plot_map: dict[str, Callable[[], None]] = {
        "angle-error": lambda: plot_angle_error_distribution(metrics_df, output_dir, dpi, ext) if metrics_df is not None else print("[WARN] Metrics CSV required for angle-error."),
        "center-error": lambda: plot_center_error_distribution(metrics_df, output_dir, dpi, ext) if metrics_df is not None else print("[WARN] Metrics CSV required for center-error."),
        "iou": lambda: plot_iou_distribution(metrics_df, output_dir, dpi, ext) if metrics_df is not None else print("[WARN] Metrics CSV required for iou."),
        "method-comparison": lambda: plot_method_quality_comparison(benchmark_df, output_dir, dpi, ext) if benchmark_df is not None else print("[WARN] Benchmark CSV required for method-comparison."),
        "fps": lambda: plot_fps_comparison(benchmark_df, output_dir, dpi, ext) if benchmark_df is not None else print("[WARN] Benchmark CSV required for fps."),
        "processing-time": lambda: plot_processing_time_comparison(benchmark_df, output_dir, dpi, ext) if benchmark_df is not None else print("[WARN] Benchmark CSV required for processing-time."),
    }

    if plots == "all":
        for fn in plot_map.values():
            fn()
    else:
        plot_map[plots]()

    print("Plot generation finished.")
    print(f"- Output dir: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BrushPose metric plots.")
    parser.add_argument("--metrics", type=Path, default=None, help="Path to per-sample metrics CSV.")
    parser.add_argument("--benchmark-results", type=Path, default=None, help="Path to benchmark_results.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", choices=["png"], default="png")
    parser.add_argument(
        "--plots",
        choices=["angle-error", "center-error", "iou", "method-comparison", "fps", "processing-time", "all"],
        default="all",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_plotting(args.metrics, args.benchmark_results, args.output_dir, args.dpi, args.format, args.plots)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
