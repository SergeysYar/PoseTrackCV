from __future__ import annotations

from pathlib import Path

import pandas as pd


def compare_methods(benchmark_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(benchmark_csv)
    summary = (
        df.groupby("method")
        .agg(mean_iou=("iou", "mean"), mean_center_error=("center_error", "mean"), mean_angle_error=("angle_error", "mean"), mean_inference_ms=("inference_time_ms", "mean"))
        .reset_index()
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary

