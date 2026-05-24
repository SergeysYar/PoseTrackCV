from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io_utils import save_json


def save_metric_reports(summary: dict[str, Any], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_dir / f"{prefix}_metrics.csv", index=False)
    save_json(out_dir / f"{prefix}_summary.json", summary)
    lines = [f"# {prefix} Evaluation Summary", ""]
    lines.extend([f"- {k}: {v}" for k, v in summary.items()])
    (out_dir / f"{prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_benchmark_reports(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "benchmark_results.csv", index=False)
    table = summary_df.to_markdown(index=False)
    (out_dir / "benchmark_report_en.md").write_text(f"# Benchmark Report\n\n{table}\n", encoding="utf-8")
    (out_dir / "benchmark_report_ru.md").write_text(f"# Отчет по бенчмарку\n\n{table}\n", encoding="utf-8")

