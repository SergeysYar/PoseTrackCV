from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_method_comparison(summary_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(summary_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["method"], df["mean_angle_error"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Mean Angle Error by Method")
    ax.set_ylabel("Angle Error (deg)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_angle_histogram(results_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(results_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["angle_error"], bins=25, color="#4c78a8")
    ax.set_title("Angle Error Distribution")
    ax.set_xlabel("Angle Error (deg)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

