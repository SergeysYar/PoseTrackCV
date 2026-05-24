"""YOLO training and optional validation entry point for BrushPose AI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - environment dependent
    yaml = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - environment dependent
    YOLO = None


@dataclass
class YOLOTrainConfig:
    """Configuration for Ultralytics training."""

    data: str
    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    device: str = "auto"
    project: str = "runs/brushpose_yolo"
    name: str = "train"
    seed: int = 42
    patience: int = 20
    workers: int = 4
    validate: bool = False


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Install with `pip install pyyaml` to use --config.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a mapping object.")
    return data


def _merge_config(cli: argparse.Namespace) -> YOLOTrainConfig:
    defaults: dict[str, Any] = {}
    if cli.config is not None:
        if not cli.config.exists():
            raise FileNotFoundError(f"Config file not found: {cli.config}")
        defaults = _load_yaml(cli.config)

    merged = {
        "data": defaults.get("data"),
        "model": defaults.get("model", "yolov8n.pt"),
        "epochs": int(defaults.get("epochs", 50)),
        "imgsz": int(defaults.get("imgsz", 640)),
        "batch": int(defaults.get("batch", 8)),
        "device": str(defaults.get("device", "auto")),
        "project": str(defaults.get("project", "runs/brushpose_yolo")),
        "name": str(defaults.get("name", "train")),
        "seed": int(defaults.get("seed", 42)),
        "patience": int(defaults.get("patience", 20)),
        "workers": int(defaults.get("workers", 4)),
        "validate": bool(defaults.get("validate", False)),
    }

    # CLI overrides config values when explicitly provided.
    for key in ["data", "model", "epochs", "imgsz", "batch", "device", "project", "name", "seed", "patience", "workers", "validate"]:
        value = getattr(cli, key)
        if value is not None:
            merged[key] = value

    if not merged["data"]:
        raise ValueError("Dataset yaml path must be provided via --data or config file.")
    return YOLOTrainConfig(**merged)


def _write_training_summary(
    cfg: YOLOTrainConfig,
    output_dir: Path,
    best_weights: Path,
    last_weights: Path,
    val_metrics: dict[str, Any] | None,
) -> Path:
    summary_path = output_dir / "training_summary.md"
    lines = [
        "# YOLO Training Summary",
        "",
        f"- Model: `{cfg.model}`",
        f"- Dataset: `{cfg.data}`",
        f"- Epochs: {cfg.epochs}",
        f"- Image size: {cfg.imgsz}",
        f"- Batch: {cfg.batch}",
        f"- Device: `{cfg.device}`",
        f"- Output directory: `{output_dir}`",
        f"- Best weights: `{best_weights}`",
        f"- Last weights: `{last_weights}`",
    ]
    if val_metrics:
        lines.append("")
        lines.append("## Validation Metrics")
        for k, v in val_metrics.items():
            lines.append(f"- {k}: {v}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def train_and_validate(cfg: YOLOTrainConfig) -> tuple[Path, Path, Path]:
    """Run YOLO training and optional validation."""
    if YOLO is None:
        raise RuntimeError("Ultralytics package is not installed. Install with `pip install ultralytics`.")

    data_path = Path(cfg.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(cfg.model)
    train_results = model.train(
        data=str(data_path),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        seed=cfg.seed,
        patience=cfg.patience,
        workers=cfg.workers,
    )

    output_dir = Path(train_results.save_dir)
    best_weights = output_dir / "weights" / "best.pt"
    last_weights = output_dir / "weights" / "last.pt"

    val_metrics: dict[str, Any] | None = None
    if cfg.validate:
        val_result = model.val(data=str(data_path), imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device)
        # Extract common metrics if present.
        val_metrics = {}
        for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if hasattr(val_result, "results_dict") and key in val_result.results_dict:
                val_metrics[key] = float(val_result.results_dict[key])
        if hasattr(val_result, "speed"):
            for k, v in val_result.speed.items():
                val_metrics[f"speed/{k}_ms"] = float(v)

    summary_path = _write_training_summary(cfg, output_dir, best_weights, last_weights, val_metrics)
    return best_weights, last_weights, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO detector for BrushPose AI.")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset.yaml.")
    parser.add_argument("--model", type=str, default=None, help="YOLO model weights, e.g., yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--validate", action="store_true", default=None)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cfg = _merge_config(args)
        print("Training configuration:")
        for k, v in asdict(cfg).items():
            print(f"- {k}: {v}")
        best_weights, last_weights, summary_path = train_and_validate(cfg)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("YOLO training finished.")
    print(f"- Best weights: {best_weights}")
    print(f"- Last weights: {last_weights}")
    print(f"- Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
