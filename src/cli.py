"""Unified CLI orchestration layer for BrushPose AI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load YAML config file safely. Returns empty dict when not available."""
    if config_path is None:
        return {}
    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}. Using CLI/default values.")
        return {}
    if yaml is None:
        print("[WARN] PyYAML is not installed. Config file will be ignored.")
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        raise ValueError(f"Invalid config file '{config_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config structure in '{config_path}': top-level YAML must be an object.")
    return data


def _cfg_get(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    sec = config.get(section, {})
    if not isinstance(sec, dict):
        return default
    return sec.get(key, default)


def run_subprocess(command: list[str], cwd: Path | None = None) -> int:
    """Run subprocess command with streamed stdout/stderr and return exit code."""
    print(f"[RUN] {' '.join(command)}")
    proc = subprocess.Popen(command, cwd=str(cwd or PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip())
    proc.wait()
    return int(proc.returncode)


def _coalesce(cli_value: Any, cfg_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (cfg_value if cfg_value is not None else default)


def cmd_prepare_data(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    mode = args.mode
    data_images = _coalesce(args.images_dir, _cfg_get(cfg, "data", "images_dir", "data/images"))
    annotations = _coalesce(args.annotations, _cfg_get(cfg, "data", "annotations", "data/annotations/annotations.csv"))
    out_dir = _coalesce(args.output_dir, "data")
    seed = int(_coalesce(args.seed, _cfg_get(cfg, "data", "seed", 42)))

    if mode == "collect":
        input_dir = _coalesce(args.input_dir, _cfg_get(cfg, "data", "raw_dir", "data/raw"))
        cmd = [
            sys.executable,
            "src/data/collect_dataset_template.py",
            "--input-dir",
            str(Path(input_dir)),
            "--output-dir",
            str(Path(data_images)),
            "--annotations-out",
            str(Path(annotations)),
            "--copy",
        ]
        return run_subprocess(cmd)

    if mode == "validate":
        report_out = Path("outputs/reports/dataset_validation.md")
        cmd = [
            sys.executable,
            "src/data/validate_dataset.py",
            "--images-dir",
            str(Path(data_images)),
            "--annotations",
            str(Path(annotations)),
            "--report-out",
            str(report_out),
        ]
        return run_subprocess(cmd)

    if mode == "split":
        split_format = _coalesce(args.format, "both")
        cmd = [
            sys.executable,
            "src/data/split_dataset.py",
            "--images-dir",
            str(Path(data_images)),
            "--annotations",
            str(Path(annotations)),
            "--output-dir",
            str(Path(out_dir)),
            "--seed",
            str(seed),
            "--format",
            str(split_format),
        ]
        return run_subprocess(cmd)

    if mode == "export-yolo":
        cmd = [
            sys.executable,
            "src/detection/export_yolo_dataset.py",
            "--images-dir",
            str(Path(data_images)),
            "--annotations",
            str(Path(annotations)),
            "--output-dir",
            str(Path(out_dir)),
            "--seed",
            str(seed),
            "--copy-images",
        ]
        return run_subprocess(cmd)

    print(f"[ERROR] Unknown prepare-data mode: {mode}")
    return 1


def cmd_train_yolo(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    data = _coalesce(args.data, _cfg_get(cfg, "yolo", "data", "data/yolo_dataset/dataset.yaml"))
    model = _coalesce(args.model, _cfg_get(cfg, "yolo", "model", "yolov8n.pt"))
    epochs = int(_coalesce(args.epochs, _cfg_get(cfg, "yolo", "epochs", 50)))
    imgsz = int(_coalesce(args.imgsz, _cfg_get(cfg, "yolo", "imgsz", 640)))
    batch = int(_coalesce(args.batch, _cfg_get(cfg, "yolo", "batch", 8)))
    device = str(_coalesce(args.device, _cfg_get(cfg, "yolo", "device", "auto")))
    project = str(_coalesce(args.project, _cfg_get(cfg, "yolo", "project", "runs/brushpose_yolo")))
    name = str(_coalesce(args.name, _cfg_get(cfg, "yolo", "name", "train")))

    cmd = [
        sys.executable,
        "src/detection/train_yolo.py",
        "--data",
        str(data),
        "--model",
        str(model),
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--device",
        device,
        "--project",
        project,
        "--name",
        name,
    ]
    if args.validate:
        cmd.append("--validate")
    return run_subprocess(cmd)


def cmd_infer(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    method = str(args.method)
    if method == "yolo":
        weights = _coalesce(args.weights, _cfg_get(cfg, "yolo", "weights"))
        if not weights:
            print("[ERROR] --weights is required for YOLO inference (or set yolo.weights in config).")
            return 1
        inp = _coalesce(args.input, _cfg_get(cfg, "inference", "input"))
        if not inp:
            print("[ERROR] --input is required.")
            return 1
        out_dir = _coalesce(args.output_dir, _cfg_get(cfg, "inference", "output_dir", "outputs/images/yolo"))
        csv_out = _coalesce(args.csv_out, _cfg_get(cfg, "inference", "csv_out", "outputs/metrics/yolo_predictions.csv"))
        conf = _coalesce(args.conf, _cfg_get(cfg, "inference", "conf", 0.25))
        iou = _coalesce(args.iou, _cfg_get(cfg, "inference", "iou", 0.5))
        imgsz = _coalesce(args.imgsz, _cfg_get(cfg, "inference", "imgsz", 640))
        cmd = [
            sys.executable,
            "src/detection/infer_yolo.py",
            "--weights",
            str(weights),
            "--input",
            str(inp),
            "--output-dir",
            str(out_dir),
            "--csv-out",
            str(csv_out),
            "--conf",
            str(conf),
            "--iou",
            str(iou),
            "--imgsz",
            str(imgsz),
        ]
        return run_subprocess(cmd)

    if method == "classical":
        inp = _coalesce(args.input, _cfg_get(cfg, "data", "test_images", "data/test/images"))
        out_dir = _coalesce(args.output_dir, _cfg_get(cfg, "classical", "output_dir", "outputs/images/classical"))
        csv_out = _coalesce(args.csv_out, _cfg_get(cfg, "classical", "csv_out", "outputs/metrics/classical_predictions.csv"))
        angle_method = str(_coalesce(args.angle_method, _cfg_get(cfg, "classical", "angle_method", "pca")))
        cmd = [
            sys.executable,
            "scripts/run_classical_cv.py",
            "--input",
            str(inp),
            "--output",
            str(out_dir),
            "--csv-out",
            str(csv_out),
            "--angle-method",
            angle_method,
        ]
        return run_subprocess(cmd)

    print(f"[ERROR] Unsupported infer method: {method}")
    return 1


def cmd_run_classical(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    inp = _coalesce(args.input, _cfg_get(cfg, "data", "test_images", "data/test/images"))
    out_dir = _coalesce(args.output, _cfg_get(cfg, "classical", "output_dir", "outputs/images/classical_cv"))
    csv_out = _coalesce(args.csv_out, _cfg_get(cfg, "classical", "csv_out", "outputs/metrics/classical_cv_predictions.csv"))
    angle_method = str(_coalesce(args.angle_method, _cfg_get(cfg, "classical", "angle_method", "pca")))
    hsv_lower = _coalesce(args.hsv_lower, _cfg_get(cfg, "classical", "hsv_lower", [0, 40, 40]))
    hsv_upper = _coalesce(args.hsv_upper, _cfg_get(cfg, "classical", "hsv_upper", [180, 255, 255]))
    min_area = _coalesce(args.min_contour_area, _cfg_get(cfg, "classical", "min_contour_area", 500))
    cmd = [
        sys.executable,
        "scripts/run_classical_cv.py",
        "--input",
        str(inp),
        "--output",
        str(out_dir),
        "--csv-out",
        str(csv_out),
        "--angle-method",
        angle_method,
        "--hsv-lower",
        str(hsv_lower[0]),
        str(hsv_lower[1]),
        str(hsv_lower[2]),
        "--hsv-upper",
        str(hsv_upper[0]),
        str(hsv_upper[1]),
        str(hsv_upper[2]),
        "--min-contour-area",
        str(min_area),
    ]
    return run_subprocess(cmd)


def cmd_evaluate(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    gt = _coalesce(args.ground_truth, _cfg_get(cfg, "evaluation", "ground_truth"))
    pred = _coalesce(args.predictions, _cfg_get(cfg, "evaluation", "predictions"))
    out_dir = _coalesce(args.output_dir, _cfg_get(cfg, "evaluation", "output_dir", "outputs/reports/eval"))
    method_name = _coalesce(args.method_name, _cfg_get(cfg, "evaluation", "method_name", "method"))
    iou_thr = _coalesce(args.iou_threshold, _cfg_get(cfg, "evaluation", "iou_threshold", 0.5))
    angle_thr = _coalesce(args.angle_threshold, _cfg_get(cfg, "evaluation", "angle_threshold", 5.0))
    report_format = _coalesce(args.report_format, _cfg_get(cfg, "evaluation", "report_format", "both"))
    if not gt or not pred:
        print("[ERROR] --ground-truth and --predictions are required (or set in config).")
        return 1
    cmd = [
        sys.executable,
        "src/evaluation/evaluate_predictions.py",
        "--ground-truth",
        str(gt),
        "--predictions",
        str(pred),
        "--output-dir",
        str(out_dir),
        "--method-name",
        str(method_name),
        "--iou-threshold",
        str(iou_thr),
        "--angle-threshold",
        str(angle_thr),
        "--report-format",
        str(report_format),
    ]
    return run_subprocess(cmd)


def cmd_benchmark(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    images_dir = _coalesce(args.images_dir, _cfg_get(cfg, "benchmark", "images_dir"))
    gt = _coalesce(args.ground_truth, _cfg_get(cfg, "benchmark", "ground_truth"))
    out_dir = _coalesce(args.output_dir, _cfg_get(cfg, "benchmark", "output_dir", "outputs"))
    methods = args.methods if args.methods else _cfg_get(cfg, "benchmark", "methods", None)
    language = _coalesce(args.language, _cfg_get(cfg, "benchmark", "language", "both"))
    yolo_weights = _coalesce(args.yolo_weights, _cfg_get(cfg, "benchmark", "yolo_weights"))
    if not images_dir or not gt:
        print("[ERROR] --images-dir and --ground-truth are required (or set in config).")
        return 1

    cmd = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--images-dir",
        str(images_dir),
        "--ground-truth",
        str(gt),
        "--output-dir",
        str(out_dir),
        "--language",
        str(language),
    ]
    if methods:
        cmd.extend(["--methods", *[str(m) for m in methods]])
    if yolo_weights:
        cmd.extend(["--yolo-weights", str(yolo_weights)])
    if args.skip_yolo_if_missing:
        cmd.append("--skip-yolo-if-missing")
    if args.iou_threshold is not None:
        cmd.extend(["--iou-threshold", str(args.iou_threshold)])
    if args.angle_threshold is not None:
        cmd.extend(["--angle-threshold", str(args.angle_threshold)])
    return run_subprocess(cmd)


def cmd_generate_report(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    benchmark_results = Path(_coalesce(args.benchmark_results, _cfg_get(cfg, "report", "benchmark_results", "outputs/metrics/benchmark_results.csv")))
    metrics_dir = Path(_coalesce(args.metrics_dir, _cfg_get(cfg, "report", "metrics_dir", "outputs/metrics/benchmark")))
    output = Path(_coalesce(args.output, _cfg_get(cfg, "report", "output", "outputs/reports/final_report_en.md")))
    language = str(_coalesce(args.language, _cfg_get(cfg, "report", "language", "en")))
    title = str(_coalesce(args.title, _cfg_get(cfg, "report", "title", "BrushPose AI Final Report")))

    if not benchmark_results.exists():
        print(f"[ERROR] Benchmark results not found: {benchmark_results}")
        return 1
    if not metrics_dir.exists():
        print(f"[ERROR] Metrics dir not found: {metrics_dir}")
        return 1

    bench_df = pd.read_csv(benchmark_results)
    method_summaries: list[dict[str, Any]] = []
    for summary_path in sorted(metrics_dir.glob("*_summary.json")):
        try:
            method_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except Exception:
            continue

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- generated_at: {datetime.utcnow().isoformat()}Z",
        f"- language: {language}",
        f"- benchmark_results: `{benchmark_results}`",
        f"- metrics_dir: `{metrics_dir}`",
        "",
        "## Benchmark Comparison",
        bench_df.to_markdown(index=False),
        "",
        "## Per-Method Summaries",
    ]
    if method_summaries:
        for sm in method_summaries:
            method = sm.get("method_name", "unknown")
            lines.append(f"### {method}")
            for k in ["detection_accuracy", "mean_iou", "mean_center_error_px", "mean_angle_error_deg", "mean_processing_time_ms", "fps"]:
                if k in sm:
                    lines.append(f"- {k}: {sm[k]}")
            lines.append("")
    else:
        lines.append("- No per-method summary JSON files found.")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Final report generated: {output}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BrushPose AI unified CLI")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Optional YAML config path.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("prepare-data", help="Run dataset preparation utilities.")
    p_data.add_argument("--mode", choices=["collect", "validate", "split", "export-yolo"], required=True)
    p_data.add_argument("--input-dir", type=Path, default=None)
    p_data.add_argument("--images-dir", type=Path, default=None)
    p_data.add_argument("--annotations", type=Path, default=None)
    p_data.add_argument("--output-dir", type=Path, default=None)
    p_data.add_argument("--seed", type=int, default=None)
    p_data.add_argument("--format", choices=["csv", "yolo", "both"], default=None)

    p_train = sub.add_parser("train-yolo", help="Train YOLO detector.")
    p_train.add_argument("--data", type=Path, default=None)
    p_train.add_argument("--model", type=str, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--imgsz", type=int, default=None)
    p_train.add_argument("--batch", type=int, default=None)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--project", type=Path, default=None)
    p_train.add_argument("--name", type=str, default=None)
    p_train.add_argument("--validate", action="store_true")

    p_infer = sub.add_parser("infer", help="Run inference with yolo or classical method.")
    p_infer.add_argument("--method", choices=["yolo", "classical"], required=True)
    p_infer.add_argument("--weights", type=Path, default=None)
    p_infer.add_argument("--input", type=Path, default=None)
    p_infer.add_argument("--output-dir", type=Path, default=None)
    p_infer.add_argument("--csv-out", type=Path, default=None)
    p_infer.add_argument("--conf", type=float, default=None)
    p_infer.add_argument("--iou", type=float, default=None)
    p_infer.add_argument("--imgsz", type=int, default=None)
    p_infer.add_argument("--angle-method", choices=["pca", "min_area_rect"], default=None)

    p_class = sub.add_parser("run-classical", help="Shortcut command for classical CV pipeline.")
    p_class.add_argument("--input", type=Path, default=None, required=False)
    p_class.add_argument("--output", type=Path, default=None)
    p_class.add_argument("--csv-out", type=Path, default=None)
    p_class.add_argument("--angle-method", choices=["pca", "min_area_rect"], default=None)
    p_class.add_argument("--hsv-lower", nargs=3, type=int, default=None)
    p_class.add_argument("--hsv-upper", nargs=3, type=int, default=None)
    p_class.add_argument("--min-contour-area", type=float, default=None)

    p_eval = sub.add_parser("evaluate", help="Evaluate predictions against ground truth.")
    p_eval.add_argument("--ground-truth", type=Path, default=None)
    p_eval.add_argument("--predictions", type=Path, default=None)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--method-name", type=str, default=None)
    p_eval.add_argument("--iou-threshold", type=float, default=None)
    p_eval.add_argument("--angle-threshold", type=float, default=None)
    p_eval.add_argument("--report-format", choices=["markdown", "json", "both"], default=None)

    p_bench = sub.add_parser("benchmark", help="Run multi-method benchmark.")
    p_bench.add_argument("--images-dir", type=Path, default=None)
    p_bench.add_argument("--ground-truth", type=Path, default=None)
    p_bench.add_argument("--output-dir", type=Path, default=None)
    p_bench.add_argument("--yolo-weights", type=Path, default=None)
    p_bench.add_argument("--methods", nargs="+", default=None)
    p_bench.add_argument("--language", choices=["en", "ru", "both"], default=None)
    p_bench.add_argument("--skip-yolo-if-missing", action="store_true")
    p_bench.add_argument("--iou-threshold", type=float, default=None)
    p_bench.add_argument("--angle-threshold", type=float, default=None)

    p_report = sub.add_parser("generate-report", help="Generate final markdown report from existing metrics.")
    p_report.add_argument("--benchmark-results", type=Path, default=None)
    p_report.add_argument("--metrics-dir", type=Path, default=None)
    p_report.add_argument("--output", type=Path, default=None)
    p_report.add_argument("--language", choices=["en", "ru"], default=None)
    p_report.add_argument("--title", type=str, default=None)

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        if args.command == "prepare-data":
            return cmd_prepare_data(args, cfg)
        if args.command == "train-yolo":
            return cmd_train_yolo(args, cfg)
        if args.command == "infer":
            return cmd_infer(args, cfg)
        if args.command == "run-classical":
            return cmd_run_classical(args, cfg)
        if args.command == "evaluate":
            return cmd_evaluate(args, cfg)
        if args.command == "benchmark":
            return cmd_benchmark(args, cfg)
        if args.command == "generate-report":
            return cmd_generate_report(args, cfg)
        print(f"[ERROR] Unknown command: {args.command}")
        return 1
    except KeyboardInterrupt:
        print("[ERROR] Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
