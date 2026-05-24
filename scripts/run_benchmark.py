"""Multi-method benchmarking orchestration for BrushPose AI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_METHODS = ["classical_min_area_rect", "classical_pca", "yolo_geometric"]


@dataclass
class BenchmarkConfig:
    images_dir: Path
    ground_truth: Path
    output_dir: Path
    yolo_weights: str | None
    methods: list[str]
    iou_threshold: float = 0.5
    angle_threshold: float = 5.0
    skip_yolo_if_missing: bool = False
    language: str = "both"


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install with `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a mapping object.")
    return data


def _merge_config(args: argparse.Namespace) -> BenchmarkConfig:
    defaults: dict[str, Any] = {}
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config not found: {args.config}")
        defaults = _load_yaml(args.config)

    methods = args.methods if args.methods else defaults.get("methods", DEFAULT_METHODS)
    cfg = BenchmarkConfig(
        images_dir=Path(args.images_dir or defaults.get("images_dir")),
        ground_truth=Path(args.ground_truth or defaults.get("ground_truth")),
        output_dir=Path(args.output_dir or defaults.get("output_dir", "outputs")),
        yolo_weights=args.yolo_weights if args.yolo_weights is not None else defaults.get("yolo_weights"),
        methods=list(methods),
        iou_threshold=float(args.iou_threshold if args.iou_threshold is not None else defaults.get("iou_threshold", 0.5)),
        angle_threshold=float(args.angle_threshold if args.angle_threshold is not None else defaults.get("angle_threshold", 5.0)),
        skip_yolo_if_missing=bool(args.skip_yolo_if_missing if args.skip_yolo_if_missing else defaults.get("skip_yolo_if_missing", False)),
        language=str(args.language or defaults.get("language", "both")),
    )
    return cfg


def _ensure_dirs(base_output: Path) -> dict[str, Path]:
    images_bench = base_output / "images" / "benchmark"
    metrics_bench = base_output / "metrics" / "benchmark"
    reports_dir = base_output / "reports"
    logs_dir = reports_dir / "benchmark_logs"
    for p in [images_bench, metrics_bench, reports_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {"images_bench": images_bench, "metrics_bench": metrics_bench, "reports_dir": reports_dir, "logs_dir": logs_dir}


def _collect_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS], key=lambda p: p.name.lower())


def _run_subprocess(cmd: list[str], log_path: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    log_path.write_text(f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n", encoding="utf-8")
    return proc.returncode, proc.stdout, proc.stderr


def _run_classical_method(method: str, cfg: BenchmarkConfig, paths: dict[str, Path]) -> tuple[Path, dict[str, Any]]:
    pred_csv = paths["metrics_bench"] / f"{method}_predictions.csv"
    image_out = paths["images_bench"] / method
    image_out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    try:
        from src.pose.classical_cv import ClassicalCVConfig, ClassicalCVPoseEstimator
    except Exception as exc:
        log_path = paths["logs_dir"] / f"{method}.log"
        angle_method = "min_area_rect" if method == "classical_min_area_rect" else "pca"
        code, _, err = _run_subprocess(
            [
                sys.executable,
                "scripts/run_classical_cv.py",
                "--input",
                str(cfg.images_dir),
                "--output",
                str(image_out),
                "--csv-out",
                str(pred_csv),
                "--angle-method",
                angle_method,
            ],
            log_path,
        )
        if code != 0:
            raise RuntimeError(f"Subprocess fallback failed: {err.strip()}")
        notes.append(f"Used subprocess fallback due to import error: {exc}")
        return pred_csv, {"status": "ok", "notes": "; ".join(notes)}

    estimator = ClassicalCVPoseEstimator(
        ClassicalCVConfig(
            use_pca_angle=method == "classical_pca",
            use_min_area_rect_angle=method == "classical_min_area_rect",
        )
    )
    rows: list[dict[str, Any]] = []
    for img_path in _collect_images(cfg.images_dir):
        pred, vis = estimator.predict_image(img_path)
        rows.append(
            {
                "filename": pred.filename,
                "status": "success" if pred.success else "failed",
                "x_min": pred.x_min,
                "y_min": pred.y_min,
                "x_max": pred.x_max,
                "y_max": pred.y_max,
                "x_center": pred.x_center,
                "y_center": pred.y_center,
                "angle_deg": pred.angle_deg,
                "confidence": pred.confidence,
                "processing_time_ms": pred.processing_time_ms,
                "message": pred.message,
            }
        )
        if vis is None:
            vis = cv2.imread(str(img_path))
        if vis is not None:
            cv2.imwrite(str(image_out / f"{img_path.stem}_{method}.jpg"), vis)
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    return pred_csv, {"status": "ok", "notes": "; ".join(notes)}


def _run_yolo_method(cfg: BenchmarkConfig, paths: dict[str, Path]) -> tuple[Path | None, dict[str, Any]]:
    method = "yolo_geometric"
    pred_csv = paths["metrics_bench"] / f"{method}_predictions.csv"
    image_out = paths["images_bench"] / method
    image_out.mkdir(parents=True, exist_ok=True)

    if not cfg.yolo_weights:
        if cfg.skip_yolo_if_missing:
            return None, {"status": "skipped", "notes": "YOLO weights are not provided; method skipped."}
        raise ValueError("YOLO weights are required for yolo_geometric. Use --yolo-weights or --skip-yolo-if-missing.")

    weights_path = Path(cfg.yolo_weights)
    if not weights_path.exists():
        if cfg.skip_yolo_if_missing:
            return None, {"status": "skipped", "notes": f"YOLO weights not found: {weights_path}"}
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    # Prefer direct API, fallback to subprocess.
    try:
        from src.detection.infer_yolo import YOLOInferConfig, run_on_images

        run_on_images(
            YOLOInferConfig(
                weights=str(weights_path),
                input=str(cfg.images_dir),
                output_dir=str(image_out),
                csv_out=str(pred_csv),
            )
        )
        return pred_csv, {"status": "ok", "notes": "Angle metrics may be unavailable for standard YOLO detection output."}
    except Exception as exc:
        log_path = paths["logs_dir"] / "yolo_geometric.log"
        code, _, err = _run_subprocess(
            [
                sys.executable,
                "src/detection/infer_yolo.py",
                "--weights",
                str(weights_path),
                "--input",
                str(cfg.images_dir),
                "--output-dir",
                str(image_out),
                "--csv-out",
                str(pred_csv),
            ],
            log_path,
        )
        if code != 0:
            if cfg.skip_yolo_if_missing:
                return None, {"status": "failed", "notes": f"YOLO execution failed: {err.strip()}"}
            raise RuntimeError(f"YOLO execution failed: {err.strip()}")
        return pred_csv, {"status": "ok", "notes": f"Used subprocess fallback due to import error: {exc}"}


def _evaluate_method(
    method: str,
    pred_csv: Path,
    cfg: BenchmarkConfig,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_dir = paths["metrics_bench"] / f"{method}_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = paths["metrics_bench"] / f"{method}_metrics.csv"
    summary_json = paths["metrics_bench"] / f"{method}_summary.json"

    try:
        from src.evaluation.evaluate_predictions import EvalConfig, evaluate

        metrics_df, summary = evaluate(
            EvalConfig(
                ground_truth=cfg.ground_truth,
                predictions=pred_csv,
                output_dir=eval_dir,
                method_name=method,
                iou_threshold=cfg.iou_threshold,
                angle_threshold=cfg.angle_threshold,
                report_format="json",
            )
        )
        metrics_df.to_csv(metrics_csv, index=False)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics_df, summary
    except Exception as exc:
        log_path = paths["logs_dir"] / f"{method}_evaluation.log"
        code, _, err = _run_subprocess(
            [
                sys.executable,
                "src/evaluation/evaluate_predictions.py",
                "--ground-truth",
                str(cfg.ground_truth),
                "--predictions",
                str(pred_csv),
                "--output-dir",
                str(eval_dir),
                "--method-name",
                method,
                "--iou-threshold",
                str(cfg.iou_threshold),
                "--angle-threshold",
                str(cfg.angle_threshold),
                "--report-format",
                "json",
            ],
            log_path,
        )
        if code != 0:
            raise RuntimeError(f"Evaluation failed for {method}: {err.strip()} | import_error={exc}")
        md = pd.read_csv(eval_dir / "metrics.csv")
        summary = json.loads((eval_dir / "summary_metrics.json").read_text(encoding="utf-8"))
        md.to_csv(metrics_csv, index=False)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return md, summary


def _pick_best(rows: list[dict[str, Any]], key: str, prefer_max: bool = True) -> str | None:
    candidates = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        candidates.append((row["method_name"], float(value)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0] if prefer_max else min(candidates, key=lambda x: x[1])[0]


def _build_report_lines(results_df: pd.DataFrame, summary_json: dict[str, Any], lang: str) -> list[str]:
    common_setup = (
        f"- Dataset path: `{summary_json['dataset_path']}`\n"
        f"- Ground truth path: `{summary_json['ground_truth_path']}`\n"
        f"- IoU threshold: `{summary_json['thresholds']['iou_threshold']}`\n"
        f"- Angle threshold: `{summary_json['thresholds']['angle_threshold']}`\n"
    )
    methods_list = "\n".join([f"- `{m}`" for m in summary_json["methods"]])
    table = results_df.to_markdown(index=False)

    en = [
        "# BrushPose AI Benchmark Report",
        "",
        "## Dataset Description",
        common_setup,
        "## Compared Methods",
        methods_list,
        "",
        "## Experimental Setup",
        "- All methods are evaluated on the same image set and ground-truth CSV.",
        "- Evaluation is executed with the shared prediction-vs-ground-truth evaluator.",
        "",
        "## Metric Notes",
        "- IoU/center/angle details are defined in `docs/en/evaluation.md`.",
        "- `map_50_proxy` is a simplified proxy, not COCO mAP.",
        "",
        "## Summary Comparison Table",
        table,
        "",
        "## Detection Quality Comparison",
        f"- Best by detection accuracy: `{summary_json.get('best_method_by_detection_accuracy')}`",
        "",
        "## Localization Quality Comparison",
        f"- Best by mean IoU: `{summary_json.get('best_method_by_mean_iou')}`",
        "",
        "## Orientation Quality Comparison",
        f"- Best by mean angle error: `{summary_json.get('best_method_by_angle_error')}`",
        "- Note: standard YOLO detection may not provide angle predictions.",
        "",
        "## Runtime Comparison",
        f"- Fastest method: `{summary_json.get('fastest_method')}`",
        "",
        "## Best Method Analysis",
        "- Select method by your deployment objective: robustness, geometric precision, or throughput.",
        "",
        "## Failure Case Analysis",
        "- Review `error_type_counts` in per-method summaries for missed detections and geometric outliers.",
        "",
        "## Practical Recommendation",
        "- For controlled tabletop scenes, classical variants provide interpretable geometric behavior.",
        "- For broader visual variability, YOLO-based detection is typically more robust.",
        "",
        "## Limitations",
        "- This benchmark compares single-object predictions per filename.",
        "- Angle metrics for YOLO can be unavailable unless geometric post-processing is enabled.",
        "",
        "## Reproducibility Command",
        "```bash",
        "python scripts/run_benchmark.py --images-dir data/test/images --ground-truth data/test/labels/annotations.csv --output-dir outputs --yolo-weights runs/brushpose_yolo/train/weights/best.pt --methods classical_min_area_rect classical_pca yolo_geometric --iou-threshold 0.5 --angle-threshold 5 --language both --skip-yolo-if-missing",
        "```",
    ]

    ru = [
        "# Отчёт по бенчмаркингу BrushPose AI",
        "",
        "## Описание датасета",
        common_setup.replace("Dataset path", "Путь к датасету").replace("Ground truth path", "Путь к разметке").replace("IoU threshold", "Порог IoU").replace("Angle threshold", "Порог угловой ошибки"),
        "## Сравниваемые методы",
        methods_list,
        "",
        "## Экспериментальная постановка",
        "- Все методы оцениваются на едином наборе изображений и общей эталонной разметке.",
        "- Оценка выполняется единым модулем сравнения предсказаний с ground truth.",
        "",
        "## Примечания к метрикам",
        "- Детальные определения IoU/ошибок центра/угла приведены в `docs/ru/evaluation.md`.",
        "- `map_50_proxy` является упрощённой proxy-метрикой, а не COCO mAP.",
        "",
        "## Сводная таблица сравнения",
        table,
        "",
        "## Сравнение качества детекции",
        f"- Лучший метод по detection_accuracy: `{summary_json.get('best_method_by_detection_accuracy')}`",
        "",
        "## Сравнение точности локализации",
        f"- Лучший метод по mean_iou: `{summary_json.get('best_method_by_mean_iou')}`",
        "",
        "## Сравнение точности ориентации",
        f"- Лучший метод по mean_angle_error: `{summary_json.get('best_method_by_angle_error')}`",
        "- Стандартная YOLO-детекция может не содержать прямого предсказания угла.",
        "",
        "## Сравнение производительности",
        f"- Наиболее быстрый метод: `{summary_json.get('fastest_method')}`",
        "",
        "## Анализ лучших результатов",
        "- Выбор метода должен учитывать целевой сценарий: устойчивость, геометрическая точность или скорость.",
        "",
        "## Анализ отказов",
        "- Для анализа типовых ошибок используйте `error_type_counts` в summary каждого метода.",
        "",
        "## Практическая рекомендация",
        "- В контролируемых tabletop-сценах классические методы обеспечивают интерпретируемость.",
        "- В более вариативных условиях детекция на базе YOLO, как правило, устойчивее.",
        "",
        "## Ограничения",
        "- Бенчмарк ориентирован на сопоставление одного предсказания на файл.",
        "- Угловые метрики для YOLO могут отсутствовать без геометрического постпроцессинга.",
        "",
        "## Команда воспроизводимости",
        "```bash",
        "python scripts/run_benchmark.py --images-dir data/test/images --ground-truth data/test/labels/annotations.csv --output-dir outputs --yolo-weights runs/brushpose_yolo/train/weights/best.pt --methods classical_min_area_rect classical_pca yolo_geometric --iou-threshold 0.5 --angle-threshold 5 --language both --skip-yolo-if-missing",
        "```",
    ]

    return en if lang == "en" else ru


def run_benchmark(cfg: BenchmarkConfig) -> int:
    if not cfg.images_dir.exists():
        print(f"[ERROR] Images directory not found: {cfg.images_dir}")
        return 1
    if not cfg.ground_truth.exists():
        print(f"[ERROR] Ground-truth file not found: {cfg.ground_truth}")
        return 1

    paths = _ensure_dirs(cfg.output_dir)
    benchmark_rows: list[dict[str, Any]] = []
    per_method_summary: dict[str, Any] = {}

    for method in cfg.methods:
        print(f"[INFO] Running method: {method}")
        try:
            if method in {"classical_min_area_rect", "classical_pca"}:
                pred_csv, run_meta = _run_classical_method(method, cfg, paths)
            elif method == "yolo_geometric":
                pred_csv, run_meta = _run_yolo_method(cfg, paths)
                if pred_csv is None:
                    benchmark_rows.append(
                        {
                            "method_name": method,
                            "total_samples": None,
                            "successful_detections": None,
                            "detection_accuracy": None,
                            "mean_iou": None,
                            "median_iou": None,
                            "map_50_proxy": None,
                            "mean_center_error_px": None,
                            "median_center_error_px": None,
                            "mean_angle_error_deg": None,
                            "median_angle_error_deg": None,
                            "angle_accuracy_at_5deg": None,
                            "mean_processing_time_ms": None,
                            "fps": None,
                            "status": run_meta["status"],
                            "notes": run_meta["notes"],
                        }
                    )
                    continue
            else:
                benchmark_rows.append(
                    {
                        "method_name": method,
                        "status": "failed",
                        "notes": f"Unknown method: {method}",
                    }
                )
                continue

            if not pred_csv.exists():
                raise RuntimeError(f"Prediction CSV was not generated: {pred_csv}")
            pred_df = pd.read_csv(pred_csv)
            if pred_df.empty:
                raise RuntimeError("Prediction CSV is empty.")

            _, summary = _evaluate_method(method, pred_csv, cfg, paths)
            per_method_summary[method] = summary
            benchmark_rows.append(
                {
                    "method_name": method,
                    "total_samples": summary.get("total_samples"),
                    "successful_detections": summary.get("successful_detections"),
                    "detection_accuracy": summary.get("detection_accuracy"),
                    "mean_iou": summary.get("mean_iou"),
                    "median_iou": summary.get("median_iou"),
                    "map_50_proxy": summary.get("map_50_proxy"),
                    "mean_center_error_px": summary.get("mean_center_error_px"),
                    "median_center_error_px": summary.get("median_center_error_px"),
                    "mean_angle_error_deg": summary.get("mean_angle_error_deg"),
                    "median_angle_error_deg": summary.get("median_angle_error_deg"),
                    "angle_accuracy_at_5deg": summary.get("angle_accuracy_at_5deg"),
                    "mean_processing_time_ms": summary.get("mean_processing_time_ms"),
                    "fps": summary.get("fps"),
                    "status": "ok",
                    "notes": run_meta.get("notes", ""),
                }
            )
        except Exception as exc:
            benchmark_rows.append(
                {
                    "method_name": method,
                    "total_samples": None,
                    "successful_detections": None,
                    "detection_accuracy": None,
                    "mean_iou": None,
                    "median_iou": None,
                    "map_50_proxy": None,
                    "mean_center_error_px": None,
                    "median_center_error_px": None,
                    "mean_angle_error_deg": None,
                    "median_angle_error_deg": None,
                    "angle_accuracy_at_5deg": None,
                    "mean_processing_time_ms": None,
                    "fps": None,
                    "status": "failed",
                    "notes": str(exc),
                }
            )
            print(f"[WARN] Method failed: {method} | {exc}")

    results_df = pd.DataFrame(benchmark_rows)
    metrics_root = cfg.output_dir / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    results_root_csv = metrics_root / "benchmark_results.csv"
    results_bench_csv = paths["metrics_bench"] / "benchmark_results.csv"
    results_df.to_csv(results_root_csv, index=False)
    results_df.to_csv(results_bench_csv, index=False)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_path": str(cfg.images_dir),
        "ground_truth_path": str(cfg.ground_truth),
        "methods": cfg.methods,
        "thresholds": {"iou_threshold": cfg.iou_threshold, "angle_threshold": cfg.angle_threshold},
        "per_method_summary": per_method_summary,
        "best_method_by_detection_accuracy": _pick_best(benchmark_rows, "detection_accuracy", prefer_max=True),
        "best_method_by_mean_iou": _pick_best(benchmark_rows, "mean_iou", prefer_max=True),
        "best_method_by_angle_error": _pick_best(
            [r for r in benchmark_rows if r.get("mean_angle_error_deg") not in {None, "unavailable"}],
            "mean_angle_error_deg",
            prefer_max=False,
        ),
        "fastest_method": _pick_best(benchmark_rows, "fps", prefer_max=True),
        "notes": "Angle metrics for standard YOLO detection can be unavailable depending on prediction outputs.",
    }
    summary_json_path = paths["metrics_bench"] / "benchmark_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    reports_dir = cfg.output_dir / "reports"
    if cfg.language in {"en", "both"}:
        (reports_dir / "benchmark_en.md").write_text(
            "\n".join(_build_report_lines(results_df, summary, "en")) + "\n",
            encoding="utf-8",
        )
    if cfg.language in {"ru", "both"}:
        (reports_dir / "benchmark_ru.md").write_text(
            "\n".join(_build_report_lines(results_df, summary, "ru")) + "\n",
            encoding="utf-8",
        )

    print("Benchmark finished.")
    print(f"- Results CSV: {results_root_csv}")
    print(f"- Summary JSON: {summary_json_path}")
    print(f"- Reports dir: {reports_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-method benchmark for BrushPose AI.")
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--yolo-weights", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--angle-threshold", type=float, default=5.0)
    parser.add_argument("--skip-yolo-if-missing", action="store_true")
    parser.add_argument("--language", choices=["en", "ru", "both"], default="both")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = _merge_config(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return run_benchmark(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
