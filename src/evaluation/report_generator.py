"""Markdown report generation for BrushPose AI evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _title(lang: str) -> str:
    return "Evaluation Report" if lang == "en" else "Отчёт по оценке"


def _labels(lang: str) -> dict[str, str]:
    if lang == "ru":
        return {
            "method": "Метод",
            "setup": "Параметры оценки",
            "dataset": "Размер датасета",
            "summary": "Сводные метрики",
            "det_quality": "Качество детекции",
            "loc_quality": "Качество локализации",
            "ori_quality": "Качество ориентации",
            "runtime": "Производительность",
            "errors": "Анализ ошибок",
            "worst_iou": "Худшие примеры по IoU",
            "worst_angle": "Худшие примеры по угловой ошибке",
            "interpretation": "Интерпретация",
            "limitations": "Ограничения текущей оценки",
        }
    return {
        "method": "Method",
        "setup": "Evaluation Setup",
        "dataset": "Dataset Size",
        "summary": "Summary Metrics",
        "det_quality": "Detection Quality",
        "loc_quality": "Localization Quality",
        "ori_quality": "Orientation Quality",
        "runtime": "Runtime Performance",
        "errors": "Error Analysis",
        "worst_iou": "Worst Samples by IoU",
        "worst_angle": "Worst Samples by Angle Error",
        "interpretation": "Interpretation",
        "limitations": "Limitations of Current Evaluation",
    }


def generate_markdown_report(
    metrics_df: pd.DataFrame,
    summary: dict[str, Any],
    output_path: Path,
    language: str = "en",
) -> None:
    """Generate markdown evaluation report in English or Russian."""
    if language not in {"en", "ru"}:
        raise ValueError("language must be 'en' or 'ru'")
    labels = _labels(language)

    method_name = summary.get("method_name", "unknown")
    total_samples = int(summary.get("total_samples", len(metrics_df)))
    iou_threshold = summary.get("iou_threshold", "n/a")
    angle_threshold = summary.get("angle_threshold", "n/a")
    center_threshold = summary.get("center_threshold", "n/a")

    summary_table = pd.DataFrame([summary]).T.reset_index()
    summary_table.columns = ["metric", "value"]

    det_table = pd.DataFrame(
        [
            {"metric": "detection_accuracy", "value": summary.get("detection_accuracy", 0.0)},
            {"metric": "precision", "value": summary.get("precision", 0.0)},
            {"metric": "recall", "value": summary.get("recall", 0.0)},
            {"metric": "f1", "value": summary.get("f1", 0.0)},
            {"metric": "map_50_proxy", "value": summary.get("map_50_proxy", 0.0)},
        ]
    )
    loc_table = pd.DataFrame(
        [
            {"metric": "mean_iou", "value": summary.get("mean_iou", 0.0)},
            {"metric": "median_iou", "value": summary.get("median_iou", 0.0)},
            {"metric": "mean_center_error_px", "value": summary.get("mean_center_error_px", 0.0)},
            {"metric": "median_center_error_px", "value": summary.get("median_center_error_px", 0.0)},
        ]
    )
    ori_table = pd.DataFrame(
        [
            {"metric": "mean_angle_error_deg", "value": summary.get("mean_angle_error_deg", "unavailable")},
            {"metric": "median_angle_error_deg", "value": summary.get("median_angle_error_deg", "unavailable")},
            {"metric": "angle_accuracy_at_5deg", "value": summary.get("angle_accuracy_at_5deg", "unavailable")},
        ]
    )
    runtime_table = pd.DataFrame(
        [
            {"metric": "mean_processing_time_ms", "value": summary.get("mean_processing_time_ms", 0.0)},
            {"metric": "median_processing_time_ms", "value": summary.get("median_processing_time_ms", 0.0)},
            {"metric": "fps", "value": summary.get("fps", 0.0)},
        ]
    )

    error_counts = summary.get("error_type_counts", {})
    error_rows = [{"error_type": k, "count": v} for k, v in sorted(error_counts.items(), key=lambda x: x[0])]
    errors_table = pd.DataFrame(error_rows) if error_rows else pd.DataFrame([{"error_type": "none", "count": 0}])

    worst_iou = metrics_df.sort_values("iou", ascending=True).head(10)[["filename", "iou", "error_type", "message"]]
    angle_available = "angle_error_deg" in metrics_df.columns and metrics_df["angle_error_deg"].notna().any()
    if angle_available:
        worst_angle = metrics_df.sort_values("angle_error_deg", ascending=False).head(10)[["filename", "angle_error_deg", "error_type", "message"]]
    else:
        worst_angle = pd.DataFrame([{"filename": "n/a", "angle_error_deg": "unavailable", "error_type": "no_angle_prediction", "message": "Angle predictions unavailable"}])

    interpretation_en = (
        "Higher IoU/precision/recall indicate stronger detection quality. "
        "Lower center and angle errors indicate better geometric consistency."
    )
    interpretation_ru = (
        "Более высокие значения IoU/precision/recall соответствуют лучшему качеству детекции. "
        "Более низкие ошибки центра и угла соответствуют более точной геометрической оценке."
    )
    limitations_en = (
        "This report evaluates filename-level matching only. "
        "It does not perform multi-object assignment, confidence-threshold sweeps, or COCO-style mAP computation."
    )
    limitations_ru = (
        "Данный отчёт выполняет сопоставление только по имени файла. "
        "Не выполняется многoобъектное назначение, sweep по confidence-порогам и COCO-подобный расчёт mAP."
    )

    lines = [
        f"# {_title(language)}",
        "",
        f"## {labels['method']}",
        f"- `{method_name}`",
        "",
        f"## {labels['setup']}",
        f"- iou_threshold: `{iou_threshold}`",
        f"- angle_threshold: `{angle_threshold}`",
        f"- center_threshold: `{center_threshold}`",
        "",
        f"## {labels['dataset']}",
        f"- total_samples: `{total_samples}`",
        "",
        f"## {labels['summary']}",
        summary_table.to_markdown(index=False),
        "",
        f"## {labels['det_quality']}",
        det_table.to_markdown(index=False),
        "",
        f"## {labels['loc_quality']}",
        loc_table.to_markdown(index=False),
        "",
        f"## {labels['ori_quality']}",
        ori_table.to_markdown(index=False),
        "",
        f"## {labels['runtime']}",
        runtime_table.to_markdown(index=False),
        "",
        f"## {labels['errors']}",
        errors_table.to_markdown(index=False),
        "",
        f"## {labels['worst_iou']}",
        worst_iou.to_markdown(index=False),
        "",
        f"## {labels['worst_angle']}",
        worst_angle.to_markdown(index=False),
        "",
        f"## {labels['interpretation']}",
        interpretation_en if language == "en" else interpretation_ru,
        "",
        f"## {labels['limitations']}",
        limitations_en if language == "en" else limitations_ru,
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

