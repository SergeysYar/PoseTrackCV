"""Evaluate prediction CSV against ground-truth annotations for BrushPose AI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure local package imports work for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_angle_error,
    compute_center_error,
    compute_detection_accuracy,
    compute_iou,
    compute_precision_recall,
    compute_processing_stats,
    summarize_numeric_metric,
)
from src.evaluation.report_generator import generate_markdown_report


@dataclass
class EvalConfig:
    ground_truth: Path
    predictions: Path
    output_dir: Path
    method_name: str
    iou_threshold: float = 0.5
    angle_threshold: float = 5.0
    center_threshold: float = 10.0
    report_format: str = "both"


def _safe_filename(value: object) -> str:
    return Path(str(value)).name.strip()


def _to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    if "x_min" not in df.columns and "x1" in df.columns:
        mapping["x1"] = "x_min"
    if "y_min" not in df.columns and "y1" in df.columns:
        mapping["y1"] = "y_min"
    if "x_max" not in df.columns and "x2" in df.columns:
        mapping["x2"] = "x_max"
    if "y_max" not in df.columns and "y2" in df.columns:
        mapping["y2"] = "y_max"
    if "x_center" not in df.columns and "center_x" in df.columns:
        mapping["center_x"] = "x_center"
    if "y_center" not in df.columns and "center_y" in df.columns:
        mapping["center_y"] = "y_center"
    df = df.rename(columns=mapping)
    if "status" not in df.columns:
        df["status"] = "detected"
    if "message" not in df.columns:
        df["message"] = ""
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    if "processing_time_ms" not in df.columns:
        df["processing_time_ms"] = np.nan
    return df


def _select_best_prediction(group: pd.DataFrame) -> tuple[pd.Series, str | None]:
    warning = None
    if len(group) <= 1:
        return group.iloc[0], warning

    warning = f"Multiple predictions found ({len(group)}); selected best candidate."
    successful = group[group["status"].astype(str).isin(["success", "detected"])]
    if not successful.empty:
        if "confidence" in successful.columns and successful["confidence"].notna().any():
            idx = successful["confidence"].astype(float).idxmax()
            return successful.loc[idx], warning
        return successful.iloc[0], warning
    return group.iloc[0], warning


def _load_inputs(cfg: EvalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.ground_truth.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {cfg.ground_truth}")
    if not cfg.predictions.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {cfg.predictions}")

    gt = pd.read_csv(cfg.ground_truth)
    pred = pd.read_csv(cfg.predictions)
    if gt.empty:
        raise ValueError("Ground-truth CSV is empty.")
    if pred.empty:
        pred = pd.DataFrame(columns=["filename", "status", "message"])

    required_gt = {"filename", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center", "angle_deg"}
    missing_gt = required_gt - set(gt.columns)
    if missing_gt:
        raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing_gt)}")

    pred = _normalize_prediction_columns(pred)
    if "filename" not in pred.columns:
        raise ValueError("Prediction CSV must contain filename column.")

    gt = gt.copy()
    pred = pred.copy()
    gt["filename_key"] = gt["filename"].map(_safe_filename)
    pred["filename_key"] = pred["filename"].map(_safe_filename)
    return gt, pred


def evaluate(cfg: EvalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate predictions vs ground truth and return per-sample metrics + summary."""
    gt_df, pred_df = _load_inputs(cfg)

    grouped_pred = {k: g.reset_index(drop=True) for k, g in pred_df.groupby("filename_key")}
    per_sample: list[dict[str, Any]] = []
    warnings: list[str] = []

    for _, gt in gt_df.iterrows():
        key = gt["filename_key"]
        gt_box = (_to_float(gt["x_min"]), _to_float(gt["y_min"]), _to_float(gt["x_max"]), _to_float(gt["y_max"]))
        gt_center = (_to_float(gt["x_center"]), _to_float(gt["y_center"]))
        gt_angle = _to_float(gt["angle_deg"])

        base_row: dict[str, Any] = {
            "filename": key,
            "method_name": cfg.method_name,
            "gt_x_min": gt_box[0],
            "gt_y_min": gt_box[1],
            "gt_x_max": gt_box[2],
            "gt_y_max": gt_box[3],
            "gt_angle_deg": gt_angle,
        }

        if key not in grouped_pred:
            row = {
                **base_row,
                "detection_success": False,
                "iou": 0.0,
                "iou_pass": False,
                "center_error_px": np.nan,
                "center_pass": False,
                "angle_error_deg": np.nan,
                "angle_pass": False,
                "confidence": np.nan,
                "processing_time_ms": np.nan,
                "pred_x_min": np.nan,
                "pred_y_min": np.nan,
                "pred_x_max": np.nan,
                "pred_y_max": np.nan,
                "pred_angle_deg": np.nan,
                "error_type": "missed_detection",
                "message": "No prediction row for filename.",
            }
            per_sample.append(row)
            continue

        pred_row, warning = _select_best_prediction(grouped_pred[key])
        if warning:
            warnings.append(f"{key}: {warning}")
        status = str(pred_row.get("status", "")).strip().lower()

        px_min = _to_float(pred_row.get("x_min"))
        py_min = _to_float(pred_row.get("y_min"))
        px_max = _to_float(pred_row.get("x_max"))
        py_max = _to_float(pred_row.get("y_max"))
        p_center_x = _to_float(pred_row.get("x_center"))
        p_center_y = _to_float(pred_row.get("y_center"))
        if p_center_x is None and None not in (px_min, px_max):
            p_center_x = (px_min + px_max) / 2.0
        if p_center_y is None and None not in (py_min, py_max):
            p_center_y = (py_min + py_max) / 2.0

        p_angle = _to_float(pred_row.get("angle_deg")) if "angle_deg" in pred_row.index else None
        conf = _to_float(pred_row.get("confidence"))
        proc_ms = _to_float(pred_row.get("processing_time_ms"))
        message = str(pred_row.get("message", ""))

        detection_success = status in {"success", "detected"}
        if status in {"failed", "no_detection"}:
            detection_success = False

        iou = 0.0
        center_error = np.nan
        angle_error = np.nan
        iou_pass = False
        center_pass = False
        angle_pass = False
        error_type = "ok"

        has_pred_box = None not in (px_min, py_min, px_max, py_max) and px_min < px_max and py_min < py_max
        has_gt_box = None not in gt_box and gt_box[0] < gt_box[2] and gt_box[1] < gt_box[3]
        if not detection_success:
            error_type = "invalid_prediction" if status == "failed" else "missed_detection"
        elif not has_pred_box or not has_gt_box:
            error_type = "invalid_prediction"
        else:
            iou = compute_iou((px_min, py_min, px_max, py_max), (gt_box[0], gt_box[1], gt_box[2], gt_box[3]))
            iou_pass = iou >= cfg.iou_threshold

            if p_center_x is not None and p_center_y is not None and None not in gt_center:
                center_error = compute_center_error((p_center_x, p_center_y), (gt_center[0], gt_center[1]))
                center_pass = center_error <= cfg.center_threshold

            if p_angle is None or gt_angle is None:
                error_type = "no_angle_prediction"
            else:
                angle_error = compute_angle_error(p_angle, gt_angle)
                angle_pass = angle_error <= cfg.angle_threshold

            if not iou_pass:
                error_type = "low_iou"
            elif not np.isnan(center_error) and not center_pass:
                error_type = "high_center_error"
            elif not np.isnan(angle_error) and not angle_pass:
                error_type = "high_angle_error"
            elif error_type != "no_angle_prediction":
                error_type = "ok"

        per_sample.append(
            {
                **base_row,
                "detection_success": bool(detection_success),
                "iou": float(iou),
                "iou_pass": bool(iou_pass),
                "center_error_px": center_error,
                "center_pass": bool(center_pass),
                "angle_error_deg": angle_error,
                "angle_pass": bool(angle_pass),
                "confidence": conf,
                "processing_time_ms": proc_ms,
                "pred_x_min": px_min,
                "pred_y_min": py_min,
                "pred_x_max": px_max,
                "pred_y_max": py_max,
                "pred_angle_deg": p_angle,
                "error_type": error_type,
                "message": (warning + " | " if warning else "") + message,
            }
        )

    metrics_df = pd.DataFrame(per_sample)
    if warnings:
        print(f"[WARN] Multiple prediction rows detected for {len(warnings)} files.")

    total = len(metrics_df)
    detection_success_flags = metrics_df["detection_success"].astype(bool).tolist()
    detection_accuracy = compute_detection_accuracy(detection_success_flags)

    tp = int(((metrics_df["iou_pass"] == True) & (metrics_df["detection_success"] == True)).sum())
    fp = int(((metrics_df["detection_success"] == True) & (metrics_df["iou_pass"] == False)).sum())
    fn = int((metrics_df["detection_success"] == False).sum())
    prf = compute_precision_recall(tp, fp, fn)

    iou_stats = summarize_numeric_metric(metrics_df["iou"].tolist())
    center_stats = summarize_numeric_metric(metrics_df["center_error_px"].dropna().tolist())

    angle_available = "angle_error_deg" in metrics_df.columns and metrics_df["angle_error_deg"].notna().any()
    if angle_available:
        angle_vals = metrics_df["angle_error_deg"].dropna().tolist()
        angle_stats = summarize_numeric_metric(angle_vals)
        angle_acc = float((pd.Series(angle_vals) <= cfg.angle_threshold).mean())
    else:
        angle_stats = {"mean": None, "median": None, "std": None, "min": None, "max": None}
        angle_acc = None

    processing_stats = compute_processing_stats(metrics_df["processing_time_ms"].dropna().tolist())
    map_50_proxy = float((metrics_df["iou"] >= 0.5).mean()) if total > 0 else 0.0
    err_counts = metrics_df["error_type"].fillna("unknown").value_counts().to_dict()

    summary: dict[str, Any] = {
        "method_name": cfg.method_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_samples": int(total),
        "successful_detections": int(sum(detection_success_flags)),
        "missed_detections": int((~metrics_df["detection_success"].astype(bool)).sum()),
        "detection_accuracy": float(detection_accuracy),
        "precision": float(prf["precision"]),
        "recall": float(prf["recall"]),
        "f1": float(prf["f1"]),
        "mean_iou": float(iou_stats["mean"]),
        "median_iou": float(iou_stats["median"]),
        "map_50_proxy": float(map_50_proxy),
        "mean_center_error_px": float(center_stats["mean"]),
        "median_center_error_px": float(center_stats["median"]),
        "mean_angle_error_deg": angle_stats["mean"] if angle_available else "unavailable",
        "median_angle_error_deg": angle_stats["median"] if angle_available else "unavailable",
        "angle_accuracy_at_5deg": angle_acc if angle_available else "unavailable",
        "mean_processing_time_ms": float(processing_stats["mean_processing_time_ms"]),
        "median_processing_time_ms": float(processing_stats["median_processing_time_ms"]),
        "fps": float(processing_stats["fps"]),
        "error_type_counts": err_counts,
        "iou_threshold": cfg.iou_threshold,
        "angle_threshold": cfg.angle_threshold,
        "center_threshold": cfg.center_threshold,
    }
    return metrics_df, summary


def _write_outputs(metrics_df: pd.DataFrame, summary: dict[str, Any], cfg: EvalConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = cfg.output_dir / "metrics.csv"
    summary_json = cfg.output_dir / "summary_metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if cfg.report_format in {"markdown", "both"}:
        generate_markdown_report(metrics_df, summary, cfg.output_dir / "benchmark_report_en.md", language="en")
        generate_markdown_report(metrics_df, summary, cfg.output_dir / "benchmark_report_ru.md", language="ru")

    if cfg.report_format in {"json", "both"}:
        # summary JSON already written above
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV against BrushPose ground-truth CSV.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--angle-threshold", type=float, default=5.0)
    parser.add_argument("--center-threshold", type=float, default=10.0)
    parser.add_argument("--report-format", choices=["markdown", "json", "both"], default="both")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = EvalConfig(
        ground_truth=args.ground_truth,
        predictions=args.predictions,
        output_dir=args.output_dir,
        method_name=args.method_name,
        iou_threshold=float(args.iou_threshold),
        angle_threshold=float(args.angle_threshold),
        center_threshold=float(args.center_threshold),
        report_format=str(args.report_format),
    )

    try:
        metrics_df, summary = evaluate(cfg)
        _write_outputs(metrics_df, summary, cfg)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Evaluation finished.")
    print(f"- Method: {cfg.method_name}")
    print(f"- Total samples: {summary['total_samples']}")
    print(f"- Detection accuracy: {summary['detection_accuracy']:.4f}")
    print(f"- Mean IoU: {summary['mean_iou']:.4f}")
    print(f"- Mean center error (px): {summary['mean_center_error_px']:.4f}")
    print(f"- Mean processing time (ms): {summary['mean_processing_time_ms']:.4f}")
    print(f"- FPS: {summary['fps']:.4f}")
    print(f"- Output dir: {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
