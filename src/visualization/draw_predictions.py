"""Drawing utilities and CLI for BrushPose prediction visualization."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLOR_PRED = (0, 255, 0)
COLOR_GT = (255, 120, 0)
COLOR_PRED_CENTER = (0, 255, 255)
COLOR_GT_CENTER = (255, 180, 80)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_ERROR = (0, 0, 255)


@dataclass
class DrawConfig:
    images_dir: Path
    predictions: Path
    output_dir: Path
    ground_truth: Path | None = None
    metrics: Path | None = None
    mode: str = "prediction"
    method_name: str = ""
    arrow_length: int = 80
    thickness: int = 2
    font_scale: float = 0.6
    output_ext: str = "jpg"


def _normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
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
    df = df.rename(columns=mapping).copy()
    if "status" not in df.columns:
        df["status"] = "detected"
    if "message" not in df.columns:
        df["message"] = ""
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    if "x_center" not in df.columns:
        df["x_center"] = np.nan
    if "y_center" not in df.columns:
        df["y_center"] = np.nan
    return df


def _safe_filename(v: object) -> str:
    return Path(str(v)).name


def _to_float(v: object) -> float | None:
    if pd.isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _compute_center_from_box(row: pd.Series) -> tuple[float | None, float | None]:
    x_min, y_min = _to_float(row.get("x_min")), _to_float(row.get("y_min"))
    x_max, y_max = _to_float(row.get("x_max")), _to_float(row.get("y_max"))
    if None in (x_min, y_min, x_max, y_max):
        return None, None
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def _draw_label(img: np.ndarray, text: str, x: int, y: int, font_scale: float) -> None:
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    y_top = max(0, y - h - baseline - 4)
    cv2.rectangle(img, (x, y_top), (x + w + 8, y + 4), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 1, cv2.LINE_AA)


def _draw_arrow(img: np.ndarray, center: tuple[float, float], angle_deg: float, color: tuple[int, int, int], length: int, thickness: int) -> None:
    rad = math.radians(angle_deg)
    x0, y0 = int(round(center[0])), int(round(center[1]))
    x1 = int(round(center[0] + length * math.cos(rad)))
    y1 = int(round(center[1] + length * math.sin(rad)))
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.22)


def draw_prediction_overlay(img: np.ndarray, row: pd.Series, cfg: DrawConfig) -> np.ndarray:
    """Draw prediction overlay: bbox, center, angle, confidence, status."""
    out = img.copy()
    status = str(row.get("status", "")).lower()
    msg = str(row.get("message", ""))

    x_min, y_min = _to_float(row.get("x_min")), _to_float(row.get("y_min"))
    x_max, y_max = _to_float(row.get("x_max")), _to_float(row.get("y_max"))
    x_center, y_center = _to_float(row.get("x_center")), _to_float(row.get("y_center"))
    if x_center is None or y_center is None:
        x_center, y_center = _compute_center_from_box(row)

    if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
        cv2.rectangle(out, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLOR_PRED, cfg.thickness)
    if x_center is not None and y_center is not None:
        cv2.circle(out, (int(round(x_center)), int(round(y_center))), 4, COLOR_PRED_CENTER, -1)

    ang = _to_float(row.get("angle_deg"))
    if ang is not None and x_center is not None and y_center is not None:
        _draw_arrow(out, (x_center, y_center), ang, COLOR_PRED, cfg.arrow_length, cfg.thickness)

    label_parts = []
    if cfg.method_name:
        label_parts.append(cfg.method_name)
    if ang is not None:
        label_parts.append(f"angle={ang:.1f}deg")
    conf = _to_float(row.get("confidence"))
    if conf is not None:
        label_parts.append(f"conf={conf:.2f}")
    if label_parts:
        anchor_x = int(x_min) if x_min is not None else 10
        anchor_y = max(24, int(y_min) - 8) if y_min is not None else 24
        _draw_label(out, " | ".join(label_parts), anchor_x, anchor_y, cfg.font_scale)

    if status in {"failed", "no_detection"} or (x_min is None and x_center is None):
        _draw_label(out, f"status={status or 'unknown'} msg={msg[:80]}", 10, out.shape[0] - 10, cfg.font_scale)
        cv2.putText(out, "DETECTION ISSUE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale + 0.1, COLOR_ERROR, 2, cv2.LINE_AA)
    return out


def draw_ground_truth_overlay(img: np.ndarray, row: pd.Series, cfg: DrawConfig) -> np.ndarray:
    """Draw ground-truth overlay: bbox, center, and angle."""
    out = img.copy()
    x_min, y_min = _to_float(row.get("x_min")), _to_float(row.get("y_min"))
    x_max, y_max = _to_float(row.get("x_max")), _to_float(row.get("y_max"))
    x_center, y_center = _to_float(row.get("x_center")), _to_float(row.get("y_center"))
    if x_center is None or y_center is None:
        x_center, y_center = _compute_center_from_box(row)

    if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
        cv2.rectangle(out, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLOR_GT, cfg.thickness)
    if x_center is not None and y_center is not None:
        cv2.circle(out, (int(round(x_center)), int(round(y_center))), 4, COLOR_GT_CENTER, -1)

    ang = _to_float(row.get("angle_deg"))
    if ang is not None and x_center is not None and y_center is not None:
        _draw_arrow(out, (x_center, y_center), ang, COLOR_GT, cfg.arrow_length, cfg.thickness)

    anchor_x = int(x_min) if x_min is not None else 10
    anchor_y = max(24, int(y_min) - 8) if y_min is not None else 24
    label = f"GT angle={ang:.1f}deg" if ang is not None else "GT"
    _draw_label(out, label, anchor_x, anchor_y, cfg.font_scale)
    return out


def draw_comparison_overlay(
    img: np.ndarray,
    pred_row: pd.Series | None,
    gt_row: pd.Series | None,
    metric_row: pd.Series | None,
    cfg: DrawConfig,
) -> np.ndarray:
    """Draw prediction + ground-truth comparison and optional metric text."""
    out = img.copy()
    if gt_row is not None:
        out = draw_ground_truth_overlay(out, gt_row, cfg)
    if pred_row is not None:
        out = draw_prediction_overlay(out, pred_row, cfg)
    if metric_row is not None:
        iou = _to_float(metric_row.get("iou"))
        ce = _to_float(metric_row.get("center_error_px"))
        ae = _to_float(metric_row.get("angle_error_deg"))
        parts = []
        if iou is not None:
            parts.append(f"IoU={iou:.3f}")
        if ce is not None:
            parts.append(f"center_err={ce:.2f}px")
        if ae is not None:
            parts.append(f"angle_err={ae:.2f}deg")
        if parts:
            _draw_label(out, " | ".join(parts), 10, out.shape[0] - 12, cfg.font_scale)
    return out


def save_prediction_viz(image_bgr: np.ndarray, pred: Any, out_path: Path, title: str = "Prediction") -> None:
    """Backward-compatible helper used by existing CLI code.

    Expected `pred` attributes:
    - x_center, y_center
    - bbox = (x_min, y_min, x_max, y_max)
    - angle_deg
    - score (optional)
    - method (optional)
    """
    row = pd.Series(
        {
            "status": "success",
            "x_min": pred.bbox[0],
            "y_min": pred.bbox[1],
            "x_max": pred.bbox[2],
            "y_max": pred.bbox[3],
            "x_center": pred.x_center,
            "y_center": pred.y_center,
            "angle_deg": getattr(pred, "angle_deg", np.nan),
            "confidence": getattr(pred, "score", np.nan),
            "message": title,
        }
    )
    cfg = DrawConfig(images_dir=Path("."), predictions=Path("."), output_dir=out_path.parent, method_name=str(getattr(pred, "method", "")))
    vis = draw_prediction_overlay(image_bgr, row, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _load_csv_or_none(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] CSV is empty: {path}")
            return None
        return df
    except Exception as exc:
        print(f"[WARN] Failed to read CSV {path}: {exc}")
        return None


def _index_by_filename(df: pd.DataFrame | None, normalize_pred: bool = False) -> dict[str, pd.DataFrame]:
    if df is None:
        return {}
    if normalize_pred:
        df = _normalize_prediction_columns(df)
    df = df.copy()
    if "filename" not in df.columns:
        return {}
    df["filename_key"] = df["filename"].map(_safe_filename)
    return {k: g.reset_index(drop=True) for k, g in df.groupby("filename_key")}


def _select_pred_row(group: pd.DataFrame) -> pd.Series:
    if len(group) == 1:
        return group.iloc[0]
    ok = group[group["status"].astype(str).str.lower().isin(["success", "detected"])]
    if not ok.empty:
        if "confidence" in ok.columns and ok["confidence"].notna().any():
            return ok.loc[ok["confidence"].astype(float).idxmax()]
        return ok.iloc[0]
    return group.iloc[0]


def _collect_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS], key=lambda p: p.name.lower())


def run_batch_draw(cfg: DrawConfig) -> int:
    """Batch render overlays for all images in directory."""
    if not cfg.images_dir.exists():
        print(f"[ERROR] Images directory not found: {cfg.images_dir}")
        return 1
    if cfg.mode in {"prediction", "comparison"} and not cfg.predictions.exists():
        print(f"[ERROR] Predictions CSV not found: {cfg.predictions}")
        return 1
    if cfg.mode in {"ground-truth", "comparison"} and cfg.ground_truth is not None and not cfg.ground_truth.exists():
        print(f"[WARN] Ground-truth CSV not found: {cfg.ground_truth}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = _load_csv_or_none(cfg.predictions) if cfg.mode in {"prediction", "comparison"} else None
    gt_df = _load_csv_or_none(cfg.ground_truth) if cfg.mode in {"ground-truth", "comparison"} else None
    metrics_df = _load_csv_or_none(cfg.metrics) if cfg.mode == "comparison" else None

    pred_idx = _index_by_filename(pred_df, normalize_pred=True)
    gt_idx = _index_by_filename(gt_df, normalize_pred=False)
    met_idx = _index_by_filename(metrics_df, normalize_pred=False)

    image_paths = _collect_images(cfg.images_dir)
    if not image_paths:
        print("[ERROR] No supported images found.")
        return 1

    processed = 0
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Unreadable image: {img_path}")
            continue
        key = img_path.name
        out = img.copy()

        pred_row = None
        if key in pred_idx:
            pred_row = _select_pred_row(pred_idx[key])
        gt_row = gt_idx[key].iloc[0] if key in gt_idx else None
        metric_row = met_idx[key].iloc[0] if key in met_idx else None

        if cfg.mode == "prediction":
            if pred_row is not None:
                out = draw_prediction_overlay(out, pred_row, cfg)
            else:
                _draw_label(out, "No prediction row for image", 10, 24, cfg.font_scale)
                cv2.putText(out, "NO PREDICTION", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale + 0.1, COLOR_ERROR, 2)
        elif cfg.mode == "ground-truth":
            if gt_row is not None:
                out = draw_ground_truth_overlay(out, gt_row, cfg)
            else:
                _draw_label(out, "No ground-truth row for image", 10, 24, cfg.font_scale)
                cv2.putText(out, "NO GROUND TRUTH", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale + 0.1, COLOR_ERROR, 2)
        else:  # comparison
            out = draw_comparison_overlay(out, pred_row, gt_row, metric_row, cfg)
            if pred_row is None and gt_row is None:
                _draw_label(out, "No prediction and no ground-truth rows", 10, 24, cfg.font_scale)

        out_file = cfg.output_dir / f"{img_path.stem}_viz.{cfg.output_ext}"
        cv2.imwrite(str(out_file), out)
        processed += 1

    print("Visualization rendering finished.")
    print(f"- Mode: {cfg.mode}")
    print(f"- Total images: {len(image_paths)}")
    print(f"- Processed images: {processed}")
    print(f"- Output dir: {cfg.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw BrushPose predictions and optional GT/metrics overlays.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--mode", choices=["prediction", "ground-truth", "comparison"], default="prediction")
    parser.add_argument("--method-name", type=str, default="")
    parser.add_argument("--arrow-length", type=int, default=80)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.6)
    parser.add_argument("--output-ext", choices=["jpg", "png"], default="jpg")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = DrawConfig(
        images_dir=args.images_dir,
        predictions=args.predictions,
        output_dir=args.output_dir,
        ground_truth=args.ground_truth,
        metrics=args.metrics,
        mode=args.mode,
        method_name=args.method_name,
        arrow_length=args.arrow_length,
        thickness=args.thickness,
        font_scale=args.font_scale,
        output_ext=args.output_ext,
    )
    try:
        return run_batch_draw(cfg)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
