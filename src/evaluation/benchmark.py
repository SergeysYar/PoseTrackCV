from __future__ import annotations

import time
from pathlib import Path

import cv2
import pandas as pd

from src.detection.infer_yolo import YOLODetector
from src.evaluation.metrics import EvalSample, angle_diff_deg, iou_xyxy, summarize
from src.pose.classical_cv import ClassicalPoseDetector
from src.pose.pose_estimator import PosePrediction
from src.utils.image_utils import list_images


def _load_gt(path: Path, w: int, h: int) -> tuple[int, int, int, int, float] | None:
    if not path.exists():
        return None
    parts = [float(x) for x in path.read_text(encoding="utf-8").strip().split()]
    if len(parts) < 5:
        return None
    _, cx, cy, bw, bh, *rest = parts
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2, (rest[0] if rest else 0.0)


def run_benchmark(image_dir: Path, label_dir: Path, out_csv: Path, yolo_weights: str = "yolov8n.pt") -> pd.DataFrame:
    classical = ClassicalPoseDetector()
    yolo = YOLODetector(weights=yolo_weights)
    rows: list[dict[str, float | str]] = []
    for image_path in list_images(image_dir):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = _load_gt(label_dir / f"{image_path.stem}.txt", w, h)
        if gt is None:
            continue
        gt_bbox = (gt[0], gt[1], gt[2], gt[3])
        gt_center = ((gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0)
        gt_angle = gt[4]

        t0 = time.perf_counter()
        c = classical.detect(img)
        c_ms = (time.perf_counter() - t0) * 1000
        preds: list[tuple[str, PosePrediction | None, float]] = []
        if c:
            preds.append(("hsv_contours", PosePrediction(c.center[0], c.center[1], c.angle_rect, c.bbox, 1.0, "hsv_contours"), c_ms))
            preds.append(("pca_orientation", PosePrediction(c.center[0], c.center[1], c.angle_pca, c.bbox, 1.0, "pca_orientation"), c_ms))
        else:
            preds.append(("hsv_contours", None, c_ms))
            preds.append(("pca_orientation", None, c_ms))

        t0 = time.perf_counter()
        y = yolo.predict(img)
        y_ms = (time.perf_counter() - t0) * 1000
        preds.append(("yolo", y[0] if y else None, y_ms))

        for method, pred, ms in preds:
            if pred is None:
                rows.append({"image": image_path.name, "method": method, "iou": 0.0, "center_error": 1e9, "angle_error": 180.0, "inference_time_ms": ms})
            else:
                center_error = ((pred.x_center - gt_center[0]) ** 2 + (pred.y_center - gt_center[1]) ** 2) ** 0.5
                rows.append(
                    {
                        "image": image_path.name,
                        "method": method,
                        "iou": iou_xyxy(pred.bbox, gt_bbox),
                        "center_error": center_error,
                        "angle_error": angle_diff_deg(pred.angle_deg, gt_angle),
                        "inference_time_ms": ms,
                    }
                )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method, g in df.groupby("method"):
        samples = [EvalSample(float(r.iou), float(r.center_error), float(r.angle_error), float(r.inference_time_ms)) for r in g.itertuples()]
        s = summarize(samples)
        s["method"] = method
        rows.append(s)
    return pd.DataFrame(rows)

