"""YOLO inference subsystem for BrushPose AI."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
try:
    import yaml
except Exception:  # pragma: no cover - environment dependent
    yaml = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - environment dependent
    YOLO = None

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
CLASS_NAME = "toothbrush"


@dataclass
class YOLOInferConfig:
    """Runtime inference configuration."""

    weights: str
    input: str
    output_dir: str = "outputs/images/yolo"
    csv_out: str = "outputs/metrics/yolo_predictions.csv"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"
    save_txt: bool = False
    save_crops: bool = False


@dataclass
class SimpleDetection:
    """Simple detection structure for compatibility and reuse."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float


class YOLODetector:
    """Small wrapper class for integration with existing project code."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 640, device: str = "auto", iou: float = 0.5) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics package is not installed.")
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.iou = iou

    def predict(self, image) -> list[SimpleDetection]:
        results = self.model.predict(image, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False)
        detections: list[SimpleDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                detections.append(
                    SimpleDetection(
                        class_id=cls_id,
                        class_name=CLASS_NAME if cls_id == 0 else f"class_{cls_id}",
                        confidence=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        center_x=(x1 + x2) / 2.0,
                        center_y=(y1 + y2) / 2.0,
                    )
                )
        return detections

    def predict_path(self, image_path: Path) -> list[SimpleDetection]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Unreadable image: {image_path}")
        return self.predict(img)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Install with `pip install pyyaml` to use --config.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a mapping object.")
    return data


def _merge_config(cli: argparse.Namespace) -> YOLOInferConfig:
    defaults: dict[str, Any] = {}
    if cli.config is not None:
        if not cli.config.exists():
            raise FileNotFoundError(f"Config file not found: {cli.config}")
        defaults = _load_yaml(cli.config)

    merged = {
        "weights": defaults.get("weights"),
        "input": defaults.get("input"),
        "output_dir": defaults.get("output_dir", "outputs/images/yolo"),
        "csv_out": defaults.get("csv_out", "outputs/metrics/yolo_predictions.csv"),
        "conf": float(defaults.get("conf", 0.25)),
        "iou": float(defaults.get("iou", 0.5)),
        "imgsz": int(defaults.get("imgsz", 640)),
        "device": str(defaults.get("device", "auto")),
        "save_txt": bool(defaults.get("save_txt", False)),
        "save_crops": bool(defaults.get("save_crops", False)),
    }
    for key in ["weights", "input", "output_dir", "csv_out", "conf", "iou", "imgsz", "device", "save_txt", "save_crops"]:
        value = getattr(cli, key)
        if value is not None:
            merged[key] = value
    if not merged["weights"]:
        raise ValueError("Weights path must be provided via --weights or config.")
    if not merged["input"]:
        raise ValueError("Input path must be provided via --input or config.")
    return YOLOInferConfig(**merged)


def _collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS], key=lambda x: x.name.lower())
    return []


def _draw_detections(image, detections: list[SimpleDetection], filename: str):
    vis = image.copy()
    for d in detections:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        cv2.circle(vis, (int(d.center_x), int(d.center_y)), 3, (0, 255, 255), -1)
        text = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(vis, text, (p1[0], max(20, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis, filename, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def _rows_for_no_detection(filename: str, width: int, height: int, ms: float, message: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "status": "no_detection",
        "class_id": None,
        "class_name": None,
        "confidence": None,
        "x1": None,
        "y1": None,
        "x2": None,
        "y2": None,
        "center_x": None,
        "center_y": None,
        "width": None,
        "height": None,
        "image_width": width,
        "image_height": height,
        "processing_time_ms": ms,
        "message": message,
    }


def run_on_images(cfg: YOLOInferConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    if YOLO is None:
        raise RuntimeError("Ultralytics package is not installed.")
    weights_path = Path(cfg.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    input_path = Path(cfg.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_dir = Path(cfg.output_dir)
    csv_out = Path(cfg.csv_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(weights=str(weights_path), conf=cfg.conf, imgsz=cfg.imgsz, device=cfg.device, iou=cfg.iou)
    image_paths = _collect_image_paths(input_path)
    if not image_paths:
        raise ValueError("No supported images found for inference.")

    rows: list[dict[str, Any]] = []
    total_detections = 0
    images_with_detections = 0
    failed_images = 0
    processed_images = 0
    confidence_values: list[float] = []
    total_ms = 0.0

    for image_path in image_paths:
        start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            failed_images += 1
            rows.append(
                {
                    "filename": image_path.name,
                    "status": "failed",
                    "class_id": None,
                    "class_name": None,
                    "confidence": None,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                    "center_x": None,
                    "center_y": None,
                    "width": None,
                    "height": None,
                    "image_width": None,
                    "image_height": None,
                    "processing_time_ms": (time.perf_counter() - start) * 1000.0,
                    "message": "OpenCV cannot read image.",
                }
            )
            continue

        h, w = image.shape[:2]
        try:
            detections = detector.predict(image)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_ms += elapsed_ms
            processed_images += 1
            vis = _draw_detections(image, detections, image_path.name)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_yolo.jpg"), vis)

            if not detections:
                rows.append(_rows_for_no_detection(image_path.name, w, h, elapsed_ms, "No detections above confidence threshold."))
            else:
                images_with_detections += 1
                total_detections += len(detections)
                for d in detections:
                    confidence_values.append(d.confidence)
                    rows.append(
                        {
                            "filename": image_path.name,
                            "status": "detected",
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "confidence": d.confidence,
                            "x1": d.x1,
                            "y1": d.y1,
                            "x2": d.x2,
                            "y2": d.y2,
                            "center_x": d.center_x,
                            "center_y": d.center_y,
                            "width": d.x2 - d.x1,
                            "height": d.y2 - d.y1,
                            "image_width": w,
                            "image_height": h,
                            "processing_time_ms": elapsed_ms,
                            "message": "OK",
                        }
                    )

            if cfg.save_txt:
                txt_out = output_dir / f"{image_path.stem}.txt"
                lines = [
                    f"{d.class_id} {d.center_x / w:.6f} {d.center_y / h:.6f} {(d.x2 - d.x1) / w:.6f} {(d.y2 - d.y1) / h:.6f} {d.confidence:.6f}"
                    for d in detections
                ]
                txt_out.write_text("\n".join(lines), encoding="utf-8")

            if cfg.save_crops:
                crops_dir = output_dir / "crops" / image_path.stem
                crops_dir.mkdir(parents=True, exist_ok=True)
                for idx, d in enumerate(detections):
                    x1, y1, x2, y2 = max(0, int(d.x1)), max(0, int(d.y1)), min(w, int(d.x2)), min(h, int(d.y2))
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(str(crops_dir / f"det_{idx:03d}.jpg"), crop)

        except Exception as exc:
            failed_images += 1
            rows.append(
                {
                    "filename": image_path.name,
                    "status": "failed",
                    "class_id": None,
                    "class_name": None,
                    "confidence": None,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                    "center_x": None,
                    "center_y": None,
                    "width": None,
                    "height": None,
                    "image_width": w,
                    "image_height": h,
                    "processing_time_ms": (time.perf_counter() - start) * 1000.0,
                    "message": str(exc),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)

    avg_conf = float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
    avg_ms = float(total_ms / processed_images) if processed_images else 0.0
    fps = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0
    summary = {
        "total_inputs": float(len(image_paths)),
        "processed_images": float(processed_images),
        "failed_images": float(failed_images),
        "images_with_detections": float(images_with_detections),
        "total_detections": float(total_detections),
        "average_confidence": avg_conf,
        "average_processing_time_ms": avg_ms,
        "approx_fps": fps,
    }
    return df, summary


def run_on_video(cfg: YOLOInferConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Optional video inference path; outputs annotated video and per-frame rows."""
    if YOLO is None:
        raise RuntimeError("Ultralytics package is not installed.")
    weights_path = Path(cfg.weights)
    input_path = Path(cfg.input)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_dir = Path(cfg.output_dir)
    csv_out = Path(cfg.csv_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(weights=str(weights_path), conf=cfg.conf, imgsz=cfg.imgsz, device=cfg.device, iou=cfg.iou)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = output_dir / f"{input_path.stem}_yolo.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    rows: list[dict[str, Any]] = []
    frame_idx = 0
    total_detections = 0
    confs: list[float] = []
    total_ms = 0.0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        detections = detector.predict(frame)
        ms = (time.perf_counter() - t0) * 1000.0
        total_ms += ms
        processed += 1
        vis = _draw_detections(frame, detections, f"{input_path.name}:{frame_idx}")
        writer.write(vis)

        if detections:
            total_detections += len(detections)
            for d in detections:
                confs.append(d.confidence)
                rows.append(
                    {
                        "filename": f"{input_path.name}:{frame_idx}",
                        "status": "detected",
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "x1": d.x1,
                        "y1": d.y1,
                        "x2": d.x2,
                        "y2": d.y2,
                        "center_x": d.center_x,
                        "center_y": d.center_y,
                        "width": d.x2 - d.x1,
                        "height": d.y2 - d.y1,
                        "image_width": width,
                        "image_height": height,
                        "processing_time_ms": ms,
                        "message": "OK",
                    }
                )
        else:
            rows.append(_rows_for_no_detection(f"{input_path.name}:{frame_idx}", width, height, ms, "No detections above confidence threshold."))
        frame_idx += 1

    cap.release()
    writer.release()
    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    avg_ms = float(total_ms / processed) if processed else 0.0
    summary = {
        "total_inputs": float(frame_idx),
        "processed_images": float(processed),
        "failed_images": 0.0,
        "images_with_detections": float(df[df["status"] == "detected"]["filename"].nunique()) if not df.empty else 0.0,
        "total_detections": float(total_detections),
        "average_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
        "average_processing_time_ms": avg_ms,
        "approx_fps": float(1000.0 / avg_ms) if avg_ms > 0 else 0.0,
    }
    return df, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO inference for BrushPose AI.")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Image path, image directory, or video path.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--csv-out", type=str, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-txt", action="store_true", default=None)
    parser.add_argument("--save-crops", action="store_true", default=None)
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = _merge_config(args)
        input_path = Path(cfg.input)
        if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            _, summary = run_on_video(cfg)
        else:
            _, summary = run_on_images(cfg)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("YOLO inference finished.")
    print(f"- Total inputs: {int(summary['total_inputs'])}")
    print(f"- Processed images: {int(summary['processed_images'])}")
    print(f"- Failed images: {int(summary['failed_images'])}")
    print(f"- Images with detections: {int(summary['images_with_detections'])}")
    print(f"- Total detections: {int(summary['total_detections'])}")
    print(f"- Average confidence: {summary['average_confidence']:.4f}")
    print(f"- Average processing time (ms): {summary['average_processing_time_ms']:.2f}")
    print(f"- Approx FPS: {summary['approx_fps']:.2f}")
    print(f"- Output images: {cfg.output_dir}")
    print(f"- Predictions CSV: {cfg.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
