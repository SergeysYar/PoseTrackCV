"""CLI runner for BrushPose classical OpenCV pose estimation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Ensure local src imports work when executed as `python scripts/run_classical_cv.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.classical_cv import ClassicalCVConfig, ClassicalCVPoseEstimator, PosePrediction

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files, key=lambda p: p.name.lower())


def _prediction_to_row(pred: PosePrediction) -> dict[str, object]:
    return {
        "filename": pred.filename,
        "status": "success" if pred.success else "failed",
        "x_min": pred.x_min,
        "y_min": pred.y_min,
        "x_max": pred.x_max,
        "y_max": pred.y_max,
        "x_center": pred.x_center,
        "y_center": pred.y_center,
        "angle_deg": pred.angle_deg,
        "area": pred.area,
        "confidence": pred.confidence,
        "processing_time_ms": pred.processing_time_ms,
        "message": pred.message,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classical OpenCV toothbrush pose estimation.")
    parser.add_argument("--input", type=Path, required=True, help="Single image path or directory.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for annotated images.")
    parser.add_argument("--csv-out", type=Path, default=Path("outputs/metrics/classical_cv_predictions.csv"))
    parser.add_argument("--hsv-lower", nargs=3, type=int, default=[0, 40, 40])
    parser.add_argument("--hsv-upper", nargs=3, type=int, default=[180, 255, 255])
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--min-contour-area", type=float, default=500.0)
    parser.add_argument("--max-contour-area-ratio", type=float, default=0.7)
    parser.add_argument("--angle-method", choices=["pca", "min_area_rect"], default="pca")
    parser.add_argument("--draw-debug", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input path does not exist: {args.input}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    cfg = ClassicalCVConfig(
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
        blur_kernel=int(args.blur_kernel),
        morph_kernel=int(args.morph_kernel),
        min_contour_area=float(args.min_contour_area),
        max_contour_area_ratio=float(args.max_contour_area_ratio),
        use_pca_angle=args.angle_method == "pca",
        use_min_area_rect_angle=args.angle_method == "min_area_rect",
        draw_debug=bool(args.draw_debug),
    )
    estimator = ClassicalCVPoseEstimator(cfg)
    image_paths = _collect_images(args.input)
    if not image_paths:
        print("[ERROR] No supported images found.")
        return 1

    rows: list[dict[str, object]] = []
    success_count = 0
    failure_count = 0
    time_sum_ms = 0.0

    for img_path in image_paths:
        pred, vis = estimator.predict_image(img_path)
        if vis is None:
            fallback = np.zeros((120, 600, 3), dtype=np.uint8)
            cv2.putText(fallback, f"Failed: {pred.message}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            vis = fallback
        out_name = f"{img_path.stem}_classical_cv.jpg"
        cv2.imwrite(str(args.output / out_name), vis)

        rows.append(_prediction_to_row(pred))
        time_sum_ms += pred.processing_time_ms
        if pred.success:
            success_count += 1
        else:
            failure_count += 1

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)

    total = len(image_paths)
    avg_ms = time_sum_ms / total if total > 0 else 0.0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    print("Classical CV batch processing finished.")
    print(f"- Total images: {total}")
    print(f"- Successful detections: {success_count}")
    print(f"- Failed detections: {failure_count}")
    print(f"- Average processing time (ms): {avg_ms:.2f}")
    print(f"- Average FPS: {avg_fps:.2f}")
    print(f"- Annotated images: {args.output}")
    print(f"- Predictions CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
