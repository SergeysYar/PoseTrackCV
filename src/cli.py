from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data.split_dataset import split_dataset
from src.data.validate_dataset import create_dataset_summary, validate_dataset
from src.detection.infer_yolo import YOLODetector
from src.detection.train_yolo import YOLOTrainConfig, train_yolo
from src.evaluation.benchmark import run_benchmark, summarize_benchmark
from src.evaluation.metrics import EvalSample, angle_diff_deg, iou_xyxy, summarize
from src.evaluation.report_generator import save_metric_reports, write_benchmark_reports
from src.pose.classical_cv import ClassicalPoseDetector
from src.pose.pose_estimator import PosePrediction
from src.utils.config_utils import load_config
from src.utils.image_utils import list_images, read_image
from src.visualization.draw_predictions import save_prediction_viz
from src.visualization.plot_metrics import plot_angle_histogram, plot_method_comparison


def _classical_predict(image) -> PosePrediction | None:
    detector = ClassicalPoseDetector()
    res = detector.detect(image)
    if res is None:
        return None
    return PosePrediction(res.center[0], res.center[1], res.angle_pca, res.bbox, 1.0, "classical_pca")


def cmd_prepare_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    split_dataset(Path(cfg["data"]["image_dir"]), Path(cfg["data"]["annotation_dir"]), Path("data"))


def cmd_validate_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    stats_csv = Path("outputs/metrics/dataset_stats.csv")
    res = validate_dataset(Path(cfg["data"]["image_dir"]), Path(cfg["data"]["annotation_dir"]), stats_csv)
    create_dataset_summary(res, Path("outputs/reports/dataset_summary.md"))


def cmd_train_yolo(args: argparse.Namespace) -> None:
    y = load_config(args.config).get("yolo", {})
    train_yolo(
        YOLOTrainConfig(
            model_name=y.get("model_name", "yolov8n.pt"),
            data_yaml=y.get("data_yaml", "data/dataset.yaml"),
            epochs=int(y.get("epochs", 50)),
            imgsz=int(y.get("imgsz", 640)),
            batch=int(y.get("batch", 16)),
            device=str(y.get("device", "cpu")),
        )
    )


def cmd_infer(args: argparse.Namespace) -> None:
    if args.method == "classical":
        image = read_image(args.image)
        pred = _classical_predict(image)
        if pred is None:
            print("No detection")
            return
        save_prediction_viz(image, pred, args.output, "Classical Inference")
        return

    cfg = load_config(args.config).get("yolo", {})
    detector = YOLODetector(
        weights=args.weights or cfg.get("model_name", "yolov8n.pt"),
        conf=float(cfg.get("conf", 0.25)),
        imgsz=int(cfg.get("imgsz", 640)),
        device=str(cfg.get("device", "cpu")),
    )
    image = read_image(args.image)
    preds = detector.predict(image)
    if not preds:
        print("No detection")
        return
    save_prediction_viz(image, preds[0], args.output, "YOLO Inference")


def cmd_run_classical(args: argparse.Namespace) -> None:
    image = read_image(args.image)
    pred = _classical_predict(image)
    if pred is None:
        raise RuntimeError("No object detected by classical pipeline.")
    save_prediction_viz(image, pred, args.output, "Classical CV")


def cmd_evaluate(args: argparse.Namespace) -> None:
    images = list_images(args.image_dir)
    detector = YOLODetector(weights=args.weights) if args.method == "yolo" else None
    samples: list[EvalSample] = []
    for image_path in images:
        image = read_image(image_path)
        h, w = image.shape[:2]
        label_path = args.label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        parts = [float(x) for x in label_path.read_text(encoding="utf-8").strip().split()]
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh, *rest = parts
        gt_bbox = (int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h))
        gt_center = ((gt_bbox[0] + gt_bbox[2]) / 2.0, (gt_bbox[1] + gt_bbox[3]) / 2.0)
        gt_angle = rest[0] if rest else 0.0
        t0 = time.perf_counter()
        if args.method == "classical":
            pred = _classical_predict(image)
        else:
            preds = detector.predict(image) if detector else []
            pred = preds[0] if preds else None
        ms = (time.perf_counter() - t0) * 1000
        if pred is None:
            samples.append(EvalSample(0.0, 1e9, 180.0, ms))
            continue
        iou = iou_xyxy(pred.bbox, gt_bbox)
        c_err = ((pred.x_center - gt_center[0]) ** 2 + (pred.y_center - gt_center[1]) ** 2) ** 0.5
        a_err = angle_diff_deg(pred.angle_deg, gt_angle)
        samples.append(EvalSample(iou, c_err, a_err, ms))
    save_metric_reports(summarize(samples), Path("outputs/metrics"), args.method)


def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    test_root = Path(cfg["data"]["test_dir"])
    y_weights = cfg.get("yolo", {}).get("model_name", "yolov8n.pt")
    raw_csv = Path("outputs/metrics/benchmark_raw.csv")
    df = run_benchmark(test_root / "images", test_root / "labels", raw_csv, y_weights)
    summary = summarize_benchmark(df)
    summary_csv = Path("outputs/metrics/benchmark_results.csv")
    summary.to_csv(summary_csv, index=False)
    write_benchmark_reports(summary, Path("outputs/reports"))
    plot_method_comparison(summary_csv, Path("outputs/plots/benchmark_methods.png"))
    plot_angle_histogram(raw_csv, Path("outputs/plots/angle_histogram.png"))


def cmd_generate_report(args: argparse.Namespace) -> None:
    rows = []
    for csv_path in Path("outputs/metrics").glob("*_metrics.csv"):
        df = pd.read_csv(csv_path)
        if not df.empty:
            d = df.iloc[0].to_dict()
            d["source"] = csv_path.name
            rows.append(d)
    out = Path("outputs/reports/final_metrics_overview.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BrushPose AI CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare-data")
    sp.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    sp.set_defaults(func=cmd_prepare_data)

    sv = sub.add_parser("validate-data")
    sv.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    sv.set_defaults(func=cmd_validate_data)

    st = sub.add_parser("train-yolo")
    st.add_argument("--config", type=Path, default=Path("configs/yolo_config.yaml"))
    st.set_defaults(func=cmd_train_yolo)

    si = sub.add_parser("infer")
    si.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    si.add_argument("--method", choices=["classical", "yolo"], default="classical")
    si.add_argument("--image", type=Path, required=True)
    si.add_argument("--weights", type=str, default="")
    si.add_argument("--output", type=Path, default=Path("outputs/images/infer.png"))
    si.set_defaults(func=cmd_infer)

    sc = sub.add_parser("run-classical")
    sc.add_argument("--image", type=Path, required=True)
    sc.add_argument("--output", type=Path, default=Path("outputs/images/classical.png"))
    sc.set_defaults(func=cmd_run_classical)

    se = sub.add_parser("evaluate")
    se.add_argument("--method", choices=["classical", "yolo"], default="classical")
    se.add_argument("--image-dir", type=Path, default=Path("data/test/images"))
    se.add_argument("--label-dir", type=Path, default=Path("data/test/labels"))
    se.add_argument("--weights", type=str, default="yolov8n.pt")
    se.set_defaults(func=cmd_evaluate)

    sb = sub.add_parser("benchmark")
    sb.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    sb.set_defaults(func=cmd_benchmark)

    sr = sub.add_parser("generate-report")
    sr.set_defaults(func=cmd_generate_report)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

