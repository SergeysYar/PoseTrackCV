# Evaluation Subsystem

## Purpose
The evaluation module measures prediction quality against ground truth for:
- detection success
- localization overlap (IoU)
- center coordinate accuracy
- orientation accuracy (if angles are available)
- runtime performance
- error distribution and worst-case samples

It is designed to work with classical CV predictions, YOLO predictions, and future hybrid pipelines.

## Input CSV Formats
Ground truth CSV:
- `filename,width,height,x_min,y_min,x_max,y_max,x_center,y_center,angle_deg,class_name`

Prediction CSV (standard):
- `filename,status,x_min,y_min,x_max,y_max,x_center,y_center,angle_deg,confidence,processing_time_ms,message`

YOLO-style prediction columns are also supported:
- `x1,y1,x2,y2,center_x,center_y`

The evaluator maps these to standard names automatically.

## Matching Logic
- Matching is done by **filename only** (`Path(...).name`), ignoring directory prefixes.
- If multiple predictions exist for a filename:
  1. prefer successful rows (`success`/`detected`)
  2. if confidence exists, select highest confidence
  3. otherwise select first successful row
- Missing predictions are marked as `missed_detection`.
- `failed`/`no_detection` rows are treated as unsuccessful detections.

## Metrics
- `IoU`: bounding-box overlap quality.
- `center_error_px`: Euclidean center distance in pixels.
- `angle_error_deg`: minimal angular difference with 180° symmetry.
- `detection_accuracy`: successful detections / total samples.
- `precision`, `recall`, `f1`
- `map_50_proxy`: share of samples with `IoU >= 0.5` (proxy, not COCO mAP).
- runtime stats: mean/median time, FPS.

## Angle Handling
If `angle_deg` is missing in prediction CSV:
- angle metrics are skipped
- summary fields are marked as `unavailable`
- affected rows use `error_type = no_angle_prediction`

## Error Analysis
Automatic categorization:
- `ok`
- `missed_detection`
- `low_iou`
- `high_center_error`
- `high_angle_error`
- `invalid_prediction`
- `missing_ground_truth`
- `no_angle_prediction`

Reports include:
- error type counts
- top worst samples by IoU
- top worst samples by angle error

## CLI Command
```bash
python src/evaluation/evaluate_predictions.py \
  --ground-truth data/test/labels/annotations.csv \
  --predictions outputs/metrics/classical_cv_predictions.csv \
  --output-dir outputs/reports/classical_cv_eval \
  --method-name classical_cv_pca \
  --iou-threshold 0.5 \
  --angle-threshold 5 \
  --report-format both
```

## Output Files
- `metrics.csv`: per-sample metrics and error tags
- `summary_metrics.json`: aggregated metrics and counts
- `benchmark_report_en.md`: English markdown report
- `benchmark_report_ru.md`: Russian markdown report

## Interpreting Results
- Higher `mean_iou`, `precision`, `recall` indicate stronger detection performance.
- Lower `mean_center_error_px` indicates better geometric localization.
- Lower `mean_angle_error_deg` indicates better orientation quality.
- Higher `fps` indicates better real-time feasibility.
