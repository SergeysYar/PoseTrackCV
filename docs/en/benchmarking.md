# Benchmarking (Multi-Method)

## Purpose
`run_benchmark.py` orchestrates a reproducible comparison of:
- `classical_min_area_rect`
- `classical_pca`
- `yolo_geometric`

The benchmark reuses existing inference and evaluation modules and aggregates results into unified tables and reports.

## Required Inputs
- `--images-dir`: test image directory
- `--ground-truth`: ground-truth CSV
- `--output-dir`: base output directory
- `--yolo-weights`: YOLO weights path (optional if skipping YOLO)

## Example Command
```bash
python scripts/run_benchmark.py \
  --images-dir data/test/images \
  --ground-truth data/test/labels/annotations.csv \
  --output-dir outputs \
  --yolo-weights runs/brushpose_yolo/train/weights/best.pt \
  --methods classical_min_area_rect classical_pca yolo_geometric \
  --iou-threshold 0.5 \
  --angle-threshold 5 \
  --language both \
  --skip-yolo-if-missing
```

## Output Structure
```text
outputs/
  images/benchmark/<method>/
  metrics/benchmark/
    <method>_predictions.csv
    <method>_metrics.csv
    <method>_summary.json
    benchmark_results.csv
    benchmark_summary.json
  metrics/benchmark_results.csv
  reports/
    benchmark_en.md
    benchmark_ru.md
    benchmark_logs/
```

## Interpreting `benchmark_results.csv`
Key columns:
- `detection_accuracy`
- `mean_iou`, `median_iou`
- `mean_center_error_px`
- `mean_angle_error_deg`
- `mean_processing_time_ms`, `fps`
- `status`, `notes`

Use `status` and `notes` to understand partial failures or unavailable metrics.

## Angle Metrics and YOLO
Standard YOLO detection predicts boxes/classes, not orientation.  
If angle output is unavailable for `yolo_geometric`, angle metrics remain empty and this is documented in `notes`.

## Typical Failure Cases
- missing image directory or ground-truth CSV
- missing YOLO weights
- unavailable Ultralytics package
- empty prediction CSV for a method
- per-method evaluation failure

The benchmark continues with remaining methods and marks failed/skipped methods in output tables.

## Reproducibility Notes
- Fix random seeds in upstream pipelines when possible.
- Keep a stable test set and immutable ground-truth CSV.
- Store per-method logs in `outputs/reports/benchmark_logs/`.
- See `docs/en/evaluation.md` for metric definitions.
