# BrushPose AI

BrushPose AI is a production-style CV portfolio project for toothbrush detection and 2D pose estimation (center + orientation) on top-view RGB images.

## Highlights
- Dual pipeline: classical CV and YOLOv8
- Unified CLI (`prepare-data`, `validate-data`, `train-yolo`, `infer`, `run-classical`, `evaluate`, `benchmark`, `generate-report`)
- Reproducible configs and modular architecture
- Bilingual docs (EN/RU)
- Benchmarking and reporting outputs (CSV/JSON/Markdown/plots)

## Quick Start
```bash
python -m venv .venv
pip install -r requirements.txt
python src/cli.py validate-data --config configs/config.yaml
python src/cli.py benchmark --config configs/config.yaml
```

## Architecture
- `src/data`: dataset collection/split/validation/conversion
- `src/detection`: YOLO train/infer abstractions
- `src/pose`: geometric and PCA-based orientation estimation
- `src/evaluation`: metrics, benchmark, reports
- `src/visualization`: overlays and plots
- `src/cli.py`: workflow entrypoint

## Metrics
- IoU, detection accuracy, precision, recall
- mAP@0.5 placeholder support
- Center error, mean/median angle error
- FPS and inference time
- Percent of samples with angle error `< 5°`

## Outputs
- Visuals: `outputs/images`, `outputs/videos`
- Metrics: `outputs/metrics`
- Reports: `outputs/reports`
- Plots: `outputs/plots`

## Roadmap
- Confidence-sweep mAP
- Video temporal smoothing
- ONNX/TensorRT export
- CI benchmark regression checks

## License
MIT

