# YOLO Subsystem (BrushPose AI)

## Purpose
The YOLO subsystem provides a production-oriented path for toothbrush detection:
- export BrushPose annotations to YOLO format
- train Ultralytics YOLO
- run optional validation
- run inference on image/folder (and optional video)
- save annotated outputs and machine-readable CSV predictions

## Expected Input Annotation
Source CSV (`BrushPose format`):
- `filename,width,height,x_min,y_min,x_max,y_max,x_center,y_center,angle_deg,class_name`

YOLO training uses only detection fields (bbox + class).  
`angle_deg` is preserved separately as split-wise CSV files.

## Export to YOLO Format
```bash
python src/detection/export_yolo_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --output-dir data/yolo_dataset \
  --seed 42 \
  --copy-images
```

Generated structure:
```text
data/yolo_dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
  angle_labels/{train_angles.csv,val_angles.csv,test_angles.csv}
  dataset.yaml
  export_report.md
```

## dataset.yaml
```yaml
path: data/yolo_dataset
train: images/train
val: images/val
test: images/test
names:
  0: toothbrush
```

## Training
```bash
python src/detection/train_yolo.py \
  --data data/yolo_dataset/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --validate
```

Outputs:
- Ultralytics run directory (`project/name`)
- `weights/best.pt`, `weights/last.pt`
- `training_summary.md`

## Validation
Validation can be executed with `--validate` in training command.  
Validation metrics are included in `training_summary.md` when available.

## Inference
```bash
python src/detection/infer_yolo.py \
  --weights runs/brushpose_yolo/train/weights/best.pt \
  --input data/test/images \
  --output-dir outputs/images/yolo \
  --csv-out outputs/metrics/yolo_predictions.csv \
  --conf 0.25
```

Options:
- `--save-txt`: save normalized YOLO txt predictions
- `--save-crops`: save detection crops
- `--config`: load defaults from YAML

## Prediction CSV Format
Columns:
- `filename,status,class_id,class_name,confidence`
- `x1,y1,x2,y2,center_x,center_y,width,height`
- `image_width,image_height,processing_time_ms,message`

Status values:
- `detected`
- `no_detection`
- `failed`

## Recommended Baseline Hyperparameters
- model: `yolov8n.pt`
- imgsz: `640`
- batch: `8` (increase on larger GPUs)
- epochs: `50`
- conf (inference): `0.25`
- iou (inference NMS): `0.5`

## Common Problems
1. `Ultralytics package is not installed`
2. `dataset.yaml not found`
3. `weights file not found`
4. `No supported images found for inference`
5. Empty export due to annotation mismatch (`class_name != toothbrush`)

## Troubleshooting
- Ensure `pip install ultralytics`.
- Verify `dataset.yaml` paths are relative to `path`.
- Check annotation numeric fields (`width,height,x_min...`) are valid.
- Confirm image files listed in CSV exist in `--images-dir`.
- Start from `yolov8n.pt` baseline before larger models.

