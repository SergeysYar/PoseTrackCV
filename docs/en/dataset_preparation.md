# Dataset Preparation for BrushPose AI

## Goal
The dataset preparation subsystem builds a reproducible dataset for toothbrush localization and orientation estimation on tabletop scenes.  
Each annotation row must include:
- bounding box: `x_min, y_min, x_max, y_max`
- center point: `x_center, y_center`
- orientation: `angle_deg` in `[0, 180]`

## Recommended Capture Setup
- Camera: fixed top-view, optical axis approximately perpendicular to table.
- Distance: consistent height to reduce scale variance.
- Background: homogeneous matte surface preferred.
- Lighting: diffuse and stable, avoid hard shadows and strong reflections.
- Object coverage: include full angle range and varied spatial positions.

## Folder Structure
```text
data/
  images/
    image_000001.jpg
    image_000002.jpg
  annotations/
    annotations.csv
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

## Annotation Format
Main file: `data/annotations/annotations.csv`

Required columns:
- `filename`
- `width`
- `height`
- `x_min`
- `y_min`
- `x_max`
- `y_max`
- `x_center`
- `y_center`
- `angle_deg`
- `class_name`

Default class: `toothbrush`

Coordinate system:
- origin at top-left pixel
- `x` increases to the right
- `y` increases downward

Angle convention:
- orientation is axial
- valid range: `[0, 180]`

## 1) Create Dataset Template
Generate normalized file names (`image_000001.jpg`, ...) and initial template CSV:

```bash
python src/data/collect_dataset_template.py \
  --input-dir data/raw \
  --output-dir data/images \
  --annotations-out data/annotations/annotations.csv \
  --copy \
  --convert-to-jpg \
  --class-name toothbrush
```

Notes:
- supported extensions: `.jpg .jpeg .png .bmp .webp`
- optional resizing: `--resize-width --resize-height`
- optional move mode: `--move` instead of `--copy`

## 2) Validate Dataset
```bash
python src/data/validate_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --report-out outputs/reports/dataset_validation.md
```

Strict validation:
```bash
python src/data/validate_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --report-out outputs/reports/dataset_validation.md \
  --strict
```

Validation checks:
- file exists and image is readable
- size in CSV matches real image size
- bbox inside image bounds
- `x_min < x_max`, `y_min < y_max`
- center inside bbox
- angle in `[0, 180]`
- non-empty class name
- duplicate filenames

If `width/height` or center values are missing, they can be inferred in non-strict mode.

## 3) Split Train/Val/Test and Export Labels
```bash
python src/data/split_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --output-dir data \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --copy-images \
  --format both
```

`--format` options:
- `csv`: split-specific `annotations.csv`
- `yolo`: YOLO txt labels + `angle_labels.csv`
- `both`: export both types

Also generated:
- `outputs/metrics/dataset_stats.csv`
- `outputs/reports/dataset_stats.md`

## 4) Convert Annotations
BrushPose CSV to YOLO:
```bash
python src/data/convert_annotations.py \
  --input data/annotations/annotations.csv \
  --images-dir data/images \
  --output-dir data/yolo_labels \
  --from-format brushpose-csv \
  --to-format yolo \
  --class-id 0
```

BrushPose CSV to pose CSV:
```bash
python src/data/convert_annotations.py \
  --input data/annotations/annotations.csv \
  --images-dir data/images \
  --output-dir data/pose_labels \
  --from-format brushpose-csv \
  --to-format pose-csv
```

## Common Mistakes
- inconsistent image and CSV dimensions after manual resizing
- invalid bbox bounds (negative values or outside image)
- incorrect angle range (using `[-180, 180]` or `[0, 360]`)
- mixed filename casing and extensions
- duplicate filenames in `annotations.csv`

## Quality Checklist
- [ ] all images are readable
- [ ] all filenames in CSV exist in `data/images`
- [ ] bbox coordinates are valid and inside image bounds
- [ ] center coordinates are coherent with bbox
- [ ] all angles are in `[0, 180]`
- [ ] class labels are non-empty and consistent
- [ ] split ratios are reproducible with fixed seed
- [ ] validation and stats reports are generated successfully
