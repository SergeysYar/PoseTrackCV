# BrushPose AI
### Toothbrush Detection and 2D Pose Estimation for Tabletop Computer Vision

Industrial-style computer vision pipeline for **object localization** and **orientation estimation** from top-view RGB imagery.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](#features)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](#training)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen.svg)](#future-improvements)

> Precision-first CV engineering for spatial reasoning on tabletop scenes.

---

## Project Overview
**BrushPose AI** is a modular computer vision system that detects a toothbrush on a homogeneous tabletop and estimates:
- object center coordinates: \((x_{center}, y_{center})\)
- orientation angle: \(\theta \in [0^\circ, 180^\circ)\)

The project combines **classical computer vision** and **deep learning detection** to provide:
- interpretable geometry-driven estimation
- configurable YOLO-based detection
- unified evaluation, benchmarking, and report generation

This repository is designed as a portfolio-grade ML/CV engineering project with reproducible workflows, structured code, and practical deployment-oriented outputs.

---

## Problem Statement
**Input**:
- Top-view RGB images or video frames containing a tabletop scene.

**Output**:
- Bounding box of the toothbrush
- Center point coordinates in image space
- Rotation angle relative to the image \(x\)-axis
- Visualization overlays and evaluation artifacts

**Core technical challenges**:
- lighting variability (shadows, highlights, specular reflection)
- low color contrast between object and background
- segmentation instability for thin elongated objects
- geometric ambiguity around near-symmetric pose states
- annotation noise in angle labels

BrushPose AI addresses these with dual pipelines and comparative benchmarking.

---

## Practical Use Cases
- **Robotics**: pre-grasp localization and orientation prior for manipulators
- **Industrial automation**: pick-and-place alignment on conveyor or static surfaces
- **Smart manufacturing**: object pose checks in assembly preparation
- **Warehouse micro-automation**: tabletop sorting and orientation-aware handling
- **Visual quality inspection**: pose consistency monitoring in packaged kits
- **Lab automation**: controlled placement validation in repetitive workflows

---

## Features
- **Toothbrush detection** from RGB tabletop scenes
- **Center localization** with pixel-level coordinates
- **Orientation estimation** via `minAreaRect` and PCA principal axis
- **Classical CV pipeline**: HSV segmentation, morphology, contours
- **YOLOv8 pipeline**: train/infer with configurable weights and thresholds
- **Visualization toolkit**: bbox, center marker, orientation arrow, angle text
- **Batch and single-image inference**
- **Evaluation subsystem** with standard detection and pose metrics
- **Benchmark framework** for method-to-method comparison
- **Dataset utilities**: validation, split, annotation conversion
- **CLI-first workflow** via `python src/cli.py ...`
- **Bilingual technical documentation** (EN/RU)

---

## Project Architecture
BrushPose AI uses modular separation across data, detection, pose, evaluation, and visualization layers.

```text
Data Collection/Validation
          |
          v
    Detection Layer
 (Classical CV / YOLOv8)
          |
          v
   Pose Estimation Layer
 (Center + Orientation)
          |
          v
  Evaluation & Benchmark
          |
          v
 Visualization & Reports
```

**Module responsibilities**

| Module | Responsibility |
|---|---|
| `src/data` | frame extraction, split generation, dataset validation, annotation conversion |
| `src/detection` | YOLO training and inference adapters |
| `src/pose` | geometry helpers, classical CV estimation, PCA orientation |
| `src/evaluation` | metrics, benchmark runner, method comparison, reporting |
| `src/visualization` | overlays, plots, comparative visual artifacts |
| `src/cli.py` | unified command-line orchestration |

Architecture image placeholder: `assets/pipeline.png`

---

## Dataset Structure
```text
data/
├── raw/
├── images/
├── annotations/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Conventions**
- Image naming: `scene_<id>.jpg` or `frame_<id>.jpg`
- Labels (YOLO): one file per image, same stem name
- Optional angle label appended after bbox tokens

**Coordinate system**
- Origin at top-left
- \(x\) grows right, \(y\) grows downward

---

## Data Collection
Recommended acquisition protocol:
- top-mounted camera with fixed optical axis normal to table
- homogeneous matte background to reduce false contours
- controlled illumination with diffusion where possible
- multi-condition capture:
  - different table colors/materials
  - varied exposure and white balance
  - multiple toothbrush orientations and positions

To improve robustness:
- include borderline hard samples (shadows, blur, reflections)
- avoid label leakage from highly repetitive scene templates
- balance orientation distribution to reduce angular bias

---

## Annotation Format
### YOLO label format
```text
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm> [angle_deg]
```

Example:
```text
0 0.532812 0.471354 0.285937 0.092708 37.5
```

### CSV format (for conversion pipelines)
```csv
image,x1,y1,x2,y2,class_id,angle_deg
frame_000123.jpg,412,265,701,332,0,37.5
```

### Angle convention
- Angle domain: \([0^\circ, 180^\circ)\)
- Orientation axis only (no directional head-tail disambiguation)

---

## Installation
### 1. Clone
```bash
git clone https://github.com/<your-org>/BrushPoseAI.git
cd BrushPoseAI
```

### 2. Create environment
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start
```bash
python src/cli.py validate-data --config configs/config.yaml
python src/cli.py run-classical --image data/images/sample.jpg --output outputs/images/classical_sample.png
python src/cli.py infer --method yolo --image data/images/sample.jpg --weights yolov8n.pt --output outputs/images/yolo_sample.png
python src/cli.py benchmark --config configs/config.yaml
```

Demo placeholders:
- `assets/demo.gif`
- `assets/results_example.png`

---

## Training
### Dataset preparation flow
1. Collect/extract images
2. Annotate bbox and optional angle
3. Convert annotations to YOLO labels
4. Split into train/val/test
5. Validate integrity and class balance

### YOLO training command
```bash
python src/cli.py train-yolo --config configs/yolo_config.yaml
```

### Typical training parameters

| Parameter | Meaning |
|---|---|
| `epochs` | number of optimization passes over training set |
| `imgsz` | input image resolution |
| `batch` | mini-batch size |
| `conf` | confidence threshold used in inference |
| `device` | CPU/GPU execution target |

Model artifacts and logs are saved to configured output directories.

---

## Inference
### Single image
```bash
python src/cli.py infer --method classical --image data/images/sample.jpg --output outputs/images/pred_classical.png
python src/cli.py infer --method yolo --image data/images/sample.jpg --weights yolov8n.pt --output outputs/images/pred_yolo.png
```

### Batch-oriented workflow
```bash
python src/cli.py evaluate --method classical --image-dir data/test/images --label-dir data/test/labels
python src/cli.py evaluate --method yolo --image-dir data/test/images --label-dir data/test/labels --weights yolov8n.pt
```

Generated outputs include overlay images, metric tables, and summary reports.

---

## Evaluation
Evaluation is performed on a held-out set with ground-truth labels.

Workflow:
1. run detector and pose estimator
2. match prediction vs ground truth
3. compute detection and pose quality metrics
4. aggregate statistics per method
5. export CSV/JSON/Markdown artifacts

Method comparison is performed through:
```bash
python src/cli.py benchmark --config configs/config.yaml
```

---

## Metrics
### Intersection-over-Union (IoU)
\[
\mathrm{IoU} = \frac{|B_{pred}\cap B_{gt}|}{|B_{pred}\cup B_{gt}|}
\]

### Center error
\[
e_c = \sqrt{(x_{pred}-x_{gt})^2 + (y_{pred}-y_{gt})^2}
\]

### Angle error (axial, 180° periodicity)
\[
e_\theta = \min\left(|\theta_{pred}-\theta_{gt}|,\ 180^\circ-|\theta_{pred}-\theta_{gt}|\right)
\]

### Additional metrics
- Precision / Recall
- mAP@0.5 (pipeline placeholder support)
- Median angle error
- Percentage of samples with \(e_\theta < 5^\circ\)
- Inference latency (ms/frame)
- Throughput (FPS)

Why these metrics matter:
- IoU validates localization quality
- center/angle errors validate geometric usability for manipulation
- latency/FPS determine real-time feasibility

---

## Example Outputs
Expected artifacts:
- predicted overlays with bbox + center + arrow + angle text
- histogram of angle errors
- benchmark comparison chart across methods
- machine-readable summaries

Placeholders:
- `assets/results_example.png`
- `assets/benchmark_plot.png`
- `assets/architecture.png`

---

## Limitations
- Sensitivity to strong illumination drift for color-segmentation pipeline
- Reduced reliability on cluttered/non-homogeneous backgrounds
- Angle ambiguity for near-symmetric visual states
- Partial occlusion can degrade contour and orientation estimates
- Cross-domain generalization depends on dataset diversity

---

## Future Improvements
- Instance segmentation-assisted orientation refinement
- Transformer-based detectors for better context modeling
- Keypoint-based pose estimation for directional disambiguation
- RGB-D integration for depth-aware pose
- Real-time optimization pipeline (TensorRT/ONNX/OpenVINO)
- ROS2 integration for robotic execution loops
- Edge deployment profiles for Jetson-class devices

---

## License
This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.

Attribution:
- Keep original license and copyright notice in derivative work
- Respect third-party dependency licenses

---

## Acknowledgements
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- Open-source computer vision community

---

## Command Examples
```bash
# Data
python src/cli.py prepare-data --config configs/config.yaml
python src/cli.py validate-data --config configs/config.yaml

# Training
python src/cli.py train-yolo --config configs/yolo_config.yaml

# Inference
python src/cli.py run-classical --image data/images/sample.jpg --output outputs/images/classical.png
python src/cli.py infer --method yolo --image data/images/sample.jpg --weights yolov8n.pt --output outputs/images/yolo.png

# Evaluation
python src/cli.py evaluate --method classical --image-dir data/test/images --label-dir data/test/labels
python src/cli.py evaluate --method yolo --image-dir data/test/images --label-dir data/test/labels --weights yolov8n.pt

# Benchmark & reports
python src/cli.py benchmark --config configs/config.yaml
python src/cli.py generate-report
```

