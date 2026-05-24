# BrushPose AI Final Report (Template)
**Project Type:** Computer Vision Engineering Case Study  
**Version:** [INSERT VERSION]  
**Date:** [INSERT DATE]  
**Author(s):** [INSERT AUTHOR NAMES]

---

## 1. Introduction
BrushPose AI addresses a practical object-pose estimation problem: detect a toothbrush in a tabletop scene and estimate its geometric state for downstream automation tasks.  
The project is motivated by real engineering needs where object localization alone is insufficient and orientation-aware reasoning is required.

This report should summarize:
- implemented methods
- quantitative performance
- runtime behavior
- failure modes
- production relevance

**Project goals (fill in):**
- [INSERT GOAL 1]
- [INSERT GOAL 2]
- [INSERT GOAL 3]

**Practical application context (fill in):**
- [INSERT APPLICATION DOMAIN]
- [INSERT DEPLOYMENT SCENARIO]

---

## 2. Problem Statement
### Task Definition
Input: top-view RGB image of a tabletop scene.  
Output:
- object bounding box \((x_{min}, y_{min}, x_{max}, y_{max})\)
- object center \((x_c, y_c)\)
- orientation angle \(\theta \in [0^\circ, 180^\circ)\)

### Environment and Constraints
- scene type: [INSERT SCENE TYPE]
- camera setup: [INSERT CAMERA SETUP]
- background assumptions: [INSERT BACKGROUND ASSUMPTION]
- photometric variability: [INSERT LIGHTING CONDITIONS]

### Required Accuracy
Define acceptable thresholds for your target use case.

| Parameter | Description |
|-----------|-------------|
| Detection success rate | [INSERT TARGET] |
| IoU threshold | [INSERT THRESHOLD] |
| Center error tolerance (px) | [INSERT THRESHOLD] |
| Angle error tolerance (deg) | [INSERT THRESHOLD] |

---

## 3. Dataset Description
### Data Acquisition
Describe capture procedure:
- camera position and height
- tabletop material/background
- lighting strategy
- image diversity strategy

### Annotation Protocol
Annotated fields:
- filename
- bbox coordinates
- center coordinates
- angle
- class label

### Split Strategy
- train: [INSERT %]
- val: [INSERT %]
- test: [INSERT %]
- random seed: [INSERT SEED]

### Dataset Statistics (fill in)
| Item | Value |
|------|-------|
| Total images | [INSERT VALUE] |
| Train images | [INSERT VALUE] |
| Val images | [INSERT VALUE] |
| Test images | [INSERT VALUE] |
| Typical resolution | [INSERT VALUE] |
| Number of classes | [INSERT VALUE] |

### Distribution Placeholders
- Class distribution: [INSERT TABLE HERE]
- Angle distribution: [INSERT HISTOGRAM HERE]
- BBox area distribution: [INSERT PLOT HERE]

### Dataset Structure
```text
data/
  images/
  annotations/annotations.csv
  train/images, train/labels
  val/images, val/labels
  test/images, test/labels
```

Figure placeholders:
- [INSERT SAMPLE DATASET IMAGE]
- [INSERT ANNOTATION EXAMPLE IMAGE]

---

## 4. Methods
### 4.1 Classical CV Pipeline
Stages:
- HSV segmentation
- morphology
- contour extraction/filtering
- center estimation
- orientation via minAreaRect or PCA

Expected strengths:
- interpretable behavior
- low compute overhead

Expected weaknesses:
- sensitivity to scene and lighting drift

### 4.2 YOLO-Based Pipeline
Stages:
- detector inference
- bbox extraction
- geometric center recovery
- optional orientation post-processing (if available)

Expected strengths:
- stronger generalization under visual variability

Expected weaknesses:
- higher compute cost and dependency on trained weights

### Method Diagram Placeholders
- [INSERT CLASSICAL PIPELINE DIAGRAM]
- [INSERT YOLO PIPELINE DIAGRAM]
- [INSERT STAGE-BY-STAGE PROCESSING VISUAL]

---

## 5. Mathematical Model (Summary)
This section is intentionally concise. Full derivations are documented in `docs/en/math_model.md`.

### Core Equations
\[
IoU = \frac{|B_{pred}\cap B_{gt}|}{|B_{pred}\cup B_{gt}|}
\]

\[
e_\theta = \min\left(|\theta_{pred}-\theta_{gt}|,\ 180^\circ-|\theta_{pred}-\theta_{gt}|\right)
\]

\[
e_c = \sqrt{(x_{pred}-x_{gt})^2 + (y_{pred}-y_{gt})^2}
\]

Additional notes:
- orientation is treated as axial (\(180^\circ\) periodicity)
- bbox and center metrics are jointly analyzed

---

## 6. Software Architecture
BrushPose AI is organized as modular subsystems:
- dataset
- detection (classical + YOLO)
- evaluation
- visualization
- benchmarking
- unified CLI

### Component Interaction (Template)
```text
Dataset -> Inference Method -> Prediction CSV -> Evaluation -> Reports/Plots
```

Architecture placeholders:
- [INSERT ARCHITECTURE DIAGRAM]
- [INSERT MODULE DEPENDENCY SCHEME]

---

## 7. Experiments
### Experimental Setup
- OS: [INSERT]
- Python: [INSERT]
- OpenCV: [INSERT]
- Ultralytics: [INSERT]
- CPU: [INSERT]
- GPU: [INSERT]

### Inference / Evaluation Configuration
- IoU threshold: [INSERT]
- angle threshold: [INSERT]
- confidence threshold: [INSERT]
- image size: [INSERT]

| Experiment | Method | Parameters | Notes |
|------------|--------|------------|-------|
| Exp-01 | classical_min_area_rect | [INSERT PARAMETERS] | [INSERT NOTE] |
| Exp-02 | classical_pca | [INSERT PARAMETERS] | [INSERT NOTE] |
| Exp-03 | yolo_geometric | [INSERT PARAMETERS] | [INSERT NOTE] |

---

## 8. Metrics
Primary metrics:
- detection accuracy
- IoU
- center error (px)
- angle error (deg)
- processing time (ms)
- FPS

Interpretation guidance:
- higher IoU and detection accuracy are better
- lower center/angle errors are better
- lower latency and higher FPS are better

| Metric | Classical CV | PCA | YOLO |
|--------|---------------|-----|------|
| Detection Accuracy | [INSERT] | [INSERT] | [INSERT] |
| Mean IoU | [INSERT] | [INSERT] | [INSERT] |
| Mean Center Error (px) | [INSERT] | [INSERT] | [INSERT] |
| Mean Angle Error (deg) | [INSERT] | [INSERT] | [INSERT/UNAVAILABLE] |
| FPS | [INSERT] | [INSERT] | [INSERT] |

---

## 9. Comparative Analysis
Summarize trade-offs among methods:
- localization quality
- orientation quality
- speed
- interpretability
- engineering complexity

### Comparison Placeholder Table
| Method | Localization Quality | Orientation Quality | Runtime | Robustness | Interpretability |
|--------|----------------------|---------------------|---------|------------|------------------|
| classical_min_area_rect | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| classical_pca | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| yolo_geometric | [INSERT] | [INSERT/UNAVAILABLE] | [INSERT] | [INSERT] | [INSERT] |

Placeholders:
- [INSERT BENCHMARK BAR CHART]
- [INSERT METHOD COMPARISON PLOT]
- [INSERT QUALITATIVE SIDE-BY-SIDE VISUALIZATION]

Recommended use cases per method:
- [INSERT RECOMMENDATION]

---

## 10. Error Analysis
Analyze representative failure modes:
- missed detections
- poor overlap (low IoU)
- center drift
- angle instability
- contour fragmentation
- lighting sensitivity
- YOLO false positives/negatives

| Error Type | Cause | Frequency | Example |
|------------|------|-----------|---------|
| missed_detection | [INSERT CAUSE] | [INSERT] | [INSERT FILE] |
| low_iou | [INSERT CAUSE] | [INSERT] | [INSERT FILE] |
| high_center_error | [INSERT CAUSE] | [INSERT] | [INSERT FILE] |
| high_angle_error | [INSERT CAUSE] | [INSERT] | [INSERT FILE] |
| invalid_prediction | [INSERT CAUSE] | [INSERT] | [INSERT FILE] |

Failure-case placeholders:
- [INSERT WORST-CASE IMAGE 1]
- [INSERT WORST-CASE IMAGE 2]
- [INSERT ERROR DISTRIBUTION HISTOGRAM]

---

## 11. Conclusion
BrushPose AI demonstrates an end-to-end CV workflow for object localization and orientation estimation with modular engineering design and reproducible evaluation.

Include final conclusions:
- achieved quality levels
- strongest method by criterion
- runtime feasibility
- production relevance

**Final outcome summary (fill in):**
- [INSERT KEY RESULT 1]
- [INSERT KEY RESULT 2]
- [INSERT KEY RESULT 3]

---

## 12. Future Work
Planned technical extensions:
- rotated bounding boxes
- keypoint-based orientation modeling
- segmentation-assisted pose extraction
- transformer detectors
- depth-aware estimation
- ROS integration
- TensorRT/ONNX optimization
- edge deployment profiles
- synthetic data augmentation
- real-time robotic integration

Prioritized roadmap (fill in):
1. [INSERT PRIORITY 1]
2. [INSERT PRIORITY 2]
3. [INSERT PRIORITY 3]

---

## Reproducibility (Quick Commands)
```bash
# dataset preparation
python src/cli.py prepare-data --mode split --images-dir data/images --annotations data/annotations/annotations.csv --output-dir data --format both

# training
python src/cli.py train-yolo --data data/yolo_dataset/dataset.yaml --model yolov8n.pt --epochs 50 --validate

# inference
python src/cli.py infer --method classical --input data/test/images --output-dir outputs/images/classical --csv-out outputs/metrics/classical_predictions.csv --angle-method pca
python src/cli.py infer --method yolo --weights runs/brushpose_yolo/train/weights/best.pt --input data/test/images --output-dir outputs/images/yolo --csv-out outputs/metrics/yolo_predictions.csv

# evaluation
python src/cli.py evaluate --ground-truth data/test/labels/annotations.csv --predictions outputs/metrics/classical_predictions.csv --output-dir outputs/reports/classical_eval --method-name classical_pca --report-format both

# benchmark
python src/cli.py benchmark --images-dir data/test/images --ground-truth data/test/labels/annotations.csv --output-dir outputs --language both --skip-yolo-if-missing
```

---

## Appendix Placeholders
- [INSERT TABLE HERE: Detailed Per-Sample Metrics]
- [INSERT BENCHMARK RESULTS]
- [INSERT YOLO DETECTION RESULT]
- [INSERT PCA ORIENTATION VISUALIZATION]
- [INSERT RUNTIME COMPARISON TABLE]

