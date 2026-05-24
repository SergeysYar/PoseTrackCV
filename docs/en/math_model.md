# BrushPose AI Mathematical Model

## 1. Introduction
This document formalizes the geometric and detection model behind **BrushPose AI**, a computer vision system for toothbrush localization and orientation estimation in tabletop scenes.  
The objective is not only to detect object presence, but to estimate a physically meaningful pose descriptor:
\[
\mathbf{p} = (x_c, y_c, \theta)
\]
where \((x_c, y_c)\) is the center and \(\theta\) is orientation.

Pose estimation is central for manipulation-aware vision pipelines. In industrial workflows (pick-and-place, inspection, robotic pre-grasping), a class label alone is insufficient; downstream control requires metric localization and angular alignment.

---

## 2. Problem Formulation
Let input image be:
\[
I \in \mathbb{R}^{H \times W \times 3}
\]
with height \(H\), width \(W\), and RGB channels.

Define:
- \(\Omega = \{(x,y)\mid x\in[0,W-1], y\in[0,H-1]\}\): image domain
- \(M:\Omega \rightarrow \{0,1\}\): object mask (foreground/background)
- \(B\): bounding box
- \(\mathbf{p}=(x_c,y_c,\theta)\): pose output

Coordinate system:
- origin at top-left pixel
- \(x\)-axis to the right
- \(y\)-axis downward
- angle domain: \(\theta \in [0^\circ,180^\circ)\)

Prediction target:
\[
f(I) \mapsto \hat{\mathbf{p}} = (\hat{x}_c,\hat{y}_c,\hat{\theta})
\]

---

## 3. Object Localization Task
Object localization can be interpreted as estimating a compact support region \(R \subset \Omega\) corresponding to the toothbrush.  
In classical CV, \(R\) is induced by segmentation and contour extraction. In YOLO-style detection, \(R\) is approximated by a learned bounding box regressor.

From a geometric perspective, localization quality determines:
- center estimation stability
- orientation estimation reliability
- downstream metric consistency

Foreground/background separation is critical because orientation is computed from shape geometry; segmentation leakage directly perturbs principal-axis estimation.

---

## 4. Bounding Box Representation
### Axis-aligned bounding box
\[
B = (x, y, w, h)
\]
where \((x,y)\) is top-left corner, \(w\) width, \(h\) height.

Equivalent corner form:
\[
B = (x_1,y_1,x_2,y_2), \quad x_2 = x_1+w,\ y_2=y_1+h
\]

Center from axis-aligned box:
\[
x_c = x + \frac{w}{2}, \qquad y_c = y + \frac{h}{2}
\]

### Rotated bounding box
A rotated rectangle may be parameterized as:
\[
B_r = (x_c, y_c, w, h, \phi)
\]
where \(\phi\) is rectangle orientation.

### YOLO-normalized representation
\[
\tilde{x}_c=\frac{x_c}{W},\quad \tilde{y}_c=\frac{y_c}{H},\quad \tilde{w}=\frac{w}{W},\quad \tilde{h}=\frac{h}{H}
\]

| Variable | Domain | Meaning |
|---|---|---|
| \(x,y\) | pixels | top-left of AABB |
| \(w,h\) | pixels | box size |
| \(x_c,y_c\) | pixels | box center |
| \(\tilde{x}_c,\tilde{y}_c,\tilde{w},\tilde{h}\) | \([0,1]\) | normalized YOLO targets |
| \(\phi,\theta\) | degrees | orientation angles |

---

## 5. Center Coordinate Calculation
BrushPose AI supports multiple center definitions.

### 5.1 Bounding-box center
\[
x_c = x + \frac{w}{2}, \qquad y_c = y + \frac{h}{2}
\]
Fast and robust, but may be biased when contour support is asymmetric.

### 5.2 Geometric contour centroid
Given contour points \(\{(x_i,y_i)\}_{i=1}^N\):
\[
x_c = \frac{1}{N}\sum_{i=1}^{N}x_i,\qquad y_c=\frac{1}{N}\sum_{i=1}^{N}y_i
\]

### 5.3 Moments-based center
For binary mask \(M(x,y)\):
\[
M_{ij}=\sum_x\sum_y x^i y^j M(x,y)
\]
\[
x_c=\frac{M_{10}}{M_{00}},\qquad y_c=\frac{M_{01}}{M_{00}}
\]
where \(M_{00}\) is foreground area.

### 5.4 Numerical considerations
- if \(M_{00}\approx 0\), centroid is unstable
- small holes/noise in mask perturb higher moments
- contour smoothing and morphology improve stability

---

## 6. Orientation Angle Definition
Orientation angle is defined w.r.t. image \(x\)-axis.  
Because toothbrush appearance is often approximately axial, orientation is modeled with \(180^\circ\) periodicity:
\[
\theta \in [0^\circ,180^\circ)
\]

Equivalent class:
\[
\theta \sim \theta + 180^\circ
\]

Conceptual diagram:
```text
y (down)
|
|      / object major axis
|    /
|  /  theta
+--------------------> x (right)
```

Ambiguity sources:
- axial symmetry (head-tail indistinguishability)
- partial visibility
- contour truncation near image boundaries

---

## 7. Contour-Based Orientation Estimation
Classical pipeline:
1. color-space conversion (BGR→HSV)
2. thresholding to obtain \(M\)
3. morphological filtering
4. contour extraction \( \partial R \)
5. contour selection by area/shape criteria

`minAreaRect` fits minimum-area rotated rectangle around contour:
\[
B_r^\star = \arg\min_{B_r \supset \partial R} \text{Area}(B_r)
\]
Its major edge direction yields orientation estimate \(\hat{\theta}_{rect}\).

Practical behavior:
- robust for elongated clean contours
- sensitive to contour fragmentation and outliers
- angle flips may occur near \(w \approx h\)

---

## 8. PCA-Based Orientation Estimation
Given contour points \(\mathbf{q}_i=[x_i,y_i]^T\), mean:
\[
\bar{\mathbf{q}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{q}_i
\]
Covariance:
\[
C=\frac{1}{N-1}\sum_{i=1}^{N}(\mathbf{q}_i-\bar{\mathbf{q}})(\mathbf{q}_i-\bar{\mathbf{q}})^T
=
\begin{bmatrix}
\sigma_x^2 & \sigma_{xy}\\
\sigma_{xy} & \sigma_y^2
\end{bmatrix}
\]
Eigen decomposition:
\[
C\mathbf{v}_k=\lambda_k\mathbf{v}_k,\quad \lambda_1\ge\lambda_2
\]
Dominant eigenvector \(\mathbf{v}_1=(v_x,v_y)\) defines principal axis:
\[
\hat{\theta}_{pca}=\operatorname{atan2}(v_y,v_x)
\]
Angle is normalized to \([0^\circ,180^\circ)\).

Why PCA works:
- estimates direction of maximal spatial variance
- naturally matches elongated object geometry
- invariant to translation

Limitations:
- unstable for near-isotropic shapes
- sensitive to large contour outliers

---

## 9. Ellipse Fitting Approach
Ellipse model approximates contour with conic:
\[
\mathcal{E}:\ Ax^2+Bxy+Cy^2+Dx+Ey+F=0
\]
fitted by least squares under ellipse constraints.

Orientation of fitted ellipse:
\[
\theta_{ell}=\frac{1}{2}\arctan\left(\frac{B}{A-C}\right)
\]
Major axis direction gives object orientation proxy.

Comparison vs PCA:
- ellipse fitting can smooth irregular contours
- requires sufficient contour points and stable fit
- may degrade for heavily non-elliptic silhouettes

---

## 10. YOLO-Based Detection Formulation
YOLO predicts object hypotheses directly from multi-scale feature maps.  
A generic prediction tuple is:
\[
(x, y, w, h, c, \mathbf{p}_{cls})
\]
where \(c\) is confidence and \(\mathbf{p}_{cls}\) class probabilities.

For single-class setup (toothbrush), this simplifies to:
\[
(x,y,w,h,c)
\]

Training objective combines:
- localization loss \(L_{box}\)
- objectness/confidence loss \(L_{obj}\)
- classification loss \(L_{cls}\)

Total:
\[
L = \lambda_{box}L_{box}+\lambda_{obj}L_{obj}+\lambda_{cls}L_{cls}
\]

BrushPose AI derives orientation after detection from ROI geometry (e.g., PCA on segmented ROI), producing a hybrid detector+geometry pipeline.

---

## 11. Intersection over Union (IoU)
\[
\operatorname{IoU}(B_p,B_g)=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}
\]

Interpretation:
- IoU \(\rightarrow 1\): strong overlap
- IoU \(\rightarrow 0\): poor localization

Thresholding:
- prediction considered matched if \(\operatorname{IoU}\ge \tau\)
- common \(\tau = 0.5\)

IoU is sensitive to both translation and scale mismatch.

---

## 12. mAP@0.5 Metric
For each confidence threshold:
- compute precision \(P=\frac{TP}{TP+FP}\)
- compute recall \(R=\frac{TP}{TP+FN}\)

Construct precision-recall curve and compute average precision:
\[
AP = \int_0^1 P(R)\,dR
\]

For multiple classes:
\[
mAP = \frac{1}{K}\sum_{k=1}^{K} AP_k
\]
In BrushPose AI (single class), mAP@0.5 corresponds to AP at IoU threshold \(0.5\).

Importance:
- combines confidence ranking and localization quality
- better reflects detector quality than raw accuracy

---

## 13. Angle Error Metrics
Naive absolute error:
\[
e_\theta = |\hat{\theta}-\theta^{gt}|
\]
For axial periodicity:
\[
e_\theta^{(180)} = \min\left(e_\theta, 180^\circ-e_\theta\right)
\]

Mean absolute angle error:
\[
MAAE = \frac{1}{N}\sum_{i=1}^{N} e_{\theta,i}^{(180)}
\]

Median angle error:
\[
MedAE = \operatorname{median}\left(\{e_{\theta,i}^{(180)}\}_{i=1}^{N}\right)
\]

Additional robustness indicator:
\[
P_{<5^\circ} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[e_{\theta,i}^{(180)}<5^\circ]
\]

---

## 14. Inference Pipeline
```text
Image Load
   -> Preprocess
      -> Detection / Segmentation
         -> Contour/ROI Extraction
            -> Center Estimation
               -> Orientation Estimation
                  -> Visualization
                     -> Metric Logging
```

Stage outputs:
- image tensor / matrix
- candidate region(s)
- geometric primitives (contours, rectangles, moments)
- final pose \((x_c,y_c,\theta)\)
- report artifacts (CSV/JSON/plots)

---

## 15. Computational Complexity
Classical CV (per image, approximate):
- threshold + morphology: \(O(HW)\)
- contour extraction: \(O(HW)\)
- PCA over \(N\) contour points: \(O(N)\) for covariance, eigendecomposition of \(2\times2\) is constant-time

YOLO inference:
- dominated by CNN forward pass
- complexity depends on architecture depth/width and resolution
- typically higher compute cost but better robustness under variability

Memory profile:
- classical pipeline: low memory footprint
- CNN pipeline: model weights + feature maps (significantly larger)

---

## 16. Error Sources and Limitations
Primary failure factors:
- illumination shifts and specular highlights
- non-homogeneous backgrounds
- mask leakage and contour breakage
- motion blur and sensor noise
- partial occlusions
- pose ambiguity for symmetric projections

Generalization risks:
- domain shift between training and deployment scenes
- under-represented orientations or backgrounds in dataset

---

## 17. Comparative Analysis of Methods
| Method | Accuracy Potential | Robustness | Speed | Interpretability | Compute Cost |
|---|---|---|---|---|---|
| Contour + `minAreaRect` | medium-high in controlled scenes | medium | high | very high | low |
| PCA on contour | high for elongated clean shapes | medium | high | high | low |
| Ellipse fitting | medium-high on smooth contours | medium | medium | medium | low-medium |
| YOLO + geometric angle | high in variable conditions | high | medium | medium | medium-high |

Engineering interpretation:
- classical methods are transparent and efficient
- deep models are more robust to photometric variability
- hybrid detection + geometry offers practical balance

---

## 18. Conclusion
BrushPose AI formulates toothbrush pose estimation as a joint localization-and-orientation problem in image space.  
The mathematical framework combines:
- geometric center estimation
- principal-axis orientation estimation
- detector quality metrics (IoU, mAP@0.5)
- angular error metrics with periodic normalization

This dual-path formulation (classical CV and YOLO-based detection) enables both interpretability and robustness, supporting research-grade benchmarking and engineering-oriented deployment analysis.

