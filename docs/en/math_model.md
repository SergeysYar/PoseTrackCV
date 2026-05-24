# Mathematical Model
For bounding box \(B=(x_1,y_1,x_2,y_2)\), center is:
\[
x_c=\frac{x_1+x_2}{2},\quad y_c=\frac{y_1+y_2}{2}
\]

PCA orientation is computed from contour covariance:
\[
\Sigma=\frac{1}{N-1}\sum_{i=1}^{N}(p_i-\bar p)(p_i-\bar p)^T
\]
Principal eigenvector \(v_1\) defines angle:
\[
\theta=\mathrm{atan2}(v_{1y},v_{1x})\in[0,180)
\]

Intersection-over-Union:
\[
\mathrm{IoU}=\frac{|B_p\cap B_{gt}|}{|B_p\cup B_{gt}|}
\]

Angle error:
\[
\Delta_\theta=\min\left(|\theta_p-\theta_{gt}|,\ 180-|\theta_p-\theta_{gt}|\right)
\]

