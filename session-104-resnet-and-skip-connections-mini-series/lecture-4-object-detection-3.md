# Object Detection 3: Why Detectors Use Rectangles

This lecture explains why object detection usually represents objects with axis-aligned rectangles instead of circles, ellipses, or arbitrary shapes.

---

## 1. Problem Formulation

Object detection localizes objects in an image:

$$
X \in \mathbb{R}^{H \times W \times C}
$$

The model predicts regions and semantic labels:

$$
\hat{Y} = \left\{\left(\hat{c}_i, \hat{b}_i, \hat{s}_i\right)\right\}_{i=1}^{M}
$$

where:

* $\hat{c}_i$ is the predicted class.
* $\hat{b}_i$ is the predicted region.
* $\hat{s}_i$ is the confidence score.

The standard region representation is the axis-aligned rectangle:

$$
b = \left(x_c, y_c, w, h\right)
$$

or equivalently:

$$
b = \left(x_{\min}, y_{\min}, x_{\max}, y_{\max}\right)
$$

The question is:

> Why is this simple rectangle so dominant?

---

## 2. Candidate Geometric Representations

### 2.1 Axis-Aligned Rectangle

An axis-aligned rectangle uses four parameters:

$$
b_{\text{rect}} = \left(x_c, y_c, w, h\right)
$$

Its sides are parallel to the image axes.

### 2.2 Circle

A circle uses three parameters:

$$
b_{\text{circle}} = \left(x_c, y_c, r\right)
$$

It assumes equal horizontal and vertical extent:

$$
w = h = 2r
$$

### 2.3 Ellipse

An ellipse can use five parameters:

$$
b_{\text{ellipse}} = \left(x_c, y_c, a, b, \theta\right)
$$

where:

* $a$ and $b$ are semi-axis lengths.
* $\theta$ is the rotation angle.

The ellipse boundary can be written as:

$$
\frac{\left(x - x_c\right)^2}{a^2}
+
\frac{\left(y - y_c\right)^2}{b^2}
=
1
$$

### 2.4 Pixel Mask

A mask predicts a binary or class label for each pixel:

$$
M \in \left\{0, 1\right\}^{H \times W}
$$

Masks are more flexible than boxes but also more expensive to annotate and predict.

---

## 3. Expressiveness Versus Practicality

Circles are too restrictive for most objects.

A person, bicycle, traffic sign, bottle, or car rarely has equal width and height. A circle often includes too much background or cuts off important parts.

Ellipses are more flexible:

$$
a \neq b
$$

and rotated ellipses can use:

$$
\theta \neq 0
$$

However, more flexibility is not automatically better for detection. The representation must be:

* easy to annotate;
* easy for a model to regress;
* easy to compare with ground truth;
* cheap to post-process;
* compatible with existing datasets and benchmarks.

Axis-aligned rectangles are not geometrically perfect, but they score well on all five requirements.

---

## 4. Parameter Count and Learning Stability

Compare the number of parameters:

| Shape | Parameters | Count |
| --- | --- | --- |
| Circle | $\left(x_c, y_c, r\right)$ | $3$ |
| Axis-aligned rectangle | $\left(x_c, y_c, w, h\right)$ | $4$ |
| Rotated rectangle | $\left(x_c, y_c, w, h, \theta\right)$ | $5$ |
| Ellipse | $\left(x_c, y_c, a, b, \theta\right)$ | $5$ |
| Mask | one value per pixel | $H \times W$ |

Rectangles are a good compromise:

* more expressive than circles;
* simpler than ellipses and masks;
* stable enough for regression.

Angles introduce extra difficulty because they are periodic:

$$
\theta \equiv \theta + 2\pi
$$

A small visual change can produce a large numerical change near the wrap-around boundary.

> [!WARNING]
> Angle regression is not impossible, but it requires careful parameterization. This is one reason ordinary object detection often avoids rotated boxes unless the task needs them.

---

## 5. Alignment with Image Grids

Convolutional feature maps live on grids:

$$
F \in \mathbb{R}^{H' \times W' \times D}
$$

Axis-aligned rectangles match this grid structure.

For a rectangle:

$$
x_{\min} \leq x \leq x_{\max}
$$

$$
y_{\min} \leq y \leq y_{\max}
$$

This makes many operations simple:

* cropping;
* indexing;
* pooling;
* clipping to image boundaries;
* computing width and height.

For an ellipse, membership requires checking:

$$
\frac{\left(x - x_c\right)^2}{a^2}
+
\frac{\left(y - y_c\right)^2}{b^2}
\leq
1
$$

This is still possible, but it is less aligned with the rectangular grid used by images and feature maps.

---

## 6. IoU Is Simple for Rectangles

Object detection relies heavily on overlap metrics.

For two boxes, Intersection over Union is:

$$
\operatorname{IoU}
=
\frac{
\operatorname{Area}\left(B_1 \cap B_2\right)
}{
\operatorname{Area}\left(B_1 \cup B_2\right)
}
$$

For axis-aligned rectangles, the intersection box is easy to compute:

$$
x_{\min}^{I} = \max\left(x_{\min}^{1}, x_{\min}^{2}\right)
$$

$$
y_{\min}^{I} = \max\left(y_{\min}^{1}, y_{\min}^{2}\right)
$$

$$
x_{\max}^{I} = \min\left(x_{\max}^{1}, x_{\max}^{2}\right)
$$

$$
y_{\max}^{I} = \min\left(y_{\max}^{1}, y_{\max}^{2}\right)
$$

Then:

$$
w_I = \max\left(0, x_{\max}^{I} - x_{\min}^{I}\right)
$$

$$
h_I = \max\left(0, y_{\max}^{I} - y_{\min}^{I}\right)
$$

$$
\operatorname{Area}\left(B_1 \cap B_2\right) = w_I h_I
$$

The computation uses only min, max, subtraction, and multiplication.

For ellipses, intersections may require nonlinear geometry or numerical approximation. That makes training, evaluation, and post-processing more complex.

---

## 7. Annotation and Human Effort

Rectangles are fast for humans to draw.

An annotator usually needs only:

1. Click or drag around the object.
2. Adjust the four sides.
3. Assign a class label.

Ellipses, rotated boxes, or masks require more careful boundary decisions. This increases cost and often decreases consistency between annotators.

Large datasets became possible partly because rectangular boxes are cheap enough to annotate at scale.

---

## 8. Dataset and Benchmark Inertia

Most major detection datasets store rectangular boxes:

* COCO;
* Pascal VOC;
* Open Images.

Evaluation tools, leaderboards, model implementations, and pre-trained checkpoints are built around rectangular IoU and rectangular annotations.

Changing the representation would require:

* new annotation tools;
* new dataset formats;
* new metrics;
* new post-processing algorithms;
* re-labeling or converting existing data.

The rectangle is therefore not only a modeling choice. It is an ecosystem standard.

---

## 9. When Rectangles Are Not Enough

Rectangles are not always the best representation.

### 9.1 Rotated Object Detection

For aerial images, shipping containers, documents, or text lines, orientation matters.

A rotated rectangle can be represented as:

$$
b_{\text{rot}} = \left(x_c, y_c, w, h, \theta\right)
$$

This can reduce background area and improve localization for long rotated objects.

### 9.2 Instance Segmentation

When shape matters, a mask is better:

$$
M_i \in \left\{0, 1\right\}^{H \times W}
$$

Masks can represent holes, curves, thin structures, and irregular boundaries.

### 9.3 Keypoint Detection

For pose or landmark tasks, points may be more useful than boxes:

$$
K = \left\{\left(x_j, y_j\right)\right\}_{j=1}^{J}
$$

Examples:

* human joints;
* facial landmarks;
* hand keypoints.

---

## 10. Relation to Segmentation

Bounding boxes are coarse object localization. Segmentation is dense object localization.

Detection output:

$$
\left(c, b, s\right)
$$

Segmentation output:

$$
Y \in \left\{0, 1, \ldots, K\right\}^{H \times W}
$$

Detection asks:

> Which rectangular region contains the object?

Segmentation asks:

> Which pixels belong to each class or object?

This is why segmentation needs stronger spatial reconstruction, which leads naturally to encoder-decoder networks and U-Net.

---

## 11. Practical Rule

Use axis-aligned rectangles when:

* object location is enough;
* fast annotation matters;
* standard detection datasets or benchmarks are used;
* the objects are reasonably captured by bounding boxes.

Use richer representations when:

* orientation matters;
* precise boundaries matter;
* the object shape is thin, curved, or irregular;
* downstream decisions depend on exact geometry.

---

## 12. Summary

Rectangles dominate object detection because they balance modeling power and practicality.

They are:

* expressive enough for many objects;
* simple to regress;
* aligned with image grids;
* efficient for IoU and NMS;
* fast to annotate;
* compatible with major datasets and benchmarks.

The next lecture moves from rectangular localization to pixel-level prediction with image segmentation.
