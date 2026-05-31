# K-Nearest Neighbors (KNN)

## *NOT for clustering*


![](./img-word2vec/knnc.jpg)

---

## 1. Why KNN Here?

After **K-Means**, it is natural to encounter another “K-based” method: **K-Nearest Neighbors (KNN)**.

Despite the similar name:

* **K-Means → Clustering (unsupervised)**
* **KNN → Prediction (supervised)**

This confusion is extremely common. So we address it early.

---

## 2. Core Idea

KNN is based on a simple principle:

> Similar inputs should have similar outputs.

Given a new data point $x$, we:

1. Compute its distance to all training points
2. Select the **K nearest neighbors**
3. Aggregate their labels to make a prediction

---

## 3. Distance Metric

The notion of “nearest” depends on the distance function.

Most common choice: **Euclidean distance**

$$
d(x, x_i) = \sqrt{\sum_{j=1}^{d} (x_j - x_{i,j})^2}
$$

Other options:

* Manhattan distance
* Cosine distance (important for embeddings later)

---

## 4. KNN for Classification

![](./img-word2vec/knna.jpg)


Each neighbor has a label.

Prediction is done via **majority voting**:

$$
\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_K(x)} \mathbf{1}(y_i = c)
$$

Interpretation:

* Look at K neighbors
* Count how many belong to each class
* Pick the most frequent class

---

## 5. KNN for Regression

Instead of labels, we have continuous values.

Prediction becomes **averaging**:

$$
\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(x)} y_i
$$

Optionally, use **distance-weighted averaging**:

$$
\hat{y} = \frac{\sum_{i} w_i y_i}{\sum_i w_i}, \quad w_i = \frac{1}{d(x, x_i)}
$$

![](./img-word2vec/knnr.jpg)


---

## 6. Key Characteristics

### No Training Phase

KNN is a **lazy learner**:

* No explicit model is learned
* All computation happens at inference time

---

### Non-Parametric

* No fixed number of parameters
* Model complexity grows with data

---

### Local Decision Making

Predictions depend only on nearby points:

* Highly flexible
* Can capture complex decision boundaries

---

## 7. Choosing K

![](./img-word2vec/knnb.jpg)


K controls the bias-variance tradeoff:

* Small K:

  * Low bias
  * High variance (sensitive to noise)

* Large K:

  * High bias
  * Low variance (over-smoothing)

---

## 8. KNN vs K-Means

| Aspect       | KNN                 | K-Means            |
| ------------ | ------------------- | ------------------ |
| Type         | Supervised          | Unsupervised       |
| Goal         | Predict label/value | Discover clusters  |
| Uses labels? | Yes                 | No                 |
| Output       | Prediction          | Cluster assignment |
| Computation  | At inference        | During training    |
