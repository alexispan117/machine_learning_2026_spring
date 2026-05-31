#  K-Means

## 1. Clustering vs. Classification

| Feature  | Classification (Supervised)       | Clustering (Unsupervised)                |
| -------- | --------------------------------- | ---------------------------------------- |
| Input    | Data + Labels                     | Data only                                |
| Goal     | Predict known categories          | Discover unknown groupings               |
| Use Case | Spam detection, medical diagnosis | Customer segmentation, anomaly detection |

---

## 2. K-Means Objective

![](./img-embedding/kmeans2.gif)


In K-means, we minimize the within-cluster sum of squares (WCSS), also called inertia:

$$
J(K) = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
$$


---

## 3. Pseudocode for K-Means

![](./img-embedding/kmeans.webp)


```text
Input: data points X, number of clusters K
Initialize: randomly select K cluster centers

repeat
    for each data point x in X:
        assign x to the nearest cluster center

    for each cluster:
        update center as the mean of assigned points

until convergence (centers do not change significantly)
Output: cluster assignments, cluster centers
```


Think of **K-Means** as placing **flags in a park**:

1. **Randomly place K flags** on the field.
2. **Assign each person to the nearest flag**.
3. **Move each flag to the average location** of its assigned people.
4. Repeat steps 2-3 until flags stop moving.

**Result:** Each flag represents the center of a cluster, and each person is assigned to their closest cluster.


---

## 4. When K-Means Works Well

K-Means is **fast and effective** when:

* Clusters are roughly **spherical** (convex)
* Clusters are roughly **equal in size**
* Distance metric (usually Euclidean) is meaningful

---

## 5. Limitations of K-Means

* Assumes **round, convex clusters**
* Fails on **non-linear or irregular shapes** (e.g., moons, spirals)

**Visual Intuition:**
K-Means draws straight lines between cluster centers. For curved or intertwined shapes, it slices through them incorrectly.

---

## 6. How to Decide K — The Elbow Method

Concept:

1. Compute the **within-cluster sum of squared distances** (inertia) for K=1,2,...
2. Plot inertia vs. K.
3. Look for the **"elbow"**: a point where adding more clusters yields diminishing returns.

> The elbow often suggests the optimal number of clusters.
