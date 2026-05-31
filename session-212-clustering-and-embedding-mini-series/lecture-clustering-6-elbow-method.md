# Elbow Method

---

## 1. Motivation: Choosing the “right” number of components

Many unsupervised learning algorithms require a key hyperparameter: the number of clusters or components.

Examples:

* K-means: number of clusters $K$
* Gaussian Mixture Models (GMM): number of Gaussian components
* Hierarchical clustering: where to cut the dendrogram
* PCA: number of principal components to keep

The Elbow Method provides a simple heuristic to select this value by analyzing how model fit improves as complexity increases.

---

## 2. Core idea of the Elbow Method

The central idea is:

As model complexity increases, the fit improves, but with diminishing returns.

We look for the “elbow point” where adding more complexity no longer gives significant improvement.

---

## 3. Example: K-means objective

In K-means, we minimize the within-cluster sum of squares (WCSS), also called inertia:

$$
J(K) = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

As $K$ increases:

* $J(K)$ always decreases
* but the marginal gain decreases

We plot:

* x-axis: number of clusters $K$
* y-axis: inertia $J(K)$

The “elbow” is the point where the curve starts flattening.

---

## 4. General intuition across models

The same principle applies beyond K-means:

Hierarchical clustering:

* Look for a sharp change in merge distance
* Cut the dendrogram at the elbow-like point

Gaussian Mixture Models (GMM):

* Look for diminishing improvement as components increase

PCA:

* Plot explained variance vs number of components
* Choose the point where additional components add little variance

Early stopping (deep learning):

* Monitor validation loss
* Stop when improvement becomes marginal or unstable
