# DBSCAN

DBSCAN is a density-based clustering method. It finds connected dense regions and marks isolated points as noise.

![](./img-embedding/dbscan.gif)

---

## 1. Motivation: When Centroids Are the Wrong Tool

K-means and GMMs describe clusters using centers. That works for compact groups, but it can fail when the structure is curved, irregular, or contaminated by outliers.

DBSCAN asks a different question:

> Which points belong to the same dense region?

This lets it discover non-convex clusters and identify noise points.

---

## 2. Parameters

DBSCAN has two main hyperparameters:

- $\epsilon$: neighborhood radius.
- $m$: minimum number of points required to form a dense neighborhood.

For a point $x^{(i)}$, define its $\epsilon$-neighborhood as:

$$
N_{\epsilon}(x^{(i)})
=
\left\{
x^{(j)}
:
\left\|x^{(j)} - x^{(i)}\right\|_2 \le \epsilon
\right\}
$$

The size of this neighborhood is $\left|N_{\epsilon}(x^{(i)})\right|$.

---

## 3. Core, Border, and Noise Points

![](./img-embedding/dbscan2.gif)

DBSCAN classifies points into three types.

| Type | Definition | Meaning |
| --- | --- | --- |
| Core point | $\left\|N_{\epsilon}(x^{(i)})\right\| \ge m$ | Point lies inside a dense region |
| Border point | Not core, but within $\epsilon$ of a core point | Point lies on the edge of a dense region |
| Noise point | Neither core nor border | Point is treated as an outlier |

Clusters are formed by connecting core points that are reachable through chains of nearby core points, then adding border points around them.

---

## 4. Algorithm

```text
Input: data matrix X, radius epsilon, minimum count m
Initialize every point as unvisited

for each unvisited point:
    mark it as visited
    find its epsilon-neighborhood

    if the neighborhood has fewer than m points:
        mark the point as noise for now
    else:
        start a new cluster
        recursively add all density-reachable points

Output: clusters and noise points
```

A point initially marked as noise can later become a border point if it is reached from a core point.

---

## 5. Choosing Parameters

The parameter $\epsilon$ controls the scale of density.

- If $\epsilon$ is too small, many points become noise.
- If $\epsilon$ is too large, separate clusters merge.

The parameter $m$ controls how strict the density requirement is.

- Larger $m$ requires stronger local evidence.
- Smaller $m$ allows smaller clusters but is more sensitive to noise.

> [!INFO]
> A common starting point is to inspect the sorted distance to each point's $m$-th nearest neighbor. A sharp bend can suggest a useful value of $\epsilon$.

---

## 6. Strengths and Limitations

DBSCAN is strong when:

- clusters have irregular shapes;
- noise and outliers matter;
- the number of clusters is unknown;
- local density is meaningful.

DBSCAN struggles when:

- clusters have very different densities;
- distance loses meaning in high dimensions;
- $\epsilon$ is hard to choose;
- the dataset is very large without efficient neighbor search.

> [!WARNING]
> DBSCAN does not remove the need for a good representation. It changes the clustering rule, but it still depends on meaningful neighborhoods.

---

## 7. Summary

DBSCAN replaces centroids with density connectivity.

The core idea is:

$$
\boxed{
\text{dense neighborhoods connected by reachability become clusters}
}
$$

This makes DBSCAN a useful alternative when cluster shape and outliers matter more than centroid interpretability.
