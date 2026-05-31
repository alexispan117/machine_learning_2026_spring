# Hierarchical Clustering

Hierarchical clustering does not return only one flat partition. It builds a tree of nested groupings, so we can inspect structure at multiple scales.

![](./img-embedding/hierarch2.gif)

---

## 1. Motivation: Beyond One Partition

K-means, GMM, and DBSCAN usually return one clustering result. In many datasets, that is too narrow.

We may want to know:

- whether small groups merge into larger themes;
- whether the data has structure at multiple resolutions;
- which clusters are close to each other;
- where a reasonable tree cut might create useful groups.

Hierarchical clustering asks:

> What nested structure does the dataset contain?

---

## 2. Dendrograms

![](./img-embedding/hierarch.gif)

The output of hierarchical clustering is a dendrogram, which is a tree.

- Leaves represent individual data points.
- Internal nodes represent merged clusters.
- The root represents all points merged into one cluster.
- The height of a merge represents the distance at which two clusters were joined.

A flat clustering is produced by cutting the tree at a chosen height.

> [!INFO]
> A dendrogram is not just a decorative plot. It records the sequence of merge decisions made by the algorithm.

---

## 3. Agglomerative Clustering

The most common form is agglomerative clustering, which is bottom-up.

```text
Input: data matrix X
Initialize each point as its own cluster

repeat
    compute distances between current clusters
    merge the two closest clusters
    update the cluster distances

until one cluster remains

Output: dendrogram
```

At the beginning, there are $n$ clusters. After each merge, the number of clusters decreases by one.

---

## 4. Divisive Clustering

Divisive clustering is top-down.

It starts with one cluster containing every point, then recursively splits clusters into smaller groups.

This approach is conceptually clean but less common in ordinary practice because deciding the best split can be expensive.

---

## 5. Linkage: Distance Between Clusters

Hierarchical clustering needs a distance between clusters, not only between points.

Let $A$ and $B$ be two clusters. A linkage rule defines $D(A,B)$.

### Single Linkage

Single linkage uses the closest pair:

$$
D_{\mathrm{single}}(A,B)
=
\min_{x \in A,\, y \in B}
\|x-y\|_2
$$

It can discover elongated structures, but it is vulnerable to chaining.

### Complete Linkage

Complete linkage uses the farthest pair:

$$
D_{\mathrm{complete}}(A,B)
=
\max_{x \in A,\, y \in B}
\|x-y\|_2
$$

It prefers compact clusters and resists chaining.

### Average Linkage

Average linkage averages all pairwise distances:

$$
D_{\mathrm{average}}(A,B)
=
\frac{1}{|A||B|}
\sum_{x \in A}
\sum_{y \in B}
\|x-y\|_2
$$

It is often a useful compromise between single and complete linkage.

### Ward Linkage

Ward linkage merges the pair that creates the smallest increase in within-cluster squared error.

If $\mu_A$ and $\mu_B$ are cluster means, the merge cost is:

$$
\Delta(A,B)
=
\frac{|A||B|}{|A|+|B|}
\|\mu_A - \mu_B\|_2^2
$$

Ward linkage is close in spirit to K-means because it favors compact variance-minimizing clusters.

> [!WARNING]
> Different linkage rules can produce very different dendrograms on the same data. The linkage rule is a modeling choice, not a formatting option.

---

## 6. Cutting the Tree

![](./img-embedding/treecut.jpg)

After building the dendrogram, we still need a flat clustering if we want cluster labels.

Two common choices are:

1. Cut by distance threshold.
2. Cut to obtain exactly $K$ clusters.

Cutting low in the tree gives many small clusters. Cutting high gives fewer larger clusters.

---

## 7. Practical Inspection

When using hierarchical clustering, inspect:

- the dendrogram merge heights;
- whether one or two merges are much larger than previous merges;
- whether linkage choice changes the conclusion;
- whether clusters contain coherent representative examples;
- whether the result remains stable after small data changes.

For embeddings, hierarchical clustering can reveal nested semantic structure. For raw high-dimensional features, it may mostly reveal noise.

---

## 8. Summary

Hierarchical clustering organizes data as a tree instead of a single partition.

The central mechanism is:

$$
\boxed{
\text{repeatedly merge the closest clusters according to a linkage rule}
}
$$

The main advantage is multi-scale inspection. The main risk is that the tree can look meaningful even when the distance metric or linkage rule is poorly matched to the data.
