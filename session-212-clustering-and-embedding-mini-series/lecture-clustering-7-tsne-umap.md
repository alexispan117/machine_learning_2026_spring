# t-SNE and UMAP

t-SNE and UMAP are visualization methods for high-dimensional data. They help us inspect embeddings, but they should not be treated as precise maps.

---

## 1. The Visualization Problem

Many modern representations live in high-dimensional spaces:

- word embeddings;
- sentence embeddings;
- image embeddings;
- neural network hidden states;
- customer or product feature vectors.

Humans cannot directly inspect points in $\mathbb{R}^d$ when $d$ is large, so we often project them into two or three dimensions:

$$
f: \mathbb{R}^{d} \rightarrow \mathbb{R}^{2}
$$

The goal is not perfect reconstruction. The goal is useful visual inspection.

---

## 2. t-SNE

![](./img-embedding/tsne8.gif)

t-SNE stands for t-distributed stochastic neighbor embedding.

Its main goal is to preserve local neighborhoods:

- points close in high-dimensional space should usually appear close in the plot;
- faraway relationships are not reliable;
- visual cluster separation is often exaggerated.

t-SNE is useful for seeing whether local groups exist in an embedding space.

> [!WARNING]
> Do not use t-SNE plots to compare global distances, cluster sizes, or absolute positions. A t-SNE plot is a visualization, not a coordinate system with trustworthy global geometry.

---

## 3. UMAP

UMAP stands for uniform manifold approximation and projection.

It also preserves local structure, but it is often faster and can preserve more global layout than t-SNE in practice.

UMAP is commonly used when:

- the dataset is large;
- we want a fast exploratory visualization;
- we want a more stable view across parameter settings;
- we want to inspect approximate manifold structure.

The word "approximate" matters. UMAP still distorts the original geometry.

---

## 4. Parameters That Matter

For t-SNE, important parameters include:

- perplexity;
- learning rate;
- random seed;
- number of iterations.

For UMAP, important parameters include:

- number of neighbors;
- minimum distance;
- distance metric;
- random seed.

Changing these parameters can change the plot.

> [!INFO]
> A good habit is to generate several plots with different seeds and parameters. If a visual pattern only appears once, treat it cautiously.

---

## 5. When to Use Them

![](./img-embedding/tsne3.gif)

Use t-SNE or UMAP for:

- exploring whether embeddings contain visible groups;
- debugging representation learning;
- spotting outliers;
- communicating approximate local structure;
- comparing embedding models qualitatively.

Do not use them for:

- measuring true cluster distances;
- proving that clusters exist;
- choosing $K$ mechanically;
- evaluating model quality by visual appearance alone.

---

## 6. Better Inspection Practice

A useful inspection workflow is:

1. Compute embeddings.
2. Visualize with t-SNE or UMAP.
3. Color points by known labels or metadata if available.
4. Inspect nearest neighbors in the original embedding space.
5. Check quantitative metrics outside the visualization.

The plot should start questions, not end them.

---

## 7. Summary

t-SNE and UMAP compress high-dimensional structure into a human-readable view.

Their central value is:

$$
\boxed{
\text{visual inspection of local neighborhood structure}
}
$$

Their central danger is overinterpretation. Use them as diagnostic tools, not as mathematical proof.
