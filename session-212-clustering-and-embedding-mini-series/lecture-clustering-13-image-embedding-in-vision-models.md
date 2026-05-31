# Image Embeddings in Vision Models

Image embeddings map pixels into vectors that are useful for classification, retrieval, clustering, and multimodal alignment.

![](./img-image/1e.jpg)

---

## 1. From Pixels to Vectors

An image is usually represented as a tensor:

$$
I \in \mathbb{R}^{H \times W \times C}
$$

where:

- $H$ is height;
- $W$ is width;
- $C$ is the number of channels.

An image encoder maps this tensor to a vector:

$$
f_{\theta}: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^{d}
$$

The embedding is:

$$
z = f_{\theta}(I)
$$

The goal is not merely to compress pixels. The goal is to produce a vector whose geometry reflects visual or semantic similarity.

---

## 2. CNN Image Embeddings

Convolutional neural networks learn image representations through local filters.

Early layers often respond to:

- edges;
- corners;
- colors;
- simple textures.

Middle and later layers can respond to:

- parts;
- shapes;
- object-level patterns;
- task-relevant visual concepts.

A typical CNN pipeline is:

$$
I
\rightarrow
\text{convolutional layers}
\rightarrow
\text{pooling}
\rightarrow
z
$$

The vector $z \in \mathbb{R}^{d}$ can then be used by a classifier, a retrieval system, or a clustering algorithm.

---

## 3. Patch Embeddings in Vision Transformers

Vision transformers convert an image into a sequence of patch vectors.

Split the image into $N$ patches. Each patch has shape:

$$
P_i \in \mathbb{R}^{p \times p \times C}
$$

Flatten the patch:

$$
\operatorname{flat}(P_i)
\in
\mathbb{R}^{1 \times p^2C}
$$

Project it into the model dimension:

$$
h_i^{(0)}
=
\operatorname{flat}(P_i)E
$$

where:

$$
E \in \mathbb{R}^{p^2C \times d_{\mathrm{model}}}
$$

The image becomes a sequence:

$$
H^{(0)}
=
\begin{bmatrix}
h_1^{(0)} \\
\vdots \\
h_N^{(0)}
\end{bmatrix}
\in
\mathbb{R}^{N \times d_{\mathrm{model}}}
$$

This is analogous to a token sequence in language modeling, except the "tokens" are image patches.

---

## 4. Global Image Embeddings

Many applications need a single vector for the whole image.

A model may obtain it by:

- pooling spatial features;
- using a special classification token;
- averaging patch representations;
- applying a projection head.

The result is:

$$
z_{\mathrm{image}} \in \mathbb{R}^{d}
$$

This global embedding is useful for:

- image search;
- duplicate detection;
- clustering;
- nearest-neighbor classification;
- contrastive learning.

---

## 5. The Loss Shapes the Geometry

Image embeddings are learned through training objectives. Different losses create different embedding spaces.

### Supervised Classification

For a labeled image $(I,y)$, a classifier predicts:

$$
P(y \mid I)
$$

The cross-entropy loss is:

$$
\mathcal{L}
=
-\log P(y \mid I)
$$

This encourages images from the same class to become easier to separate from images in other classes, but the embedding may focus only on features useful for the label set.

### Contrastive Learning

Contrastive learning compares image views or image-text pairs.

For a positive pair $(z_i,z_j)$, a common objective is:

$$
\mathcal{L}_i
=
-
\log
\frac{
\exp(\operatorname{sim}(z_i,z_j)/\tau)
}{
\sum_{k=1}^{B}
\exp(\operatorname{sim}(z_i,z_k)/\tau)
}
$$

where $B$ is the batch size and $\tau$ is the temperature.

This shapes the embedding space around similarity rather than fixed class labels.

### Reconstruction

Autoencoding objectives ask the model to reconstruct the input:

$$
\mathcal{L}
=
\left\|I-\hat{I}\right\|_2^2
$$

This may preserve detailed visual information, but it does not automatically guarantee semantic separation.

---

## 6. End-to-End Learning

An image embedding is usually not hand-designed. It emerges as part of an end-to-end system:

$$
I
\rightarrow
\text{encoder}
\rightarrow
z
\rightarrow
\text{task head}
\rightarrow
\hat{y}
$$

The loss sends gradients back through the whole chain:

$$
\frac{\partial \mathcal{L}}{\partial z}
\rightarrow
\frac{\partial \mathcal{L}}{\partial \theta}
$$

This updates convolution filters, patch projections, transformer blocks, and projection heads.

---

## 7. Inspecting Image Embeddings

Useful inspection questions include:

- Do nearest neighbors show the same object, style, or label?
- Do clusters correspond to useful concepts?
- Are embeddings sensitive to background or lighting?
- Do small transformations preserve similarity?
- Are mistakes systematic for certain classes or image types?

> [!WARNING]
> A visually impressive embedding plot can hide shortcut learning. Always inspect nearest neighbors and failure cases in the original image space.

---

## 8. Summary

Image embeddings convert spatial pixel data into useful vectors.

The central pattern is:

$$
\boxed{
I
\rightarrow
f_{\theta}(I)
=
z_{\mathrm{image}}
\in
\mathbb{R}^{d}
}
$$

The quality of the embedding depends on architecture, data, objective, and the similarity notion used during training or evaluation.
