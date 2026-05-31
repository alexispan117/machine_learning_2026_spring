# Image Embedding in Vision Models (From Pixels to Vectors)

---

## 1. From Images to Representation Vectors

![](./img-image/1e.jpg)


Image embedding is the process of mapping raw pixel data into a continuous vector space.

The goal is not just compression, but representation:

* similar images should be close in vector space
* different images should be far apart
* the representation should be useful for downstream tasks

Formally, we learn a function:

$$
f: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^{d}
$$

or, in some models, a sequence of vectors.

---

## 2. Image as Input Tensor

An image is represented as:

$$
I \in \mathbb{R}^{H \times W \times C}
$$

where:

* $H$: height
* $W$: width
* $C$: channels (RGB)

Unlike text, images are continuous and spatially structured, so we must define a way to convert them into tokens or features.

---

## 3. Patch Embedding (Vision Transformers)

A common approach is to split the image into patches.

Each patch:

$$
P_i \in \mathbb{R}^{p \times p \times C}
$$

After flattening:

$$
\text{flatten}(P_i) \in \mathbb{R}^{p^2 C}
$$

We project each patch into a vector:

$$
h_i^{(0)} = \text{flatten}(P_i) \cdot E
$$

where:

$$
E \in \mathbb{R}^{p^2 C \times d}
$$

and:

$$
h_i^{(0)} \in \mathbb{R}^{d}
$$

This produces a **sequence**:

$$
H^{(0)} = (h_1^{(0)}, \dots, h_N^{(0)})
$$

Each vector represents a local region of the image.

---

## 4. CNN-Based Image Embedding

In convolutional networks, embeddings are learned implicitly.

A convolution layer learns hierarchical features:

* edges in early layers
* textures and parts in middle layers
* objects in deeper layers

The final output is often pooled into a single **vector**.

---

## 5. Global Image Embedding

For classification or retrieval, the model produces a single vector:

$$
z \in \mathbb{R}^{d}
$$

This vector summarizes the entire image.

It is used for:

* image classification
* similarity search
* contrastive learning

---

## 6. Learning Image Embeddings End-to-End

Image embeddings are not learned in isolation. They are learned as part of a full model, where the embedding is an intermediate representation optimized through a task objective.

The key idea of end-to-end learning is:

All components—from raw pixels to final output—are trained jointly using gradient-based optimization.

---

### 6.1 Forward pipeline

An image passes through a sequence of transformations:

$$
I \rightarrow \text{Encoder} \rightarrow z \rightarrow \text{Head} \rightarrow \hat{y}
$$

where:

* $I$ is the input image
* the encoder produces an embedding $z \in \mathbb{R}^d$
* the head maps $z$ to task-specific outputs $\hat{y}$

Examples:

* CNN:
  $$
  I \rightarrow \text{Conv layers} \rightarrow \text{Pooling} \rightarrow z
  $$

* Vision Transformer:
  $$
  I \rightarrow \text{Patch embedding} \rightarrow \text{Transformer} \rightarrow z
  $$

---

### 6.2 Loss defines what the embedding should represent

The embedding $z$ is not directly supervised. Instead, it is shaped indirectly by the loss function.

Different objectives produce different embedding geometries.

#### (1) Supervised classification

$$
\mathcal{L} = -\log P(y \mid I)
$$

Effect on embedding:

* samples from the same class are pulled together
* different classes are pushed apart


#### (2) Contrastive learning

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Effect on embedding:

* positive pairs collapse
* negatives spread out
* embedding space becomes structured by similarity

#### (3) Self-supervised / reconstruction 

Example:

$$
\mathcal{L} = |I - \hat{I}|^2
$$

Effect:

* embedding preserves detailed information
* focuses more on reconstruction than discrimination

---

### 6.3 Backpropagation: how embeddings are learned

The critical mechanism is gradient flow.

The loss provides gradients:

$$
\frac{\partial \mathcal{L}}{\partial z}
$$

These gradients propagate backward through the entire model:

$$
z \rightarrow \text{encoder} \rightarrow \text{pixels}
$$

This updates:

* patch embedding matrix $E$
* convolution filters
* transformer weights

As a result:

The embedding function $f_\theta(I)$ is gradually shaped so that $z$ becomes useful for minimizing the task loss.

---

### 6.4 Key intuition

The embedding is not explicitly designed.

It emerges as the internal representation that best supports the objective.

* Change the loss → change the geometry of $z$
* Change the architecture → change how features are extracted

