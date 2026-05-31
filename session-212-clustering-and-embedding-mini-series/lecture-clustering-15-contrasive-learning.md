# Contrastive Learning

Contrastive learning trains representations by comparing positive pairs against negative pairs.

![](./img-image/contrastive-learning.webp)

---

## 1. Motivation

In high-dimensional data, clustering raw inputs often fails because distance does not represent meaning.

Contrastive learning attacks the earlier problem at the representation level:

$$
\boxed{
\text{learn an embedding space first, then use distance or similarity}
}
$$

Instead of requiring class labels, contrastive learning builds a training signal from relationships between examples.

---

## 2. Positive and Negative Pairs

![](./img-image/cl.jpg)

The method depends on two types of pairs.

| Pair Type | Meaning | Example |
| --- | --- | --- |
| Positive pair | Two views that should represent the same underlying item or concept | Two augmented crops of one image |
| Negative pair | Two views treated as different | Views from different images |

The training goal is:

- pull positive embeddings together;
- push negative embeddings apart.

This creates structure in the embedding space.

---

## 3. Encoder and Normalization

Let an encoder map an input to an embedding:

$$
z_i = f_{\theta}(x_i)
$$

Often the embedding is normalized:

$$
\bar{z}_i
=
\frac{z_i}{\|z_i\|_2}
$$

Then cosine similarity becomes a dot product:

$$
\operatorname{sim}(\bar{z}_i,\bar{z}_j)
=
\bar{z}_i \bar{z}_j^\top
$$

Normalization keeps the model from winning only by increasing vector norms.

---

## 4. InfoNCE Loss

![](./img-image/cl2.jpg)

Assume $i$ is an anchor and $j$ is its positive partner. In a batch of $B$ candidates, the InfoNCE loss is:

$$
\boxed{
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
}
$$

where $\tau > 0$ is the temperature.

The numerator rewards the positive pair. The denominator compares the positive against competing examples.

---

## 5. Temperature

The temperature $\tau$ controls how sharp the comparison is.

- Smaller $\tau$ makes the model focus strongly on the hardest distinctions.
- Larger $\tau$ makes the distribution smoother.

Temperature changes the geometry of the learned space. It is not just a numerical detail.

---

## 6. What Counts as a Positive Pair

The most important design choice is the construction of positive pairs.

For images, positives may come from:

- random crop;
- color jitter;
- blur;
- horizontal flip;
- two views of the same object.

For text, positives may come from:

- different dropout masks;
- neighboring contexts;
- paraphrases;
- paired query-document data;
- equivalent translations.

For image-text models, positives are matched image-caption pairs.

The positive-pair rule defines what invariances the model learns.

> [!WARNING]
> Bad positive pairs teach bad invariances. If two views remove information needed for the task, the representation may become insensitive to important differences.

---

## 7. Connection to Clustering

Contrastive learning can make clustering easier because it reshapes the embedding space before clustering.

The pipeline becomes:

$$
\text{data}
\rightarrow
\text{contrastive encoder}
\rightarrow
\text{embedding space}
\rightarrow
\text{clustering}
$$

If positive pairs reflect useful semantic identity, then related examples tend to move closer. K-means, KNN, and retrieval methods can then operate on a better geometry.

---

## 8. Common Failure Modes

Contrastive learning can fail when:

- positives are too weak or too aggressive;
- negatives include false negatives from the same semantic class;
- batch size is too small for useful comparison;
- the model learns shortcuts from augmentation artifacts;
- the learned similarity does not match the downstream task.

Good inspection includes nearest neighbors, retrieval examples, cluster quality, and sensitivity to augmentation choices.

---

## 9. Summary

Contrastive learning is representation learning through comparison.

The central rule is:

$$
\boxed{
\text{positive pairs close, negative pairs far}
}
$$

It does not replace clustering. It often creates the embedding geometry that makes clustering meaningful.
