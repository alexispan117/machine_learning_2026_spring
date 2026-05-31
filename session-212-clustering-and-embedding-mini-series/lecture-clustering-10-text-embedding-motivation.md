# Text Embedding Motivation

Text embeddings solve the problem of turning discrete language symbols into continuous vectors that neural networks and similarity methods can use.

![](./img-word2vec/embedding-the-cat-sat-on-a-mat.jpg)

Online visualizations:

- https://projector.tensorflow.org/
- https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html

---

## 1. The Discrete-to-Continuous Gap

Human language is made of discrete symbols:

- words;
- subword tokens;
- punctuation;
- phrases;
- documents.

Neural networks operate on numbers. The basic question is:

$$
\boxed{
\text{How do discrete tokens become continuous vectors?}
}
$$

This is the entry point for modern NLP.

---

## 2. One-Hot Encoding

Suppose the vocabulary has size $V$. A one-hot vector for token $i$ is:

$$
e_i \in \mathbb{R}^{V}
$$

where:

- the $i$-th entry is $1$;
- every other entry is $0$.

Example in code:

```python
import numpy as np

vocab = ["the", "cat", "sat", "on", "mat"]
cat_index = 1

cat_one_hot = np.zeros(len(vocab))
cat_one_hot[cat_index] = 1
```

This encodes identity, but not meaning.

---

## 3. Why One-Hot Vectors Are Weak

One-hot vectors have three major problems.

### High Dimension

The vector dimension equals the vocabulary size.

| Vocabulary | Dimension |
| --- | --- |
| Small vocabulary | $10{,}000$ |
| Medium tokenizer | $50{,}000$ |
| Large multilingual tokenizer | $200{,}000$ or more |

The vectors are mostly zeros.

### No Similarity

For two different one-hot vectors $e_i$ and $e_j$:

$$
e_i e_j^\top = 0
\quad
\text{when}
\quad
i \ne j
$$

This means "cat" and "dog" are as unrelated as "cat" and "spreadsheet" under the one-hot dot product.

### No Generalization

The model cannot know that two tokens play similar roles unless it learns that from data in later layers.

The input representation gives no help.

---

## 4. Dense Embeddings

A dense embedding replaces a sparse one-hot vector with a learned vector:

$$
x_i \in \mathbb{R}^{V}
\rightarrow
e_i \in \mathbb{R}^{d}
$$

where $d \ll V$.

The embedding matrix is:

$$
E \in \mathbb{R}^{V \times d}
$$

The embedding for token $i$ is row $i$ of $E$:

$$
e_i = E[i]
$$

This is a lookup operation, not a hand-designed semantic rule.

---

## 5. The Distributional Hypothesis

The guiding idea is:

> Words that appear in similar contexts tend to have related meanings.

For example, "cat" and "dog" may appear near words such as "pet," "food," "fur," and "sleep." A learning algorithm can use these context patterns to place them closer in embedding space.

This idea motivates Word2Vec in the next lecture.

---

## 6. Neural Computation

Dense embeddings make text usable by neural networks.

A token sequence becomes a matrix:

$$
(i_1,\ldots,i_T)
\rightarrow
H^{(0)} \in \mathbb{R}^{T \times d}
$$

where row $t$ is the embedding of token $i_t$:

$$
h_t^{(0)} = E[i_t]
$$

In a transformer implementation, the lookup often appears as a single layer call:

```python
x = token_embedding(token_ids)
```

The mathematical operation is still the same: token IDs select rows from a learned matrix.

---

## 7. What Makes a Good Text Embedding

A useful text embedding should support:

- semantic similarity;
- syntactic regularities;
- downstream prediction;
- retrieval;
- clustering;
- robustness to surface variation.

Different objectives produce different embedding geometries. Word2Vec, transformer language models, sentence encoders, and multimodal models all learn embeddings, but they do not learn the same space.

---

## 8. Summary

One-hot vectors represent token identity. Dense embeddings represent tokens in a learned continuous space.

The transition is:

$$
\boxed{
\text{sparse identity vector}
\rightarrow
\text{dense learned representation}
}
$$

This is the bridge from symbolic text to neural representation learning.
