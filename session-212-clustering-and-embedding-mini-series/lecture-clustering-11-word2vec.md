# Word2Vec

Word2Vec learns word embeddings by training a small neural model to predict context relationships.

![](./img-word2vec/w2v1.jpg)

Paper:

- https://arxiv.org/pdf/1301.3781

---

## 1. Motivation

One-hot vectors encode word identity, but not meaning.

Word2Vec asks a simple question:

> Can we learn word meaning by predicting which words appear near each other?

This follows the distributional hypothesis:

> Words used in similar contexts tend to have related meanings.

---

## 2. Notation

Let:

- $V$ be the vocabulary size;
- $d$ be the embedding dimension;
- $w_t$ be the token at position $t$ in a corpus;
- $c$ be the context-window size;
- $v_w \in \mathbb{R}^{1 \times d}$ be the input embedding of word $w$;
- $u_w \in \mathbb{R}^{d \times 1}$ be the output embedding of word $w$.

The context window around $w_t$ is:

$$
w_{t-c},\ldots,w_{t-1},w_t,w_{t+1},\ldots,w_{t+c}
$$

---

## 3. Two Training Tasks

![](./img-word2vec/cbowvsskipgram.jpg)

Word2Vec has two classic variants.

### Skip-Gram

Skip-gram predicts context words from a center word:

$$
w_t \rightarrow w_{t+j}
\quad
\text{where}
\quad
-c \le j \le c
\quad
\text{and}
\quad
j \ne 0
$$

Each center word produces several training pairs.

![](./img-word2vec/skipgram.jpg)

### CBOW

CBOW, or continuous bag of words, predicts the center word from surrounding context words:

$$
\{w_{t+j}: -c \le j \le c,\ j \ne 0\}
\rightarrow
w_t
$$

![](./img-word2vec/cbow.jpg)

Skip-gram is often better for rare words. CBOW is often faster and smoother.

---

## 4. Embedding Lookup

Word2Vec uses a lookup table:

$$
E \in \mathbb{R}^{V \times d}
$$

For an input word $w_i$, the one-hot vector $x_i \in \mathbb{R}^{1 \times V}$ selects one row:

$$
h = x_i E
$$

Because $x_i$ is one-hot, this is equivalent to:

$$
h = E[i]
$$

The selected row is the word embedding.

![](./img-word2vec/w2v.jpg)

---

## 5. Softmax Prediction

For skip-gram, we want to predict an output word $w_o$ from an input word $w_i$.

The logit for candidate word $w$ is:

$$
s_w = v_{w_i} u_w
$$

The softmax probability is:

$$
P(w_o \mid w_i)
=
\frac{\exp(v_{w_i}u_{w_o})}
{\sum_{w=1}^{V}\exp(v_{w_i}u_w)}
$$

The model learns embeddings that make real center-context pairs score higher than unlikely pairs.

---

## 6. Skip-Gram Objective

![](./img-word2vec/skipgram2.jpg)

The skip-gram objective maximizes context likelihood:

$$
\boxed{
\mathcal{L}
=
\sum_{t=1}^{T}
\sum_{\substack{-c \le j \le c \\ j \ne 0}}
\log P(w_{t+j} \mid w_t)
}
$$

In training, we usually minimize the negative log likelihood:

$$
\mathcal{J} = -\mathcal{L}
$$

---

## 7. CBOW Objective

CBOW first combines context embeddings. A simple version averages them:

$$
h_t =
\frac{1}{2c}
\sum_{\substack{-c \le j \le c \\ j \ne 0}}
v_{w_{t+j}}
$$

Then it predicts the center word:

$$
P(w_t \mid \text{context})
=
\operatorname{softmax}(h_t U)
$$

where $U \in \mathbb{R}^{d \times V}$ contains output vectors as columns.

---

## 8. Negative Sampling

Computing the full softmax over all $V$ words can be expensive.

Negative sampling replaces the full vocabulary prediction with a binary discrimination task:

- real center-context pair: label $1$;
- randomly sampled fake pair: label $0$.

For one positive pair $(w_i,w_o)$ and negative samples $w_1^-,\ldots,w_m^-$, the objective is:

$$
\log \sigma(v_{w_i}u_{w_o})
+
\sum_{r=1}^{m}
\log \sigma(-v_{w_i}u_{w_r^-})
$$

This is much cheaper than a full softmax and works well in practice.

---

## 9. What the Model Learns

Word2Vec learns two sets of vectors:

- input vectors $v_w$;
- output vectors $u_w$.

After training, the input vectors are commonly used as word embeddings.

The learned space often captures:

- semantic similarity;
- syntactic similarity;
- analogy-like directions;
- topical neighborhoods.

![](./img-word2vec/manwomankingqueen.jpg)

---

## 10. Connection to Clustering

Word2Vec turns sparse symbolic tokens into dense vectors:

$$
\boxed{
\text{word}
\rightarrow
\text{embedding}
\rightarrow
\text{nearest neighbors or clusters}
}
$$

This makes clustering more meaningful because distance can reflect context-based similarity rather than token identity.

---

## 11. From Word2Vec to Transformers

Word2Vec learns static embeddings: each word type gets a single vector.

Transformers still use embedding lookup tables, but the final representation of a token becomes contextual. The vector for a token can change depending on the surrounding sequence.

That is the transition:

$$
\boxed{
\text{static word embedding}
\rightarrow
\text{contextual token representation}
}
$$
