# Embedding and Unembedding Layers in Transformers

Transformers begin by mapping token IDs into vectors and end by mapping hidden vectors back to vocabulary logits.

![](./img-word2vec/transformer-explained.jpg)

---

## 1. From Word2Vec to Transformers

Word2Vec learns useful static word vectors from context prediction.

Transformers keep the embedding idea but change the setting:

- tokens are often subword units, not whole words;
- embeddings are trained jointly with the whole model;
- hidden states become contextual after self-attention;
- the model predicts the next token through an unembedding layer.

The full pipeline is:

$$
\boxed{
\text{token IDs}
\rightarrow
\text{embedding}
\rightarrow
\text{transformer blocks}
\rightarrow
\text{unembedding}
\rightarrow
\text{logits}
}
$$

---

## 2. Token IDs

Let a sequence contain $T$ tokens:

$$
x_1,\ldots,x_T
$$

Each token ID belongs to a vocabulary:

$$
x_t \in \{1,\ldots,V\}
$$

At this stage, token IDs are discrete symbols. The integer $57$ is not closer in meaning to $58$ than to $9000$. The model needs a learned vector interface.

---

## 3. The Embedding Layer

The token embedding matrix is:

$$
E \in \mathbb{R}^{V \times d_{\mathrm{model}}}
$$

The embedding of token $x_t$ is row $x_t$ of $E$:

$$
h_t^{(0)}
=
E[x_t]
\in
\mathbb{R}^{1 \times d_{\mathrm{model}}}
$$

The whole sequence becomes:

$$
H^{(0)}
=
\begin{bmatrix}
h_1^{(0)} \\
\vdots \\
h_T^{(0)}
\end{bmatrix}
\in
\mathbb{R}^{T \times d_{\mathrm{model}}}
$$

This is the input to the transformer blocks.

---

## 4. Positional Information

![](./img-word2vec/transformer2017.jpg)

The embedding lookup alone does not encode order. A transformer therefore adds positional information.

With learned or fixed position vectors $p_t$:

$$
\tilde{h}_t^{(0)}
=
h_t^{(0)} + p_t
$$

The model then processes:

$$
\tilde{H}^{(0)}
\rightarrow
H^{(1)}
\rightarrow
\cdots
\rightarrow
H^{(L)}
$$

where $L$ is the number of transformer blocks.

---

## 5. Contextual Hidden States

After self-attention, a token representation is no longer just its lookup vector.

The final hidden state:

$$
h_t^{(L)}
\in
\mathbb{R}^{1 \times d_{\mathrm{model}}}
$$

depends on:

- the token at position $t$;
- previous tokens in causal language modeling;
- attention patterns;
- model parameters across all layers.

This is why transformer representations are contextual.

---

## 6. The Unembedding Layer

![](./img-word2vec/transformer_decoding_2.gif)

To predict a vocabulary distribution, the model maps each final hidden state back to vocabulary-sized logits.

Let:

$$
U \in \mathbb{R}^{d_{\mathrm{model}} \times V}
$$

Then:

$$
\ell_t
=
h_t^{(L)} U
\in
\mathbb{R}^{1 \times V}
$$

The vector $\ell_t$ contains one logit per vocabulary token.

The probability of the next token is:

$$
P(x_{t+1}=j \mid x_{\le t})
=
\frac{\exp(\ell_{t,j})}
{\sum_{r=1}^{V}\exp(\ell_{t,r})}
$$

---

## 7. Training Objective

For causal language modeling, the loss is usually next-token cross entropy:

$$
\boxed{
\mathcal{L}
=
-
\sum_{t=1}^{T-1}
\log P(x_{t+1} \mid x_{\le t})
}
$$

Gradients from this loss update:

- token embeddings $E$;
- positional parameters if learned;
- attention and feed-forward layers;
- unembedding matrix $U$.

Everything is trained end-to-end.

---

## 8. Weight Tying

Some transformers tie the embedding and unembedding weights.

Under weight tying, the output vectors are shared with the input token embeddings:

$$
U = E^\top
$$

This is shape-compatible because:

$$
E \in \mathbb{R}^{V \times d_{\mathrm{model}}}
\quad
\text{and}
\quad
E^\top \in \mathbb{R}^{d_{\mathrm{model}} \times V}
$$

The geometric interpretation is:

$$
\ell_{t,j}
=
h_t^{(L)} E[j]^\top
$$

The model gives a high logit to token $j$ when the hidden state aligns with that token's embedding.

---

## 9. Embedding Versus Hidden State

It is important to separate two ideas.

| Object | Meaning |
| --- | --- |
| Token embedding $E[x_t]$ | Initial learned vector for token ID $x_t$ |
| Hidden state $h_t^{(L)}$ | Contextual representation after transformer layers |
| Logits $\ell_t$ | Scores over possible next tokens |

> [!WARNING]
> The embedding vector for a token is not the same thing as the final contextual hidden state at a position. The hidden state has absorbed information from context.

---

## 10. Summary

The transformer embedding layer maps discrete token IDs into vectors. The unembedding layer maps contextual vectors back to vocabulary logits.

The central pattern is:

$$
\boxed{
x_t
\rightarrow
E[x_t]
\rightarrow
h_t^{(L)}
\rightarrow
\ell_t
}
$$

This connects language modeling to the same embedding geometry used for similarity, clustering, and retrieval.
