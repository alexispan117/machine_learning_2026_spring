# RNN One-to-Many: Conditional Sequence Generation

This lecture studies the one-to-many pattern: one input representation generates a sequence. This appears in image captioning, music generation, conditional text generation, and action planning.

![](./img/rnntypes.jpg)

---

## 1. The One-to-Many Setting

The model receives one input:

$$
x
$$

and produces a sequence:

$$
\hat{y}_{1:T}
=
\left(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T\right)
$$

The mapping is:

$$
x \rightarrow \hat{y}_{1:T}
$$

The input $x$ acts as a condition or seed.

Examples:

* image embedding to caption;
* topic vector to sentence;
* class label to music sequence;
* goal representation to action sequence.

---

## 2. Decoder State

A one-to-many model uses a recurrent decoder state:

$$
s_t \in \mathbb{R}^{1 \times d_s}
$$

We use $s_t$ instead of $h_t$ to emphasize that this recurrent state belongs to a generator or decoder.

At each step, the model produces logits:

$$
o_t = g\left(s_t\right)
$$

Then:

$$
\hat{y}_t = \operatorname{softmax}\left(o_t\right)
$$

for discrete token generation.

---

## 3. Method A: Initial-State Conditioning

The input $x$ is used once to initialize the decoder:

$$
s_0 =
\phi\left(x W_{\text{init}} + b_{\text{init}}\right)
$$

Then generation unfolds:

$$
s_t =
f_{\text{dec}}\left(e_{t-1}, s_{t-1}\right)
$$

where $e_{t-1}$ is the embedding of the previous output token.

The prediction is:

$$
o_t = s_t W_o + b_o
$$

This approach says:

> Put the condition into the initial state, then let the decoder dynamics generate the sequence.

---

## 4. Method B: Repeated Conditioning

The input $x$ can also be fed at every step:

$$
s_t =
f_{\text{dec}}\left(
\operatorname{Concat}\left(e_{t-1}, x\right),
s_{t-1}
\right)
$$

This approach says:

> Keep reminding the decoder of the condition.

Repeated conditioning is often stronger when the generated sequence is long or the condition is easy to forget.

---

## 5. Autoregressive Generation

Most one-to-many models are autoregressive.

At inference time:

$$
\hat{y}_t
\sim
P_{\theta}\left(y_t \mid \hat{y}_{<t}, x\right)
$$

The next prediction depends on previous generated tokens.

The sequence probability factorizes as:

$$
\boxed{
P_{\theta}\left(y_{1:T} \mid x\right)
=
\prod_{t=1}^{T}
P_{\theta}\left(y_t \mid y_{<t}, x\right)
}
$$

Generation stops when the model emits an end token or reaches a maximum length.

---

## 6. Teacher Forcing

During training, we usually feed the ground-truth previous token:

$$
s_t =
f_{\text{dec}}\left(e\left(y_{t-1}^{\text{gt}}\right), s_{t-1}\right)
$$

This is teacher forcing.

During inference, the model feeds its own previous prediction:

$$
s_t =
f_{\text{dec}}\left(e\left(\hat{y}_{t-1}\right), s_{t-1}\right)
$$

This creates a train-test mismatch.

> [!WARNING]
> Teacher forcing stabilizes training, but inference is harder because the model must recover from its own mistakes.

---

## 7. Error Accumulation

Autoregressive generation can drift.

If an early prediction is wrong:

$$
\hat{y}_{t-1} \neq y_{t-1}^{\text{gt}}
$$

then the next state is conditioned on the wrong token:

$$
s_t =
f_{\text{dec}}\left(e\left(\hat{y}_{t-1}\right), s_{t-1}\right)
$$

This can cause later predictions to become worse.

This is called exposure bias or error accumulation.

---

## 8. Decoding Choices

At each step, the model outputs a probability distribution.

Common decoding methods include:

### 8.1 Greedy Decoding

Choose the highest-probability token:

$$
\hat{y}_t =
\arg\max_y
P_{\theta}\left(y \mid \hat{y}_{<t}, x\right)
$$

### 8.2 Sampling

Sample from the distribution:

$$
\hat{y}_t
\sim
P_{\theta}\left(y \mid \hat{y}_{<t}, x\right)
$$

### 8.3 Beam Search

Keep several high-probability partial sequences and expand them step by step.

Beam search is useful when the output space is large and greedy decoding is too narrow.

---

## 9. Example: Image Captioning

An image encoder produces a vector:

$$
x = \operatorname{Encoder}\left(\text{image}\right)
$$

The decoder generates:

$$
\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T
$$

where each $\hat{y}_t$ is a word or subword token.

Initial-state conditioning:

$$
s_0 = \phi\left(x W_{\text{init}} + b_{\text{init}}\right)
$$

Repeated conditioning:

$$
s_t =
f_{\text{dec}}\left(
\operatorname{Concat}\left(e_{t-1}, x\right),
s_{t-1}
\right)
$$

---

## 10. Summary

One-to-many RNNs generate a sequence from one condition.

The central factorization is:

$$
\boxed{
P_{\theta}\left(y_{1:T} \mid x\right)
=
\prod_{t=1}^{T}
P_{\theta}\left(y_t \mid y_{<t}, x\right)
}
$$

The next lecture generalizes from one input condition to many input tokens and many output tokens.
