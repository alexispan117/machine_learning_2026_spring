## Question: From Hidden States to Token Probabilities

A Transformer block produces a contextual representation $h_i \in \mathbb{R}^{1 \times d_{\text{model}}}$ at token position $i$. To predict a token, the model projects this representation into vocabulary space, producing logits $o_i \in \mathbb{R}^{1 \times |\mathcal{V}|}$ and probabilities $p_i=\operatorname{softmax}(o_i)$.

1. Explain why $h_i$ itself is not yet a word or token prediction.
2. Write the softmax formula for $p_{ic}$, where $c$ is a vocabulary class index.
3. Suppose the correct token id is $t_i$ and the model assigns $p_{i,t_i}=0.80$. Write the one-hot cross-entropy loss for this token position.
4. Compare this with the case $p_{i,t_i}=0.05$. Which loss is larger, and what does that mean about the model's prediction?
5. Draw the pipeline $h_i \rightarrow o_i \rightarrow p_i \rightarrow \ell_i$, labeling the role of each object.

## Question: Cross-Entropy, Likelihood, and the Softmax Gradient

For vocabulary prediction at token position $i$, let $o_i$ be the output logits, $p_i=\operatorname{softmax}(o_i)$, and $y_i$ be the target distribution over the vocabulary. The per-token cross-entropy is

$$
\ell_i=-\sum_{c=0}^{|\mathcal{V}|-1}y_{ic}\log p_{ic}.
$$

1. If $y_i$ is one-hot with correct token id $t_i$, simplify the expression for $\ell_i$.
2. For a sequence with valid prediction positions $\mathcal{I}$, write the average loss $\mathcal{L}$ using $\ell_i$.
3. Explain why minimizing this cross-entropy is equivalent to increasing the probability assigned to the observed tokens.
4. State the gradient formula $\frac{\partial \ell_i}{\partial o_{ic}}$ for softmax followed by cross-entropy.
5. Interpret the expression "prediction minus target" for an overestimated wrong token and for an underestimated correct token.

## Question: Masked Language Modeling Loss

In masked language modeling, an encoder receives a corrupted input sequence $\tilde{x}$ where some original tokens have been hidden. Let $\mathcal{M}$ be the set of masked positions, and let $t_i$ be the original token id that should be recovered at masked position $i$.

1. Explain why directly asking an encoder to predict every original token without hiding any input token would create a trivial copying problem.
2. Write the per-token loss $\ell_i$ for a masked position $i \in \mathcal{M}$ using the predicted probability $p_{i,t_i}$.
3. Write the total masked language modeling loss over the masked-position set $\mathcal{M}$.
4. Explain why unmasked positions usually do not contribute to this loss, even though they still help provide context.
5. In words, describe what kind of representation this objective encourages the encoder to learn.

## Question: Decoder-Only and Encoder-Decoder Prediction Alignment

Autoregressive language models predict future target tokens from visible previous tokens. A decoder-only model learns next-token prediction, while an encoder-decoder model learns a target sequence conditioned on an input sequence.

1. For decoder-only causal language modeling with sequence $x_1,\ldots,x_T$, write the factorization of $P(x_1,\ldots,x_T)$ into next-token conditional probabilities.
2. Show the input-label shift for a short decoder-only sequence $x_1,x_2,x_3,x_4$.
3. Explain why a causal mask is still required during parallel training when the full sequence is available.
4. For an encoder-decoder model with source sequence $x$ and target sequence $y_1,\ldots,y_{T_d}$, write the conditional factorization of $P(y_1,\ldots,y_{T_d}\mid x)$.
5. Compare the visible information used by the decoder in the two settings: decoder-only prediction versus encoder-decoder conditional prediction.

## Question: Label Smoothing and Valid Loss Positions

For vocabulary prediction, label smoothing replaces a hard one-hot target with a softer target distribution. Use the convention

$$
y^{(\text{LS})}_{ic}=(1-\varepsilon)\,\mathbf{1}[c=t_i]+\frac{\varepsilon}{|\mathcal{V}|},
$$

where $t_i$ is the correct token id and $\varepsilon\in[0,1)$.

1. For $|\mathcal{V}|=4$, $\varepsilon=0.2$, and correct token id $t_i=2$, compute all four target probabilities $y^{(\text{LS})}_{i0},y^{(\text{LS})}_{i1},y^{(\text{LS})}_{i2},y^{(\text{LS})}_{i3}$.
2. Write the smoothed cross-entropy loss $\ell_i^{(\text{LS})}$ for one token position.
3. Explain why label smoothing can reduce overconfident predictions.
4. Let $r_i \in \{0,1\}$ indicate whether token position $i$ should be counted in the loss. Write the weighted average loss using $r_i$ and $\ell_i$.
5. Give one example of a setting where some positions should be ignored in the loss, and explain why.
