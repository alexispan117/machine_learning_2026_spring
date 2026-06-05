## Question: FFN Parameters and Computation Compared with Attention

Consider a Transformer block with model dimension $d$, sequence length $n$, and feed-forward hidden width $d_{\text{ff}}=4d$. Ignore biases for this question.

1. The attention projections for $Q$, $K$, $V$, and output projection contain about $4d^2$ parameters. The FFN contains $W_1\in\mathbb{R}^{d\times d_{\text{ff}}}$ and $W_2\in\mathbb{R}^{d_{\text{ff}}\times d}$. Compute the FFN parameter count in terms of $d$.
2. Using your answer, compare the number of FFN parameters with the number of attention projection parameters.
3. Explain why attention has a sequence-length interaction cost involving $n^2$, while the FFN is applied independently to each of the $n$ positions.
4. For $d=768$ and $d_{\text{ff}}=3072$, compute the number of FFN weight parameters.
5. Explain why a Transformer block can spend a large fraction of its parameters in the FFN while still needing attention for cross-position communication.
