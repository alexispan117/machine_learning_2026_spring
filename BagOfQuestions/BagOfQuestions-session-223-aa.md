## Question: Feed-Forward Network Structure in a Transformer Block

A position-wise feed-forward network (FFN) transforms each token representation independently. For one token represented as a row vector $x \in \mathbb{R}^{1 \times d_{\text{model}}}$, a standard FFN uses expansion, non-linearity, and contraction:

$$
\operatorname{FFN}(x)=f(xW_1+b_1)W_2+b_2.
$$

1. State the shapes of $W_1$, $b_1$, $W_2$, and $b_2$ when the hidden width is $d_{\text{ff}}=4d_{\text{model}}$.
2. Explain why the first linear layer is called an expansion and the second linear layer is called a contraction.
3. If $d_{\text{model}}=256$ and $d_{\text{ff}}=1024$, compute the number of trainable parameters in the FFN, including biases.
4. Explain why applying the same FFN to every token position does not move information between different positions.
5. Draw a schema showing one token vector going through expansion, activation, and contraction, and label the dimensions at each step.
