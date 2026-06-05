## Question: Activation Functions Inside the Feed-Forward Network

The activation function inside a feed-forward network makes the transformation non-linear. Without this activation, two linear layers would collapse into one linear transformation.

1. Write the formula for $\operatorname{ReLU}(x)$ and explain why it is computationally simple.
2. State the derivative of ReLU for $x>0$ and for $x<0$. Explain why ReLU avoids vanishing gradients on the positive side.
3. Explain the dying ReLU problem and how Leaky ReLU changes the negative side of the function.
4. Write the formula for $\operatorname{ReLU}^2(x)=\max(0,x)^2$, then compute its value for $x=-2$, $x=0$, and $x=3$.
5. GELU is often used in Transformer FFNs. In words, compare GELU with ReLU in terms of smoothness and gradient behavior near zero.
