## Question: FFN as a Sparse Knowledge Memory

One useful interpretation of Transformer feed-forward networks is that they behave like large key-value memories. In the expression

$$
\operatorname{FFN}(x)=f(xW_1+b_1)W_2+b_2,
$$

the expansion layer decides which intermediate neurons activate, and the contraction layer combines their contributions back into the model dimension.

1. Explain the key-value interpretation of $W_1$, the activation function $f$, and $W_2$.
2. Why can a large FFN store many patterns or facts even though each token activates only part of the intermediate layer?
3. Describe what sparse activation means in this setting.
4. Explain why different input representations may activate different groups of FFN neurons.
5. Compare this implicit sparse behavior with the idea of a mixture-of-experts model that explicitly routes inputs to selected experts.
