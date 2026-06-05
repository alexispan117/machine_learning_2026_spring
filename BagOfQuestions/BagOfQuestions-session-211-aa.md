## Question: From Text to Token IDs and Embeddings

A language model cannot directly process raw strings. It first needs a tokenizer that segments text into tokens and maps each token to an integer ID. Let the tokenized sequence have length $T$, vocabulary size $V$, and embedding dimension $d$.

1. Draw the full pipeline

$$
\boxed{\text{text}\rightarrow\text{tokens}\rightarrow\text{token IDs}\rightarrow\text{embeddings}\rightarrow\text{model}}
$$

and explain the role of each step.
2. If the token IDs are $x_1,\ldots,x_T$ with $x_t\in\{0,1,\ldots,V-1\}$, explain why each ID must be inside this range.
3. Let the embedding matrix be $E\in\mathbb{R}^{V\times d}$. Write the formula for selecting the embedding vector $e_t$ for token ID $x_t$, and state the shape of $e_t$.
4. For $V=50{,}000$ and $d=768$, compute the number of parameters in the token embedding matrix.
5. Explain why a trained model and its tokenizer should be treated as a coupled system rather than two interchangeable parts.
