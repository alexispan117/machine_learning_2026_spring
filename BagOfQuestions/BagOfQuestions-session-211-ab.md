## Question: Tokenization Granularity and Compute

Tokenization granularity controls how large each token is. Character-level tokenization uses very small units, word-level tokenization uses larger units, and subword tokenization is a compromise between coverage and efficiency.

1. Tokenize the string `unbelievable` in three possible ways: character-level, word-level, and subword-level. The subword split does not need to be unique; give one reasonable example.
2. For the string `Shanghai Jiao Tong University`, compare which tokenization strategy is likely to give the shortest sequence and which is likely to have the best coverage for rare or unseen strings.
3. Self-attention has a sequence-length cost that scales like $O(T^2)$. Explain why a tokenizer that produces many more tokens can make the same text more expensive to process.
4. Define the average characters-per-token metric

$$
\mathrm{CPT}
=
\frac{\text{number of characters in the corpus}}
{\text{number of tokens in the corpus}}.
$$

What does a low value of $\mathrm{CPT}$ suggest?
5. Explain why subword tokenization became a practical compromise for open-ended text.
