# Practice: Tokenization

This practice file turns the tokenization mini-series into concrete inspection tasks.

Reference:

- https://huggingface.co/learn/llm-course/en/chapter6/5

---

## 1. Inspect a Real Tokenizer

Use an online tokenizer visualization or a local tokenizer library.

Tokenize the following strings:

```text
I love Shanghai.
Shanghai Jiao Tong University
unbelievable
tokenization
rare_scientific_term_123
Pudong     airport
```

Record:

- the token strings;
- the token IDs;
- the number of tokens $T$;
- surprising splits;
- whether spaces or punctuation become their own tokens.

---

## 2. Compare Granularity

For each string, estimate how many tokens would be produced by:

- character-level tokenization;
- word-level tokenization;
- subword tokenization.

Use this table:

| String | Character Tokens | Word Tokens | Subword Tokens | Main Observation |
| --- | --- | --- | --- | --- |
| `I love Shanghai.` |  |  |  |  |
| `Shanghai Jiao Tong University` |  |  |  |  |
| `unbelievable` |  |  |  |  |
| `rare_scientific_term_123` |  |  |  |  |

Then answer:

1. Which strategy gives the shortest sequences?
2. Which strategy has the best coverage?
3. Which strategy is the best compromise for open-world text?

---

## 3. Manual BPE Exercise

Use this tiny corpus:

```text
low lower lowest
new newer newest
```

Start from character tokens:

```text
l o w
l o w e r
l o w e s t
n e w
n e w e r
n e w e s t
```

Perform three BPE merge steps.

For each step, write:

- the most frequent adjacent pair;
- the new token created;
- the updated corpus representation.

The merge rule is:

$$
(a^*,b^*)
=
\arg\max_{(a,b)}
\operatorname{count}(a,b)
$$

---

## 4. Vocabulary and Embedding Cost

Suppose a language model has embedding dimension $d = 768$.

Compute the number of embedding parameters for each vocabulary size:

| Vocabulary Size $V$ | Embedding Parameters $Vd$ |
| --- | --- |
| $10{,}000$ |  |
| $30{,}000$ |  |
| $50{,}000$ |  |
| $100{,}000$ |  |

Then answer:

1. What happens to embedding parameters when $V$ doubles?
2. Why might a larger vocabulary reduce sequence length $T$?
3. Why might a smaller vocabulary improve coverage only if it has byte or character fallback?

---

## 5. Special Token Design

Design a simple chat format using special tokens.

Your format should represent:

- system instruction;
- user message;
- assistant response;
- end of message.

Write one example conversation in your format. Prefer a short Shanghai-related conversation, such as asking for a one-day plan near the Bund or comparing Pudong and Xuhui.

Then explain:

1. Which tokens mark role boundaries?
2. Which token marks the end of a message?
3. What could go wrong if the model was trained with one format but served with another?

---

## 6. Tokenization Beyond Text

An image has size $224 \times 224 \times 3$. A vision transformer uses patches of size $16 \times 16$.

Compute the number of image patch tokens:

$$
N =
\frac{224}{16}
\cdot
\frac{224}{16}
$$

Then answer:

1. How many patch tokens are produced?
2. What happens to $N$ if the patch size becomes $8 \times 8$?
3. Why does smaller patch size increase compute?

---

## 7. Reflection

Write a short paragraph answering:

> Tokenization is not just preprocessing. It is part of model design.

Use at least two examples from the mini-series.
