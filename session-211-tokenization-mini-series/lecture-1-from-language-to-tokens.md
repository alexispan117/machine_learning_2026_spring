# From Language to Tokens


Online Tokenizer visualization:
- https://tiktokenizer.vercel.app/?model=gpt2

![](./img/embedding-the-cat-sat-on-a-mat.jpg)

---

## 1. The Fundamental Constraint

All neural networks operate on numbers, not strings.

A Transformer does not “see” words or sentences. It receives a sequence of discrete symbols:

$$
\text{input} = [x_1, x_2, \dots, x_T], \quad x_t \in \{0,1,\dots, V-1\}
$$

where:

* $T$ is the sequence length
* $V$ is the vocabulary size
* each $x_t$ is an integer index

This immediately imposes a requirement:

$$
\text{text} \;\longrightarrow\; \text{discrete tokens} \;\longrightarrow\; \text{integer IDs}
$$

This transformation is not optional. It is the **entry point** of all language models.

We call this transformation **tokenization**.

---

## 2. Tokenization as an Interface

Tokenization is best understood as an **interface layer** between human language and machine computation.

The full pipeline is:

$$
\text{text} \;\rightarrow\; \text{tokens} \;\rightarrow\; \text{IDs} \;\rightarrow\; \text{embeddings} \;\rightarrow\; \text{Transformer}
$$

Each stage serves a precise role:

1. **Tokenizer**
   Splits raw text into discrete units (tokens)

2. **Vocabulary**
   Maps each token to a unique integer ID

3. **Embedding Layer**
   Converts IDs into continuous vectors

If we denote the embedding table as:

$$
E \in \mathbb{R}^{V \times d}
$$

then each token is mapped via lookup:

$$
x_t \;\longrightarrow\; E[x_t] \in \mathbb{R}^{1 \times d}
$$

At this point, language has been fully converted into geometry.

---

## 3. What Is a “Token”?

A token is not necessarily a word.

It is a **design choice** that determines how text is decomposed.

Consider the same idea across languages:

| Language | Text         | Tokenization (example)   |
| -------- | ------------ | ------------------------ |
| English  | unbelievable | `un` + `believ` + `able` |
| Chinese  | 我喜欢上海大学      | `我` + `喜欢` + `上海` + `大学` |
| French   | Je t'aime    | `Je` + `t'` + `aime`     |

Tokens can be:

* full words
* subwords
* characters
* punctuation
* whitespace markers

This choice directly affects how the model perceives structure and meaning.

---

## 4. Tokenization Defines the Computational Budget

Self-attention scales as:

$$
O(T^2)
$$

So the number of tokens $T$ is not just a representation detail — it determines compute cost.

Different tokenization strategies produce very different sequence lengths:

| Text                               | Character-level | Word-level | Subword |
| ---------------------------------- | --------------- | ---------- | ------- |
| The cat sat                        | 11              | 3          | 3       |
| unbelievable                       | 12              | 1          | 3       |
| I love natural language processing | 34              | 5          | 6–8     |

A finer granularity increases $T$, which increases cost quadratically.

A coarser granularity reduces $T$, but harms flexibility.

Tokenization is therefore a **compute–representation trade-off**.

---

## 5. The OOV Problem and the Rise of Subwords

A naive word-level vocabulary fails in open-world settings.

Natural language contains:

* rare scientific terms
* new names and entities
* typos and noise
* code and URLs
* multilingual mixtures

A fixed word vocabulary inevitably produces **out-of-vocabulary (OOV)** tokens.

Subword tokenization solves this by decomposing words into reusable units.

Instead of failing on:

* “pneumonoultramicroscopicsilicovolcanoconiosis”

a subword tokenizer represents it as a sequence of known pieces.

This ensures:

* full coverage
* compositional generalization
* robustness to noise

This is why modern LLMs almost universally use **subword tokenization**.

---

## 6. Special Tokens as Structural Signals

Tokenization does not only encode content — it also encodes structure.

Most vocabularies include reserved tokens:

| Token            | Role                          |
| ---------------- | ----------------------------- |
| `<bos>` / `<s>`  | sequence start                |
| `<eos>` / `</s>` | sequence end                  |
| `<pad>`          | padding                       |
| `<unk>`          | unknown token                 |
| `<mask>`         | masked prediction             |
| `[CLS]`, `[SEP]` | classification and separation |

These tokens act as a **hidden grammar**.

They define:

* where sequences begin and end
* how multiple inputs are separated
* how tasks are formatted

In modern systems, even conversation is tokenized:

```
<|system|>...
<|user|>...
<|assistant|>...
```

The model does not “understand roles” abstractly — it learns them through tokens.

---

## 7. Tokenization Is Not Preprocessing — It Is Modeling

Two models with identical Transformer architectures can behave very differently if their tokenizers differ.

Tokenization determines:

* what counts as a unit of meaning
* how information is segmented
* how long sequences become
* how efficiently the model trains

In this sense, tokenization is part of the **model design**, not just data cleaning.

---

## 8. Vocabulary Size and Parameter Trade-offs

The vocabulary size $V$ directly affects model parameters through the embedding table:

$$
\text{Embedding parameters} = V \times d
$$

Examples:

* GPT-style: $\sim 50\text{K} \times 768$ → tens of millions
* BERT-base: $\sim 30\text{K} \times 768$

This leads to a trade-off:

| Vocabulary | Effect                                  |
| ---------- | --------------------------------------- |
| Small $V$  | longer sequences, cheaper embeddings    |
| Large $V$  | shorter sequences, expensive embeddings |

The tokenizer therefore shapes both:

* **memory footprint**
* **runtime cost**
