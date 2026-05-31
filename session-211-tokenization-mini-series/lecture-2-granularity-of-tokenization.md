# Tokenization Granularity

Tokenization granularity controls how much text each token represents. The main trade-off is coverage versus efficiency.

Online tokenizer visualization:

- https://tiktokenizer.vercel.app/

Online resource:

- https://huggingface.co/learn/llm-course/en/chapter6/5

---

## 1. The Granularity Spectrum

At one extreme, every character or byte can be a token. At the other extreme, every word or phrase can be a token.

Modern language models usually choose the middle: subword tokens.

The design question is:

$$
\boxed{
\text{How large should a reusable text unit be?}
}
$$

The answer affects vocabulary size $V$, sequence length $T$, compute, robustness, and semantic structure.

---

## 2. Character-Level Tokenization

Character-level tokenization treats each character as a token.

Example:

```text
Shanghai -> S h a n g h a i
```

Strengths:

- small vocabulary;
- strong coverage;
- robust to new words and typos;
- simple to understand.

Weaknesses:

- long sequences;
- expensive attention because of $O(T^2)$ scaling;
- model must learn word-like patterns from many small units;
- less efficient use of context length.

Character-level tokenization is universal, but it asks the model to do more work.

---

## 3. Byte-Level Tokenization

Byte-level tokenization starts from bytes rather than characters.

With a byte vocabulary, any text that can be encoded as bytes can be represented. This is useful for:

- multilingual text;
- symbols;
- unusual punctuation;
- code;
- noisy web text.

Byte-level BPE, used by several GPT-style tokenizers, combines byte-level coverage with learned merges so common patterns become larger tokens.

> [!INFO]
> Byte-level tokenization is a practical coverage strategy. It avoids true unknown characters by falling back to bytes.

---

## 4. Word-Level Tokenization

Word-level tokenization treats each word as a token.

Example:

```text
I love Shanghai -> I love Shanghai
```

Strengths:

- short sequences;
- human-readable tokens;
- each token often has clear semantic content.

Weaknesses:

- very large vocabulary;
- severe out-of-vocabulary risk;
- poor handling of names, typos, morphology, and code;
- difficult multilingual coverage.

Word-level tokenization is efficient when the vocabulary is complete, but natural language is open-ended.

---

## 5. Subword Tokenization

Subword tokenization decomposes rare words while keeping frequent words or fragments intact.

Example:

```text
unbelievable -> un believ able
```

Subword tokenization gives a practical compromise:

- common words can remain single tokens;
- rare words can be decomposed;
- vocabulary size stays manageable;
- sequence length stays shorter than character-level tokenization;
- unknown-token collapse is reduced or avoided.

This is why subword tokenization became the standard choice for transformer language models.

---

## 6. Out-of-Vocabulary Collapse

Out-of-vocabulary collapse happens when many different strings map to the same unknown token.

In a word-level tokenizer:

```text
Pudong -> <unk>
Xuhui -> <unk>
ShanghaiTech -> <unk>
```

The model receives the same ID for distinct objects.

In a subword tokenizer, these strings can be decomposed into known pieces. Even if the split is imperfect, the model receives more information than a single unknown token.

---

## 7. Language Differences

Tokenization is not one-size-fits-all.

English has visible word spaces, but it also has morphology:

- prefixes;
- suffixes;
- compounds;
- inflections.

Chinese does not mark word boundaries with spaces, so tokenization often involves segmentation. Some languages have rich inflection, productive compounding, or writing systems that make simple whitespace splitting unreliable.

The tokenizer must fit the language mixture in the training data.

> [!WARNING]
> A tokenizer trained mostly on one language can be inefficient or unfairly costly for another language. More tokens for the same content means less effective context and more compute.

---

## 8. Measuring Efficiency

A simple efficiency metric is average characters per token:

$$
\boxed{
\mathrm{CPT}
=
\frac{\text{number of characters in the corpus}}
{\text{number of tokens in the corpus}}
}
$$

Low $\mathrm{CPT}$ means the tokenizer uses many tokens per character span. High $\mathrm{CPT}$ means each token covers more characters on average.

This metric is useful, but it is incomplete. A good tokenizer should also preserve meaningful structure and avoid systematic inefficiency for important data types.

---

## 9. The Main Trade-Off

| Granularity | Coverage | Efficiency | Main Use |
| --- | --- | --- | --- |
| Character | Very high | Low | Simple universality |
| Byte | Very high | Low to medium | Robust open-world text |
| Word | Low to medium | High | Closed vocabularies |
| Subword | High | Medium to high | Modern LLMs |

The practical goal is a tokenizer that:

- represents any input;
- keeps $T$ manageable;
- keeps $V$ manageable;
- handles rare and multilingual text reasonably;
- supports the model's training objective.

---

## 10. Summary

Tokenization granularity controls the balance between vocabulary size and sequence length.

The central lesson is:

$$
\boxed{
\text{fine tokens improve coverage, coarse tokens improve efficiency}
}
$$

Subword tokenization wins because it balances both pressures.
