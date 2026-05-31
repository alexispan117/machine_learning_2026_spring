# Tokenization Granularity: The Discrete Interface of Transformers

Online Tokenizer visualization:
- https://tiktokenizer.vercel.app/


Online resources:
- https://huggingface.co/learn/llm-course/en/chapter6/5




---

## 1. The Granularity Spectrum

The fundamental challenge in tokenization is finding the right unit of analysis. We balance **Coverage** (the ability to represent any input) against **Efficiency** (the amount of information packed into a single sequence).

### 2.1 Character-Level: The Universal Atom
At this level, every single character (including spaces and punctuation) is a token.
* **The Advantage:** Perfect coverage. Since the vocabulary is limited to the character set of a language (e.g., ~100 for English), there is no such thing as an "unknown" word. It is naturally robust to misspellings and new terminology.
* **The Computational Tax:** Sequence length ($T$) explodes. Because Transformer attention scales quadratically $O(T^2)$, processing a character-level document is significantly more expensive than processing a word-level one. Furthermore, the model must "waste" layers learning that `c` + `a` + `t` represents a specific animal before it can begin high-level reasoning.

### 2.2 Word-Level: The Semantic Molecule
Every space-separated word is treated as a unique token.
* **The Advantage:** High semantic density. Each token carries a clear, human-interpretable meaning, and sequence lengths are short.
* **The "Unknown" Crisis:** This approach suffers from the **Out-of-Vocabulary (OOV)** problem. Language is infinite; a word-level dictionary can never contain every name, technical term, or typo. Anything missing is mapped to a special `<unk>` token.

### 2.3 Subword-Level: The Modern Standard
Subword tokenization (e.g., BPE, WordPiece, Unigram) decomposes frequent words into wholes and rare words into meaningful fragments.
* **Example:** `unbelievable` $\to$ `un` + `believ` + `able`
* **Synthesis:** It provides the best of both worlds: the efficiency of words for common text and the fallback safety of characters for rare text.

---

## 2. The Information Catastrophe of OOV

The OOV problem in word-level tokenization is more than a technicality—it is a form of "information erasure."

Consider a vocabulary that does not include specific names. In a word-level system:
* `Alicia` $\to$ `<unk>`
* `Alexander` $\to$ `<unk>`
* `Alexandria` $\to$ `<unk>`

To the model, these three distinct entities are represented by the exact same ID. It cannot distinguish a person from a city or one name from another. In contrast, a subword tokenizer breaks them down:
* `Alicia` $\to$ `Ali` + `cia`
* `Alexander` $\to$ `Alex` + `ander`
* `Alexandria` $\to$ `Alex` + `and` + `ria`

The model can now see the shared `Alex` prefix and the distinct suffixes, allowing it to maintain identity and even infer grammatical or semantic relationships between similar strings.

---

## 3. Linguistic Divergence: English vs. Chinese

Tokenization is not "one size fits all." Different languages present different structural challenges.

| Feature | English (Alpha-Phonetic) | Chinese (Logographic) |
| :--- | :--- | :--- |
| **Boundary** | Clear (Whitespace) | Ambiguous (No spaces) |
| **Morphology** | High (Prefixes/Suffixes) | Low (Root-based) |
| **OOV Risk** | High for new words | Low (Characters are semantic) |
| **Avg. Chars/Token** | ~4.0 - 4.5 | ~1.5 - 2.0 |

In English, tokenization focuses on stripping suffixes (like `-ing` or `-ed`). In Chinese, the tokenizer must perform **segmentation**. Because there are no spaces, the same sequence of characters can often be split in multiple ways, and the choice of split drastically changes the model's representation of the meaning.

---

## 4. Measuring Efficiency: The "Chars per Token" Metric

The efficiency of a tokenizer is often measured by its compression ratio. We define **Average Characters per Token (CPT)** as:

$$CPT = \frac{\text{Total Characters in Corpus}}{\text{Total Tokens in Corpus}}$$

* **Low CPT (~1.0):** Indicates a character-level or very fine-grained subword tokenizer. This wastes the context window and increases compute.
* **High CPT (>5.0):** Indicates a word-level or coarse tokenizer. This maximizes context but risks frequent OOV encounters.

For modern LLMs, the target is usually a "sweet spot" where common words are single tokens, but the model can still zoom in to the character level when it encounters something bizarre.

---

## 5. Why Subwords Won the Transformer Era

The shift to subword tokenization after 2017 was driven by two architectural realities of the Transformer:

1.  **The Context Window is Finite:** We need to fit as much information as possible into the $N$ tokens of the context window. Word-level is best for this but breaks on rare words. Subword-level offers near-word efficiency with a 100% success rate on representation.
2.  **Attention is Expensive:** By keeping sequence lengths manageable (shorter than character-level), subword tokenization keeps the $O(T^2)$ cost of the self-attention mechanism from becoming the primary bottleneck during training and inference.
