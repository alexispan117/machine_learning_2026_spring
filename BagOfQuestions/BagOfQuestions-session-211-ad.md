## Question: BPE, WordPiece, and Unknown Tokens

BPE and WordPiece are both subword tokenization methods. They learn reusable text pieces from a corpus, but their selection criteria are different.

1. State the main criterion used by BPE when choosing the next merge.
2. Explain, at a high level, how WordPiece differs from BPE: why is it often described as likelihood-oriented or association-oriented rather than purely frequency-driven?
3. In WordPiece-style notation, explain the meaning of a continuation marker in a split such as `playing -> play ##ing`.
4. A word-level tokenizer maps three rare strings to the same token:

```text
Shanghaidaxue -> <unk>
Pudongxinqu -> <unk>
Nanchenlu -> <unk>
```

Explain what information is lost in this unknown-token collapse.
5. Explain how subword tokenization can reduce this collapse even when the resulting split is not linguistically perfect.
