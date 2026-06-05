## Question: Manual Byte-Pair Encoding Merges

Byte-pair encoding (BPE) starts from small units such as characters or bytes and repeatedly merges the most frequent adjacent pair. At each step it chooses

$$
(a^*,b^*)
=
\arg\max_{(a,b)}
\operatorname{count}(a,b).
$$

Consider this tiny corpus, already split into character tokens:

```text
l o w
l o w e r
l o w e s t
n e w
n e w e r
n e w e s t
```

1. Count the adjacent token pairs in the initial corpus and identify one most frequent pair.
2. Perform the first BPE merge and write the updated corpus representation.
3. Recompute pair counts after the first merge, then perform a second merge.
4. Explain why BPE must recompute pair counts after each merge instead of using only the initial counts.
5. In words, explain why frequent fragments such as `low` or `new` may become useful vocabulary entries.
