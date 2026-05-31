# FFN as Knowledge Memory: Storing Facts in Transformers

## 1. The Memory Hypothesis

Recent research has revealed a surprising insight: **Feed-Forward Networks (FFNs) in Transformers act as key-value memories**, storing vast amounts of factual knowledge.

This lecture explores how FFNs function as distributed databases of learned information.

---

## 2. The Scale of FFN Parameters

### Parameter Distribution in Transformers

For a typical Transformer layer with $d_{model} = 768$ and $d_{ff} = 3072$:

| Component | Parameters | Fraction |
|-----------|------------|----------|
| **Attention** | $4 \times d_{model}^2 \approx 2.4M$ | ~30% |
| **FFN** | $2 \times d_{model} \times d_{ff} \approx 4.7M$ | ~60% |
| **Layer Norm, etc.** | ~$2 \times d_{model} \approx 1.5K$ | ~0.1% |
| **Total per layer** | ~7.1M | 100% |

**Key insight**: FFN contains **approximately 2/3 of all parameters** in a Transformer!

### Scaling to Full Models

| Model | Total Parameters | FFN Parameters | Percentage |
|-------|------------------|----------------|------------|
| BERT-Base | 110M | ~70M | ~64% |
| GPT-3 | 175B | ~115B | ~66% |
| nanochat | 85M | ~55M | ~65% |

---

## 3. FFN as Key-Value Memory

### The Mathematical Structure

Decompose the FFN:

$$
\text{FFN}(x) = \text{activation}(x W_1 + b_1) W_2 + b_2
$$

Where:
- $W_1 \in \mathbb{R}^{d_{model} \times 4d_{model}}$ (expansion)
- $W_2 \in \mathbb{R}^{4d_{model} \times d_{model}}$ (contraction)
- $x \in \mathbb{R}^{1 \times d_{model}}$ is a row vector

**Key-Value Interpretation**:
- **$W_1$**: "Keys" — determines which neurons activate
- **activation**: "Threshold" — non-linear filtering
- **$W_2$**: "Values" — what each neuron contributes to output

---

## 4. Empirical Evidence

### Study: Knowledge Storage in FFN

Research by Geva et al. (2021) analyzed BERT's FFN layers:

**Methodology**:
1. Feed factual prompts through BERT
2. Measure which FFN neurons activate
3. Correlate neuron activation with knowledge types

**Findings**:

| Neuron Type | Example Triggers |
|-------------|------------------|
| **Entity neurons** | Country names, person names |
| **Relation neurons** | "capital of", "born in" |
| **Attribute neurons** | Colors, numbers, dates |
| **Syntactic neurons** | POS tags, syntactic patterns |

---

## 5. Knowledge Localization

### Which Layers Store What?

Knowledge is distributed across layers:

| Layer Range | Knowledge Type |
|-------------|----------------|
| **Early layers** | Syntactic patterns, word relationships |
| **Middle layers** | Factual knowledge, entity attributes |
| **Late layers** | Task-specific representations, semantic roles |


---

## 6. Sparse Activation

### The Sparsity Pattern

FFN neurons exhibit **high sparsity**: only a small fraction activate for any given input.

```
Typical activation patterns:
- Average sparsity: 80-95% (only 5-20% of neurons active)
- Different inputs activate different neurons
- Similar inputs activate similar neuron sets
```

### Implications

1. **Efficiency**: Most neurons are "off" at any time
2. **Specialization**: Neurons develop specific functions
3. **Capacity**: Large FFNs can store many facts (sparse coding)

---

## 7. The Mixture of Experts Connection

### From Dense to Sparse

Standard FFN: All parameters used for all inputs

```
Input → [All 3072 neurons] → Output
```

Mixture of Experts (MoE): Only subset of parameters used

```
Input → [Router] → [Selected experts] → Output
```

### FFN as Implicit MoE

The sparse activation of FFN suggests it naturally implements a **dense version of mixture of experts**:

- Each neuron = one "expert"
- Activation pattern = routing decision
- Different inputs → Different expert combinations

### Future: Explicit MoE

Modern models (Switch Transformer, GLaM) use explicit MoE:
- Multiple FFN "experts"
- Learned router selects top-k experts
- Even larger capacity with same computation
