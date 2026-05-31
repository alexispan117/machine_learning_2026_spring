# Feed-Forward Networks


![](./img/transformer-explained.jpg)


## 1. Beyond Attention: The Other Half of the Transformer

We have extensively studied attention—the mechanism that routes information between tokens. But attention alone is not enough. Enter the **Feed-Forward Network (FFN)**—the "expert" that processes information at each position independently.

---

## 2. What is the FFN?



### The Structure

A standard FFN consists of:
1. **Linear expansion**: Project to higher dimension (typically 4×)
2. **Non-linear activation**: Apply element-wise non-linearity
3. **Linear contraction**: Project back to model dimension

Mathematically:

$$
\text{FFN}(x) = \text{activation}(x W_1 + b_1) W_2 + b_2
$$

Where:
- $W_1 \in \mathbb{R}^{d_{model} \times 4d_{model}}$ (expansion)
- $W_2 \in \mathbb{R}^{4d_{model} \times d_{model}}$ (contraction)
- $x \in \mathbb{R}^{1 \times d_{model}}$ is a row vector

---

## 3. Why Expand Then Contract?

### The Dimensionality Argument

```
Input:     768-dim ───→ FFN ───→ 768-dim
                ↓              ↑
            [3072-dim]  (intermediate)
```

**Why 4× expansion?**

1. **Increased capacity**: More parameters to store information
2. **Overcomplete representation**: Similar to overcomplete dictionaries in signal processing
3. **Empirically optimal**: Found through extensive experimentation

### Analogy: Expert Consultation

Imagine consulting a panel of experts:
- **Input**: Your question (768-dim representation)
- **Expansion**: Ask 3072 specialists (each gets the question)
- **Activation**: Each specialist contributes their expertise
- **Contraction**: Synthesize the panel's input into a coherent answer

---

## 4. Position Independence

### Key Property

Unlike attention, FFN is applied **identically to every position**.

This means:
- No information flows between positions in FFN
- The same "expert" evaluates each token
- Each position's representation is transformed independently

### Why This Design?

**Separation of concerns**:
- **Attention**: "Where should I look?" (routing)
- **FFN**: "What should I do with what I found?" (processing)

By separating these, the model can:
1. First collect relevant information (attention)
2. Then process it independently (FFN)

---

## 5. Implementation Details

### Standard Implementation

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model=768, d_ff=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()  # or ReLU, Swish, etc.
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Expand
        x = self.linear1(x)  # (batch, seq_len, d_ff)
        
        # Activate
        x = self.activation(x)
        x = self.dropout(x)
        
        # Contract
        x = self.linear2(x)  # (batch, seq_len, d_model)
        
        return x
```

### Connection to nanochat

nanochat uses a simplified FFN with ReLU² (squared ReLU) (see `./nanochat/gpt.py`):

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard 4x expansion
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        # ReLU squared: modern activation that's smooth and fast
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
```

Key observations:
- **4× expansion**: `4 * config.n_embd` is the standard
- **No bias**: Modern architectures often omit bias for simplicity
- **ReLU²**: Smooth activation (no sharp corners like ReLU)

---

## 6. Computational Analysis

### FLOPs Comparison: Attention vs FFN

For sequence length $n$ and model dimension $d$:

| Component | Operations | Parameters |
|-----------|------------|------------|
| **Attention** | $O(n^2 \cdot d)$ | $4d^2$ (Q, K, V, O projections) |
| **FFN** | $O(n \cdot d \cdot d_{ff})$ | $2d \cdot d_{ff}$ |

With $d_{ff} = 4d$:
- FFN operations: $O(n \cdot 4d^2)$
- FFN parameters: $8d^2$

**Key insight**: FFN has **twice the parameters** of attention but is **faster** (no quadratic term in sequence length)!
