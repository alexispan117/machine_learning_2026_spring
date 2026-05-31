# Attention and FFN: Division of Labor

## 1. Two Complementary Systems

A Transformer block combines two distinct operations:
- **Multi-Head Attention (MHA)**: Routes information between positions
- **Feed-Forward Network (FFN)**: Processes information at each position

Understanding their division of labor is key to understanding how Transformers work.

---

## 2. Attention: The Router

### What Attention Does

1. **Aggregates context**: Combines information from relevant positions
2. **Enables long-range dependencies**: Any token can attend to any other
3. **Dynamic routing**: Different inputs → different attention patterns

### What Attention Does NOT Do

1. **No information creation**: Only redistributes existing representations
2. **No deep processing**: Linear combination of values
3. **No knowledge storage**: Parameters only for computing weights

---

## 3. FFN: The Processor

### What FFN Does

1. **Non-linear transformation**: Applies complex functions to representations
2. **Knowledge retrieval**: Accesses stored facts and patterns
3. **Feature extraction**: Transforms representations into task-relevant features

### What FFN Does NOT Do

1. **No cross-position communication**: Each position processed independently
2. **No dynamic routing**: Same network for all positions
3. **No context awareness**: Relies on attention for context

---

## 4. Why Both Are Necessary

### Attention-Only Model

```python
class AttentionOnly(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
    
    def forward(self, x):
        # Only attention, no FFN
        return self.attention(x, x, x)[0]
```

**Limitations**:
- Can only redistribute existing information
- Cannot create new features
- Limited expressiveness

### FFN-Only Model

```python
class FFNOnly(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        # Only FFN, no attention
        return self.ffn(x)
```

**Limitations**:
- No context from other positions
- Each token processed in isolation
- Cannot resolve ambiguities using context

### Combined: The Transformer

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Attention: Get context
        attn_out, _ = self.attention(self.norm1(x), x, x)
        x = x + attn_out
        
        # FFN: Process
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x
```

