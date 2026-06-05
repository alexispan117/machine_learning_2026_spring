## Question: Attention and FFN Division of Labor

A Transformer block usually contains multi-head attention followed by a position-wise feed-forward network. These two components have different jobs: attention routes information between token positions, while the FFN processes the representation at each position.

1. Explain what information movement attention can perform that an FFN alone cannot perform.
2. Explain what kind of non-linear processing an FFN can perform that attention alone does not provide.
3. Suppose the representation at position $i$ has already collected context from other positions through attention. Explain how the FFN can process that contextual information while still acting independently at each position.
4. Draw a simple Transformer block schema with attention, residual connection, FFN, and residual connection. Label which part routes across positions and which part processes each position separately.
5. Give one limitation of an attention-only block and one limitation of an FFN-only block.
