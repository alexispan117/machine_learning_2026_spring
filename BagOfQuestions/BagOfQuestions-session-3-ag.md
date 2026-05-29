## Question: DeepSeek Mini Mixture-of-Experts

A simplified DeepSeek-inspired model uses a tiny mixture-of-experts layer. The model stores several small expert networks, but for one token it may use only some of them.

There are $4$ experts. Each expert is a small two-layer network: first a dense layer from $10$ inputs to $100$ hidden units, then a dense layer from $100$ hidden units back to $10$ outputs. Each dense layer has one bias per output unit. A router chooses experts using a dense layer from $10$ inputs to $4$ expert scores, also with one bias per output score.

1. Count the parameters in the first dense layer of one expert.
2. Count the parameters in the second dense layer of one expert, then compute the total parameters in one expert.
3. Count the parameters in all $4$ experts together.
4. Count the parameters in the router and then compute the total stored trainable parameters in this mixture-of-experts layer.
5. If only $2$ of the $4$ experts are used for one token, explain why the model still stores parameters for all $4$ experts.
