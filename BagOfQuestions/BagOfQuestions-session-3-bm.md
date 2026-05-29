## Question: DeepSeek-Style Model Size Comparison

A team describes two tiny DeepSeek-style language-model demos. Both models use a vocabulary of $1000$ tokens and output one score for each possible next token.

Model A uses hidden width $10$. It has one dense layer from $10$ inputs to $20$ hidden units, followed by an output dense layer from $20$ hidden units to $1000$ vocabulary scores. Model B doubles these internal sizes: one dense layer from $20$ inputs to $40$ hidden units, followed by an output dense layer from $40$ hidden units to $1000$ vocabulary scores. Every dense layer has one bias per output unit.

1. For Model A, count the parameters in the first dense layer and in the output layer.
2. For Model B, count the parameters in the first dense layer and in the output layer.
3. Compute the total counted parameters for Model A and for Model B.
4. Which part is larger in each model: the small internal dense layer or the output vocabulary layer? Explain using the numbers.
