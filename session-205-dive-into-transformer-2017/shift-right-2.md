In the original Transformer paper ("Attention Is All You Need"), **"shift right"** refers to how the **decoder input** is constructed from the target sequence.

## What it means

The target sentence is shifted one position to the right, with a **start-of-sequence token** (`<<SOS>` / `<BOS>`) inserted at the beginning. For example:

| Target (labels) | `<SOS>` | I | am | fine | `<EOS>` |
|---|---|---|---|---|---|
| **Decoder input** (shifted right) | `<SOS>` | I | am | fine |  |
| **What the model predicts** | I | am | fine | `<EOS>` |  |

So the decoder receives `<SOS> I am fine`, and is trained to predict `I am fine <EOS>` at each corresponding position.

## Why it's done

1. **Autoregressive property**: The decoder must predict position *i* using only information from positions **<< i**. Shifting right enforces this causal structure — the model never sees the token it's supposed to predict at the current step.

2. **Teacher forcing during training**: Instead of feeding the model its own (possibly wrong) previous predictions one-by-one, we feed the entire ground-truth target sequence shifted right. Combined with **masked (causal) self-attention**, this lets all positions train in parallel while still respecting the auto-regressive constraint.

3. **Inference vs. training**: During actual generation (inference), there is no ground truth to shift — you literally generate one token, append it to the input, and run the decoder again. The "shift right" label in the paper's diagram describes the conceptual data flow.

In short: **shift right = prepend `<SOS>` to the target so the decoder learns to predict the next token from left to right.**


"Shifted right" is just the parallelized, matrix-friendly implementation of **Teacher Forcing** for architectures that don't use recurrent loops.

Here is how the two concepts map to each other:

* **The Classic RNN Way:** In a traditional sequence-to-sequence RNN, Teacher Forcing means that at time step $t$, instead of feeding the model's actual (and potentially wrong) prediction from step $t-1$, you force-feed it the ground-truth token from step $t-1$. You do this sequentially, step by step, in a loop.
* **The Transformer Way:** Because Transformers don't have a time loop during training and process the entire sequence at once (parallelization), you can't inject the ground truth step-by-step. Instead, you create a single tensor of the entire ground-truth sequence, shift it right by one position, and hand the whole thing to the decoder.

By shifting the target matrix right, every single token position across the entire sequence is simultaneously fed its correct prior context. Paired with the causal mask, it achieves the exact same result as classic teacher forcing, but optimized for the GPU.