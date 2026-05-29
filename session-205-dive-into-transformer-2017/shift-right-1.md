In the Transformer paper (*Attention Is All You Need*), the phrase **"Outputs (shifted right)"** refers to a critical trick used during **training** to ensure the model learns to predict the next word auto-regressively without "cheating" by looking ahead.

Here is exactly what it means, why it’s necessary, and how it works.

---

### The Core Concept

During inference (generating text), a Transformer decoder emits tokens one by one. But during training, we want to train the model in parallel on the entire target sentence at once to maximize GPU efficiency.

To do this, we take the target sequence and **shift it to the right by inserting a special start-of-sequence token** (often denoted as `<BOS>`, `<SOS>`, or `<pad>`) at the very beginning.

### Why It's Necessary

If you fed the raw target sentence directly into the decoder without shifting it, the model at position `i` would see the actual answer for position `i`. The network would quickly realize it doesn't need to learn any complex linguistic patterns—it just needs to copy the token at its current position.

By shifting the input right, you ensure that when the model is processing the input at position `i`, the target label it is evaluated against is actually the token at position `i + 1`.

---

### Step-by-Step Example

Imagine your target translation or text sequence is: **"The cat sat"**

If we align the decoder's inputs and the target labels (what we calculate the loss against), the shift looks like this:

| Time Step | Decoder Input (Shifted Right) | Target Label (Expected Output) | What the Model Is Learning |
| --- | --- | --- | --- |
| **1** | `<BOS>` | **The** | Given the start of the sentence, predict the first word. |
| **2** | `<BOS>` **The** | **cat** | Given "The", predict the next word "cat". |
| **3** | `<BOS>` **The cat** | **sat** | Given "The cat", predict "sat". |
| **4** | `<BOS>` **The cat sat** | `<EOS>` | Given the full phrase, predict the end-of-sequence. |

*Note: In code, this is usually achieved by taking the target tensor, stripping the very last token, and prepending the `<BOS>` token.*

---

### How it pairs with the "Causal Mask"

Shifting right handles the horizontal alignment, but because the Transformer uses self-attention, the token `<BOS>` at step 1 could technically still "look ahead" at "The", "cat", and "sat" inside the self-attention layer.

To prevent this, the **Shifted Right Input** is paired with a **Causal Mask** (an upper-triangular matrix of $-\infty$). The mask blocks the attention mechanism from looking at any future tokens.

Together, **Shifting Right** + **Causal Masking** perfectly simulate the step-by-step nature of text generation while allowing the entire sentence to be processed by the GPU in a single parallel pass.