# Activation Functions


## 1. The Role of Non-Linearity

Without activation functions, a neural network would be a giant linear model.

Activation functions introduce **non-linearity**, enabling networks to learn complex patterns.

---

## 2. ReLU: The Revolutionary Simplicity

### The Problem Before ReLU

Early networks used sigmoid or tanh:

#### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

#### Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Problem**: Saturation
- When inputs are large, gradients approach zero
- Deep networks become impossible to train (vanishing gradients)

### The ReLU Solution

$$
\text{ReLU}(x) = \max(0, x)
$$


### Why ReLU Works

| Property | Benefit |
|----------|---------|
| **Non-saturating** | Gradient = 1 for positive inputs |
| **Computationally cheap** | Simple max operation |
| **Induces sparsity** | ~50% of neurons inactive |
| **Biological inspiration** | Mimics neuron firing thresholds |

### The Derivative

$$
\frac{d}{dx} \text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}
$$

**Key insight**: No vanishing gradient for positive activations!

---

## 3. ReLU's Problems and Variants

### Problem 1: The Dying ReLU

When a neuron's weights push it into the negative region:
- Output = 0
- Gradient = 0
- Never recovers ("dead neuron")

### Leaky ReLU

$$
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

Small negative slope ($\alpha$) keeps gradients flowing.

### PReLU (Parametric ReLU)

Learn the negative slope:

$$
\text{PReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

Where $\alpha$ is a learned parameter.

### ELU (Exponential Linear Unit)

$$
\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
$$

**Benefits**:
- Smooth for negative values
- Mean activation closer to zero (helps training)



---

## 4. Swish and the Search for Smoothness

### Swish (2017)

Discovered by neural architecture search:

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**Properties**:
- Smooth (unlike ReLU's sharp corner)
- Non-monotonic (slight dip for negative x)
- Self-gated: output = input × gate (sigmoid)

---

## 5. GELU: The Transformer Standard

### Gaussian Error Linear Unit

Introduced in 2016, became standard in BERT and GPT:

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

Where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

### Approximations

**Approximation 1** (used in practice):

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

**Approximation 2** (simpler):

$$
\text{GELU}(x) \approx x \cdot \sigma(1.702x)
$$

### Why GELU Works Better

| Property | ReLU | GELU |
|----------|------|------|
| **Smoothness** | Sharp corner at 0 | Smooth everywhere |
| **Gradients** | Abrupt change | Gradual transition |
| **Stochastic interpretation** | None | Probabilistic regularization |
| **Training stability** | Good | Better |

**The smoothness of GELU allows for better gradient flow and more stable optimization.**

---

## 6. ReLU²: nanochat's Choice

### The Squared Variant

nanochat uses ReLU² (also called squared ReLU):

$$
\text{ReLU}^2(x) = \max(0, x)^2
$$


### Why ReLU²?

1. **Smooth for positive values**: Differentiable everywhere (except 0)
2. **Faster than GELU**: No exponentials or error functions
3. **Good performance**: Competitive with GELU in practice
4. **Simplicity**: Easy to implement and understand
