# AdamW: Decoupled Weight Decay for Transformers

Training a transformer is not only about finding a direction that reduces loss. It is also about keeping the parameter values controlled during a long and expensive optimization process.

AdamW became a standard optimizer because it separates these two jobs:

$$
\boxed{
\text{loss minimization}
\quad
\text{and}
\quad
\text{weight decay}
}
$$

The central idea is simple:

$$
\boxed{
\text{AdamW applies Adam's adaptive update and weight decay as two separate mechanisms.}
}
$$

---

## 1. The Problem

Modern transformer training usually involves:

- many layers;
- large matrix-shaped parameters;
- noisy mini-batch gradients;
- long training schedules;
- learning-rate warmup and decay;
- large optimizer memory states.

Adam is strong because it adapts the step size coordinate by coordinate. However, this adaptive scaling creates a subtle problem: adding an $L_2$ penalty to the loss is no longer the same as applying ordinary weight decay.

That is the reason AdamW exists.

> [!INFO]
> The `W` in AdamW stands for weight decay. The defining feature is decoupled weight decay, not merely the presence of an $L_2$-like term.

---

## 2. Notation

Let:

- $W$ be a generic trainable weight matrix;
- $\mathcal{B}$ be the current mini-batch;
- $\mathcal{L}_{\mathcal{B}}(W)$ be the mini-batch loss;
- $g = \frac{\partial \mathcal{L}_{\mathcal{B}}}{\partial W}$ be the mini-batch gradient;
- $\eta > 0$ be the learning rate;
- $\lambda \ge 0$ be the weight decay coefficient;
- $m$ be Adam's first moment estimate;
- $v$ be Adam's second moment estimate;
- $\beta_1$ and $\beta_2$ be exponential decay rates;
- $\epsilon$ be a small numerical-stability constant.

We use $W \leftarrow \cdots$ for an in-place parameter update.

For a whole model, $W$ can be read as one parameter tensor at a time. The same logic applies to attention projection matrices, feedforward matrices, embedding matrices, and other trainable tensors.

---

## 3. Weight Decay in Plain Gradient Descent

Start with an $L_2$-regularized objective:

$$
\mathcal{J}(W)
=
\mathcal{L}_{\mathcal{B}}(W)
+
\frac{\lambda}{2}\lVert W\rVert_F^2
$$

The gradient is:

$$
\frac{\partial \mathcal{J}}{\partial W}
=
g
+
\lambda W
$$

Plain gradient descent on this objective gives:

$$
W \leftarrow W - \eta(g+\lambda W)
$$

Rearrange the update:

$$
\boxed{
W \leftarrow (1-\eta\lambda)W - \eta g
}
$$

This reveals two separate-looking effects:

- $(1-\eta\lambda)W$ shrinks the weight matrix;
- $-\eta g$ follows the mini-batch gradient.

For plain gradient descent or mini-batch SGD, $L_2$ regularization and weight decay are mathematically equivalent.

---

## 4. Adam in One Page

Adam smooths gradients and adapts step sizes per coordinate.

First moment:

$$
m^{(t)}
\leftarrow
\beta_1 m^{(t-1)}
+
(1-\beta_1)g
$$

Second moment:

$$
v^{(t)}
\leftarrow
\beta_2 v^{(t-1)}
+
(1-\beta_2)g^2
$$

Here $g^2$ means element-wise square.

Bias correction:

$$
\hat{m}^{(t)}
=
\frac{m^{(t)}}{1-(\beta_1)^t}
$$

$$
\hat{v}^{(t)}
=
\frac{v^{(t)}}{1-(\beta_2)^t}
$$

Adaptive direction:

$$
u^{(t)}
=
\frac{\hat{m}^{(t)}}{\sqrt{\hat{v}^{(t)}}+\epsilon}
$$

Adam update:

$$
\boxed{
W \leftarrow W - \eta u^{(t)}
}
$$

Adam's power comes from the denominator $\sqrt{\hat{v}^{(t)}}+\epsilon$. Coordinates with consistently large squared gradients receive smaller effective steps; coordinates with smaller squared gradients can receive larger effective steps.

---

## 5. Why Naive $L_2$ Regularization Breaks Inside Adam

If we add an $L_2$ penalty to the loss and then use Adam, the gradient entering Adam becomes:

$$
g_{\mathrm{coupled}}
=
g
+
\lambda W
$$

Adam then builds moments from this coupled gradient:

$$
m^{(t)}
\leftarrow
\beta_1 m^{(t-1)}
+
(1-\beta_1)g_{\mathrm{coupled}}
$$

$$
v^{(t)}
\leftarrow
\beta_2 v^{(t-1)}
+
(1-\beta_2)g_{\mathrm{coupled}}^2
$$

The decay term $\lambda W$ is now mixed into both $m^{(t)}$ and $v^{(t)}$. It is also adaptively scaled coordinate by coordinate.

The result is no longer clean shrinkage:

$$
\lambda W
\longrightarrow
\text{Adam moments}
\longrightarrow
\text{adaptive scaling}
\longrightarrow
\text{parameter update}
$$

> [!WARNING]
> In Adam, adding an $L_2$ penalty to the loss couples regularization with adaptive scaling. This is not equivalent to ordinary weight decay.

---

## 6. AdamW as Two Flows

AdamW keeps the loss-gradient flow separate from the weight-decay flow.

Loss-gradient flow:

$$
\mathcal{L}_{\mathcal{B}}
\longrightarrow
g
\longrightarrow
m^{(t)}, v^{(t)}
\longrightarrow
u^{(t)}
\longrightarrow
-\eta u^{(t)}
$$

Weight-decay flow:

$$
W
\longrightarrow
-\eta\lambda W
$$

Combined AdamW update:

$$
\boxed{
W \leftarrow W - \eta u^{(t)} - \eta\lambda W
}
$$

Equivalently:

$$
\boxed{
W \leftarrow (1-\eta\lambda)W - \eta u^{(t)}
}
$$

The important distinction is:

$$
\boxed{
m^{(t)} \text{ and } v^{(t)} \text{ are computed from } g,
\text{ not from } g+\lambda W.
}
$$

---

## 7. Coupled Adam Versus AdamW

| Question | Naive Adam with $L_2$ | AdamW |
|---|---|---|
| Gradient used for Adam moments | $g+\lambda W$ | $g$ |
| Decay enters first moment $m$ | Yes | No |
| Decay enters second moment $v$ | Yes | No |
| Decay is adaptively scaled | Yes | No |
| Weight decay interpretation | Entangled | Direct shrinkage |
| Practical transformer baseline | Usually not intended | Standard choice |

AdamW makes $\lambda$ easier to reason about. It controls a direct shrinkage term instead of a term hidden inside Adam's moment estimates.

---

## 8. Why AdamW Fits Transformer Training

Transformers benefit from AdamW because they are large, deep, and sensitive to optimizer details.

AdamW supports the usual transformer training stack:

- adaptive optimization through Adam moments;
- explicit weight decay for magnitude control;
- learning-rate warmup;
- cosine or linear learning-rate decay;
- gradient clipping;
- mixed precision;
- distributed training;
- parameter groups with different decay rules.

The clean mental split is:

$$
\text{Adam part}
=
\text{follow the loss landscape}
$$

$$
\text{weight-decay part}
=
\text{control parameter size}
$$

This split is especially useful when models are trained for many steps and small optimizer differences compound over time.

---

## 9. Parameter Groups

In transformer training, weight decay is usually not applied to every parameter.

Parameters commonly decayed:

- attention projection weights;
- feedforward network weights;
- other large matrix weights.

Parameters commonly excluded:

- bias terms;
- LayerNorm scale parameters;
- RMSNorm scale parameters;
- other small normalization parameters.

Embedding matrices are recipe-dependent. Some training setups decay them; others put them in a no-decay group.

The practical reason is that large weight matrices usually represent most of the model capacity, while biases and normalization parameters mainly control shifts and scales.

---

## 10. Learning Rate Schedules Still Matter

AdamW decouples weight decay from Adam's adaptive moments, but the decay term is often still multiplied by the current learning rate:

$$
-\eta\lambda W
$$

If $\eta$ changes over time, the effective shrinkage also changes over time.

During warmup:

$$
\eta \text{ is small}
\quad
\Rightarrow
\quad
\text{decay is small}
$$

During cosine decay:

$$
\eta \text{ decreases}
\quad
\Rightarrow
\quad
\text{decay also decreases}
$$

So AdamW separates weight decay from adaptive moments, but not necessarily from the learning-rate schedule.

---

## 11. Optimizer Memory Cost

AdamW stores several tensors for each trainable parameter:

- the parameter $W$;
- the gradient $g$;
- the first moment $m$;
- the second moment $v$.

For very large models, this optimizer state can consume enormous memory. This memory pressure motivates many newer optimizer methods.

The main research directions are:

- store less optimizer state;
- use lower precision for optimizer state;
- reduce the number of training steps;
- use matrix structure instead of treating all coordinates independently.

---

## 12. Frontier Optimizer Directions

AdamW remains the default reference point, but optimizer research for large language models is very active. The methods below are useful to know because they attack different bottlenecks.

### 12.1 Memory-Efficient Adaptive Optimizers

These methods aim to keep Adam-like behavior while reducing optimizer-state memory.

| Method | Main idea | What to remember |
|---|---|---|
| Adafactor | Factorize second-moment statistics for matrix-shaped parameters | Much lower memory than Adam for large matrices |
| Adam-mini | Use fewer distinct learning-rate statistics by grouping parameters | Attempts to preserve AdamW-like behavior with less second-moment state |
| 8-bit Adam | Store optimizer states in low precision | Saves memory with quantized optimizer states |
| Paged Adam | Move optimizer states through memory more carefully | Useful when GPU memory is tight |
| GaLore | Project gradients into a low-rank subspace for optimizer updates | Reduces optimizer memory while still updating full parameters |
| Q-GaLore | Combine low-rank gradient projection with quantization | Pushes memory reduction further |
| VLoRP-style methods | Vary low-rank projection granularity or subspaces | Explore more flexible low-rank optimizer states |

The shared question is:

$$
\boxed{
\text{Can we keep most of AdamW's benefit while storing less optimizer state?}
}
$$

### 12.2 Sign-Based Optimizers

Lion is the best-known recent sign-based optimizer. It uses momentum-like information, then applies a sign update.

The rough flow is:

$$
\text{momentum signal}
\longrightarrow
\operatorname{sign}(\text{signal})
\longrightarrow
\text{parameter update}
$$

Lion does not need a full second-moment tensor like Adam. This can reduce memory and simplify the update, but it often needs different learning-rate and weight-decay tuning.

### 12.3 Curvature-Aware Optimizers

These methods try to use information about curvature, not only first-order gradients.

| Method | Main idea | What to remember |
|---|---|---|
| Sophia | Use a lightweight diagonal Hessian estimate and clipping | Designed to improve step efficiency in language model pretraining |
| Shampoo | Precondition matrix-shaped parameters with matrix statistics | More structure-aware than coordinate-wise Adam |
| Distributed Shampoo | Scale Shampoo-style preconditioning to distributed training | Makes second-order-inspired updates more practical |
| SOAP | Combine Shampoo-style preconditioning with Adam-like behavior | Tries to stabilize and improve Shampoo-like training |

The shared question is:

$$
\boxed{
\text{Can curvature information reduce training steps enough to justify extra computation?}
}
$$

### 12.4 Matrix-Aware and Orthogonalized Updates

Transformer weights are often matrices, but Adam treats their coordinates mostly independently. Matrix-aware methods try to exploit the shape of these parameters.

Muon-style methods use momentum and then orthogonalize matrix updates, often through Newton-Schulz iterations. The rough idea is:

$$
\text{matrix-shaped gradient}
\longrightarrow
\text{orthogonalized update}
\longrightarrow
\text{better-conditioned training direction}
$$

This family is attractive because it tries to improve training dynamics without simply adding more Adam-style state tensors.

### 12.5 Practical Warning

Frontier optimizers can be valuable, but they are not free upgrades. They may require:

- new learning-rate ranges;
- new weight-decay settings;
- more careful stability checks;
- special distributed implementations;
- architecture-specific tuning.

> [!WARNING]
> AdamW remains the baseline because it is stable, widely supported, and well understood. A newer optimizer should be chosen because it solves a concrete bottleneck, not because it is newer.

---

## 13. Practical Decision Map

Use AdamW when:

- a stable default is needed;
- training recipes already assume AdamW;
- optimizer memory is acceptable;
- reproducibility matters.

Consider memory-efficient Adam variants when:

- optimizer state is the main bottleneck;
- model parameters fit, but Adam states do not;
- lower precision or low-rank optimizer states are acceptable.

Consider curvature-aware or matrix-aware methods when:

- reducing the number of training steps matters more than per-step simplicity;
- the implementation is mature enough for the training setup;
- tuning time is available.

The conservative strategy is:

$$
\boxed{
\text{Start from AdamW, identify the bottleneck, then choose a specialized optimizer if needed.}
}
$$

---

## 14. Common Mistakes

### Mistake 1: Calling Adam with an $L_2$ Penalty AdamW

If $g+\lambda W$ enters Adam's moment estimates, the method is coupled Adam with $L_2$ regularization, not AdamW.

### Mistake 2: Decaying Every Parameter

Biases and normalization parameters are often excluded from weight decay in transformer training. Parameter groups matter.

### Mistake 3: Thinking AdamW Removes All Interactions

AdamW separates decay from Adam's adaptive moments, but the decay term is often still multiplied by the learning rate.

### Mistake 4: Treating Frontier Optimizers as Automatic Improvements

Newer optimizers may reduce memory or improve step efficiency, but they can also introduce new hyperparameters, implementation complexity, and scale-specific behavior.

---

## 15. Further Reading

- AdamW: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Adafactor: [Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
- Lion: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)
- Sophia: [A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342)
- GaLore: [Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- Adam-mini: [Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793)
- Shampoo: [Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)
- Scalable Shampoo-style methods: [Scalable Second Order Optimization for Deep Learning](https://arxiv.org/abs/2002.09018)
- SOAP: [Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321)
- Muon at LLM scale: [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)

---

## 16. Final Picture

AdamW uses Adam's adaptive direction:

$$
u^{(t)}
=
\frac{\hat{m}^{(t)}}{\sqrt{\hat{v}^{(t)}}+\epsilon}
$$

where $\hat{m}^{(t)}$ and $\hat{v}^{(t)}$ are computed from:

$$
g = \frac{\partial \mathcal{L}_{\mathcal{B}}}{\partial W}
$$

Then it applies decoupled weight decay:

$$
\boxed{
W \leftarrow (1-\eta\lambda)W - \eta
\frac{\hat{m}^{(t)}}{\sqrt{\hat{v}^{(t)}}+\epsilon}
}
$$

One-sentence summary:

$$
\boxed{
\text{AdamW keeps adaptive optimization and weight decay separate.}
}
$$