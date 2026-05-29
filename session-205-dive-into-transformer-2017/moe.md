# Mixture-of-Experts (MoE)


---

## 1. Motivation: Why Dense Models Become Expensive

Traditional feedforward neural networks are typically **dense**:

* Every layer activates for every token
* Every parameter participates in computation
* Increasing capacity usually means increasing computation cost

Suppose we double model size:

* More parameters
* More FLOPs
* Higher memory usage
* Slower inference

This creates a scaling problem.

> [!INFO]
> Large language models need more capacity, but activating every parameter for every token becomes computationally expensive.

MoE attempts to solve this problem through **conditional computation**.

---

## 2. Core Idea of Mixture-of-Experts

Instead of using one large network for every token:

* Store multiple smaller networks called **experts**
* Use a routing mechanism
* Activate only a subset of experts

The idea becomes:

$$
\text{Output} = \sum_{i \in \text{selected experts}} w_i E_i(x)
$$

where:

* $E_i$ = expert network
* $w_i$ = routing weight
* only selected experts are computed

This creates a model that is:

* large in parameter count
* sparse in computation

---

## 3. Dense vs Sparse Computation

### Dense Model

Every token activates:

$$
\text{All parameters} \rightarrow \text{active}
$$

Computation cost grows with parameter count.

### Sparse MoE Model

Only selected experts activate:

$$
\text{Subset of parameters} \rightarrow \text{active}
$$

Parameter count may grow significantly while computation increases slowly.

---

## 4. Components of an MoE Layer

A typical MoE layer contains:

### Experts

Experts are small neural networks.

For expert $i$:

$$
E_i(x)
$$

Each expert learns specialized behavior.

Examples:

* syntax expert
* reasoning expert
* multilingual expert
* code expert

### Router

Router decides which experts should process the token.

Router produces expert scores:

$$
r = Wx + b
$$

where:

* input dimension = token representation size
* output dimension = number of experts

Higher score means greater preference.

### Top-$k$ Selection

Router usually chooses only several experts:

$$
\text{Top-}k(r)
$$

Examples:

* Top-1 routing
* Top-2 routing

If there are 64 experts:

* only 2 may activate

---

## 5. Example Flow of an MoE Layer

Assume:

* 4 experts
* top-2 routing

Input token:

$$
x
$$

Router computes:

$$
r = [0.2, 1.8, 0.5, 2.1]
$$

Top-2 experts:

* Expert 4
* Expert 2

Only these experts execute:

$$
y = w_2E_2(x) + w_4E_4(x)
$$

Experts 1 and 3 remain inactive.

---

## 6. Why MoE Increases Capacity

Suppose:

* One expert contains 10M parameters
* 64 experts exist

Stored capacity:

$$
64 \times 10M = 640M
$$

If only 2 experts activate:

effective computation:

$$
2 \times 10M = 20M
$$

Therefore:

* huge stored capacity
* relatively small computation cost

> [!INFO]
> MoE separates **stored capacity** from **activated computation**.

---

## 7. Why Parameter Counting Matters

In dense models:

$$
\text{Stored Parameters}
=\text{Active Parameters}
$$

In MoE:

$$
\text{Stored Parameters}
\neq
\text{Active Parameters}
$$

This distinction creates two different questions:

### Total Stored Parameters

How many parameters exist in memory?

### Activated Parameters

How many parameters execute for one token?

> [!WARNING]
> In MoE systems, inactive experts still consume memory because their weights must remain available for future routing decisions.

---

## 8. Parameter Counting for Dense Layers

For a dense layer:

Input dimension:

$$
d_{\text{in}}
$$

Output dimension:

$$
d_{\text{out}}
$$

Parameter count:

$$
d_{\text{in}} \times d_{\text{out}}
+
d_{\text{out}}
$$

because:

* weights:

$$
d_{\text{in}} \times d_{\text{out}}
$$

* biases:

$$
d_{\text{out}}
$$

This formula will be repeatedly used when counting expert parameters.

---

## 9. Router Parameter Counting

Router is usually another dense layer.

Router parameters:

$$
d_{\text{input}}
\times
N_{\text{experts}}
+
N_{\text{experts}}
$$

where:

* $d_{\text{input}}$ = token dimension
* $N_{\text{experts}}$ = number of experts

Router size is usually much smaller than total expert size.

---

## 10. Trade-Offs of MoE

### Advantages

* larger capacity
* sparse computation
* scalable models
* expert specialization

### Disadvantages

* routing complexity
* load imbalance
* communication overhead
* harder distributed training

> [!WARNING]
> If routing sends most tokens to only a few experts, some experts may become overloaded while others rarely learn.
