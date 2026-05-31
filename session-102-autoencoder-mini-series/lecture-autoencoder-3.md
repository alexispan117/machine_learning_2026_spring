# The Reparameterization Trick

The reparameterization trick lets gradients flow through stochastic latent sampling in variational autoencoders.

![](./img/z1.jpg)

---

## 1. The Training Problem

In a VAE, the encoder predicts:

$$
q_{\phi}(z \mid x)
=
\mathcal{N}
\left(
\mu_{\phi}(x),
\operatorname{diag}
\left(
\sigma_{\phi}^2(x)
\right)
\right)
$$

Then we sample:

$$
z \sim q_{\phi}(z \mid x)
$$

The decoder reconstructs:

$$
\hat{x}=g_{\theta}(z)
$$

The problem is that naive sampling creates a stochastic node between encoder outputs and the loss.

---

## 2. Why Naive Sampling Blocks Learning

The loss depends on the sampled latent variable:

$$
\mathcal{L}
=
\mathcal{L}
\left(
\hat{x},
x
\right)
$$

and:

$$
\hat{x}=g_{\theta}(z)
$$

But if we treat:

$$
z \sim \mathcal{N}(\mu,\sigma^2)
$$

as an opaque random draw, it is not a deterministic differentiable function of $\mu$ and $\sigma$.

Backpropagation needs gradients such as:

$$
\frac{\partial \mathcal{L}}{\partial \mu}
$$

and:

$$
\frac{\partial \mathcal{L}}{\partial \sigma}
$$

Naive sampling hides the path those gradients should use.

---

## 3. Separate Randomness from Parameters

The trick is to sample noise from a fixed distribution:

$$
\epsilon \sim \mathcal{N}(0,I)
$$

Then construct:

$$
\boxed{
z = \mu + \sigma \odot \epsilon
}
$$

Now randomness comes from $\epsilon$, while $\mu$ and $\sigma$ appear in a deterministic differentiable formula.

![](./img/z3.jpg)

---

## 4. Gradient Flow

After reparameterization:

$$
z(\mu,\sigma,\epsilon)
=
\mu+\sigma\odot\epsilon
$$

For a fixed sampled $\epsilon$:

$$
\frac{\partial z}{\partial \mu}=1
$$

and:

$$
\frac{\partial z}{\partial \sigma}=\epsilon
$$

This restores a differentiable computational path from the reconstruction loss back into the encoder.

![](./img/z2.jpg)

---

## 5. Log Variance Form

Implementations usually predict log variance:

$$
\ell = \log \sigma^2
$$

Then:

$$
\sigma
=
\exp
\left(
\frac{1}{2}\ell
\right)
$$

Sampling becomes:

$$
\boxed{
z
=
\mu
+
\exp
\left(
\frac{1}{2}\ell
\right)
\odot
\epsilon
}
$$

This avoids directly predicting a positive standard deviation.

---

## 6. PyTorch Implementation

```python
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + std * eps
    return z
```

This code corresponds exactly to:

$$
z
=
\mu
+
\exp
\left(
\frac{1}{2}\log\sigma^2
\right)
\odot
\epsilon
$$

The sampled tensor `eps` has no learnable parameters. The gradient flows through `mu` and `std`.

---

## 7. VAE Forward Pass

A minimal VAE forward pass has this structure:

```python
def forward(self, x):
    hidden = self.encoder(x)
    mu = self.mu_head(hidden)
    log_var = self.log_var_head(hidden)
    z = reparameterize(mu, log_var)
    x_hat = self.decoder(z)
    return x_hat, mu, log_var
```

The model returns $\mu$ and $\log\sigma^2$ because the loss needs them for the KL term.

---

## 8. VAE Loss in Code

The VAE loss combines reconstruction and KL divergence:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{recon}}
+
D_{\mathrm{KL}}
\left(
q_{\phi}(z \mid x)
\|
p(z)
\right)
$$

For diagonal Gaussian posterior and standard normal prior:

$$
D_{\mathrm{KL}}(q\|p)
=
\frac{1}{2}
\sum_{j=1}^{k}
\left(
\mu_j^2
+
\sigma_j^2
-
\log\sigma_j^2
-
1
\right)
$$

Using `log_var`, this can be implemented as:

```python
def vae_loss(x_hat, x, mu, log_var):
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = 0.5 * torch.sum(
        mu.pow(2) + log_var.exp() - log_var - 1
    )
    return recon + kl
```

---

## 9. Common Debugging Checks

When training a VAE, inspect:

- reconstruction samples;
- random samples from $z \sim \mathcal{N}(0,I)$;
- interpolation between latent points;
- average reconstruction loss;
- average KL divergence;
- ranges of $\mu$ and $\log\sigma^2$;
- whether the decoder ignores $z$.

> [!WARNING]
> If the KL term collapses to nearly zero and reconstructions ignore the latent code, the model may have posterior collapse. This is common with powerful decoders.

---

## 10. Latent Interpolation

Latent interpolation checks whether the learned space is smooth.

Given two latent points $z_a$ and $z_b$, interpolate:

$$
z(t)
=
(1-t)z_a + tz_b
$$

where:

$$
t \in [0,1]
$$

Decode each $z(t)$ and inspect whether outputs change smoothly.

The folder includes interpolation examples such as:

```text
latent-space-interpolation/digit_1_to_2.gif
latent-space-interpolation/digit_5_to_9.gif
```

---

## 11. Why This Trick Matters

The reparameterization trick changes:

$$
z \sim \mathcal{N}(\mu,\sigma^2)
$$

into:

$$
z = \mu + \sigma \odot \epsilon,
\quad
\epsilon \sim \mathcal{N}(0,I)
$$

The distribution is the same, but the computational graph is trainable.

---

## 12. Summary

The reparameterization trick is the bridge between probabilistic sampling and gradient-based learning.

The central formula is:

$$
\boxed{
z
=
\mu
+
\sigma
\odot
\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
}
$$

Without this trick, the VAE encoder would not receive a useful reconstruction gradient through sampled latent variables.
