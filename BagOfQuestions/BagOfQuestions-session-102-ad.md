## Question: Variational Autoencoder Objective

A variational autoencoder maps each input to a distribution over latent variables instead of a single deterministic latent point. For each input $x$, the encoder predicts

$$
q_{\phi}(z\mid x)
=
\mathcal{N}
\left(
\mu_{\phi}(x),
\operatorname{diag}(\sigma_{\phi}^2(x))
\right),
$$

and the prior is usually $p(z)=\mathcal{N}(0,I)$.

1. Explain why a deterministic autoencoder does not automatically provide a smooth or sampleable latent space.
2. If the latent dimension is $k$, state the shapes of $\mu_{\phi}(x)$ and $\log\sigma_{\phi}^2(x)$ for one input example, and explain why the encoder head has $2k$ outputs.
3. Write the VAE loss as reconstruction loss plus a KL-divergence term.
4. Write the closed-form KL divergence for a diagonal Gaussian posterior and standard normal prior.
5. Explain what can happen if the KL term is too weak, and what can happen if it is too strong.
