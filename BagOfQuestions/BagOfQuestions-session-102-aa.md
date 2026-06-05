## Question: Autoencoder Architecture and Reconstruction Loss

An autoencoder learns a representation by compressing an input into a latent code and reconstructing the original input. For one flattened MNIST image, let $x\in\mathbb{R}^{1\times784}$, latent code $z\in\mathbb{R}^{1\times k}$, and reconstruction $\hat{x}\in\mathbb{R}^{1\times784}$.

1. Write the encoder mapping $z=f_{\theta}(x)$ and decoder mapping $\hat{x}=g_{\phi}(z)$, then combine them into one full autoencoder formula.
2. For the architecture `784 -> 128 -> 32 -> 128 -> 784`, identify the bottleneck layer and explain why it creates pressure to learn useful compressed information.
3. Write the mean squared reconstruction loss for a dataset $X\in\mathbb{R}^{n\times d}$.
4. Explain why the target in a standard autoencoder training pair is the input itself.
5. Draw a schema of the encoder, latent code, decoder, and reconstruction, labeling the main dimensions for the MNIST example.
