## Question: Denoising Autoencoders and Anomaly Detection

A denoising autoencoder receives a corrupted input but learns to reconstruct the clean original input. The same reconstruction idea can also be used for anomaly detection.

1. Let $x_{\mathrm{noisy}}=x+\epsilon$. Write the denoising autoencoder prediction $\hat{x}$ using an encoder $f_{\theta}$ and decoder $g_{\phi}$.
2. In denoising training, should the reconstruction loss compare $\hat{x}$ with $x_{\mathrm{noisy}}$ or with the clean $x$? Explain why.
3. Explain how denoising encourages the latent code to preserve stable structure rather than random noise.
4. For anomaly detection, define a reconstruction-error score $E(x)$ and a threshold rule using $\tau$.
5. Give one reason why reconstruction error is useful for anomaly detection and one reason why it is not a perfect anomaly detector.
