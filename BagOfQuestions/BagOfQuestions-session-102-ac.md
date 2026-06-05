## Question: Linear Autoencoders, PCA, and Latent Interpolation

An undercomplete linear autoencoder uses a linear encoder and decoder with squared reconstruction error. This connects neural dimensionality reduction to PCA. Nonlinear autoencoders can also produce latent spaces that can be inspected by interpolation.

1. Write the linear encoder and decoder formulas using $z=xW_e$ and $\hat{x}=zW_d$.
2. Explain the connection between an undercomplete linear autoencoder trained with squared reconstruction error and the top principal-component subspace from PCA.
3. Why can a nonlinear autoencoder learn representations that PCA cannot capture?
4. Given two latent points $z_a$ and $z_b$, write the linear interpolation formula $z(t)$ for $t\in[0,1]$.
5. Explain what it means if decoded interpolation samples change smoothly from one digit-like image to another.
