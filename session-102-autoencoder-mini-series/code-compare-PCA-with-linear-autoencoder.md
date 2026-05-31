Open Practice session.

General idea:

Do it on the mnist dataset.


```python
# PCA as a baseline: compare with linear autoencoder
from sklearn.decomposition import PCA

# 1. Standard PCA approach
pca = PCA(n_components=32)
X_reduced = pca.fit_transform(images.flatten(start_dim=1))
X_reconstructed_pca = pca.inverse_transform(X_reduced)

# 2. Linear Autoencoder reconstruction
# The MSE of a fully trained Linear AE should converge 
# to the same value as the PCA reconstruction error.
```

