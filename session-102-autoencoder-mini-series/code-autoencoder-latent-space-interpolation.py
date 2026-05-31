# -*- coding: utf-8 -*-
"""
Latent Space Interpolation with AutoEncoders using MNIST CSV
Complete Python script for training an AutoEncoder on MNIST data and generating GIFs of latent space interpolations.
"""

# Online Visualization:
# - https://introduction-to-autoencoders.vercel.app/

# !IMPORTANT!
# Compare this with:
# https://keras.io/examples/generative/vae/
# TELL ME WHAT IS DIFFERENT BETWEEN THEM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as imageio
import io
import gc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

# ------------------------------------------------------------------------------
# 1. Load MNIST CSV data
#    Converts pixel values to float32 and normalizes to [0, 1]
# ------------------------------------------------------------------------------
train_data = pd.read_csv("mnist_train.csv")
y = train_data.iloc[:, 0].to_numpy(np.int64)  # Labels
X = train_data.iloc[:, 1:].to_numpy(np.float32) / 255.0  # Pixel data normalized
print("Dataset shape:", X.shape)  # Expected: (6000, 784)
os.makedirs("latent-space-interpolation", exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Define Dataset and DataLoader for PyTorch
#    Simple dataset that returns tensors for training
# ------------------------------------------------------------------------------
class MNISTDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx])


dataset = MNISTDataset(X)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# ------------------------------------------------------------------------------
# 3. Define AutoEncoder model with latent dimension 2
#    Latent dim 2 keeps space interpretable for easy interpolation visualization
# ------------------------------------------------------------------------------
latent_dim = 2


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),  # Output pixel values in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# ------------------------------------------------------------------------------
# 4. Train the AutoEncoder model
#    Uses MSE loss and Adam optimizer over multiple epochs
# ------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        loss = criterion(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(loader):.6f}")

# ------------------------------------------------------------------------------
# 5. Encode entire dataset into latent space
#    Extracts latent vectors for all images to use in interpolation
# ------------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X).to(device)
    latent_vectors = model.encoder(X_tensor).cpu().numpy()
print("Latent shape:", latent_vectors.shape)  # Expected: (6000, 2)


# ------------------------------------------------------------------------------
# 6. Helper function to find a random index for a given digit label
# ------------------------------------------------------------------------------
def get_digit_index(digit):
    indices = np.where(y == digit)[0]
    return np.random.choice(indices)


# ------------------------------------------------------------------------------
# 7. Core latent interpolation function
#    Linearly interpolates between two latent vectors z1 and z2
#    z1: latent vector of digit A, z2: latent vector of digit B
#    steps: number of interpolation points (default 40)
#    Returns a list of decoded images (28x28 arrays) for each interpolation step
# ------------------------------------------------------------------------------
def latent_interpolation(z1, z2, steps=40):
    interpolations = []
    for t in np.linspace(0, 1, steps):
        # Linear interpolation in latent space: z(t) = (1 - t)*z1 + t*z2
        # At t=0: digit A, at t=1: digit B, intermediate t: smooth blend
        z = (1 - t) * z1 + t * z2
        z_tensor = torch.tensor(z).float().to(device)
        with torch.no_grad():
            recon = model.decoder(z_tensor).cpu().numpy()
        # Reshape flattened 784 vector back to 28x28 image
        interpolations.append(recon.reshape(28, 28))
    return interpolations


# ------------------------------------------------------------------------------
# 8. GIF generator for interpolation between two digits
#    Creates a GIF showing morph from digit_a to digit_b
#    Uses buffer-based approach to save memory and generate clean frames
# ------------------------------------------------------------------------------
def make_interpolation_gif(digit_a, digit_b):
    idx_a = get_digit_index(digit_a)
    idx_b = get_digit_index(digit_b)
    z1 = latent_vectors[idx_a]
    z2 = latent_vectors[idx_b]
    images = latent_interpolation(z1, z2, steps=40)
    frames = []
    dpi, figsize = 150, (3, 3)
    for step, img in enumerate(images):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{digit_a} → {digit_b}\nstep {step+1}/{len(images)}", fontsize=10)
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        buf.close()
        plt.close(fig)
        gc.collect()
    filename = f"./latent-space-interpolation/digit_{digit_a}_to_{digit_b}.gif"
    print("Saving", filename)
    imageio.mimsave(filename, frames, duration=0.08, fps=12)
    print("Saved", filename)


# ------------------------------------------------------------------------------
# 9. Generate specific interpolation GIFs for fixed digit pairs
# ------------------------------------------------------------------------------
make_interpolation_gif(1, 7)
make_interpolation_gif(2, 8)
make_interpolation_gif(3, 5)
make_interpolation_gif(4, 9)

# ------------------------------------------------------------------------------
# 10. Generate additional GIFs for random digit pairs
#     Creates 10 random pairs and generates GIFs for each
# ------------------------------------------------------------------------------
pairs = []
digits = list(range(10))
for _ in range(10):
    a, b = np.random.choice(digits, 2, replace=False)
    pairs.append((a, b))
print("Random pairs:", pairs)
for a, b in pairs:
    make_interpolation_gif(a, b)
