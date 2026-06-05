## Question: Tokenization Beyond Text

Tokenization means converting raw data into a sequence of units that a model can process. For text, tokens are often discrete IDs. For images, audio, video, or biological sequences, tokens may be patches, frames, tiles, motifs, or learned codebook entries.

1. An image has size $224\times224\times3$, and a vision transformer uses patches of size $16\times16$. Compute the number of image patch tokens

$$
N=\frac{224}{16}\cdot\frac{224}{16}.
$$

2. If the patch size changes from $16\times16$ to $8\times8$, explain what happens to the number of patch tokens and why this affects attention compute.
3. For audio, describe the pipeline

$$
\text{waveform}\rightarrow\text{spectrogram}\rightarrow\text{time-frequency tiles}\rightarrow\text{embeddings}.
$$

4. In a discrete codebook method, an encoder maps each region representation $h_i$ to the nearest codebook vector:

$$
z_i=\arg\min_k\left\|h_i-c_k\right\|_2^2.
$$

Explain how this makes a non-text input look more like a sequence of token IDs.
5. Give one example of information that a poor tokenization scheme might hide before the model ever sees the data.
