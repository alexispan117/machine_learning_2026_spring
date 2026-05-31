# Multimodal Alignment

## Unifying Images and Text in a Shared Semantic Space

![](./img-embedding/multimodelembedding.jpg)


- Paper: https://arxiv.org/abs/2103.00020


---

## 1. From Embeddings to a Deeper Question

In previous lecture, we established that:

* Images can be mapped to vectors (image embeddings)
* Text can be mapped to vectors (text embeddings)

These embeddings make similarity meaningful **within each modality**.


### But a limitation remains

* Image embeddings live in one space
* Text embeddings live in another space

> We still cannot directly compare an image with a piece of text

---

## 2. The Key Idea: A Unified Space

![](./img-word2vec/clipopenai2.jpg)


If both images and text can be represented as vectors, a natural question arises:

> Can we place them in the **same vector space**?

### Goal

Construct a shared space where:

* An image of a cat → vector \( v_{\text{image}} \)
* The word “cat” → vector \( v_{\text{text}} \)

And enforce:

> \( v_{\text{image}} \approx v_{\text{text}} \)

---

### General Principle

* Matching image–text pairs → close
* Non-matching pairs → far

> This is the core idea of **multimodal alignment**

---

## 3. What Is Multimodal Alignment?

Multimodal alignment is the process of:

> Learning a representation where **different types of data share the same semantic geometry**


### “Modality” means:

* Images
* Text
* Audio
* Video


### Alignment means:

> Similar concepts across modalities are mapped to nearby points

---

## 4. CLIP (Contrastive Language–Image Pre-training; By OpenAI, 2021)


![](./img-embedding/clip.jpg)


### 4.1 Training Data

CLIP is trained on pairs:

* Image: a cat
* Text: “a photo of a cat”


### 4.2 Two Encoders

CLIP consists of:

* An image encoder → maps image to a vector
* A text encoder → maps text to a vector


### 4.3 Learning Objective

The model learns by solving the following task:

> Given many images and many texts, match the correct pairs


### Contrastive Learning

![](./img-image/cl2.jpg)


* Correct image–text pair → pulled closer
* Incorrect pairs → pushed apart


> Over time, the model organizes the space so that
> **alignment emerges naturally**

---

## 5. What the Shared Space Looks Like



After training, the embedding space has the following structure:


### Alignment Across Modalities

* Image of a dog ↔ “dog” → close
* Image of a car ↔ “car” → close


### Semantic Organization

* “cat”, “kitten”, “a small cat” → nearby
* All corresponding images → nearby


> The space encodes **concepts**, not just data types

---

## 6. What This Enables


![](./img-image/clip3.jpg)

### 6.1 Cross-Modal Retrieval


#### Text → Image

* Input: “a red car”
* Output: relevant images


#### Image → Text

* Input: an image
* Output: best matching descriptions



### 6.2 Zero-Shot Classification

Instead of training a classifier:

* Represent class labels as text embeddings
* Compare image embedding to them


> No task-specific training required

---

## 7. A New Capability: Cross-Modal Clustering



### Traditional Clustering

* Cluster images using image embeddings
* Fully data-driven


### Multimodal Clustering

> Use **text embeddings as cluster centers**


### Example

Define text vectors for:

* “cat”
* “dog”
* “car”


Then:

* Map each image into the shared space
* Assign it to the nearest text vector


> This becomes a form of clustering guided by language
