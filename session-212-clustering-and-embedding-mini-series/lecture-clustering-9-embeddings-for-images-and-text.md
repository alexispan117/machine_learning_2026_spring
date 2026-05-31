#  Embeddings for Images and Text

$$\text{Making Similarity Meaningful}$$

Online visualization:
- https://projector.tensorflow.org/
- https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html

---


## 1. Image Embeddings: From Pixels to Meaning



![](./img-image/1e.jpg)

An image embedding is:

> A **vector** that captures the **content of an image**, not its raw pixels


In embedding space:

* Images of cats → cluster together
* Cats vs cars → well separated


> Now distance becomes meaningful


![](./img-embedding/2.jpg)



---

## 2. Text Embeddings: From Words to Meaning



![](./img-embedding/3.jpg)

A text embedding is:

> A vector representing the **meaning of a word, sentence, or document**


Embeddings can capture relationships like:

* king − man  ≈ queen - woman


> This shows that the space encodes **structure**, not just labels



![](./img-embedding/5.jpg)
![](./img-embedding/manwomankingqueen.jpg)


In embedding space:

* Similar words → close
* Similar sentences → grouped


> Meaning becomes geometrically represented

---

## 3. A Unified View: Embeddings as Geometry of Meaning

![](./img-embedding/4.jpg)



We can now unify both cases:


### Image Embeddings

* Input: pixels
* Output: semantic vector


### Text Embeddings

* Input: tokens
* Output: semantic vector



### Shared Goal

> Learn a space where:

* Geometry reflects meaning
* Distance reflects similarity

---

## 4. Why This Fixes Clustering


### Before (Raw Space)

* Distance is unreliable
* Clusters are unclear
* Algorithms fail


### After (Embedding Space)

* Structure emerges
* Clusters become separable
* Distance is meaningful
