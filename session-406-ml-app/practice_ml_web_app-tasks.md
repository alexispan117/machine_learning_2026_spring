# Course Tutorial and Lab Guide: Machine Learning Web Apps

This guide covers the end-to-end development, local deployment, cloud migration, and containerization of two machine learning applications: the **MNIST Digit Predictor** and the **UFO Project**.

* **Video Playlist:** [Bilibili Video Tutorial Series](https://space.bilibili.com/472463946/lists/211561?type=season)
* **Repository:** [Gitee - ML Web App](https://gitee.com/lundechen/machine_learning_web_app)
* **Core Stack:** FastAPI, Uvicorn, Streamlit

---

## Core Projects Overview

### 1. MNIST Digit Predictor

* **Frontend:** Built with Streamlit, utilizing a **drawable canvas** component for real-time user drawings.
* **Backend:** Powered by **FastAPI** and **Uvicorn** running on a local server.
* **Goal:** Capture canvas drawings, send them to the backend API, and return the model's digit prediction.

### 2. UFO Project

* **Migration Task:** Transition the existing project from Flask to **FastAPI + Uvicorn**.
* **Resources:**
* Guidelines: [`practice-UFO-flask-app.md`](https://www.google.com/search?q=./practice-UFO-flask-app.md)
* Reference Solution: [`practice-UFO-flask-app-solution`](https://www.google.com/search?q=practice-UFO-flask-app-solution)



> Note: Some legacy code might be outdated, but you can leverage modern AI tools to debug and update dependencies. Follow the roadmap step by step.

---

## The 3-Task Roadmap

Each task is time-boxed to approximately 1 hour. Apply the video workflows to **both** the MNIST and UFO projects.

* **Video Playlist:** [Bilibili Video Tutorial Series](https://space.bilibili.com/472463946/lists/211561?type=season)
* **Repository:** [Gitee - ML Web App](https://gitee.com/lundechen/machine_learning_web_app)

### Task 1: Local Deployment Workflow (1 Hour)

Master the core architecture and get both applications running seamlessly on your local machine.

* **Video 1:** Get Inspired!
* **Video 2:** Training models using Jupyter Notebooks
* **Video 3:** Interacting with trained models via Streamlit
* **Video 4:** Local Deployment: System Architecture Overview
* **Video 5:** Local Deployment: Building the FastAPI Backend & Uvicorn
* **Video 6:** Local Deployment: Designing the Streamlit Frontend
* **Video 7:** Local Deployment: Localhost Integration Demo
* **UFO Project Task:** Apply this entire 7-step workflow to the UFO Project.

### Task 2: Cloud Deployment (1 Hour)

Move your local applications to a live production environment.

* **Video 8:** Cloud Deployment: Purchasing and connecting to a Cloud VM
* **Video 9:** Cloud Deployment: Configuring FastAPI & the Backend on the cloud
* **Video 10:** Cloud Deployment: Deploying Streamlit & the Frontend
* **UFO Project Task:** Apply this cloud migration workflow to the UFO Project.

### Task 3: Containerization with Docker (1 Hour)

Standardize your applications for reliable deployment anywhere.

* **Video 11:** Learning the fundamentals of Docker
* **Video 12:** Cloud Deployment using Docker containers
* **UFO Project Task:** Containerize and deploy the UFO Project using Docker as well.

---

## The MNIST Optimization Challenge (Open-Ended)

The baseline model for the MNIST canvas drawing often yields imperfect or incorrect predictions due to variations in live user drawings. **Your goal is to improve this accuracy.**

### Suggested Approaches

Previous students have successfully solved this by exploring:

1. **Data Augmentation:** Introducing rotations, scaling, and noise to the training set to mimic realistic mouse/canvas drawings.
2. **Deeper Architectures:** Building and fine-tuning deep Convolutional Neural Networks (CNNs).

### Reference Implementation Code

You can examine these legacy files from the repository to understand the original layout:

* [Frontend Script (`frontend.py`)](https://gitee.com/lundechen/machine_learning_web_app/blob/master/frontend.py)
* [Backend Script (`backend.py`)](https://gitee.com/lundechen/machine_learning_web_app/blob/master/backend.py)
* [Monolithic App (`app.py`)](https://gitee.com/lundechen/machine_learning_web_app/blob/master/app.py)

---

## Gallery (for reference, not necessarily good)

Check out these historical repositories from past students to see different approaches to the assignment:

* [Student Project Reference 1 (EmelinePer)](https://github.com/EmelinePer/Practice_5)
* [Student Project Reference 2 (EmporioSabo)](https://github.com/EmporioSabo/MNIST-Digit-Recognizer)
* https://github.com/EmporioSabo/UFO-Sighting-Predictor

