# AI vs Human Image Detection

A deep learning-based model to detect whether an image is **AI-generated or Real**, along with an explanation of the reasoning behind the model’s decision.

---

## Overview

With the increasing realism standards of image generative models, like Midjourney, DALL-E, etc. It is getting harder to distinguish between AI-generated and real images.

This project focuses on building a **binary image classifier** and augmenting it with **interpretability**, so that the prediction is not just a label, but also something that can be understood.

---

## Technical Blog

I wrote a complete breakdown of this project here: [My First Machine Learning Project](https://firstproj-blog.hashnode.dev/my-first-machine-learning-project-ai-vs-human-image-detection)

---

## Model Details

* Architecture: **ResNet18 (fine-tuned)**
* Task: Binary Classification (**AI vs REAL**)
* Framework: **PyTorch**
* Input Size: `224 × 224`
* Output:

  * Probability score
  * Predicted class
  * Short explanation based on attention distribution (The Reasoning Behind Its Decision)

---

## Dataset

* Total Images: **200,000+**
* Sources:

  * [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images?resource=download)
  * [Dragon dataset](https://huggingface.co/datasets/lesc-unifi/dragon/viewer/Large)
  * [Space Images](https://bit.ly/4sdOe5j)
  * [Div2k High Res dataset](https://bit.ly/4bP0Pae)
  * [Face Detection Dataset](https://bit.ly/4m6IXuN)

### Preprocessing

* Resize to `224 × 224`
* Normalize using ImageNet mean/std
* Removed low-quality / grayscale / noisy samples manually.

---

## Tech Stack

* **Backend:** Flask
* **Model:** PyTorch + torchvision

---

## Features

* Upload image and get prediction instantly
* Displays confidence score
* Provides a simple reason for the prediction

---

## Project Structure

```
AI_Vs_Human_IMG_Detection/

│
├── app.py
├── predict.py
├── trained_resnet18_binary.pth
└── README.md
```

---

## Deployement Link 

- In Process...

---

## Workflow

1. User uploads an image
2. Image is sent to Flask API
3. Model processes the image
4. Grad-CAM heatmap is generated
5. Prediction + confidence + Reason is returned to the user.

---

## Future Improvements

* Train on larger and more diverse datasets
* Improve robustness on highly realistic AI images
* Replace heuristic explanations with learned explanations
* Deploy full pipeline on cloud

---
