# deepfake-detection-vitam
# Vision-Transformer-Classifier

A Vision Transformer (ViT) for end-to-end image classification on custom datasets, using transfer learning and mixed precision training with PyTorch.

---

## Vision Transformer Project

This repository contains the implementation of a **Vision Transformer (ViT-B/16)** deep learning model designed for robust image classification tasks. The model leverages powerful self-attention mechanisms to capture global context in images and uses transfer learning for efficient training on custom datasets.

---

## Project Overview

Vision Transformers (ViTs) have demonstrated state-of-the-art performance on various image classification benchmarks by replacing traditional convolutional layers with multi-head self-attention.  
This project uses a pre-trained ViT-B/16 from torchvision and fine-tunes it for a custom multi-class image classification problem.

The complete pipeline includes:
- **Flexible Data Loading:** Uses PyTorch’s `ImageFolder` for easy dataset setup.
- **Transfer Learning:** Loads ImageNet weights and replaces the output head for custom classes.
- **Mixed Precision Training:** Boosts training speed and reduces memory usage.
- **Evaluation Tools:** Generates classification reports and confusion matrices for model performance analysis.
- **Visualization:** Includes helpers to visualize images and results.

---

## Features

 **Pre-trained Vision Transformer:** Leverages ViT-B/16 with ImageNet weights for strong feature extraction.

 **Custom Output Head:** Easily adjustable for any number of classes.

 **Mixed Precision Training:** Uses `torch.cuda.amp` for faster training on GPUs.

 **Early Stopping:** Includes early stopping logic to prevent overfitting.

 **Evaluation Reports:** Computes accuracy, precision, recall, F1-score, and saves confusion matrix heatmaps.

 **Reusable Helpers:** Utilities for visualizing images and loading saved models.

---

## Repository Structure

| File | Description |
|------|--------------|
| `dataset.py` | Defines data loaders using torchvision’s `ImageFolder` and data transforms. |
| `model.py` | Builds the Vision Transformer model with a custom output head. |
| `train.py` | Contains the training loop, mixed precision logic, and early stopping. |
| `evaluate.py` | Provides functions to compute accuracy, generate classification reports, and plot/save confusion matrices. |
| `utils.py` | Includes helper functions for visualizing images and loading trained models. |
| `figures/` | Directory to save plots such as the confusion matrix heatmap. |

---

## Dataset

This project uses the **Deepfake and Real Images Validation** dataset available on Hugging Face:
 [Deepfake-and-real-images-validation](https://huggingface.co/datasets/JamieWithofs/Deepfake-and-real-images-validation)

You can download it directly from the Hugging Face link above.  
The dataset contains both real and deepfake images, which can be used for binary image classification.
