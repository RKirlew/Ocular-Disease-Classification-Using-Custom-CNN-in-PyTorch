# Ocular Disease Classification Using Custom CNN in PyTorch

## Overview

This repository contains a PyTorch-based deep learning model for classifying ocular diseases from retinal fundus images. The model leverages a custom Convolutional Neural Network (CNN) architecture to achieve high accuracy in disease classification.

## Table of Contents

- [Project Introduction](#project-introduction)
- [Model Architecture](#model-architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Learning Outcomes](#learning-outcomes)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Introduction

In the field of medical image analysis, the identification and classification of ocular diseases from retinal images play a crucial role in early diagnosis and treatment. This project focuses on building a robust machine learning model that can accurately classify various ocular diseases from image data.

---

## Model Architecture

The model uses a custom CNN architecture consisting of:

- **Convolutional Layers**: Capturing features from images.
- **Pooling Layers**: Reducing spatial dimensions to improve performance.
- **Fully Connected Layers**: Mapping features to disease classes.

The key components include:
- `Conv2d` layers for feature extraction.
- MaxPooling layers to downsample the data.
- Fully connected layers for classification.

**Example of a basic model structure**:

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Dataset Preparation

The dataset consists of images organized by disease category. Each disease has its own folder under the main dataset directory.

**Structure Example:**

/content/Fundus/1000images/  
    ├── disease_1/  
    ├── disease_2/  
    └── ...

Images are preprocessed using resizing, normalization, and tensor conversion before being fed into the model.

---

## Training and Evaluation

The training process involves:

- Splitting the dataset into training and validation sets.
- Using a custom CNN model with Adam optimizer and Cross-Entropy loss function.
- Iteratively training the model for multiple epochs.

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
```
## Learning Outcomes

Through this project, I gained a deeper understanding of:

- Convolutional Neural Networks (CNNs): How to design, train, and optimize CNN architectures for image classification tasks.
- PyTorch: Building and managing neural networks, using libraries for model training, evaluation, and saving/loading states.
- Dataset Handling: Efficiently managing image datasets with PyTorch's Dataset and DataLoader functionalities.
- Model Optimization: Techniques for improving model performance such as data augmentation, regularization, and parameter tuning.

---

## Technologies Used

- **PyTorch**: Deep learning framework for building, training, and optimizing models.
- **Torchvision**: For image datasets and transformations.
- **Python**: Programming language for building machine learning pipelines.

---

## How to Use

- Clone the repository:

  ```bash
  git clone https://github.com/RKirlew/ocular-disease-classification.git

