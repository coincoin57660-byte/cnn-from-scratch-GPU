# CNN From Scratch with PyTorch

A convolutional neural network implemented almost entirely from scratch using only tensor operations from PyTorch.

The goal of this project is to understand how deep learning works internally by implementing:

- Convolution
- Max Pooling
- ReLU
- Fully Connected layers
- Forward propagation
- Backpropagation
- Gradient descent

without relying on `torch.nn.Conv2d`, `torch.nn.Linear`, `torch.optim`, or automatic differentiation.

---

## Features

- Custom convolution layer
- Custom max pooling layer
- Manual backpropagation
- GPU support (CUDA)
- MNIST-like dataset
- Cross-entropy loss
- He (Kaiming) initialization

---

## Network

Input

28×28 grayscale image

↓

Conv(1→32, 3×3)

↓

ReLU

↓

MaxPool

↓

Conv(32→64, 3×3)

↓

ReLU

↓

MaxPool

↓

Flatten

↓

Linear 3136 → 750

↓

ReLU

↓

Linear 750 → 150

↓

ReLU

↓

Linear 150 → 10

---

## Performance

GPU tested:

- NVIDIA RTX 4070
- CUDA 12.6

Training is entirely executed on GPU.

---

## Project goals

This project was created for educational purposes to understand the mathematics and implementation behind convolutional neural networks.
