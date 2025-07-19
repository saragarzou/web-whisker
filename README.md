# Whisker: Draw a Digit, Get That Many Cats

![status](https://img.shields.io/badge/status-Work%20in%20Progress-red)

[Whisker](https://web-whisker.vercel.app/) is a web app that uses a custom-built neural network to recognize handwritten digits (0–9). When a user draws a number and submits it, Whisker predicts the digit and displays that number of cats on screen.

---

## Neural Network Architecture

The digit classifier is a simple feedforward neural network with:

- **Input layer**: 784 neurons (28x28 pixels flattened)
- **Hidden layers**:
  - 1st hidden layer: 128 neurons (ReLU)
  - 2nd hidden layer: 64 neurons (ReLU)
- **Output layer**: 10 neurons (Softmax), representing digits 0 to 9
