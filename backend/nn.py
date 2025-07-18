import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from typing import Tuple


class NeuralNetwork:
    def __init__(self, layer_sizes: Tuple[int, ...]):
        """
        Initialize the neural network with given layer sizes.
        Parameters:
            layer_sizes (Tuple[int, ...]): Number of neurons per layer including input and output.
        """
        self.layer_sizes = layer_sizes
        self.parameters = {}
        self.L = len(layer_sizes) - 1 
        self._init_parameters()

    def _init_parameters(self):
        """Initialize weights and biases using He initialization."""
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l - 1]) * np.sqrt(2 / self.layer_sizes[l - 1])
            self.parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

    @staticmethod
    def relu(Z):
        """Apply ReLU activation function."""
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        """Compute derivative of ReLU."""
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        """Apply softmax function to output layer."""
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    @staticmethod
    def one_hot(Y, num_classes=10):
        """Convert labels to one-hot encoded format."""
        one_hot_Y = np.zeros((Y.size, num_classes))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def forward(self, X):
        """
        Perform forward propagation through the network.
        Returns a cache of intermediate activations and linear combinations.
        """
        cache = {"A0": X}
        for l in range(1, self.L):
            Z = self.parameters[f"W{l}"] @ cache[f"A{l - 1}"] + self.parameters[f"b{l}"]
            A = self.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        ZL = self.parameters[f"W{self.L}"] @ cache[f"A{self.L - 1}"] + self.parameters[f"b{self.L}"]
        AL = self.softmax(ZL)
        cache[f"Z{self.L}"] = ZL
        cache[f"A{self.L}"] = AL
        return cache

    def compute_loss(self, AL, Y):
        """
        Compute the cross-entropy loss.
        Parameters:
            AL (np.ndarray): Output probabilities
            Y (np.ndarray): Ground truth labels
        Returns:
            float: Cross-entropy loss
        """
        m = Y.shape[0]
        one_hot_Y = self.one_hot(Y)
        loss = -np.sum(one_hot_Y * np.log(AL + 1e-8)) / m
        return loss

    def backward(self, cache, X, Y):
        """
        Perform backward propagation to compute gradients.
        Returns a dictionary of gradients for all parameters.
        """
        grads = {}
        m = X.shape[1]
        one_hot_Y = self.one_hot(Y)
        dZL = cache[f"A{self.L}"] - one_hot_Y
        grads[f"dW{self.L}"] = dZL @ cache[f"A{self.L - 1}"].T / m
        grads[f"db{self.L}"] = np.sum(dZL, axis=1, keepdims=True) / m

        for l in reversed(range(1, self.L)):
            dZ = (self.parameters[f"W{l + 1}"].T @ dZL) * self.relu_derivative(cache[f"Z{l}"])
            dW = dZ @ cache[f"A{l - 1}"].T / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db
            dZL = dZ

        return grads

    def update_parameters(self, grads, learning_rate):
        """Update weights and biases using gradient descent."""
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
        
    def save_weights(self, filename):
        """Save model weights and biases to a .npz file."""
        np.savez(filename, **self.parameters)

    def load_weights(self, filename):
        """Load model weights and biases from a .npz file."""
        data = np.load(filename)
        self.parameters = {k: data[k] for k in data}

    def predict(self, X):
        """Predict class labels for given inputs."""
        cache = self.forward(X)
        AL = cache[f"A{self.L}"]
        return np.argmax(AL, axis=0)

    def accuracy(self, preds, labels):
        """Compute accuracy of predictions."""
        return np.mean(preds == labels)

    def train(self, X, Y, X_val, Y_val, iterations=1000, learning_rate=0.1):
        """
        Train the neural network using full-batch gradient descent.
        Logs loss and accuracy every 100 iterations.
        """
        for i in range(iterations):
            cache = self.forward(X)
            loss = self.compute_loss(cache[f"A{self.L}"], Y)
            grads = self.backward(cache, X, Y)
            self.update_parameters(grads, learning_rate)

            if i % 100 == 0 or i == iterations - 1:
                train_acc = self.accuracy(self.predict(X), Y)
                val_acc = self.accuracy(self.predict(X_val), Y_val)
                print(f"[{i:04d}] Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    def visualize_prediction(self, X, Y, index):
        """Display a single image and the model's prediction."""
        image = X[:, index, None]
        label = Y[index]
        prediction = self.predict(image)[0]
        plt.imshow(image.reshape(28, 28) * 255, cmap="gray")
        plt.axis("off")
        plt.title(f"Predicted: {prediction} | Label: {label}")
        plt.show()


def preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    Splits training data into training and validation sets.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val = x_train[:50000], x_train[50000:]
    y_train, y_val = y_train[:50000], y_train[50000:]
    X_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
    X_val = x_val.reshape(x_val.shape[0], -1).T / 255.0
    X_test = x_test.reshape(x_test.shape[0], -1).T / 255.0
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data()
    model = NeuralNetwork(layer_sizes=(784, 128, 64, 10))
    model.train(X_train, Y_train, X_val, Y_val, iterations=1000, learning_rate=0.05)

    test_preds = model.predict(X_test)
    test_acc = model.accuracy(test_preds, Y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    for idx in range(4):
        model.visualize_prediction(X_test, Y_test, idx)
