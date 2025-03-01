"""Contains MLPModel Implementation"""

import numpy as np

from src.activation_function import ReLUActivation, SigmoidActivation

from .module import Module


class MLP(Module):
    """Multi-Layer Perceptron Model"""

    def __init__(self, layers: list[int], seed=None):
        np.random.seed(seed)
        self.layers = layers
        self.hidden_activation = ReLUActivation()
        self.output_activation = SigmoidActivation()

        self.weights = []
        self.biases = []

        self.activations = None
        self.zs = None

        self.has_run_forward = False

        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            if i < len(layers) - 2:
                # Kaiming Initialization for ReLU (https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
                std = np.sqrt(2.0 / n_in)
            else:
                # Xavier Initialization for Sigmoid (https://arxiv.org/abs/1502.01852)
                std = 2 / (n_in + n_out)
            W = np.random.normal(0, std, size=(n_out, n_in))
            b = np.zeros((n_out, 1))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        self.zs = []
        a = x
        for i in range(len(self.layers) - 1):
            W = self.weights[i]
            b = self.biases[i]
            z = W @ a + b
            self.zs.append(z)
            if i < len(self.layers) - 2:
                a = self.hidden_activation.forward(z)
            else:
                a = self.output_activation.forward(z)
            self.activations.append(a)
        return a

    def backward(self, x, y, lr=0.001) -> float:
        """TODO: Replace this with something that is aligned with the pytorch implementation."""
        out = self.forward(x)

        delta = out - y

        grad_w = [None] * (len(self.layers) - 1)
        grad_b = [None] * (len(self.layers) - 1)

        grad_w[-1] = delta @ self.activations[-2].T
        grad_b[-1] = np.mean(delta, axis=1, keepdims=True)

        for i in range(len(self.layers) - 2, 0, -1):
            z = self.zs[i - 1]
            d_act = self.hidden_activation.derivative(z)
            delta = (self.weights[i].T @ delta) * d_act

            grad_w[i - 1] = delta @ self.activations[i - 1].T
            grad_b[i - 1] = np.mean(delta, axis=1, keepdims=True)

        for i in range(len(self.layers) - 1):
            self.weights[i] -= lr * grad_w[i]
            self.biases[i] -= lr * grad_b[i]

        eps = 1e-9
        loss = -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))
        return loss
