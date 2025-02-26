import numpy as np

from .constants import activation_function_map


class MLPModel:
    def __init__(
        self, layers, hidden_activation: str, output_activation: str, seed=None
    ):
        np.random.seed(seed)
        self.layers = layers
        self.hidden_activation = activation_function_map[hidden_activation]()
        self.output_activation = activation_function_map[output_activation]()

        self.weights = []
        self.biases = []

        self.activations = None
        self.zs = None

        self.has_run_forward = False

        # He initialization for ReLU layers
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i + 1]
            if i < len(layers) - 2:
                # Hidden layer => ReLU => use He init
                std = np.sqrt(2.0 / fan_in)
            else:
                # Output layer => Sigmoid => use something moderate
                std = np.sqrt(1.0 / fan_in)
            W = np.random.normal(0, std, size=(fan_out, fan_in))
            b = np.zeros((fan_out, 1))
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

    def backprop(self, x, y, lr=0.001):
        out = self.forward(x)

        delta = out - y

        grad_w = [None] * (len(self.layers) - 1)
        grad_b = [None] * (len(self.layers) - 1)

        grad_w[-1] = delta @ self.activations[-2].T
        grad_b[-1] = np.mean(delta, axis=1, keepdims=True)

        for i in range(len(self.layers) - 2, 0, -1):
            z = self.zs[i - 1]
            # derivative of hidden activation
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
