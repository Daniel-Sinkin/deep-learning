"""Contains the activation Functions that we use."""

from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    """Activation Function"""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass applying the activation function element-wise."""

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of Forward pass for backpropagation."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ReLUActivation(ActivationFunction):
    """ReLU"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)


class SigmoidActivation(ActivationFunction):
    """Sigmoid"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid function inplace on a specifically allocated memory chunk, this is
        around 40% faster than the naive `return 1 / (1 + np.exp(-x))` implementation.
        """
        arr = np.zeros_like(x, dtype=np.float32)
        np.exp(-x, out=arr)
        np.add(arr, np.float32(1.0), out=arr)
        np.reciprocal(arr, out=arr)
        return arr

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class TanhActivation(ActivationFunction):
    """Tanh Activation Function"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Compute the hyperbolic tangent element-wise and cast to float32.
        return np.tanh(x).astype(np.float32)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # The derivative of tanh is 1 - tanh^2(x)
        t = np.tanh(x).astype(np.float32)
        return 1 - t * t


class IdentityActivation(ActivationFunction):
    """Linear Activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x
