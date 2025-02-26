from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray: ...


class ReLUActivation(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)


class SigmoidActivation(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)
