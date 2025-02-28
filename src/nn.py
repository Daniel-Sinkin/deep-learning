"""Contains the Baseclass for a Neural Model building block."""

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Baseclass for a Neural Model building block."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the model"""

    @abstractmethod
    def backward(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        """Does the backward pass. Returns the loss."""

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)
