"""Contains MLPModel Implementation"""

import numpy as np

from src.activation_function import TanhActivation
from src.nn import Model


class RNNModel(Model):
    """Recurrent Neural Network Model using float32 throughout."""

    def __init__(self, d_in: int, d_out: int, seed=None):
        _rng = np.random.default_rng(seed=seed)

        self.activation = TanhActivation()

        assert isinstance(d_in, int) and d_in >= 1
        assert isinstance(d_out, int) and d_out >= 1

        self.d_in = d_in
        self.d_out = d_out

        # Use float32 for all arrays.
        self.h = np.zeros((d_out,), dtype=np.float32)
        self.b = np.zeros((d_out,), dtype=np.float32)

        self.W_xh = _rng.normal(size=(d_out, d_in)).astype(np.float32)
        self.W_hh = _rng.normal(size=(d_out, d_out)).astype(np.float32)

        self.has_run_forward = False

    def _forward__VALIDATE(self, x) -> None:
        assert len(x.shape) == 3
        batch_size, sequence_length, d_in_sample = x.shape
        assert (
            isinstance(batch_size, int) and batch_size == 1
        ), "Currently only batches of size 1 are supported."
        assert isinstance(sequence_length, int) and sequence_length >= 1
        assert d_in_sample == self.d_in, "Input size of sample is wrong."

    def forward(self, x) -> np.ndarray:
        self._forward__VALIDATE(x)
        _, sequence_length, _ = x.shape
        ys = np.zeros((sequence_length, self.d_out), dtype=np.float32)
        sample = x[0]
        for t in range(sequence_length):
            self.h = self.activation(
                self.W_xh @ sample[t, :] + self.W_hh @ self.h + self.b
            )
            ys[t, :] = self.h

        assert ys.shape == (sequence_length, self.d_out)
        return ys

    def backward(self, x, y, lr=0.001) -> float:
        raise NotImplementedError
