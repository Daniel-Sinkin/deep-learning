"""Contains GRU Implementation"""

import numpy as np

from .activation_function import SigmoidActivation, TanhActivation
from .module.module import Module


class GRU(Module):
    """Gated Recurrent Unit"""

    def __init__(self, d_in: int, d_out: int, seed=None):
        assert isinstance(d_in, int) and d_in >= 1
        assert isinstance(d_out, int) and d_out >= 1

        self.d_out = d_out
        self.d_in = d_in

        self.sigmoid_activation = SigmoidActivation()
        self.tanh_activation = TanhActivation()

        _rng = np.random.default_rng(seed=seed)

        self.h = np.zeros(shape=(d_out,))

        # Update Gate
        self.W_z = _rng.normal(size=(d_out, d_out + d_in))
        self.b_z = _rng.normal(size=(d_out,))

        # Reset Gate
        self.W_r = _rng.normal(size=(d_out, d_out + d_in))
        self.b_r = _rng.normal(size=(d_out,))

        # h Gate
        self.W_h = _rng.normal(size=(d_out, d_out + d_in))
        self.b_h = _rng.normal(size=(d_out,))

        self.has_run_forward = False

    def _forward__VALIDATE(self, x) -> np.ndarray:
        assert len(x.shape) == 3
        batch_size, sequence_length, d_in_sample = x.shape

        assert isinstance(batch_size, int) and batch_size == 1
        assert isinstance(sequence_length, int) and sequence_length >= 1
        assert isinstance(d_in_sample, int) and d_in_sample >= 1
        assert d_in_sample == self.d_in

    def forward(self, x) -> np.ndarray:
        self._forward__VALIDATE(x)

        sample = x[0]
        for sample_seq in sample:
            concatted = np.hstack([self.h, sample_seq])

            z = self.sigmoid_activation(self.W_z @ concatted + self.b_z)
            r = self.sigmoid_activation(self.W_r @ concatted + self.b_r)
            h_tilde = self.tanh_activation(
                self.W_h @ np.hstack([r * self.h, sample_seq])
            )
            self.h = (1 - z) * self.h + z * h_tilde

    def backward(self, x, y, lr=0.001) -> float:
        raise NotImplementedError("GRU Backward pass is out of scope.")
