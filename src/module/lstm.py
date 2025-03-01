"""Contains LSTM Implementation"""

import numpy as np

from src.activation_function import SigmoidActivation, TanhActivation

from .module import Module


class LSTM(Module):
    """Long Short-Term Memory"""

    def __init__(self, d_in: int, d_out: int, seed=None):
        assert isinstance(d_in, int) and d_in >= 1
        assert isinstance(d_out, int) and d_out >= 1

        self.d_out = d_out
        self.d_in = d_in

        self.sigmoid_activation = SigmoidActivation()
        self.tanh_activation = TanhActivation()

        _rng = np.random.default_rng(seed=seed)

        self.h = np.zeros(shape=(d_out,))
        self.C = np.zeros(shape=(d_out,))

        # Forget Gate
        self.W_f = _rng.normal(size=(d_out, d_out + d_in))
        self.b_f = _rng.normal(size=(d_out,))

        # Update Gate (Input + New Candidate)
        self.W_i = _rng.normal(size=(d_out, d_out + d_in))
        self.b_i = _rng.normal(size=(d_out,))
        self.W_C = _rng.normal(size=(d_out, d_out + d_in))
        self.b_C = _rng.normal(size=(d_out,))

        # Output Gate
        self.W_o = _rng.normal(size=(d_out, d_out + d_in))
        self.b_o = _rng.normal(size=(d_out,))

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

            f = self.sigmoid_activation(self.W_f @ concatted + self.b_f)
            i = self.sigmoid_activation(self.W_i @ concatted + self.b_i)
            C_tilde = self.tanh_activation(self.W_C @ concatted + self.b_C)
            o = self.sigmoid_activation(self.W_o @ concatted + self.b_o)

            self.C = f * self.C + i * C_tilde
            self.h = o * self.tanh_activation(self.C)

    def backward(self, x, y, lr=0.001) -> float:
        raise NotImplementedError("LSTM Backward pass is out of scope.")
