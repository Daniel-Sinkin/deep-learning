"""Implementation of CNN Model."""

import numpy as np

from src.activation_function import SigmoidActivation
from src.functions import conv2d

from .module import Module


class CNN(Module):
    """Convolutional Neural Network with Simoid activation function."""

    def __init__(
        self,
        filter_y: int,
        filter_x: int,
        c_in: int,
        c_out: int,
        padding: int = 0,
        stride: int = 1,
        seed=None,
    ):
        self._check_args(
            filter_y=filter_y,
            filter_x=filter_x,
            c_in=c_in,
            c_out=c_out,
            padding=padding,
            stride=stride,
        )

        self.padding = padding
        self.stride = stride

        self.filter_y = filter_y
        self.filter_x = filter_x
        self.c_in = c_in
        self.c_out = c_out

        # Xavier Initialization for Sigmoid
        n_in = self.filter_y * self.filter_x * self.c_in
        n_out = self.filter_y * self.filter_x * self.c_out
        weight_std = 2 / (n_in + n_out)

        self.bias = np.zeros(shape=(c_out,), dtype=np.float32)

        _rng = np.random.default_rng(seed=seed)
        self.W = _rng.normal(
            loc=0.0, scale=weight_std, size=(c_out, c_in, filter_y, filter_x)
        ).astype(np.float32)

        self.activation = SigmoidActivation()

    def _check_args(
        self,
        filter_y: int,
        filter_x: int,
        c_in: int,
        c_out: int,
        padding: int,
        stride: int,
    ) -> None:
        """Checks initialization arguments via asserts"""
        assert isinstance(filter_y, int) and filter_y >= 1
        assert isinstance(filter_x, int) and filter_x >= 1
        assert isinstance(c_in, int) and c_in >= 1
        assert isinstance(c_out, int) and c_out >= 1
        assert isinstance(padding, int) and padding >= 0
        assert isinstance(stride, int) and stride >= 1

    def compute_output_size(self, input_dim: int, is_y: bool = True) -> int:
        """Uses the formula for output size: O = (I - F + 2 * P) / S + 1"""
        i = input_dim
        f = self.filter_y if is_y else self.filter_x
        p = self.padding
        s = self.stride
        return (i - f + 2 * p) // s + 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 4
        batch_size, c_in, h, w = x.shape

        assert c_in == self.c_in, "Input Channel dimension does not match!"

        output_y = self.compute_output_size(h, is_y=True)
        output_x = self.compute_output_size(w, is_y=False)
        output = np.zeros(shape=(batch_size, self.c_out, output_y, output_x))
        for sample_idx in range(batch_size):
            for channel_out_idx in range(self.c_out):
                output[sample_idx, channel_out_idx, :, :] = conv2d(
                    sample=x[sample_idx, :, :, :],
                    filter_=self.W,
                    bias=self.bias[channel_out_idx],
                    padding=self.padding,
                    stride=self.stride,
                )
        return self.activation(output)

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        raise NotImplementedError("Backward pass for CNN out of scope for now.")
