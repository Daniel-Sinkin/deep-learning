"""
Contains helper functions.
"""

import numpy as np


def conv2d(
    sample: np.ndarray,
    filter_: np.ndarray,
    bias: float,
    padding: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """Implements 2d convolution with 0.0 padding (logical, avoids re-allocating sample)"""
    input_y, input_x = sample.shape
    filter_y, filter_x = filter_.shape

    output_y = (input_y - filter_y + 2 * padding) // stride + 1
    output_x = (input_x - filter_x + 2 * padding) // stride + 1

    output = np.zeros((output_y, output_x), dtype=sample.dtype)
    patch = np.zeros((filter_y, filter_x), dtype=sample.dtype)

    for y in range(output_y):
        for x in range(output_x):
            patch.fill(0.0)

            pos_y, pos_x = y * stride - padding, x * stride - padding

            start_y = max(0, -pos_y)
            end_y = min(filter_y, input_y - pos_y)
            start_x = max(0, -pos_x)
            end_x = min(filter_x, input_x - pos_x)

            patch[start_y:end_y, start_x:end_x] = sample[
                max(0, pos_y) : max(0, pos_y) + (end_y - start_y),
                max(0, pos_x) : max(0, pos_x) + (end_x - start_x),
            ]
            output[y, x] = (patch * filter_).sum() + bias
    return output
