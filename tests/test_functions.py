"""Pytests for the functions.py file"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.functions import conv2d


@pytest.mark.parametrize(
    "input_shape, filter_shape, bias, padding, stride",
    [
        ((5, 5), (3, 3), 0.0, 0, 1),
        ((7, 7), (3, 3), 1.0, 1, 1),
        ((8, 8), (5, 5), -0.5, 2, 2),
        ((6, 6), (2, 2), 0.2, 0, 2),
        ((4, 4), (3, 3), 0.0, 1, 1),
        ((2, 2), (4, 4), 0.0, 1, 1),
    ],
)
def test_conv2d(input_shape, filter_shape, bias, padding, stride):
    # Generate random input and filter
    np.random.seed(42)
    sample = np.random.randn(*input_shape).astype(np.float32)
    filter_ = np.random.randn(*filter_shape).astype(np.float32)

    # Compute output using your function
    numpy_output = conv2d(sample, filter_, bias, padding, stride)

    # Convert to PyTorch tensors
    sample_torch = (
        torch.tensor(sample).unsqueeze(0).unsqueeze(0)
    )  # Add batch & channel dims
    filter_torch = torch.tensor(filter_).unsqueeze(0).unsqueeze(0)

    # Compute PyTorch output
    torch_output = F.conv2d(
        sample_torch,
        filter_torch,
        bias=torch.tensor([bias]),
        stride=stride,
        padding=padding,
    )
    torch_output = torch_output.squeeze().numpy()

    # Compare results
    np.testing.assert_allclose(numpy_output, torch_output, rtol=1e-5, atol=1e-5)
