import numpy as np
import pytest
import torch

from src.module.rnn import RNN


@pytest.fixture
def sample_input():
    """Fixture to provide a sample input sequence along with input/output dimensions."""
    d_in = 3
    d_out = 2
    batch_size = 1
    sequence_length = 5
    input_seq = np.array(
        [
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
            ]
        ]
    )
    return input_seq, d_in, d_out


@pytest.fixture
def numpy_rnn(sample_input):
    """Fixture that creates the numpy-based RNN model with a fixed seed."""
    _, d_in, d_out = sample_input
    model = RNN(d_in, d_out, seed=42)
    return model


@pytest.fixture
def torch_rnn(numpy_rnn, sample_input):
    """Fixture that creates a PyTorch RNN and initializes weights to match the numpy model."""
    _, d_in, d_out = sample_input
    model = torch.nn.RNN(
        input_size=d_in, hidden_size=d_out, nonlinearity="tanh", batch_first=False
    )

    # Copy weights from numpy model
    with torch.no_grad():
        model.weight_ih_l0.copy_(torch.from_numpy(numpy_rnn.W_xh))
        model.weight_hh_l0.copy_(torch.from_numpy(numpy_rnn.W_hh))
        model.bias_ih_l0.copy_(torch.from_numpy(numpy_rnn.b))
        model.bias_hh_l0.fill_(0)  # numpy model does not use a second bias term

    return model


def test_forward_output_shape(numpy_rnn, sample_input):
    """Test that the numpy model forward pass returns the expected output shape."""
    input_seq, _, d_out = sample_input
    output = numpy_rnn.forward(input_seq)
    sequence_length = input_seq.shape[1]
    assert output.shape == (sequence_length, d_out)


def test_forward_equivalence(numpy_rnn, torch_rnn, sample_input):
    """Test that the numpy model and PyTorch RNN produce similar outputs."""
    input_seq, _, _ = sample_input

    # Get output from the numpy model
    output_numpy = numpy_rnn.forward(input_seq)

    # Convert input to PyTorch format: (sequence_length, batch_size, d_in)
    input_tensor = torch.from_numpy(input_seq).float().transpose(0, 1)

    # Run through PyTorch RNN
    output_torch, _ = torch_rnn(input_tensor)

    # Compare outputs
    np.testing.assert_allclose(
        output_numpy, output_torch.squeeze(1).detach().numpy(), rtol=1e-5, atol=1e-6
    )


def test_invalid_input_shape(numpy_rnn):
    """Test that passing an input with the wrong dimensions triggers an assertion error."""
    invalid_input = np.array([[0.1, 0.2, 0.3]])  # Should be 3D
    with pytest.raises(AssertionError):
        numpy_rnn.forward(invalid_input)


def test_invalid_batch_size(numpy_rnn, sample_input):
    """Test that an input with an unsupported batch size (other than 1) triggers an assertion error."""
    input_seq, _, _ = sample_input
    input_seq_invalid = np.repeat(input_seq, 2, axis=0)  # Batch size 2
    with pytest.raises(AssertionError):
        numpy_rnn.forward(input_seq_invalid)
