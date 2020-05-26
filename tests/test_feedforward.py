import torch
from gpt2.modeling.feedforward import PositionwiseFeedForward


def test_the_shape_from_feedforward_layer():
    # Create feed-forward layer.
    layer = PositionwiseFeedForward(dims=16, rate=4, dropout=0.1)

    # Check shape-invariance for multi-dimension tensor.
    input_tensor = torch.zeros((3, 5, 9, 10, 16))
    assert layer(input_tensor).shape == input_tensor.shape

    # Check shape-invariance for vector (which is a special case).
    input_tensor = torch.zeros((16,))
    assert layer(input_tensor).shape == input_tensor.shape
