import torch
from gpt2.modeling import Swish, PositionwiseFeedForward


def test_swish_output_shape():
    layer = Swish()

    assert layer(torch.zeros((10, 20, 200))).shape == (10, 20, 200)
    assert layer(torch.zeros((10, 20, 2, 1, 200))).shape == (10, 20, 2, 1, 200)


def test_positionwise_feedforward_output_shape():
    layer = PositionwiseFeedForward(dims=16, rate=4, dropout=0.1)

    assert layer(torch.zeros((10, 20, 16))).shape == (10, 20, 16)
    assert layer(torch.zeros((10, 20, 2, 1, 16))).shape == (10, 20, 2, 1, 16)
    assert layer(torch.zeros((16,))).shape == (16,)
