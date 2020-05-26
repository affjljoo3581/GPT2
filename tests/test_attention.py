import torch
from gpt2.modeling.attention import (BaseAttention,
                                     MultiHeadAttention,
                                     AttentionBlock)


def test_the_shape_from_base_attention_layer():
    # Create single attention layer.
    layer = BaseAttention()

    # Check the shape of attention.
    q = torch.zeros((3, 8, 4, 10, 16))
    k = torch.zeros((3, 8, 4, 20, 16))
    v = torch.zeros((3, 8, 4, 20, 32))
    assert layer(q, k, v).shape == (3, 8, 4, 10, 32)

    # Test with masking tensor.
    mask = torch.zeros((3, 8, 4, 10, 20)).bool()
    assert layer(q, k, v, mask).shape == (3, 8, 4, 10, 32)


def test_base_attention_layer_with_simple_data():
    # Create single attention layer.
    layer = BaseAttention().eval()

    # Test for all-one tensors.
    q = torch.ones((3, 16))
    k = torch.ones((4, 16))
    v = torch.ones((4, 1))
    assert layer(q, k, v).sum(-1).mean() == 1

    # Test for masking tensor.
    v = torch.arange(4, dtype=q.dtype).unsqueeze(-1)
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
                        dtype=torch.bool)
    expected = torch.tensor(([0], [0.5], [1.5]))
    assert (layer(q, k, v, mask) == expected).all()


def test_the_shape_from_multihead_attention_layer():
    # Create multi-headed attention layer.
    layer = MultiHeadAttention(heads=2)

    # Check the shape of attention.
    q = torch.zeros((10, 16))
    k = torch.zeros((20, 16))
    v = torch.zeros((20, 32))
    assert layer(q, k, v).shape == (10, 32)

    q = torch.zeros((3, 8, 4, 10, 16))
    k = torch.zeros((3, 8, 4, 20, 16))
    v = torch.zeros((3, 8, 4, 20, 32))
    assert layer(q, k, v).shape == (3, 8, 4, 10, 32)

    # Test with masking tensor.
    mask = torch.zeros((3, 8, 4, 10, 20)).bool()
    assert layer(q, k, v, mask).shape == (3, 8, 4, 10, 32)


def test_multihead_attention_layer_with_simple_data():
    # Create multi-headed attention layer.
    layer = MultiHeadAttention(heads=2).eval()

    # Test for all-one tensors.
    q = torch.ones((3, 16))
    k = torch.ones((4, 16))
    v = torch.ones((4, 2))
    assert layer(q, k, v).mean() == 1

    v = torch.tensor([[1, 2], [1, 2], [1, 2], [1, 2]], dtype=q.dtype)
    expected = torch.tensor([[1, 2], [1, 2], [1, 2]])
    assert (layer(q, k, v) == expected).all()

    # Test for masking tensor.
    v = torch.tensor([[0, 0], [1, 2], [2, 4], [3, 6]], dtype=q.dtype)
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
                        dtype=torch.bool)
    expected = torch.tensor(([0, 0], [0.5, 1], [1.5, 3]))
    assert (layer(q, k, v, mask) == expected).all()


def test_the_shape_from_attention_block():
    # Create attention block layer.
    layer = AttentionBlock(heads=2, dims=16)

    # Check the shape of attention.
    q = torch.zeros((10, 16))
    k = torch.zeros((20, 16))
    v = torch.zeros((20, 16))
    x, past = layer(q, k, v)

    assert x.shape == (10, 16)
    assert past[0].shape == (20, 16)
    assert past[1].shape == (20, 16)

    # Test for reusing `past` key-value tensors.
    q = torch.zeros((5, 16))
    k = torch.zeros((7, 16))
    v = torch.zeros((7, 16))
    x, past = layer(q, k, v, past=past)

    assert x.shape == (5, 16)
    assert past[0].shape == (27, 16)
    assert past[1].shape == (27, 16)
