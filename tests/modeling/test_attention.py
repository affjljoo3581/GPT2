import torch
from gpt2.modeling import BaseAttention, MultiHeadAttention, AttentionLayer


def test_base_attention_output_shape():
    layer = BaseAttention()

    q = torch.zeros((3, 8, 4, 10, 16))
    k = torch.zeros((3, 8, 4, 20, 16))
    v = torch.zeros((3, 8, 4, 20, 32))
    assert layer(q, k, v).shape == (3, 8, 4, 10, 32)

    # The shape of output tensor is not affected by masking tensor.
    mask = torch.zeros((3, 8, 4, 10, 20)).bool()
    assert layer(q, k, v, mask).shape == (3, 8, 4, 10, 32)


def test_base_attention_output_values():
    layer = BaseAttention().eval()

    q = torch.ones((3, 16))
    k = torch.ones((4, 16))
    v = torch.arange(4).float().unsqueeze(-1)
    assert layer(q, k, v).tolist() == [[1.5], [1.5], [1.5]]

    # Due to the masking tensors, the attended outputs would be different.
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]).bool()
    assert layer(q, k, v, mask).tolist() == [[0], [0.5], [1.5]]


def test_multihead_attention_output_shape():
    layer = MultiHeadAttention(heads=2)

    q = torch.zeros((10, 16))
    k = torch.zeros((20, 16))
    v = torch.zeros((20, 32))
    assert layer(q, k, v).shape == (10, 32)

    q = torch.zeros((3, 8, 4, 10, 16))
    k = torch.zeros((3, 8, 4, 20, 16))
    v = torch.zeros((3, 8, 4, 20, 32))
    assert layer(q, k, v).shape == (3, 8, 4, 10, 32)

    # The shape of output tensor is not affected by masking tensor.
    mask = torch.zeros((3, 8, 4, 10, 20)).bool()
    assert layer(q, k, v, mask).shape == (3, 8, 4, 10, 32)


def test_multihead_attention_output_values():
    layer = MultiHeadAttention(heads=2).eval()

    q = torch.ones((3, 16))
    k = torch.ones((4, 16))
    v1 = torch.ones((4, 2))
    v2 = torch.tensor([[1, 2], [1, 2], [1, 2], [1, 2]]).float()

    assert layer(q, k, v1).tolist() == [[1, 1], [1, 1], [1, 1]]
    assert layer(q, k, v2).tolist() == [[1, 2], [1, 2], [1, 2]]

    # Due to the masking tensors, the attended outputs would be different.
    v = torch.tensor([[0, 0], [1, 2], [2, 4], [3, 6]]).float()
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]).bool()
    assert layer(q, k, v, mask).tolist() == [[0, 0], [0.5, 1], [1.5, 3]]


def test_attention_layer_output_shape():
    layer = AttentionLayer(heads=2, dims=16)

    q = torch.zeros((10, 16))
    k = torch.zeros((20, 16))
    v = torch.zeros((20, 16))
    x, past = layer(q, k, v)

    assert x.shape == (10, 16)
    assert past[0].shape == (20, 16)
    assert past[1].shape == (20, 16)

    q = torch.zeros((5, 16))
    k = torch.zeros((7, 16))
    v = torch.zeros((7, 16))
    x, past = layer(q, k, v, past=past)

    assert x.shape == (5, 16)
    assert past[0].shape == (27, 16)
    assert past[1].shape == (27, 16)
