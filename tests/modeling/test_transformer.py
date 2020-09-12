import torch
from gpt2.modeling import TransformerLayer, Transformer


def test_transformer_layer_output_shape():
    layer = TransformerLayer(heads=2, dims=16, rate=4).eval()

    x, past = layer(torch.zeros((10, 16)))
    assert x.shape == (10, 16)
    assert past[0].shape == (10, 16)
    assert past[1].shape == x.shape

    x, past = layer(torch.zeros((3, 2, 10, 16)))
    assert x.shape == (3, 2, 10, 16)
    assert past[0].shape == (3, 2, 10, 16)
    assert past[1].shape == (3, 2, 10, 16)

    # Previously calculated attention keys and values would be concatenated to
    # the current ones.
    x, past = layer(torch.zeros((3, 2, 5, 16)), past=past)
    assert x.shape == (3, 2, 5, 16)
    assert past[0].shape == (3, 2, 15, 16)
    assert past[1].shape == (3, 2, 15, 16)

    # Previously calculated attention values are also masked by masking tensor.
    x, past = layer(torch.zeros((3, 2, 3, 16)),
                    past=past,
                    mask=torch.zeros((3, 2, 3, 18)).bool())
    assert x.shape == (3, 2, 3, 16)
    assert past[0].shape == (3, 2, 18, 16)
    assert past[1].shape == (3, 2, 18, 16)


def test_transformer_output_shape():
    model = Transformer(layers=2, pad_idx=0, words=80, seq_len=100, heads=2,
                        dims=16, rate=4, bidirectional=False).eval()

    x, past = model(torch.randint(80, (10,)))
    assert x.shape == (10, 80)
    for p in past:
        assert p[0].shape == (10, 16)
        assert p[1].shape == (10, 16)

    x, past = model(torch.randint(80, (2, 7, 5, 10)))
    assert x.shape == (2, 7, 5, 10, 80)
    for p in past:
        assert p[0].shape == (2, 7, 5, 10, 16)
        assert p[1].shape == (2, 7, 5, 10, 16)

    # Previously calculated attention keys and values would be concatenated to
    # the current ones.
    x, past = model(torch.randint(80, (2, 7, 5, 7)), past=past)
    assert x.shape == (2, 7, 5, 7, 80)
    for p in past:
        assert p[0].shape == (2, 7, 5, 17, 16)
        assert p[1].shape == (2, 7, 5, 17, 16)


def test_transformer_generating_sequence():
    model = Transformer(layers=2, pad_idx=0, words=80, seq_len=100, heads=2,
                        dims=16, rate=4, bidirectional=False).eval()

    past = None
    for _ in range(10):
        x, past = model(torch.randint(80, (1,)), past=past)

        # The output tensor is a distribution of next-word.
        assert x.shape == (1, 80)

    # All keys and values should be stacked.
    for p in past:
        assert p[0].shape == (10, 16)
        assert p[1].shape == (10, 16)
