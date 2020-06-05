import torch
from gpt2.modeling.gpt2 import DecoderBlock, GPT2


def test_the_shape_from_decoder_block():
    # Create transformer-based decoder layer.
    layer = DecoderBlock(heads=2, dims=16, rate=4).eval()

    # Check shape-invariance for various shape.
    input_tensor = torch.zeros((10, 16))
    output_tensor, past = layer(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert past[0].shape == input_tensor.shape
    assert past[1].shape == input_tensor.shape

    input_tensor = torch.zeros((3, 2, 10, 16))
    output_tensor, past = layer(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert past[0].shape == input_tensor.shape
    assert past[1].shape == input_tensor.shape

    # Test for using `past` tensors.
    input_tensor = torch.zeros((3, 2, 5, 16))
    output_tensor, past = layer(input_tensor, past)
    assert output_tensor.shape == input_tensor.shape
    assert past[0].shape == (3, 2, 15, 16)
    assert past[1].shape == (3, 2, 15, 16)

    # Test for masking tensor.
    input_tensor = torch.zeros((3, 2, 3, 16))
    masking_tensor = torch.zeros((3, 2, 3, 18), dtype=torch.bool)
    output_tensor, past = layer(input_tensor, past, masking_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert past[0].shape == (3, 2, 18, 16)
    assert past[1].shape == (3, 2, 18, 16)


def test_the_shape_from_gpt2_model():
    # Create GPT-2 model.
    model = GPT2(layers=2,
                 pad_idx=0,
                 words=80,
                 seq_len=100,
                 heads=2,
                 dims=16,
                 rate=4)

    # Check if the model predicts well for single vector.
    input_tensor = torch.randint(80, (10,), dtype=torch.long)
    output_tensor, past = model(input_tensor)
    assert output_tensor.shape == (10, 80)
    assert len(past) == 2
    for p in past:
        assert p[0].shape == (10, 16)
        assert p[1].shape == (10, 16)

    # Check if the model predicts well for multi-dimensional tensor.
    input_tensor = torch.randint(80, (2, 7, 5, 10), dtype=torch.long)
    output_tensor, past = model(input_tensor)
    assert output_tensor.shape == (2, 7, 5, 10, 80)
    assert len(past) == 2
    for p in past:
        assert p[0].shape == (2, 7, 5, 10, 16)
        assert p[1].shape == (2, 7, 5, 10, 16)

    # Test for single vector with `past` tensors.
    input_tensor = torch.randint(80, (2, 7, 5, 10), dtype=torch.long)
    _, past = model(input_tensor)
    input_tensor = torch.randint(80, (2, 7, 5, 7), dtype=torch.long)
    output_tensor, past = model(input_tensor, past)

    assert output_tensor.shape == (2, 7, 5, 7, 80)
    assert len(past) == 2
    for p in past:
        assert p[0].shape == (2, 7, 5, 17, 16)
        assert p[1].shape == (2, 7, 5, 17, 16)

    # Consider the case of generating sentence.
    past = None
    for _ in range(10):
        input_tensor = torch.randint(80, (1,), dtype=torch.long)
        output_tensor, past = model(input_tensor, past)

    assert output_tensor.shape == (1, 80)
    assert len(past) == 2
    for p in past:
        assert p[0].shape == (10, 16)
        assert p[1].shape == (10, 16)
