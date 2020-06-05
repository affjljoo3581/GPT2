import torch
import itertools
from gpt2.modeling.masking import PadMasking, FutureMasking


def test_the_shape_from_pad_masking_layer():
    # Create pad-masking layer.
    layer = PadMasking(pad_idx=0)

    # Test for various-dimensional tensors.
    input_tensor = torch.randint(8000, (30,), dtype=torch.long)
    assert layer(input_tensor).shape == (1, 30)

    input_tensor = torch.randint(8000, (16, 30,), dtype=torch.long)
    assert layer(input_tensor).shape == (16, 1, 30)

    input_tensor = torch.randint(8000, (2, 5, 6, 4), dtype=torch.long)
    assert layer(input_tensor).shape == (2, 5, 6, 1, 4)

    # Check if the masks are shifted by `offset`.
    input_tensor = torch.randint(8000, (30,), dtype=torch.long)
    assert layer(input_tensor, offset=5).shape == (1, 35)

    input_tensor = torch.randint(8000, (16, 30,), dtype=torch.long)
    assert layer(input_tensor, offset=7).shape == (16, 1, 37)

    input_tensor = torch.randint(8000, (2, 5, 6, 4), dtype=torch.long)
    assert layer(input_tensor, offset=2).shape == (2, 5, 6, 1, 6)


def test_pad_tokens_are_masked_well():
    # Create pad-masking layer.
    layer = PadMasking(pad_idx=0)

    # Check if pad tokens are masked well.
    input_tensor = torch.tensor([0, 0, 1, 0, 5, 2])
    expected = torch.tensor([[1, 1, 0, 1, 0, 0]], dtype=torch.bool)
    assert (layer(input_tensor) == expected).all()

    # Test for multi-dimension tensor.
    input_tensor = (torch.tensor([0, 0, 1, 0, 5, 2])
                    .view(1, 1, -1)
                    .expand(3, 5, -1))
    expected = torch.tensor([[1, 1, 0, 1, 0, 0]], dtype=torch.bool)
    for i, j in itertools.product(range(3), range(5)):
        assert (layer(input_tensor)[i, j] == expected).all()

    # Check if masks are shifted.
    input_tensor = torch.tensor([0, 0, 1, 0, 5, 2])
    expected = torch.tensor([[0, 0, 1, 1, 0, 1, 0, 0]], dtype=torch.bool)
    assert (layer(input_tensor, offset=2) == expected).all()

    # Test the mask-shifting for multi-dimension tensor.
    input_tensor = (torch.tensor([0, 0, 1, 0, 5, 2])
                    .view(1, 1, -1)
                    .expand(3, 5, -1))
    expected = torch.tensor([[0, 1, 1, 0, 1, 0, 0]], dtype=torch.bool)
    for i, j in itertools.product(range(3), range(5)):
        assert (layer(input_tensor, offset=1)[i, j] == expected).all()


def test_the_shape_from_future_masking():
    # Create future-masking layer.
    layer = FutureMasking()

    # Test for various-dimensional tensors.
    input_tensor = torch.randint(8000, (30,), dtype=torch.long)
    assert layer(input_tensor).shape == (30, 30)

    input_tensor = torch.randint(8000, (16, 30,), dtype=torch.long)
    assert layer(input_tensor).shape == (1, 30, 30)

    input_tensor = torch.randint(8000, (2, 5, 6, 4), dtype=torch.long)
    assert layer(input_tensor).shape == (1, 1, 1, 4, 4)

    # Check if the masks are shifted by `offset`.
    input_tensor = torch.randint(8000, (30,), dtype=torch.long)
    assert layer(input_tensor, offset=5).shape == (30, 35)

    input_tensor = torch.randint(8000, (16, 30,), dtype=torch.long)
    assert layer(input_tensor, offset=7).shape == (1, 30, 37)

    input_tensor = torch.randint(8000, (2, 5, 6, 4), dtype=torch.long)
    assert layer(input_tensor, offset=2).shape == (1, 1, 1, 4, 6)


def test_invariance_of_future_masking_layer():
    # Create future-masking layer.
    layer = FutureMasking()

    # Check if the maskings from different token ids are same.
    input_tensor_1 = torch.randint(8000, (3, 7, 30), dtype=torch.long)
    input_tensor_2 = torch.randint(8000, (3, 7, 30), dtype=torch.long)
    assert (layer(input_tensor_1) == layer(input_tensor_2)).all()

    # Test the invariance for single vectors.
    input_tensor_1 = torch.randint(8000, (30,), dtype=torch.long)
    input_tensor_2 = torch.randint(8000, (30,), dtype=torch.long)
    assert (layer(input_tensor_1) == layer(input_tensor_2)).all()


def test_future_tokens_are_masked_well():
    # Create future-masking layer.
    layer = FutureMasking()

    input_tensor = torch.randint(8000, (3,), dtype=torch.long)
    expected = torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                            dtype=torch.bool)
    assert (layer(input_tensor) == expected).all()

    input_tensor = torch.randint(8000, (3,), dtype=torch.long)
    expected = torch.tensor([[0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0]],
                            dtype=torch.bool)
    assert (layer(input_tensor, offset=2) == expected).all()
