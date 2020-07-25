import torch
from gpt2.modeling import PadMasking, FutureMasking


def test_pad_masking_output_shape():
    layer = PadMasking(pad_idx=0)

    x1 = torch.randint(8000, (30,))
    x2 = torch.randint(8000, (16, 30))
    x3 = torch.randint(8000, (2, 5, 6, 4))

    assert layer(x1).shape == x1.shape + x1.shape[-1:]
    assert layer(x2).shape == x2.shape + x2.shape[-1:]
    assert layer(x3).shape == x3.shape + x3.shape[-1:]

    # The expanded values also be masked.
    x1 = torch.randint(8000, (30,))
    x2 = torch.randint(8000, (16, 30))
    x3 = torch.randint(8000, (2, 5, 6, 4))

    assert layer(x1, offset=5).shape == x1.shape + (x1.size(-1) + 5,)
    assert layer(x2, offset=7).shape == x2.shape + (x2.size(-1) + 7,)
    assert layer(x3, offset=2).shape == x3.shape + (x3.size(-1) + 2,)


def test_pad_masking_output_values():
    layer = PadMasking(pad_idx=0)

    x1 = torch.tensor([0, 0, 1, 0, 5, 2])
    x2 = torch.tensor([[[0, 0, 1, 0, 5, 2]]]).expand(3, 5, -1)

    assert layer(x1).long().tolist() == [[1, 1, 0, 1, 0, 0]] * 6
    assert layer(x2).long().tolist() == [[[[1, 1, 0, 1, 0, 0]] * 6] * 5] * 3

    # Since the tokens added before the sequence are not paddings, they would
    # be excluded in masking.
    assert (layer(x1, offset=2).long().tolist()
            == [[0, 0, 1, 1, 0, 1, 0, 0]] * 6)
    assert (layer(x2, offset=2).long().tolist()
            == [[[[0, 0, 1, 1, 0, 1, 0, 0]] * 6] * 5] * 3)


def test_future_masking_output_shape():
    layer = FutureMasking()

    x1 = torch.randint(8000, (30,))
    x2 = torch.randint(8000, (16, 30))
    x3 = torch.randint(8000, (2, 5, 6, 4))

    assert layer(x1).shape == x1.shape + x1.shape[-1:]
    assert layer(x2).shape == x2.shape + x2.shape[-1:]
    assert layer(x3).shape == x3.shape + x3.shape[-1:]

    # The expanded values also be masked.
    x1 = torch.randint(8000, (30,))
    x2 = torch.randint(8000, (16, 30))
    x3 = torch.randint(8000, (2, 5, 6, 4))

    assert layer(x1, offset=5).shape == x1.shape + (x1.size(-1) + 5,)
    assert layer(x2, offset=7).shape == x2.shape + (x2.size(-1) + 7,)
    assert layer(x3, offset=2).shape == x3.shape + (x3.size(-1) + 2,)


def test_future_masking_output_value():
    layer = FutureMasking()

    x1 = torch.randint(8000, (3, 7, 30))
    x2 = torch.randint(8000, (3, 7, 30))
    x3 = torch.randint(8000, (30,))
    x4 = torch.randint(8000, (30,))

    assert (layer(x1) == layer(x2)).all()
    assert (layer(x3) == layer(x4)).all()

    # The expanded values also be masked.
    x = torch.randint(8000, (3,))
    assert layer(x).bool().tolist() == [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    assert (layer(x, offset=2).bool().tolist()
            == [[0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
