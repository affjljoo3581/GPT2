import torch
from gpt2.modeling.embedding import PositionalEmbedding, TokenEmbedding


def test_the_shape_from_positional_embedding_layer():
    # Create positional embedding layer.
    layer = PositionalEmbedding(num_embeddings=100, embedding_dim=32)

    # Check shape-invariance for grouped batch tensors.
    input_tensor = torch.randint(8000, (10, 6, 100), dtype=torch.long)
    assert layer(input_tensor).shape == input_tensor.shape + (32,)

    # Check shape-invarance for single matrix.
    input_tensor = torch.randint(8000, (6, 10), dtype=torch.long)
    assert layer(input_tensor).shape == input_tensor.shape + (32,)


def test_position_invariance_of_positional_embedding():
    # Create positional embedding layer.
    layer = PositionalEmbedding(num_embeddings=100, embedding_dim=32)

    # Compare with two different tensors of which lengths are same.
    input_tensor_1 = torch.randint(8000, (10, 6, 100), dtype=torch.long)
    input_tensor_2 = torch.randint(8000, (10, 6, 100), dtype=torch.long)
    assert (layer(input_tensor_1) == layer(input_tensor_2)).all()

    # Check if embedded positional informations are padded by ``offset``.
    input_tensor_1 = torch.randint(8000, (10,), dtype=torch.long)
    input_tensor_2 = torch.randint(8000, (5,), dtype=torch.long)
    for i in range(5):
        assert (layer(input_tensor_1)[i:i+5]
                == layer(input_tensor_2, offset=i)).all()

    input_tensor_1 = torch.randint(8000, (8, 5, 10), dtype=torch.long)
    input_tensor_2 = torch.randint(8000, (8, 5, 5), dtype=torch.long)
    for i in range(5):
        assert (layer(input_tensor_1)[..., i:i+5, :]
                == layer(input_tensor_2, offset=i)).all()


def test_the_shape_from_token_embedding():
    # Create token embedding layer.
    layer = TokenEmbedding(num_embeddings=80, embedding_dim=32)

    # Check shape-invariance for multi-dimension tensor.
    input_tensor = torch.randint(80, (3, 6, 10), dtype=torch.long)
    assert layer(input_tensor).shape == input_tensor.shape + (32,)

    # Check shape-invariance for single vector.
    input_tensor = torch.randint(80, (10,), dtype=torch.long)
    assert layer(input_tensor).shape == input_tensor.shape + (32,)

    # Check for transposed embedding matrix.
    input_tensor = torch.zeros((20, 30, 32))
    assert layer(input_tensor, transposed=True).shape == (20, 30, 80)
