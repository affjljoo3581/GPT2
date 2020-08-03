import torch
import torch.nn as nn
from gpt2.modeling import PositionalEmbedding, TokenEmbedding


def test_positional_embedding_load_state_dict():
    layer_32 = PositionalEmbedding(num_embeddings=32, embedding_dim=16)
    layer_64 = PositionalEmbedding(num_embeddings=64, embedding_dim=16)

    # Reduce the embedding matrix to decrease the sequence length.
    layer_32.load_state_dict(layer_64.state_dict())
    assert (layer_32.weight == layer_64.weight[:32]).all()

    layer_32.reset_parameters()
    layer_64.reset_parameters()

    # Expand the embedding matrix to increase the sequence length.
    layer_64.load_state_dict(layer_32.state_dict())
    assert (layer_32.weight == layer_64.weight[:32]).all()


def test_positional_embedding_load_state_dict_with_wrapper():
    layer_32 = nn.Sequential(
        PositionalEmbedding(num_embeddings=32, embedding_dim=16))
    layer_64 = nn.Sequential(
        PositionalEmbedding(num_embeddings=64, embedding_dim=16))

    # Reduce the embedding matrix to decrease the sequence length.
    layer_32.load_state_dict(layer_64.state_dict())
    assert (layer_32[0].weight == layer_64[0].weight[:32]).all()

    layer_32[0].reset_parameters()
    layer_64[0].reset_parameters()

    # Expand the embedding matrix to increase the sequence length.
    layer_64.load_state_dict(layer_32.state_dict())
    assert (layer_32[0].weight == layer_64[0].weight[:32]).all()


def test_positional_embedding_output_shape():
    layer = PositionalEmbedding(num_embeddings=100, embedding_dim=32)

    x1 = torch.randint(8000, (10, 6, 100))
    x2 = torch.randint(8000, (10, 6))

    assert layer(x1).shape == x1.shape + (32,)
    assert layer(x2).shape == x2.shape + (32,)


def test_positional_embedding_output_value():
    layer = PositionalEmbedding(num_embeddings=100, embedding_dim=32)

    x1 = torch.randint(8000, (10, 6, 100))
    x2 = torch.randint(8000, (10, 6, 100))
    assert (layer(x1) == layer(x2)).all()

    x1 = torch.randint(8000, (10,))
    x2 = torch.randint(8000, (5,))
    for offset in range(5):
        assert (layer(x1)[offset:offset+5] == layer(x2, offset=offset)).all()

    x1 = torch.randint(8000, (8, 5, 10,))
    x2 = torch.randint(8000, (8, 5, 5,))
    for offset in range(5):
        assert (layer(x1)[..., offset:offset+5, :]
                == layer(x2, offset=offset)).all()


def test_token_embedding_output_shape():
    layer = TokenEmbedding(num_embeddings=80, embedding_dim=32)

    x1 = torch.randint(80, (3, 6, 10))
    x2 = torch.zeros((20, 30, 32))

    assert layer(x1).shape == x1.shape + (32,)
    assert layer(x2, transposed=True).shape == x2.shape[:-1] + (80,)
