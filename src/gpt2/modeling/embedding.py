import torch
import torch.nn as nn


class PositionalEmbedding(nn.Embedding):
    """Implementation of positional embedding layer.

    Positional embedding layer encodes representations with their positional
    informations. Each position is embedded to a ``embedding_dim`` dimensional
    vector. If the input sequence is shifted or padded, the positional indices
    would be adjusted equvalently with the given ``offset`` of the sequence.

    Note:
        The parameter ``num_embeddings`` implies the length of each sequence.
    """
    def forward(self, x: torch.Tensor, offset: int = 0):
        """Embed positional information to vectors.

        Arguments:
            x (tensor): Input tensor of shape `(..., seq_len)`.
            offset (int): The offset of input sequences.

        Returns:
            An embedded tensor of shape `(..., seq_len, embedding_dims)`.
        """
        # Create position indices tensor.
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        # Embed the position indices to vectors.
        return super().forward(position)


class EmbeddingBlock(nn.Module):
    """Embedding layer with positional encoding.

    Arguments:
        words (int): The number of words in vocabulary.
        seq_len (int): The maximum length of input sequences.
        dims (int): The dimension of embedded vectors.
        dropout (float): The probability that each vector is dropped.
    """
    def __init__(self, words: int, seq_len: int, dims: int,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(words, dims)
        self.position_embedding = PositionalEmbedding(seq_len, dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """Embed each word to the vector with its positional information.

        Arguments:
            x (tensor): Input tensor of shape `(..., seq_len)`.
            offset (int): The offset of input sequences.

        Returns:
            An embedded tensor of shape `(..., seq_len, dims)`.
        """
        x = self.token_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
