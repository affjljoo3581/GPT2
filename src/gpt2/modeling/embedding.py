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
    def reset_parameters(self):
        """Initialize embedding matrix."""
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Embed positional information to vectors.

        Arguments:
            x (tensor): Input tensor of shape `(..., seq_len)`.
            offset (int): The offset of input sequences.

        Returns:
            An embedded tensor of shape `(..., seq_len, embedding_dim)`.
        """
        # Create position indices tensor.
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        # Embed the position indices to vectors.
        return super().forward(position)


class TokenEmbedding(nn.Embedding):
    """Implementation of token embedding layer.

    Note:
        The parameter ``num_embeddings`` implies the number of subwords in
        vocabulary.
    """
    def reset_parameters(self):
        """Initialize embedding matrix."""
        nn.init.normal_(self.weight, std=0.02)

    def forward(self,
                x: torch.Tensor,
                transposed: bool = False) -> torch.Tensor:
        """Embed subword tokens to vectors.

        Arguments:
            x (tensor): Input tensor of shape `(..., seq_len)` or
                `(..., seq_len, embedding_dim)`.
            transposed (bool): The boolean determining whether to transpose the
                embedding matrix.

        Returns:
            An embedded tensor of shape `(..., seq_len, embedding_dim)` or a
            logit tensor of shape `(..., seq_len, num_embeddings)`.

        Note:
            If ``transposed==True`` then the input tensor would be projected to
            the vocabulary space.
        """
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))

        return super().forward(x)
