import torch
import torch.nn as nn
from typing import Dict


class PositionalEmbedding(nn.Embedding):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, embedding_dim)
    ===========================================================================
    """
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def _load_from_state_dict(self,
                              state_dict: Dict[str, torch.Tensor],
                              prefix: str,
                              *args,
                              **kwargs):
        weight = state_dict[f'{prefix}weight']

        # Reduce or expand the positional embedding matrix to increase or
        # decrease the total sequence length.
        if weight.size(0) < self.num_embeddings:
            weight = torch.cat((weight, self.weight[weight.size(0):]), dim=0)
        elif weight.size(0) > self.num_embeddings:
            weight = weight[:self.num_embeddings]

        state_dict[f'{prefix}weight'] = weight
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        return super().forward(position)


class TokenEmbedding(nn.Embedding):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long or float  (..., seq_len)
                                    or (..., seq_len, embedding_dim)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, embedding_dim)
                                    or (..., seq_len, num_embeddings)
    ===========================================================================
    """
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self,
                x: torch.Tensor,
                transposed: bool = False) -> torch.Tensor:
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))
        else:
            return super().forward(x)
