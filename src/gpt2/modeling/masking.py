import torch
import torch.nn as nn


class PadMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., 1, seq_len + offset)
    ===========================================================================
    """
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.size()[:-1] + (1, offset,),
                              dtype=torch.bool, device=x.device)
        return torch.cat((shifted, is_pad), dim=-1)


class FutureMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (...1, seq_len, seq_len + offset)
    ===========================================================================
    """
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(-1)

        # Create upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)
        return future.view((1,) * (x.ndim - 1) + future.size())
