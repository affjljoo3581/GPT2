import torch
import torch.nn as nn


class PadMasking(nn.Module):
    """Implementation of pad-masking layer.

    Pad-masking tempts the model to avoid attending to values which are padded
    to the sequences.

    Arguments:
        pad_idx (int): The index of pad token to ignore in attention.
    """
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Create masking tensor to ignore paddings.

        Arguments:
            x (tensor): Input tensor of shape `(..., seq_len)`.
            offset (int): The offset of input sequences.

        Returns:
            A masking tensor of shape `(..., 1, seq_len + offset)`.
        """
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.size()[:-1] + (1, offset,),
                              dtype=torch.bool, device=x.device)
        return torch.cat((shifted, is_pad), dim=-1)


class FutureMasking(nn.Module):
    """Implementation of future-masking layer.

    Future-masking helps preventing the model from attending to the future
    tokens in the sequences.
    """
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Create masking tensor to ignore the future tokens.

        Arguments:
            x (tensor): Input tensor of shape `(..., seq_len)`.
            offset (int): The offset of input sequences.

        Returns:
            A masking tensor of shape `(...1, seq_len, seq_len + offset)`.
        """
        seq_len = x.size(-1)

        # Create upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)
        return future.view((1,) * (x.ndim - 1) + future.size())


class MaskingBlock(nn.Module):
    """Integrated masking layer.

    Arguments:
        pad_idx (int): The index of pad token to ignore in attention.
    """
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Create combined masking tensor.

        Arguments:
            x (tensor): Input tensor of shape `(..., seq_len)`.
            offset (int): The offset of input sequences.

        Returns:
            A masking tensor of shape `(..., seq_len, seq_len + offset)
        """
        return self.pad_masking(x, offset) + self.future_masking(x, offset)
