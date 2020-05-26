import torch.nn as nn


class PositionwiseFeedForward(nn.Sequential):
    """Position-wise feed-forward layer.

    Arguments:
        dims (int): The dimension of input tensor.
        rate (int): The increase rate of dimensionality in bottleneck.
        dropout (float): The probability that each element is dropped.

    Note:
        The output tensor has same shape as the input tensor.
    """
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__(
            nn.Linear(dims, dims * rate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims * rate, dims))
