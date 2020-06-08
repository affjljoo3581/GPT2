import torch.nn as nn


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__(
            nn.Linear(dims, dims * rate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims * rate, dims))
