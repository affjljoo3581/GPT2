import torch
import torch.nn as nn


class PositionalEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # Create position indices tensor.
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        # Embed the position indices to vectors.
        return super().forward(position)


class TokenEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self,
                x: torch.Tensor,
                transposed: bool = False) -> torch.Tensor:
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))

        return super().forward(x)
