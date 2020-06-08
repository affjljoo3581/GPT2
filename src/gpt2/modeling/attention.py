import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class BaseAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate attention from the given queries, keys and values.

        Arguments:
            q (tensor): Input query tensor of shape `(..., query_len, dims)`.
            k (tensor): Input key tensor of shape `(..., kv_len, dims)`.
            v (tensor): Input value tensor of shape `(..., kv_len, dims)`.
            mask (tensor): Masking tensor of shape `(..., query_len, kv_len)`.

        Returns:
            An attention tensor which has the same shape as ``q``.
        """
        # Calculate attention weight logits.
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            x += mask.float() * x.new_tensor(-1e9, dtype=torch.float32)

        # Apply softmax and dropout layer.
        x = self.dropout(x.softmax(-1))

        # Return weighted sum of values.
        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate multi-headed attentions.

        Arguments:
            q (tensor): Input query tensor of shape `(..., query_len, dims)`.
            k (tensor): Input key tensor of shape `(..., kv_len, dims)`.
            v (tensor): Input value tensor of shape `(..., kv_len, dims)`.
            mask (tensor): Masking tensor of shape `(..., query_len, kv_len)`.

        Returns:
            An attention tensor which has the same shape as ``q``.
        """
        # Split each input tensor into multi-heads.
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads))
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads))
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        # Add new axis to the masking tensor.
        if mask is not None:
            mask = mask.unsqueeze(-3)

        # Calculate attentions and merge multi-heads.
        return (super().forward(q, k, v, mask)
                .transpose(-3, -2)
                .contiguous()
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads)))


class AttentionBlock(nn.Module):
    def __init__(self, heads: int, dims: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                past: Optional[Tuple[torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Calculate multi-headed attentions with linear projections.

        Arguments:
            q (tensor): Input query tensor of shape `(..., query_len, dims)`.
            k (tensor): Input key tensor of shape `(..., kv_len, dims)`.
            v (tensor): Input value tensor of shape `(..., kv_len, dims)`.
            past (tuple): The tuple of previously calculated key-value tensors
                of shape `(..., past_len, dims)`.
            mask (tensor): Masking tensor of shape `(..., query_len, kv_len)`.

        Returns:
            * An attention tensor which has same shape as ``q``.
            * A tuple of projected key-value tensors of shape
              `(..., past_len + kv_len, dims)`.
        """
        # Project input tensors.
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        # Reuse previously calculated keys and values.
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        # Calculate multi-headed attention and apply linear projection.
        x = self.linear(self.attn(q, k, v, mask))

        return x, (k, v)
