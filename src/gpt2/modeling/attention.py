import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class BaseAttention(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Calculate attention weight logits.
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            x += mask.float() * x.new_tensor(-1e9, dtype=torch.float32)

        # Apply softmax and dropout layer.
        x = self.dropout(x.softmax(-1))

        # Return weighted sum of values.
        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """
    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    past (tuple)
                    float           (..., past_len, dims)
                    float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    present (tuple)
                    float           (..., past_len + query_len, dims)
                    float           (..., past_len + query_len, dims)
    ===========================================================================
    """
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
