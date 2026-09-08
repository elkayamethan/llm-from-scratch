import torch
import torch.nn.functional as F

from torch import nn, Tensor
from common.types import AttentionImpl
from typing import get_args


class MultiHeadAttention(nn.Module):
    """A multi-head causal attention module with dropout and a final linear projection layer"""

    def __init__(
            self,
            d_in: int,
            d_out: int,
            num_heads: int,
            context_length: int,
            *,
            drop_rate: float = 0.0,
            kqv_bias: bool = False,
            attention_impl: AttentionImpl = "manual",
    ) -> None:
        super().__init__() 
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")
        if attention_impl not in get_args(AttentionImpl):
            raise ValueError(f"attention_impl must be one of {get_args(AttentionImpl)}, received: {attention_impl}")

        self.attention_impl = attention_impl
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.W_k = nn.Linear(in_features=d_in, out_features=d_out, bias=kqv_bias)
        self.W_q = nn.Linear(in_features=d_in, out_features=d_out, bias=kqv_bias)
        self.W_v = nn.Linear(in_features=d_in, out_features=d_out, bias=kqv_bias)
        self.final_proj = nn.Linear(in_features=d_out, out_features=d_out)

        self.dropout = nn.Dropout(p=drop_rate)

        if attention_impl == "manual":
            self.mask: Tensor
            self.register_buffer(
                'mask',
                torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
                persistent=False,
            )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape
        if num_tokens > self.context_length:
            raise ValueError("num_tokens cannot exceed context_length")

        keys: Tensor = self.W_k(x)
        queries: Tensor = self.W_q(x)
        values: Tensor = self.W_v(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        if self.attention_impl == "manual":
            context_vectors = self._manual_attention(queries, keys, values, num_tokens)
        else:
            context_vectors = self._sdpa_attention(queries, keys, values)
        context_vectors = context_vectors.reshape(batch_size, num_tokens, self.d_out)

        return self.final_proj(context_vectors)


    def _manual_attention(self, queries: Tensor, keys: Tensor, values: Tensor, num_tokens: int) -> Tensor:
        """Manually computes causal attention"""

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores_masked = attention_scores.masked_fill(
            mask=self.mask[:num_tokens, :num_tokens], 
            value=-torch.inf,
        )
        attention_weights = torch.softmax(
            attention_scores_masked / self.head_dim**0.5,
            dim=-1,
        )
        attention_weights = self.dropout(attention_weights)
        context_vectors = (attention_weights @ values).transpose(1,2)
        
        return context_vectors

    def _sdpa_attention(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """Fused implementation using 'F.scaled_dot_product_attention', uses flash attention on CUDA with bf16/fp16.
        
        Note: falls back to unfused on MPS/CPU or when head_dim % 8 != 0 or head_dim > 256."""

        context_vectors = F.scaled_dot_product_attention(
            queries, 
            keys, 
            values,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        return context_vectors.transpose(1,2)
