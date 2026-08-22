import torch
from torch import nn, Tensor

# NOTE: Has been made obsolete by MultiHeadAttention
class CausalAttention(nn.Module):
    """A single-head causal attention module with dropout"""

    def __init__(
            self,
            d_in: int,
            d_out: int,
            context_length: int,
            *,
            dropout: float = 0.5,
            kqv_bias: bool = False,
    ) -> None:
        super().__init__() 
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length

        self.W_k = nn.Linear(in_features=d_in, out_features=d_out, bias=kqv_bias)
        self.W_q = nn.Linear(in_features=d_in, out_features=d_out, bias=kqv_bias)
        self.W_v = nn.Linear(in_features=d_in, out_features=d_out, bias=kqv_bias)

        self.dropout = nn.Dropout(p=dropout)

        self.mask: Tensor
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        num_tokens = x.shape[1]
        if num_tokens > self.context_length:
            raise ValueError("num_tokens cannot exceed context_length")

        keys: Tensor = self.W_k(x)
        queries: Tensor = self.W_q(x)
        values: Tensor = self.W_v(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores_masked = attention_scores.masked_fill(
            mask=self.mask[:num_tokens, :num_tokens], 
            value=-torch.inf,
        )
        attention_weights = torch.softmax(
            attention_scores_masked / self.d_out**0.5,
            dim=-1,
        )
        attention_weights = self.dropout(attention_weights)

        return attention_weights @ values
