import torch
from torch import nn, Tensor

class MultiHeadAttention(nn.Module):
    """A multi-head causal attention module with dropout and a final linear projection layer"""

    def __init__(
            self,
            d_in: int,
            d_out: int,
            num_heads: int,
            context_length: int,
            *,
            drop_rate: float = 0.5,
            kqv_bias: bool = False,
    ) -> None:
        super().__init__() 
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

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

        self.mask: Tensor
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
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
        context_vectors = context_vectors.reshape(batch_size, num_tokens, self.d_out)

        return self.final_proj(context_vectors)
