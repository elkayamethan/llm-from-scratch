from torch import nn, Tensor
from architecture.blocks.activation import GELU
from architecture.blocks.attention import MultiHeadAttention
from architecture.blocks.normalization import LayerNorm
from common.config.schemas import TransformerConfig


class Transformer(nn.Module):
    """A transformer block module"""

    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=cfg.embedding_dimension,
            d_out=cfg.embedding_dimension,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            drop_rate=cfg.drop_rate,
            kqv_bias=cfg.kqv_bias,
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.embedding_dimension, 4 * cfg.embedding_dimension),
            GELU(),
            nn.Linear(4 * cfg.embedding_dimension, cfg.embedding_dimension),
        )
        self.norm1 = LayerNorm(cfg.embedding_dimension)
        self.norm2 = LayerNorm(cfg.embedding_dimension)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
