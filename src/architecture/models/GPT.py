import torch

from torch import nn, Tensor
from common.config.schemas import GPTConfig
from architecture.blocks import Transformer
from architecture.blocks.normalization import LayerNorm

class GPT(nn.Module):
    """A basic GPT model skeleton"""

    def __init__(
        self,
        cfg: GPTConfig,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.positional_embedding = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.embedding_dropout = nn.Dropout(cfg.drop_rate)

        self.transformer_blocks = nn.Sequential(*[Transformer(cfg) for _ in range(cfg.n_transformer_layers)])
        self.final_norm = LayerNorm(cfg.embedding_dim)
        self.out = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        tok_embeds = self.token_embedding(idx)
        pos_embeds = self.positional_embedding(torch.arange(idx.shape[1], device=idx.device))

        x = tok_embeds + pos_embeds
        x = self.embedding_dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)

        return logits
