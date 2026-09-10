import math
import torch

from torch import nn, Tensor
from common.config.schemas import GPTConfig
from architecture.blocks import Transformer
from architecture.blocks.normalization import LayerNorm

class GPT(nn.Module):
    """A basic GPT-2 model skeleton, with weight tying."""

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

        self.out.weight = self.token_embedding.weight

        # Order matters
        self.apply(self._init_weights)
        self._init_transformer_blocks()

    def forward(self, idx: Tensor) -> Tensor:
        tok_embeds = self.token_embedding(idx)
        pos_embeds = self.positional_embedding(torch.arange(idx.shape[1], device=idx.device))

        x = tok_embeds + pos_embeds
        x = self.embedding_dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)

        return logits

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 initialization: N(0, 0.02) weights, zero biases"""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_transformer_blocks(self) -> None:
        for module in self.transformer_blocks:
            assert isinstance(module, Transformer), f"Expected 'Transformer' modules in self.transformer_blocks, received: {type(module).__name__}"
            nn.init.normal_(module.attention.final_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * len(self.transformer_blocks)))

            ff_final_linear = module.ff[2]
            assert isinstance(ff_final_linear, nn.Linear), f"Final layer in Transformer's feed-forward expected to be of type nn.Linear, received: {type(ff_final_linear).__name__}"
            nn.init.normal_(ff_final_linear.weight, mean=0.0, std=0.02 / math.sqrt(2 * len(self.transformer_blocks)))
