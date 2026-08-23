from pydantic import Field

from common.config.schemas.transformer import TransformerConfig


class GPTConfig(TransformerConfig):
    """Configuration for a GPT-style model"""

    vocab_size: int = Field(gt=0)
    n_transformer_layers: int = Field(gt=0)
