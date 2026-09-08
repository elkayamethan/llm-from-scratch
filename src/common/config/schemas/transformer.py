from pydantic import BaseModel, ConfigDict, Field, model_validator
from common.types import AttentionImpl


class TransformerConfig(BaseModel):
    """Configuration for a transformer block"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_length: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    drop_rate: float = Field(ge=0.0, lt=1.0)
    kqv_bias: bool
    attention_impl: AttentionImpl

    @model_validator(mode="after")
    def check_embedding_dim_and_n_heads(self) -> "TransformerConfig":
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError(f"embedding_dim must be divisible by n_heads, received: embedding_dim={self.embedding_dim}, n_heads={self.n_heads}")

        head_dim = self.embedding_dim // self.n_heads
        if self.attention_impl == "sdpa" and (head_dim % 8 != 0 or head_dim > 256):
            raise ValueError(f"When attention_impl is 'sdpa', head_dim must be a multiple of 8 and <= 256, received: embedding_dim={self.embedding_dim}, n_heads={self.n_heads}")

        return self
