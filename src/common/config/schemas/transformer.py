from pydantic import BaseModel, ConfigDict, Field, model_validator


class TransformerConfig(BaseModel):
    """Configuration for a transformer block"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_length: int = Field(gt=0)
    embedding_dimension: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    drop_rate: float = Field(ge=0.0, lt=1.0)
    kqv_bias: bool

    @model_validator(mode="after")
    def check_heads_divide_embedding_dimension(self) -> "TransformerConfig":
        if self.embedding_dimension % self.n_heads != 0:
            raise ValueError("embedding_dimension must be divisible by n_heads")

        return self
