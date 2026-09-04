from pydantic import BaseModel, ConfigDict, Field, model_validator

class TrainingConfig(BaseModel):
    """Configuration of a training run"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_epochs: int = Field(gt=0)
    max_lr: float = Field(gt=0)
    min_lr: float = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0)
    grad_clip_norm: float | None = Field(gt=0, default=1.0)
    eval_freq: int = Field(gt=0)
    eval_batches: int = Field(gt=0)

    @model_validator(mode="after")
    def check_lr(self) -> "TrainingConfig":
        if self.min_lr > self.max_lr:
            raise ValueError(f"max_lr >= min_lr must hold, received: min_lr={self.min_lr}, max_lr={self.max_lr}")

        return self
