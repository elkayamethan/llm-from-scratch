from pydantic import BaseModel, ConfigDict, Field, model_validator

class TrainingConfig(BaseModel):
    """Training configuration"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_epochs: int = Field(gt=0)
    max_lr: float = Field(gt=0)
    min_lr: float = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0)
    grad_clip_norm: float | None = Field(gt=0, default=1.0)
    use_bf16: bool = True
    eval_freq: int = Field(gt=0)
    eval_batches: int = Field(gt=0)
    grad_accum_steps: int = Field(gt=0, default=1)
    compile: bool = False
    batch_size: int = Field(gt=0)
    eps: float = Field(gt=0, default=1e-8)
    checkpoint_freq: int = Field(gt=0)
    log_freq: int = Field(gt=0)
    betas: tuple[float, float] = (0.9, 0.95)

    @model_validator(mode="after")
    def check_lr(self) -> "TrainingConfig":
        if self.min_lr > self.max_lr:
            raise ValueError(f"max_lr >= min_lr must hold, received: min_lr={self.min_lr}, max_lr={self.max_lr}")

        return self

    @model_validator(mode="after")
    def check_betas(self) -> "TrainingConfig":
        beta1, beta2 = self.betas
        if beta1 < 0 or beta1 >= 1 or beta2 < 0 or beta2 >= 1:
            raise ValueError(f"betas must each be in [0, 1), received: {self.betas}")

        return self
